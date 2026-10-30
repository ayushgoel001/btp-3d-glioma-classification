from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.nn import functional
from tqdm import tqdm

from .utils import MODALITIES, sha256_file


@dataclass(frozen=True)
class PreprocessConfig:
    spatial_size: tuple[int, int, int] = (96, 96, 96)
    crop_mode: str = "roi"
    margin: int = 16
    canonical_orientation: bool = True

    def validate(self) -> None:
        if len(self.spatial_size) != 3 or any(value < 16 for value in self.spatial_size):
            raise ValueError("spatial_size must contain three integers >= 16")
        if self.crop_mode not in {"roi", "brain"}:
            raise ValueError("crop_mode must be 'roi' or 'brain'")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")


def _load_canonical(path: str, canonical: bool) -> nib.Nifti1Image:
    image = nib.load(path)
    return nib.as_closest_canonical(image) if canonical else image


def _validate_geometry(images: dict[str, nib.Nifti1Image]) -> None:
    reference = images[MODALITIES[0]]
    for name, image in images.items():
        if image.shape != reference.shape:
            raise ValueError(f"Shape mismatch: {name}={image.shape}, reference={reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=1e-3):
            raise ValueError(f"Affine mismatch for modality {name}")
        if not np.allclose(
            image.header.get_zooms()[:3], reference.header.get_zooms()[:3], atol=1e-4
        ):
            raise ValueError(f"Voxel-spacing mismatch for modality {name}")


def _bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot crop an empty foreground mask")
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))  # type: ignore[return-value]


def _normalize_nonzero(volume: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    normalized = np.zeros_like(volume, dtype=np.float32)
    foreground = volume != 0
    values = volume[foreground].astype(np.float32)
    if values.size == 0:
        raise ValueError("MRI modality contains no non-zero voxels")
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        raise ValueError("MRI modality has near-zero variance in its foreground")
    normalized[foreground] = (values - mean) / std
    return normalized, {"mean": mean, "std": std, "nonzero_voxels": int(values.size)}


def _resize(array: np.ndarray, spatial_size: tuple[int, int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))[None]
    align_corners = False if mode == "trilinear" else None
    resized = functional.interpolate(
        tensor, size=spatial_size, mode=mode, align_corners=align_corners
    )
    return resized[0].cpu().numpy()


def preprocess_paths(
    modality_paths: dict[str, str],
    seg_path: str | None,
    config: PreprocessConfig,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    config.validate()
    images = {
        name: _load_canonical(path, config.canonical_orientation)
        for name, path in modality_paths.items()
    }
    if set(images) != set(MODALITIES):
        raise ValueError(f"Exactly these modalities are required: {MODALITIES}")
    if seg_path:
        images["seg"] = _load_canonical(seg_path, config.canonical_orientation)
    _validate_geometry(images)

    reference = images[MODALITIES[0]]
    arrays = {name: np.asarray(images[name].dataobj, dtype=np.float32) for name in MODALITIES}
    segmentation = np.asarray(images["seg"].dataobj, dtype=np.int16) if "seg" in images else None
    if config.crop_mode == "roi":
        if segmentation is None:
            raise ValueError("ROI preprocessing requires a segmentation mask")
        crop_mask = segmentation > 0
    else:
        crop_mask = np.logical_or.reduce([array != 0 for array in arrays.values()])
    crop = _bbox(crop_mask, config.margin)

    normalized: list[np.ndarray] = []
    normalization: dict[str, dict[str, float]] = {}
    for name in MODALITIES:
        array, statistics = _normalize_nonzero(arrays[name])
        normalized.append(array[crop])
        normalization[name] = statistics
    stacked = np.stack(normalized, axis=0)
    processed = _resize(stacked, config.spatial_size, "trilinear").astype(np.float32)
    processed_seg = None
    if segmentation is not None:
        resized_seg = _resize(
            segmentation[crop][None].astype(np.float32), config.spatial_size, "nearest"
        )
        processed_seg = resized_seg[0].astype(np.uint8)

    metadata: dict[str, Any] = {
        "preprocess_config": asdict(config),
        "original_shape": list(reference.shape),
        "original_spacing": [float(value) for value in reference.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(reference.affine)),
        "crop_bounds": [[part.start, part.stop] for part in crop],
        "cropped_shape": list(stacked.shape[1:]),
        "normalization": normalization,
        "modalities": list(MODALITIES),
    }
    return processed, processed_seg, metadata


def _cache_one(
    record: dict[str, Any], output_dir: str, config: PreprocessConfig, overwrite: bool
) -> dict[str, Any]:
    patient_id = str(record["patient_id"])
    destination = Path(output_dir) / f"{patient_id}.npz"
    if destination.exists() and not overwrite:
        return {
            "patient_id": patient_id,
            "cache_path": str(destination.resolve()),
            "status": "existing",
        }
    modality_paths = {name: str(record[name]) for name in MODALITIES}
    image, segmentation, metadata = preprocess_paths(modality_paths, str(record["seg"]), config)
    metadata.update(
        {
            "patient_id": patient_id,
            "label": int(record["label"]),
            "label_name": str(record["label_name"]),
            "source_sha256": {
                name: sha256_file(path)
                for name, path in {**modality_paths, "seg": str(record["seg"])}.items()
            },
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        payload: dict[str, Any] = {
            "image": image.astype(np.float16),
            "label": np.asarray(int(record["label"]), dtype=np.int64),
            "metadata": np.asarray(json.dumps(metadata)),
        }
        if segmentation is not None:
            payload["seg"] = segmentation
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"patient_id": patient_id, "cache_path": str(destination.resolve()), "status": "written"}


def preprocess_manifest(
    manifest_csv: str | Path,
    output_dir: str | Path,
    config: PreprocessConfig,
    workers: int = 1,
    overwrite: bool = False,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    required = {"patient_id", "label", "label_name", "seg", *MODALITIES}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    records = manifest.to_dict(orient="records")
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for record in tqdm(records, desc="Preprocessing patients"):
            results.append(_cache_one(record, str(output_dir), config, overwrite))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_cache_one, record, str(output_dir), config, overwrite)
                for record in records
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Preprocessing patients"
            ):
                results.append(future.result())
    cache = pd.DataFrame(results)
    enriched = manifest.merge(
        cache[["patient_id", "cache_path"]], on="patient_id", how="left", validate="one_to_one"
    )
    output_manifest = Path(output_dir) / "manifest_cached.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_manifest, index=False)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and cache BraTS patient volumes")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spatial-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--crop-mode", choices=("roi", "brain"), default="roi")
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(tuple(args.spatial_size), args.crop_mode, args.margin)
    manifest = preprocess_manifest(
        args.manifest, args.output_dir, config, args.workers, args.overwrite
    )
    print(f"Cached {len(manifest)} patients at {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
