from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .utils import LABEL_TO_INDEX, MODALITIES


def _find_volume(subject_dir: Path, patient_id: str, suffix: str) -> Path | None:
    candidates = [
        subject_dir / f"{patient_id}_{suffix}.nii.gz",
        subject_dir / f"{patient_id}_{suffix}.nii",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            # Reject interrupted mirror downloads early. A filename alone is
            # not evidence that the NIfTI header is readable.
            nib.load(path)
        except (nib.filebasedimages.ImageFileError, OSError):
            continue
        return path.resolve()
    return None


def _subject_directories(data_root: Path) -> dict[str, list[Path]]:
    """Index every directory for a subject.

    Public mirrors sometimes mix extracted, archived, and nested layouts.  A
    subject may therefore have modalities distributed across two otherwise
    equivalent directories; retaining every candidate avoids depending on
    filesystem traversal order.
    """
    directories: dict[str, list[Path]] = {}
    for path in data_root.rglob("BraTS20_Training_*"):
        if path.is_dir():
            directories.setdefault(path.name, []).append(path)
    return directories


def _find_subject_volume(
    subject_dirs: list[Path], patient_id: str, suffix: str
) -> Path | None:
    for subject_dir in subject_dirs:
        volume = _find_volume(subject_dir, patient_id, suffix)
        if volume is not None:
            return volume
    return None


def _assign_fixed_splits(
    frame: pd.DataFrame, seed: int, val_fraction: float, test_fraction: float
) -> pd.DataFrame:
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be positive and sum to less than 1")
    train_val, test = train_test_split(
        frame.index,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame["label"],
    )
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=frame.loc[train_val, "label"],
    )
    frame = frame.copy()
    frame["split"] = ""
    frame.loc[train, "split"] = "train"
    frame.loc[val, "split"] = "val"
    frame.loc[test, "split"] = "test"
    return frame


def build_manifest(
    data_root: str | Path,
    mapping_csv: str | Path,
    output_csv: str | Path,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    folds: int = 0,
    strict: bool = False,
) -> pd.DataFrame:
    data_root = Path(data_root)
    mapping = pd.read_csv(mapping_csv)
    required_mapping = {"Grade", "BraTS_2020_subject_ID"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError(f"Mapping CSV must contain {sorted(required_mapping)}")
    if mapping["BraTS_2020_subject_ID"].duplicated().any():
        raise ValueError("Mapping CSV contains duplicate BraTS 2020 patient IDs")
    unexpected = set(mapping["Grade"].dropna().unique()) - set(LABEL_TO_INDEX)
    if unexpected:
        raise ValueError(f"Unexpected grade labels: {sorted(unexpected)}")

    subject_dirs = _subject_directories(data_root)
    rows: list[dict[str, object]] = []
    incomplete: list[str] = []
    for record in mapping.itertuples(index=False):
        patient_id = str(record.BraTS_2020_subject_ID)
        grade = str(record.Grade)
        patient_dirs = subject_dirs.get(patient_id)
        if patient_dirs is None:
            incomplete.append(patient_id)
            continue
        paths = {
            modality: _find_subject_volume(patient_dirs, patient_id, modality)
            for modality in MODALITIES
        }
        paths["seg"] = _find_subject_volume(patient_dirs, patient_id, "seg")
        if any(path is None for path in paths.values()):
            incomplete.append(patient_id)
            continue
        rows.append(
            {
                "patient_id": patient_id,
                "label": LABEL_TO_INDEX[grade],
                "label_name": grade,
                **{name: str(path) for name, path in paths.items()},
                "dataset": "BraTS2020_training",
            }
        )

    if strict and incomplete:
        raise FileNotFoundError(
            f"Missing or incomplete subjects ({len(incomplete)}): {incomplete[:10]}"
        )
    if len(rows) < 4 or len({row["label"] for row in rows}) < 2:
        raise ValueError("At least two complete patients from each of two classes are required")

    frame = pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)
    if frame["patient_id"].duplicated().any():
        raise AssertionError("A patient appears more than once in the manifest")
    frame = _assign_fixed_splits(frame, seed, val_fraction, test_fraction)
    frame["fold"] = -1
    if folds:
        if folds < 2:
            raise ValueError("folds must be zero or at least two")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (_, validation_indices) in enumerate(splitter.split(frame, frame["label"])):
            frame.loc[validation_indices, "fold"] = fold

    frame["split_seed"] = seed
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def audit_manifest(manifest_csv: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    audit_rows: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        images = {name: nib.load(getattr(row, name)) for name in (*MODALITIES, "seg")}
        shapes = {name: image.shape for name, image in images.items()}
        spacings = {
            name: tuple(float(x) for x in image.header.get_zooms()[:3])
            for name, image in images.items()
        }
        reference = images[MODALITIES[0]]
        shapes_match = len(set(shapes.values())) == 1
        affines_match = all(
            np.allclose(reference.affine, image.affine, atol=1e-3) for image in images.values()
        )
        spacings_match = len(set(spacings.values())) == 1
        affine_hash = hashlib.sha256(
            np.asarray(reference.affine, dtype=np.float64).tobytes()
        ).hexdigest()[:16]
        seg_values = sorted(int(value) for value in np.unique(images["seg"].dataobj))
        audit_rows.append(
            {
                "patient_id": row.patient_id,
                "shape": "x".join(str(value) for value in reference.shape),
                "spacing": "x".join(f"{value:g}" for value in spacings[MODALITIES[0]]),
                "orientation": "".join(nib.aff2axcodes(reference.affine)),
                "shapes_match": shapes_match,
                "affines_match": affines_match,
                "spacings_match": spacings_match,
                "seg_values": ";".join(str(value) for value in seg_values),
                "affine_hash": affine_hash,
                "valid": shapes_match
                and affines_match
                and spacings_match
                and set(seg_values).issubset({0, 1, 2, 4}),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if output_csv:
        destination = Path(output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(destination, index=False)
    if not audit["valid"].all():
        invalid = audit.loc[~audit["valid"], "patient_id"].tolist()
        raise ValueError(f"Manifest audit failed for: {invalid}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally audit a BraTS 2020 patient manifest"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--audit-output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.data_root,
        args.mapping_csv,
        args.output,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        folds=args.folds,
        strict=args.strict,
    )
    print(manifest.groupby(["split", "label_name"]).size().unstack(fill_value=0))
    print(f"Wrote {len(manifest)} patient records to {args.output}")
    if args.audit_output:
        audit = audit_manifest(args.output, args.audit_output)
        print(f"Audited {len(audit)} patients; all modality geometry checks passed")


if __name__ == "__main__":
    main()
