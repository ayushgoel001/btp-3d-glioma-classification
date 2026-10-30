from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from btp3d.dataset import CachedBraTSDataset, assert_disjoint_splits
from btp3d.manifest import build_manifest
from btp3d.metrics import classification_metrics
from btp3d.model import CompactResNet3D, ModelConfig
from btp3d.preprocess import PreprocessConfig, preprocess_manifest, preprocess_paths


def _write_subject(root: Path, patient_id: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    subject = root / patient_id
    subject.mkdir(parents=True)
    shape = (24, 26, 22)
    grid = np.indices(shape)
    brain = ((grid[0] - 12) ** 2 / 100 + (grid[1] - 13) ** 2 / 120 + (grid[2] - 11) ** 2 / 80) < 1
    tumor = ((grid[0] - 13) ** 2 + (grid[1] - 12) ** 2 + (grid[2] - 10) ** 2) < 12
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    for index, modality in enumerate(("flair", "t1", "t1ce", "t2")):
        volume = np.zeros(shape, dtype=np.float32)
        volume[brain] = rng.normal(100 + index * 20, 15, size=brain.sum())
        volume[tumor] += 30 + index * 5
        nib.save(nib.Nifti1Image(volume, affine), subject / f"{patient_id}_{modality}.nii.gz")
    segmentation = np.zeros(shape, dtype=np.uint8)
    segmentation[tumor] = 2
    nib.save(nib.Nifti1Image(segmentation, affine), subject / f"{patient_id}_seg.nii.gz")


def test_preprocessing_and_model(tmp_path: Path) -> None:
    patient_id = "BraTS20_Training_001"
    _write_subject(tmp_path, patient_id, 1)
    paths = {
        name: str(tmp_path / patient_id / f"{patient_id}_{name}.nii.gz")
        for name in ("flair", "t1", "t1ce", "t2")
    }
    image, segmentation, metadata = preprocess_paths(
        paths,
        str(tmp_path / patient_id / f"{patient_id}_seg.nii.gz"),
        PreprocessConfig((32, 32, 32), "roi", 3),
    )
    assert image.shape == (4, 32, 32, 32)
    assert segmentation is not None and segmentation.shape == (32, 32, 32)
    assert set(np.unique(segmentation)).issubset({0, 2})
    assert metadata["orientation"] == ["R", "A", "S"]
    model = CompactResNet3D(ModelConfig(base_channels=4, group_norm_groups=4))
    output = model(torch.from_numpy(image)[None])
    assert output.shape == (1, 2)
    output.sum().backward()


def test_manifest_cache_and_metrics(tmp_path: Path) -> None:
    mapping_rows = []
    for index in range(1, 9):
        patient_id = f"BraTS20_Training_{index:03d}"
        _write_subject(tmp_path / "raw", patient_id, index)
        mapping_rows.append(
            {"Grade": "LGG" if index <= 4 else "HGG", "BraTS_2020_subject_ID": patient_id}
        )
    mapping = tmp_path / "mapping.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping, index=False)
    manifest_path = tmp_path / "manifest.csv"
    manifest = build_manifest(
        tmp_path / "raw", mapping, manifest_path, val_fraction=0.25, test_fraction=0.25
    )
    assert_disjoint_splits(manifest)
    cached = preprocess_manifest(
        manifest_path,
        tmp_path / "cache",
        PreprocessConfig((24, 24, 24), "brain", 0),
    )
    dataset = CachedBraTSDataset(cached[cached.split == "train"])
    assert len(dataset) == 4
    assert dataset[0]["image"].shape == (4, 24, 24, 24)
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.7, 0.9]))
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
