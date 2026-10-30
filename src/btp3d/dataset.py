from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class RandomMRI3DAugment:
    """Conservative training-only transforms for RAS-oriented volumes."""

    def __init__(
        self, flip_probability: float = 0.5, noise_std: float = 0.03, intensity_scale: float = 0.10
    ):
        self.flip_probability = flip_probability
        self.noise_std = noise_std
        self.intensity_scale = intensity_scale

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # After canonical RAS conversion, tensor dimension 1 is the left-right axis.
        if random.random() < self.flip_probability:
            image = torch.flip(image, dims=(1,))
        if self.intensity_scale > 0:
            scales = torch.empty((image.shape[0], 1, 1, 1)).uniform_(
                1.0 - self.intensity_scale, 1.0 + self.intensity_scale
            )
            image = image * scales
        if self.noise_std > 0 and random.random() < 0.5:
            foreground = image != 0
            image = image + torch.randn_like(image) * self.noise_std * foreground
        return image


class CachedBraTSDataset(Dataset[dict[str, object]]):
    def __init__(self, records: pd.DataFrame, augment: bool = False):
        if records["patient_id"].duplicated().any():
            raise ValueError("Dataset contains duplicate patient IDs")
        if not records["cache_path"].map(lambda value: Path(value).is_file()).all():
            missing = records.loc[
                ~records["cache_path"].map(lambda value: Path(value).is_file()), "patient_id"
            ].tolist()
            raise FileNotFoundError(f"Missing cached tensors for: {missing[:10]}")
        self.records = records.reset_index(drop=True).copy()
        self.transform = RandomMRI3DAugment() if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records.iloc[index]
        with np.load(row.cache_path, allow_pickle=False) as payload:
            image = torch.from_numpy(payload["image"].astype(np.float32))
            cached_label = int(payload["label"])
        if cached_label != int(row.label):
            raise ValueError(f"Cached/manifest label mismatch for {row.patient_id}")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": cached_label, "patient_id": str(row.patient_id)}


def assert_disjoint_splits(frame: pd.DataFrame) -> None:
    if frame["patient_id"].duplicated().any():
        duplicate_ids = (
            frame.loc[frame["patient_id"].duplicated(False), "patient_id"].unique().tolist()
        )
        raise ValueError(f"Patient leakage: IDs occur in multiple rows: {duplicate_ids[:10]}")
    expected = {"train", "val", "test"}
    observed = set(frame["split"].unique())
    if not expected.issubset(observed):
        raise ValueError(f"Expected train/val/test splits, observed {sorted(observed)}")
