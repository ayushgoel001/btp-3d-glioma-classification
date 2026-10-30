from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import MODALITIES


def create_qa_figure(cache_path: str | Path, output_path: str | Path) -> None:
    with np.load(cache_path, allow_pickle=False) as payload:
        image = payload["image"].astype(np.float32)
        segmentation = payload.get("seg", None)
        metadata = json.loads(payload["metadata"].item())
    if image.shape[0] != 4:
        raise ValueError(f"Expected four modalities, received {image.shape}")
    if segmentation is not None and np.any(segmentation > 0):
        tumor_by_slice = (segmentation > 0).sum(axis=(0, 1))
        slice_index = int(tumor_by_slice.argmax())
    else:
        slice_index = image.shape[-1] // 2
    figure, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for channel, modality in enumerate(MODALITIES):
        axes[channel].imshow(image[channel, :, :, slice_index].T, cmap="gray", origin="lower")
        axes[channel].set_title(modality.upper())
        axes[channel].axis("off")
    axes[4].imshow(image[0, :, :, slice_index].T, cmap="gray", origin="lower")
    if segmentation is not None:
        overlay = np.ma.masked_where(
            segmentation[:, :, slice_index].T == 0, segmentation[:, :, slice_index].T
        )
        axes[4].imshow(overlay, cmap="autumn", alpha=0.55, origin="lower", interpolation="nearest")
    axes[4].set_title("FLAIR + mask")
    axes[4].axis("off")
    figure.suptitle(
        f"{metadata['patient_id']} | {metadata['label_name']} | crop {metadata['cropped_shape']} -> {image.shape[1:]}"
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create visual QA panels from cached patient tensors"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-split", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest)
    for split, records in frame.groupby("split"):
        for row in records.head(args.per_split).itertuples(index=False):
            create_qa_figure(
                row.cache_path, Path(args.output_dir) / f"{split}_{row.patient_id}.png"
            )
    print(f"QA figures written to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
