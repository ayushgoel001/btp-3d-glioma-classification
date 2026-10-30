"""Download original BraTS 2020 training files from the public Kaggle mirror.

The official UPenn page remains the primary dataset source. This helper exists
only because the historical registration endpoint may be inconvenient for a
time-limited reproducibility run. Review the mirror's terms before use.
"""

from __future__ import annotations

import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import kagglehub
import nibabel as nib
import pandas as pd
from tqdm import tqdm

HANDLE = "awsaf49/brats20-dataset-training-validation"
PREFIX = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")
REMOTE_OVERRIDES = {
    ("BraTS20_Training_355", "seg"): f"{PREFIX}/BraTS20_Training_355/W39_1998.09.19_Segm.nii"
}


def _relative_file(patient_id: str, modality: str) -> str:
    return f"{PREFIX}/{patient_id}/{patient_id}_{modality}.nii"


def _already_available(output_dir: Path, patient_id: str, modality: str) -> bool:
    filename = f"{patient_id}_{modality}.nii"
    candidates = (
        output_dir / patient_id / filename,
        output_dir / PREFIX / patient_id / filename,
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            nib.load(path)
        except (nib.filebasedimages.ImageFileError, OSError):
            continue
        return True
    return False


def download(mapping_csv: str | Path, output_dir: str | Path, workers: int = 8) -> None:
    # Use direct file downloads. Colab's cache resolver can ignore output_dir
    # and fail through its local /kagglehub/datasets/{is_supported,mount} service.
    # KaggleHub checks this setting when selecting a resolver for each request.
    os.environ["DISABLE_COLAB_CACHE"] = "true"
    output_dir = Path(output_dir)
    mapping = pd.read_csv(mapping_csv)
    patient_ids = mapping["BraTS_2020_subject_ID"].astype(str).tolist()
    tasks = [
        (patient_id, modality)
        for patient_id in patient_ids
        for modality in MODALITIES
        if not _already_available(output_dir, patient_id, modality)
    ]
    failures: list[tuple[str, str, str]] = []

    def fetch(patient_id: str, modality: str) -> str:
        remote_path = REMOTE_OVERRIDES.get(
            (patient_id, modality), _relative_file(patient_id, modality)
        )
        downloaded = kagglehub.dataset_download(
            HANDLE,
            path=remote_path,
            force_download=True,
            output_dir=str(output_dir),
        )
        if (patient_id, modality) in REMOTE_OVERRIDES:
            destination = output_dir / PREFIX / patient_id / f"{patient_id}_{modality}.nii"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, destination)
        return downloaded

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(fetch, patient_id, modality): (patient_id, modality)
            for patient_id, modality in tasks
        }
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="BraTS files"):
            patient_id, modality = future_to_task[future]
            try:
                future.result()
            # KaggleHub can surface transport, archive, or filesystem exceptions;
            # keep the other independent downloads running and report every failure.
            except Exception as error:  # noqa: BLE001
                failures.append((patient_id, modality, str(error)))
    if failures:
        preview = "\n".join(f"{patient_id}/{modality}: {error}" for patient_id, modality, error in failures[:20])
        raise RuntimeError(f"{len(failures)} downloads failed; rerun to retry.\n{preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download original BraTS 2020 NIfTI training files")
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.mapping_csv, args.output_dir, args.workers)
