from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import classification_metrics
from .model import ModelConfig
from .train import TrainingConfig, train_model
from .utils import save_json


def cross_validate(
    cached_manifest: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict[str, object]:
    """Run deterministic patient-level rotating train/validation/test folds.

    Fold k is the held-out test fold, fold (k + 1) mod K is used only for
    checkpoint selection, and all remaining folds are training data. Every
    patient therefore appears in a test set exactly once.
    """
    frame = pd.read_csv(cached_manifest)
    folds = sorted(int(value) for value in frame["fold"].unique() if int(value) >= 0)
    if len(folds) < 3 or folds != list(range(len(folds))):
        raise ValueError("Manifest must contain contiguous fold IDs and at least three folds")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics: list[dict[str, object]] = []
    out_of_fold_predictions: list[pd.DataFrame] = []
    for test_fold in folds:
        validation_fold = (test_fold + 1) % len(folds)
        fold_frame = frame.copy()
        fold_frame["split"] = "train"
        fold_frame.loc[fold_frame.fold == validation_fold, "split"] = "val"
        fold_frame.loc[fold_frame.fold == test_fold, "split"] = "test"
        fold_dir = output_dir / f"fold_{test_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest = fold_dir / "manifest.csv"
        fold_frame.to_csv(fold_manifest, index=False)
        metrics = train_model(fold_manifest, fold_dir, model_config, training_config)
        metrics["fold"] = test_fold
        metrics["validation_fold"] = validation_fold
        fold_metrics.append(metrics)
        predictions = pd.read_csv(fold_dir / "test_predictions.csv")
        predictions["fold"] = test_fold
        out_of_fold_predictions.append(predictions)

    predictions = pd.concat(out_of_fold_predictions, ignore_index=True)
    if predictions["patient_id"].duplicated().any() or len(predictions) != len(frame):
        raise AssertionError("Out-of-fold predictions must contain every patient exactly once")
    predictions.to_csv(output_dir / "out_of_fold_predictions.csv", index=False)
    aggregate = classification_metrics(
        predictions["label"].to_numpy(), predictions["probability_hgg"].to_numpy()
    )
    keys = ("balanced_accuracy", "macro_f1", "sensitivity_hgg", "specificity_lgg", "roc_auc")
    aggregate["fold_summary"] = {
        key: {
            "mean": float(np.mean([float(metrics[key]) for metrics in fold_metrics])),
            "std": float(np.std([float(metrics[key]) for metrics in fold_metrics], ddof=1)),
        }
        for key in keys
    }
    aggregate["fold_metrics"] = fold_metrics
    aggregate["seed"] = training_config.seed
    save_json(aggregate, output_dir / "cross_validation_metrics.json")
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rotating patient-level cross-validation")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--group-norm-groups", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = cross_validate(
        args.manifest,
        args.output_dir,
        ModelConfig(
            base_channels=args.base_channels,
            group_norm_groups=args.group_norm_groups,
            dropout=args.dropout,
        ),
        TrainingConfig(
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.patience,
            num_workers=args.num_workers,
            amp=not args.no_amp,
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
