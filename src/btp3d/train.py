from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_curve
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CachedBraTSDataset, assert_disjoint_splits
from .metrics import classification_metrics
from .model import CompactResNet3D, ModelConfig
from .utils import INDEX_TO_LABEL, save_json, seed_everything, seed_worker


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 60
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    num_workers: int = 2
    amp: bool = True


def _make_loader(records: pd.DataFrame, config: TrainingConfig, training: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        CachedBraTSDataset(records, augment=training),
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    accumulation: int = 1,
    amp: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    targets: list[int] = []
    probabilities: list[float] = []
    patient_ids: list[str] = []
    for step, batch in enumerate(loader):
        inputs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)
                backward_loss = loss / accumulation
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(backward_loss).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        probabilities.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy().tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())
        patient_ids.extend(list(batch["patient_id"]))
    return float(np.mean(losses)), np.asarray(targets), np.asarray(probabilities), patient_ids


def _plots(
    history: pd.DataFrame, predictions: pd.DataFrame, metrics: dict[str, Any], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(history["epoch"], history["train_loss"], label="Train")
    axis.plot(history["epoch"], history["val_loss"], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Training history")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(metrics["confusion_matrix_lgg_hgg"])
    figure, axis = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set(xlabel="Predicted", ylabel="True", title="Patient-level confusion matrix")
    axis.set_xticklabels(["LGG", "HGG"])
    axis.set_yticklabels(["LGG", "HGG"], rotation=0)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    if predictions["label"].nunique() == 2:
        false_positive, true_positive, _ = roc_curve(
            predictions["label"], predictions["probability_hgg"]
        )
        figure, axis = plt.subplots(figsize=(5, 4.5))
        axis.plot(false_positive, true_positive, label=f"AUC = {metrics['roc_auc']:.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set(
            xlabel="False-positive rate", ylabel="True-positive rate", title="Patient-level ROC"
        )
        axis.legend(loc="lower right")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "roc_curve.png", dpi=180)
        plt.close(figure)


def train_model(
    cached_manifest: str | Path,
    output_dir: str | Path,
    model_config: ModelConfig,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    seed_everything(training_config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(cached_manifest)
    assert_disjoint_splits(frame)
    train_records = frame[frame.split == "train"].copy()
    validation_records = frame[frame.split == "val"].copy()
    test_records = frame[frame.split == "test"].copy()
    for name, records in (
        ("train", train_records),
        ("val", validation_records),
        ("test", test_records),
    ):
        if records["label"].nunique() != 2:
            raise ValueError(f"{name} split does not contain both classes")

    train_loader = _make_loader(train_records, training_config, training=True)
    validation_loader = _make_loader(validation_records, training_config, training=False)
    test_loader = _make_loader(test_records, training_config, training=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CompactResNet3D(model_config).to(device)
    class_counts = train_records["label"].value_counts().reindex([0, 1]).to_numpy(dtype=np.float32)
    class_weights = class_counts.sum() / (2.0 * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=training_config.amp and device.type == "cuda")

    history_rows: list[dict[str, float]] = []
    best_score = -math.inf
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"
    started = time.time()
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_true, train_probability, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation=training_config.gradient_accumulation,
            amp=training_config.amp,
        )
        validation_loss, validation_true, validation_probability, _ = _run_epoch(
            model, validation_loader, criterion, device, amp=training_config.amp
        )
        train_metrics = classification_metrics(train_true, train_probability)
        validation_metrics = classification_metrics(validation_true, validation_probability)
        score = float(validation_metrics["balanced_accuracy"])
        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "val_balanced_accuracy": score,
                "val_macro_f1": validation_metrics["macro_f1"],
                "val_roc_auc": validation_metrics["roc_auc"],
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={validation_loss:.4f} "
            f"val_bal_acc={score:.4f} val_auc={validation_metrics['roc_auc']:.4f}"
        )
        if score > best_score + 1e-6:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.export_config(),
                    "training_config": asdict(training_config),
                    "preprocess_config": json.loads(
                        np.load(train_records.iloc[0].cache_path)["metadata"].item()
                    )["preprocess_config"],
                    "class_to_index": {"LGG": 0, "HGG": 1},
                    "best_validation_balanced_accuracy": best_score,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if (
            training_config.early_stopping_patience > 0
            and stale_epochs >= training_config.early_stopping_patience
        ):
            print(f"Early stopping after {epoch} epochs")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_true, test_probability, patient_ids = _run_epoch(
        model, test_loader, criterion, device, amp=training_config.amp
    )
    test_metrics = classification_metrics(test_true, test_probability)
    test_metrics.update(
        {
            "test_loss": test_loss,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_balanced_accuracy": float(
                checkpoint["best_validation_balanced_accuracy"]
            ),
            "elapsed_seconds": float(time.time() - started),
            "device": str(device),
            "seed": training_config.seed,
            "class_weights_lgg_hgg": class_weights.tolist(),
            "split_counts": {
                name: records["label_name"].value_counts().to_dict()
                for name, records in (
                    ("train", train_records),
                    ("val", validation_records),
                    ("test", test_records),
                )
            },
        }
    )
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "label": test_true,
            "label_name": [INDEX_TO_LABEL[int(value)] for value in test_true],
            "probability_hgg": test_probability,
            "prediction": (test_probability >= 0.5).astype(int),
            "prediction_name": [INDEX_TO_LABEL[int(value)] for value in (test_probability >= 0.5)],
        }
    )
    predictions["correct"] = predictions["label"] == predictions["prediction"]
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "history.csv", index=False)
    save_json(test_metrics, output_dir / "test_metrics.json")
    _plots(history, predictions, test_metrics, output_dir)
    return test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact 3D ResNet baseline")
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
    model_config = ModelConfig(
        base_channels=args.base_channels,
        group_norm_groups=args.group_norm_groups,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        amp=not args.no_amp,
    )
    metrics = train_model(args.manifest, args.output_dir, model_config, training_config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
