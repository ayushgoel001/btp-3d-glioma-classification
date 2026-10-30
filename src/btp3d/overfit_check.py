from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CachedBraTSDataset
from .model import CompactResNet3D, ModelConfig
from .utils import save_json, seed_everything


def run_overfit_check(
    cached_manifest: str | Path,
    output_dir: str | Path,
    samples_per_class: int = 4,
    epochs: int = 200,
    seed: int = 42,
    base_channels: int = 4,
) -> dict[str, object]:
    """Memorize a tiny balanced subset to detect optimization/pipeline bugs."""
    seed_everything(seed)
    frame = pd.read_csv(cached_manifest)
    selected = pd.concat(
        [group.sample(samples_per_class, random_state=seed) for _, group in frame.groupby("label")],
        ignore_index=True,
    )
    if len(selected) != samples_per_class * 2:
        raise ValueError("Not enough cached patients from both classes")
    dataset = CachedBraTSDataset(selected, augment=False)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig(base_channels=base_channels, group_norm_groups=4, dropout=0.0)
    model = CompactResNet3D(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    final_loss = float("inf")
    final_accuracy = 0.0
    completed_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
            logits = model(batch["image"].to(device))
            labels = batch["label"].to(device)
            final_loss = float(criterion(logits, labels).cpu())
            final_accuracy = float((logits.argmax(1) == labels).float().mean().cpu())
        completed_epoch = epoch
        if final_accuracy == 1.0 and final_loss < 0.02:
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(selected.iloc[0].cache_path, allow_pickle=False) as payload:
        preprocess_config = json.loads(payload["metadata"].item())["preprocess_config"]
    checkpoint_path = output_dir / "overfit_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model.export_config(),
            "preprocess_config": preprocess_config,
            "class_to_index": {"LGG": 0, "HGG": 1},
            "epoch": completed_epoch,
            "purpose": "pipeline_debugging_only",
        },
        checkpoint_path,
    )
    reloaded = CompactResNet3D(model_config).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model_state"])
    reloaded.eval()
    with torch.inference_mode():
        reload_logits = reloaded(batch["image"].to(device))
    reload_matches = bool(torch.equal(logits.argmax(1), reload_logits.argmax(1)))
    result = {
        "passed": final_accuracy == 1.0 and final_loss < 0.02 and reload_matches,
        "n_patients": len(dataset),
        "epochs": completed_epoch,
        "final_loss": final_loss,
        "training_accuracy": final_accuracy,
        "checkpoint_reload_predictions_match": reload_matches,
        "patient_ids": selected["patient_id"].tolist(),
        "warning": "Memorization check only; these are not evaluation metrics.",
    }
    save_json(result, output_dir / "overfit_check.json")
    if not result["passed"]:
        raise RuntimeError(f"Tiny-subset overfit check failed: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the eight-patient pipeline memorization check")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_overfit_check(
        args.manifest,
        args.output_dir,
        samples_per_class=args.samples_per_class,
        epochs=args.epochs,
        seed=args.seed,
        base_channels=args.base_channels,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
