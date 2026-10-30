from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import CompactResNet3D, ModelConfig
from .preprocess import PreprocessConfig, preprocess_paths
from .utils import INDEX_TO_LABEL, MODALITIES


def predict_patient(
    checkpoint_path: str | Path,
    modality_paths: dict[str, str],
    seg_path: str | None = None,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CompactResNet3D(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    preprocess_config = PreprocessConfig(**checkpoint["preprocess_config"])
    image, _, metadata = preprocess_paths(modality_paths, seg_path, preprocess_config)
    tensor = torch.from_numpy(image)[None].to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    prediction = int(probabilities.argmax())
    return {
        "prediction": INDEX_TO_LABEL[prediction],
        "probability_lgg": float(probabilities[0]),
        "probability_hgg": float(probabilities[1]),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "preprocessing": metadata,
        "warning": "Academic research output only; not for clinical use.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HGG/LGG inference for one multimodal NIfTI subject"
    )
    parser.add_argument("--checkpoint", required=True)
    for modality in MODALITIES:
        parser.add_argument(f"--{modality}", required=True)
    parser.add_argument("--seg", help="Required when the checkpoint uses ROI cropping")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {name: getattr(args, name) for name in MODALITIES}
    result = predict_patient(args.checkpoint, paths, args.seg)
    rendered = json.dumps(result, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
