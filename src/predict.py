"""CPU inference helpers for single face images."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError

from src.faces_model import get_transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = {0: "FAKE", 1: "REAL"}


def configured_weights_path() -> Path:
    """Resolve MODEL_WEIGHTS relative to the project root unless it is absolute."""
    configured = Path(os.environ.get("MODEL_WEIGHTS", "models/effnet_b0_smoke.pth"))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


@lru_cache(maxsize=1)
def load_model() -> torch.nn.Module:
    """Load and cache the configured EfficientNet checkpoint on CPU."""
    weights_path = configured_weights_path()
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. "
            "Set MODEL_WEIGHTS to a valid EfficientNet-B0 state_dict file."
        )

    from src.faces_model import build_model

    model = build_model(pretrained=False).cpu()
    try:
        state_dict: Any = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise RuntimeError(f"Could not load model weights from {weights_path}: {error}") from error
    model.eval()
    return model


def _read_image(path_or_bytes: str | Path | bytes | bytearray) -> tuple[bytes, Image.Image]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw_bytes = bytes(path_or_bytes)
    else:
        path = Path(path_or_bytes)
        try:
            raw_bytes = path.read_bytes()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Image file not found: {path}") from error
        except OSError as error:
            raise OSError(f"Could not read image file {path}: {error}") from error

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.load()
            image = source.convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError("Unreadable image: file is not a valid supported image.") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"Unreadable image: {error}") from error
    return raw_bytes, image


def predict_image(
    path_or_bytes: str | Path | bytes | bytearray,
    model: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Predict fake/real probabilities and hash the exact input bytes."""
    raw_bytes, image = _read_image(path_or_bytes)
    inference_model = model if model is not None else load_model()
    inference_model.to("cpu")
    inference_model.eval()
    image_tensor = get_transforms(train=False)(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(inference_model(image_tensor), dim=1)[0]
    p_fake = float(probabilities[0].item())
    p_real = float(probabilities[1].item())
    return {
        "verdict": LABELS[0] if p_fake >= 0.5 else LABELS[1],
        "p_fake": p_fake,
        "p_real": p_real,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify a face image as fake or real.")
    parser.add_argument("image_path", type=Path, help="Path to a JPG/PNG image")
    args = parser.parse_args()
    try:
        result = predict_image(args.image_path)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        parser.exit(status=1, message=f"Error: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
