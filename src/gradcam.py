"""Manual Grad-CAM visualization for EfficientNet-B0 face predictions."""

from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.faces_model import get_transforms


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1),
        ],
        axis=-1,
    )


def gradcam_overlay(
    image: Image.Image | bytes | bytearray,
    model: nn.Module,
    target_class: int = 0,
) -> Image.Image:
    """Blend a normalized Grad-CAM map for one class over the original RGB image."""
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 (fake) or 1 (real)")
    if isinstance(image, (bytes, bytearray)):
        with Image.open(io.BytesIO(bytes(image))) as source:
            source.load()
            original = source.convert("RGB")
    else:
        original = image.convert("RGB")

    model_device = torch.device("cpu")
    model.to(model_device)
    target_layer = model.features[-1]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(save_activation)
    previous_training = model.training
    parameter_grad_states = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.eval()
        input_tensor = get_transforms(train=False)(original).unsqueeze(0).to(model_device)
        input_tensor.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Grad-CAM expects a two-class model output")
            logits[0, target_class].backward()
        if not activations or not gradients or gradients[0] is None:
            raise RuntimeError("Grad-CAM did not receive gradients from the target feature block")

        activation = activations[0].detach()
        gradient = gradients[0].detach()
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap_tensor = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
        heatmap_tensor = torch.nn.functional.interpolate(
            heatmap_tensor,
            size=original.size[::-1],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        peak = heatmap_tensor.max()
        if not torch.isfinite(peak) or peak.item() <= 0:
            raise RuntimeError("Grad-CAM heatmap is all zeros or non-finite")
        heatmap = (heatmap_tensor / peak).cpu().numpy()
        if not np.any(heatmap > 0):
            raise RuntimeError("Grad-CAM heatmap is all zeros")

        heat_rgb = (_jet_colormap(heatmap) * 255).astype(np.uint8)
        colorized = Image.fromarray(heat_rgb, mode="RGB")
        return Image.blend(original, colorized, alpha=0.4)
    finally:
        hook.remove()
        for parameter, requires_grad in zip(model.parameters(), parameter_grad_states):
            parameter.requires_grad_(requires_grad)
        model.zero_grad(set_to_none=True)
        model.train(previous_training)
