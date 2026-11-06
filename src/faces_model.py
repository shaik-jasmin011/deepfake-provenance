"""Transfer-learning utilities for the face authenticity dataset."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms

from src.datasets import CsvImageDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(pretrained: bool = True) -> nn.Module:
    """Build EfficientNet-B0 with a two-class dropout classification head."""
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, 2))
    return model


def get_transforms(train: bool) -> transforms.Compose:
    """Return mild training augmentation or deterministic ImageNet evaluation transforms."""
    operations: list[Any] = [transforms.Resize(224)]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def make_loader(
    csv_path: str | Path,
    train: bool,
    batch_size: int,
    limit: int | None = None,
    seed: int = 42,
) -> DataLoader:
    """Create a deterministic, optionally stratified CSV image data loader."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    dataset = CsvImageDataset(csv_path, transform=get_transforms(train))
    dataset_size = len(dataset)
    if limit is not None:
        if limit < 1 or limit > dataset_size:
            raise ValueError(f"limit must be between 1 and {dataset_size}, got {limit}")
        if limit < dataset_size:
            labels = dataset.samples["label"]
            if labels.value_counts().min() < 2:
                raise ValueError("Stratified sampling requires at least two rows per class")
            selected, _ = train_test_split(
                np.arange(dataset_size),
                train_size=limit,
                random_state=seed,
                stratify=labels,
            )
            dataset = Subset(dataset, selected.tolist())

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,
        generator=generator,
    )


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    patience: int = 3,
    device: torch.device | str | None = None,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Train a classifier, schedule on validation loss, and restore its best weights."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    set_seed(seed)
    device = _resolve_device(device)
    model.to(device)

    head = model.classifier
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("classifier.") and parameter.requires_grad
    ]
    if not head_parameters and not backbone_parameters:
        raise ValueError("Model has no trainable parameters")

    parameter_groups: list[dict[str, Any]] = []
    if backbone_parameters:
        parameter_groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_lr if backbone_lr is not None else lr,
            }
        )
    if head_parameters:
        parameter_groups.append(
            {"params": head_parameters, "lr": head_lr if head_lr is not None else lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_count += labels.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device, dtype=torch.long)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_count += labels.size(0)

        if train_count == 0 or val_count == 0:
            raise ValueError("Training and validation loaders must each contain images")
        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        scheduler.step(val_loss)
        epoch_seconds = time.perf_counter() - started
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "train_acc": train_correct / train_count,
                "val_loss": val_loss,
                "val_acc": val_correct / val_count,
                "seconds": epoch_seconds,
            }
        )
        print(
            f"Epoch {epoch + 1}/{epochs} - train loss={train_loss:.4f}, "
            f"acc={train_correct / train_count:.4f}; val loss={val_loss:.4f}, "
            f"acc={val_correct / val_count:.4f}; seconds={epoch_seconds:.1f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    model.load_state_dict(best_weights)
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return mean cross-entropy, accuracy, labels, and fake-class probabilities."""
    device = _resolve_device(device)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    labels_all: list[np.ndarray] = []
    fake_probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    total = 0
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            logits = model(images)
            loss_sum += criterion(logits, labels).item()
            probabilities = torch.softmax(logits, dim=1)
            labels_all.append(labels.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            fake_probabilities.append(probabilities[:, 0].cpu().numpy())
            total += labels.size(0)
    if total == 0:
        raise ValueError("Evaluation loader contains no images")
    labels_array = np.concatenate(labels_all)
    prediction_array = np.concatenate(predictions)
    return {
        "loss": loss_sum / total,
        "accuracy": float(accuracy_score(labels_array, prediction_array)),
        "labels": labels_array,
        "predictions": prediction_array,
        "p_fake": np.concatenate(fake_probabilities),
    }
