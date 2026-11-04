"""Training, evaluation, and early-stopping helpers for image classifiers."""

from copy import deepcopy
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Train for one epoch and return sample-weighted loss and accuracy."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader contains no samples")
    return total_loss / sample_count, correct / sample_count


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, Tensor]:
    """Evaluate once and return sample-weighted loss, accuracy, and CPU probabilities."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    correct = 0
    sample_count = 0
    probabilities: list[Tensor] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size
            probabilities.append(torch.softmax(logits, dim=1).cpu())

    if sample_count == 0:
        raise ValueError("Evaluation loader contains no samples")
    return total_loss / sample_count, correct / sample_count, torch.cat(probabilities)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    patience: int,
) -> dict[str, Any]:
    """Train with validation-loss early stopping and restore the best model weights."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    history: dict[str, Any] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rate": [],
    }
    best_loss = float("inf")
    best_epoch = 0
    best_weights = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_weights = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_weights)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_loss
    return history
