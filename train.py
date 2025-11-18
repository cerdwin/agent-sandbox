# train.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    batch_size: int = 128
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    early_stopping_patience: int = 5


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: str,
) -> DataLoader:
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig | None = None,
) -> nn.Module:
    """Train the model and return the best-validation model."""
    if config is None:
        config = TrainConfig()

    device = torch.device(config.device)
    model = model.to(device)

    train_loader = _make_loader(
        X_train, y_train, batch_size=config.batch_size, shuffle=True, device=config.device
    )
    val_loader = _make_loader(
        X_val, y_val, batch_size=512, shuffle=False, device=config.device
    )

    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_val_acc = 0.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        model.train()
        for xb, yb in train_loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == yb.long()).sum().item()
                total += yb.numel()
        val_acc = correct / max(total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
