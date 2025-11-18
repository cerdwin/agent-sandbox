# evaluate.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.inference_mode()
def accuracy(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str = "cpu",
) -> float:
    model = model.to(device)
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)

    logits = model(X_t)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()

    correct = (preds == y_t).sum().item()
    total = y_t.numel()
    return correct / max(total, 1)


def evaluate_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = "cpu",
) -> dict:
    """Return a dict with train/val/test accuracy and main score."""
    results = {
        "train_accuracy": accuracy(model, X_train, y_train, device=device),
        "val_accuracy": accuracy(model, X_val, y_val, device=device),
        "test_accuracy": accuracy(model, X_test, y_test, device=device),
    }
    # Main metric used by the leaderboard
    results["score"] = results["test_accuracy"]
    return results
