
---

### `data_loader.py`

A simple self-contained dataset: if `data.npz` doesn’t exist, generate it once.

```python
# data_loader.py
from __future__ import annotations

import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data.npz"


def _generate_dataset(
    n_samples: int = 5000,
    n_features: int = 20,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic binary classification dataset.

    The generative process is intentionally simple but not degenerate.
    """
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    true_w = rng.normal(size=(n_features,))
    logits = X @ true_w
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < probs).astype(np.int64)
    return X, y


def load_splits(
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 0,
):
    """Return (X_train, y_train, X_val, y_val, X_test, y_test)."""
    if DATA_PATH.exists():
        data = np.load(DATA_PATH)
        return (
            data["X_train"],
            data["y_train"],
            data["X_val"],
            data["y_val"],
            data["X_test"],
            data["y_test"],
        )

    X, y = _generate_dataset(random_state=random_state)
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATA_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
