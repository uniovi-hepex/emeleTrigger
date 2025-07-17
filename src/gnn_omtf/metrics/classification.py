from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

__all__ = ["classification_metrics", "accuracy", "precision_recall_f1"]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 0:
            return np.array([])
        x = x.numpy()
    return np.asarray(x)


def classification_metrics(
    y_pred,
    y_true,
    *,
    threshold: float = 0.5,
    multi_class: bool = False,
) -> dict:
    """
    Metrics for either binary (*multi_class=False*) or multi‑class.

    * `y_pred` may be raw logits, probabilities, or class indices
    * `y_true` must be class indices (0 … C‑1)
    """
    # to numpy
    y_pred = _to_numpy(y_pred)
    y_true = _to_numpy(y_true).astype(int).ravel()

    # convert predictions
    if not multi_class:
        # logits → probability
        if y_pred.dtype.kind in "f":
            if y_pred.min() < 0 or y_pred.max() > 1:
                y_pred = 1 / (1 + np.exp(-y_pred))
            y_pred = (y_pred > threshold).astype(int)
        label_list = [0, 1]
    else:
        if y_pred.ndim > 1:
            y_pred = y_pred.argmax(axis=1)
        y_pred = y_pred.astype(int)
        label_list = np.unique(y_true)

    # compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=label_list)

    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cm":        cm.tolist(),
    }


def accuracy(
    y_pred,
    y_true,
    *,
    threshold: float = 0.5,
) -> float:
    """Compute binary accuracy after optional sigmoid + threshold."""
    y_pred_np = _to_numpy(y_pred)
    y_true_np = _to_numpy(y_true).astype(int).ravel()
    if y_pred_np.dtype.kind in "f":
        if y_pred_np.min() < 0 or y_pred_np.max() > 1:
            y_pred_np = 1 / (1 + np.exp(-y_pred_np))
        y_pred_np = (y_pred_np > threshold).astype(int)
    return accuracy_score(y_true_np, y_pred_np)


def precision_recall_f1(
    y_pred,
    y_true,
    *,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Compute (precision, recall, f1) for binary classification."""
    y_pred_np = _to_numpy(y_pred)
    y_true_np = _to_numpy(y_true).astype(int).ravel()
    if y_pred_np.dtype.kind in "f":
        if y_pred_np.min() < 0 or y_pred_np.max() > 1:
            y_pred_np = 1 / (1 + np.exp(-y_pred_np))
        y_pred_np = (y_pred_np > threshold).astype(int)
    p = precision_score(y_true_np, y_pred_np, zero_division=0)
    r = recall_score(y_true_np, y_pred_np, zero_division=0)
    f = f1_score(y_true_np, y_pred_np, zero_division=0)
    return p, r, f
