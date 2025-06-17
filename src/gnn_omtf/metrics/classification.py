from __future__ import annotations
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

__all__ = ["classification_metrics", "accuracy", "precision_recall_f1"]

def classification_metrics(y_true, y_pred, *, threshold: float = 0.5) -> dict:
    """Return a dict with accuracy, precision, recall, F1 and confusion-matrix."""
    if y_pred.ndim:
        y_pred = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "cm":        cm,
    }

# minimal stubs for CLI import paths
def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))

def precision_recall_f1(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
