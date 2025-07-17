from .regression import (
    regression_metrics,  # <--- keep this as the function
    mse, rmse, bias, resolution, regression_summary,
)
from .classification import (
    classification_metrics,     # ← NEW ‼
    accuracy,
    precision_recall_f1,
)
# Move the dict to a different name
regression_metric_fns = {
    "mse": mse,
    "rmse": rmse,
    "bias": bias,
    "resolution": resolution,
}

classification_metrics = {
    "accuracy": accuracy,
    "precision_recall_f1": precision_recall_f1,
}

__all__ = [
    "regression_metrics",      # ← function
    "regression_metric_fns",   # ← dict
    "classification_metrics",
    "regression_summary",
]
    