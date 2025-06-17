from __future__ import annotations
import torch


def regression_metrics(pred: torch.Tensor, truth: torch.Tensor) -> dict:
    pred, truth = pred.float(), truth.float()
    residuals   = pred - truth
    mse_val = torch.mean(residuals ** 2).item()
    r2    = 1 - torch.sum(residuals ** 2) / torch.sum((truth - truth.mean()) ** 2)
    maxe  = torch.max(torch.abs(residuals)).item()
    kl    = _kl_div(pred, truth)
    return {"mse": mse_val, "r2": r2.item(), "max_err": maxe, "kl": kl}

# --------------------------------------------------------------------------- #
# single-call convenience function  (your original snippet)
# --------------------------------------------------------------------------- #
def regression_summary(pred: torch.Tensor, truth: torch.Tensor) -> dict[str, float]:
    """Return a dict with multiple metrics at once."""
    pred, truth = pred.float(), truth.float()
    resid = pred - truth
    mse  = torch.mean(resid ** 2).item()
    r2   = 1 - torch.sum(resid ** 2) / torch.sum((truth - truth.mean()) ** 2)
    maxe = torch.max(torch.abs(resid)).item()
    kl   = _kl_div(pred, truth)
    return {"mse": mse, "r2": r2.item(), "max_err": maxe, "kl": kl}

# --------------------------------------------------------------------------- #
# individual metric callables (used by CLI / pruning code)
# --------------------------------------------------------------------------- #
def mse(pred, truth):           return regression_summary(pred, truth)["mse"]
def rmse(pred, truth):          return torch.sqrt(torch.tensor(mse(pred, truth))).item()
def bias(pred, truth):          return torch.mean(pred - truth).item()
def resolution(pred, truth):    return torch.std(pred - truth, unbiased=False).item()

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _kl_div(p, q):
    p, q = p / p.sum(), q / q.sum()
    mask = (p > 0) & (q > 0)
    return torch.sum(p[mask] * torch.log(p[mask] / q[mask])).item()
