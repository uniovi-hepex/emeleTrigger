#  src/gnn_omtf/training/loop.py
from __future__ import annotations
from collections.abc import Callable
import torch
from torch.amp import autocast, GradScaler        # ← NEW import path

# ────────────────────────────────────────────────────────────────────
def _match_shapes(out: torch.Tensor, y: torch.Tensor, loss_fn: Callable):
    """
    • BCEWithLogitsLoss requires identical shapes → flatten both to (N,).
    • All other losses keep the original tensors untouched.
    """
    if isinstance(loss_fn, torch.nn.BCEWithLogitsLoss):
        return out.view(-1), y.view(-1).float()
    return out, y


# ────────────────────────────────────────────────────────────────────
def _step_autocast(model, batch, loss_fn, use_amp: bool, device):
    """Forward pass under optional autocast – returns (loss, logits)."""
    with autocast(device.type, enabled=use_amp):
        logits       = model(batch)                      # [N] or [N, 1]
        logits, targ = _match_shapes(logits, batch.y, loss_fn)
        loss         = loss_fn(logits, targ)
    return loss, logits


# ────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimiser, loss_fn: Callable,
                    device: torch.device, *, amp: bool = False) -> float:
    model.train()
    total_loss = 0.0
    use_amp    = amp and device.type == "cuda"
    scaler     = GradScaler(device.type, enabled=use_amp)
    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad(set_to_none=True)

        loss, _ = _step_autocast(model, batch, loss_fn, use_amp, device)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


# ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn: Callable, device: torch.device,
             *, amp: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    use_amp    = amp and device.type == "cuda"

    for batch in loader:
        batch = batch.to(device)
        loss, _ = _step_autocast(model, batch, loss_fn, use_amp, device)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)
