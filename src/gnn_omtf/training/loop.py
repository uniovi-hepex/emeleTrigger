#  src/gnn_omtf/training/loop.py
from __future__ import annotations
"""One-epoch helpers – no business logic."""
from collections.abc import Callable
import torch

# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model,
    loader,
    optimiser,
    loss_fn: Callable,
    device: torch.device,
    *,
    amp: bool = False,
) -> float:
    """Single training epoch – returns mean loss."""
    model.train()
    total_loss = 0.0

    use_amp = amp and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad(set_to_none=True)

        # forward & loss -----------------------------------------------------
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(batch)
            loss = loss_fn(out, batch.y)

        # backward -----------------------------------------------------------
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            optimiser.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)

# --------------------------------------------------------------------------- #
# Validation / Test                                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn: Callable,
    device: torch.device,
    *,
    amp: bool = False,
) -> float:
    """Evaluate on *loader* – returns mean loss."""
    model.eval()
    total_loss = 0.0

    use_amp = amp and device.type == "cuda"

    for batch in loader:
        batch = batch.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out  = model(batch)
            loss = loss_fn(out, batch.y)
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)
