from __future__ import annotations


"""Reusable callbacks for the GNN training loop.

These callbacks are **framework-agnostic**: they don’t depend on Lightning,
Keras, or PyTorch Ignite – they just receive the trainer object and a metrics
dict every epoch.
"""

"""
How the callbacks plug in

    Instantiate them in trainer.py (or your Optuna objective):

from gnn_omtf.training import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=cfg.patience),
    ModelCheckpoint(path=cfg.checkpoint, monitor="val_loss"),
]

Call their hooks each epoch inside the training loop:

metrics = {"train_loss": train_loss, "val_loss": val_loss}
for cb in callbacks:
    cb.on_epoch_end(self, epoch, metrics)
if any(getattr(cb, "should_stop", False) for cb in callbacks):
    break
"""

import logging
from pathlib import Path
from typing import Dict

import torch

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Generic base-class
# --------------------------------------------------------------------------- #
class Callback:
    """API everyone follows.  Add more hook-methods if you need them."""

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):  # noqa: D401
        pass  # pragma: no cover

    def on_train_end(self, trainer):  # noqa: D401
        pass  # pragma: no cover


# --------------------------------------------------------------------------- #
# Concrete helpers
# --------------------------------------------------------------------------- #
class EarlyStopping(Callback):
    """Stop training when the monitored metric has not improved for *patience* epochs."""

    def __init__(self, patience: int = 10, monitor: str = "val_loss", mode: str = "min"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = float("inf") if mode == "min" else -float("inf")
        self.counter = 0
        self.should_stop = False

    # ------------------------------------------------------------------ #
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        current = metrics[self.monitor]
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            log.info("EarlyStopping counter %d/%d", self.counter, self.patience)
            if self.counter >= self.patience:
                self.should_stop = True
                log.info("EarlyStopping triggered – best %s=%.5f", self.monitor, self.best)


class ModelCheckpoint(Callback):
    """Save best-model weights according to a monitored metric."""

    def __init__(
        self,
        path: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
    ):
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self.best = float("inf") if mode == "min" else -float("inf")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        current = metrics[self.monitor]
        improved = current < self.best if self.mode == "min" else current > self.best
        if improved:
            self.best = current
            torch.save(trainer.model.state_dict(), self.path)
            log.info("✔  Saved best model → %s  (%s = %.5f)", self.path, self.monitor, self.best)

        # optionally keep a rolling “last epoch” checkpoint
        if self.save_last:
            torch.save(trainer.model.state_dict(), self.path.with_suffix(".last.ckpt"))
