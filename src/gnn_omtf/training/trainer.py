from __future__ import annotations

import logging
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..models import BaseGNN
from .loop import train_one_epoch, evaluate
from ..metrics.regression import regression_metrics
from ..metrics.classification import classification_metrics

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
class GraphTrainer:
    """
    Train/validate/test a single GNN and return metrics.

    The task (``"regression"`` or ``"classification"``) is read from
    ``cfg.task`` – default **regression** for backward compatibility.
    """

    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.task = getattr(cfg, "task", "classification").lower()
        self.device = self._select_device(cfg.device)
        self._set_seed(cfg.seed)
        log.info("Initialised trainer for **%s** task", self.task)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _select_device(device_str: str) -> torch.device:
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log.info("Random seed set to %d", seed)

    # ---------------------------------------------------------------- fit loop
    def fit(self, dataset) -> dict:
        # ── 0. discard graphs that ended up with 0 nodes after transforms ──
        keep = [i for i, g in enumerate(dataset) if g.x.numel() > 0]
        if len(keep) != len(dataset):               # warn once
            log.warning("Dropped %d empty graphs", len(dataset) - len(keep))
        dataset = torch.utils.data.Subset(dataset, keep)
        train_hist, val_hist = [], []

        # 70 / 15 / 15 split
        lengths = [int(0.7 * len(dataset)), int(0.15 * len(dataset))]
        lengths.append(len(dataset) - sum(lengths))
        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, lengths, generator=torch.Generator().manual_seed(self.cfg.seed)
        )

        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=self.cfg.batch_size)
        test_loader  = DataLoader(test_set,  batch_size=self.cfg.batch_size)

        # ── build model ────────────────────────────────────────────────────
        model = BaseGNN.get(
            self.cfg.model, **getattr(self.cfg, "model_args", {})
        ).to(self.device)

        opt      = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        loss_fn  = self._get_loss()
        best_val = float("inf")
        epochs_no_improve = 0

        ckpt_dir   = Path(self.cfg.out_dir)
        ckpt_full  = ckpt_dir / "best.ckpt"
        ckpt_state = ckpt_dir / "best_state_dict.pt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn,
                                      self.device, amp=self.cfg.amp)
            va_loss = evaluate(model, val_loader, loss_fn, self.device)

            train_hist.append(tr_loss)
            val_hist.append(va_loss)
            log.info("Epoch %3d  train %.5f | val %.5f", epoch, tr_loss, va_loss)

            if va_loss < best_val:
                best_val, epochs_no_improve = va_loss, 0
                torch.save(model,              ckpt_full)   # full module
                torch.save(model.state_dict(), ckpt_state)  # weights only
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    log.info("Early‑stopping triggered.")
                    break

        # ── evaluate best model on test split ──────────────────────────────
        best_model = torch.load(ckpt_full, map_location=self.device, weights_only=False)
        test_loss  = evaluate(best_model, test_loader, loss_fn, self.device)

        preds, truth = self._gather_outputs(best_model, test_loader)
        task_metrics = self._compute_metrics(preds, truth)

        return dict(
            train=train_hist,
            val=val_hist,
            test_loss=float(test_loss),
            **task_metrics,
        )

    # ---------------------------------------------------------------- task‑specific helpers
    def _get_loss(self):
        if self.task == "classification":
            # binary logits → BCEWithLogitsLoss
            return torch.nn.BCEWithLogitsLoss()
        return torch.nn.MSELoss()              # default regression

    def _compute_metrics(self, preds: torch.Tensor, truth: torch.Tensor) -> dict:
        if self.task == "classification":
            # convert logits → probabilities in (0,1)
            probs = torch.sigmoid(preds).cpu().numpy().ravel()
            labels = truth.cpu().numpy().astype(int).ravel()
            return classification_metrics(labels, probs)
        # regression
        return regression_metrics(preds, truth)

    # ---------------------------------------------------------------- gather outs
    def _gather_outputs(self, model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds.append(model(batch).cpu())
                trues.append(batch.y.cpu())
        return torch.cat(preds), torch.cat(trues)
