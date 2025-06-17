# ──────────────────────────────────────────────────────────────────────────────
# src/gnn_omtf/training/trainer.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import logging, random, torch, numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from ..models import BaseGNN
from .loop import train_one_epoch, evaluate
from ..metrics.regression import regression_metrics

log = logging.getLogger(__name__)


class GraphTrainer:
    """Train/validate/test a single GNN and return metrics."""
    def __init__(self, cfg):
        self.cfg     = cfg
        self.device  = self._select_device(cfg.device)
        self._set_seed(cfg.seed)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _select_device(device_str):
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log.info("Random seed set to %d", seed)

    # ---------------------------------------------------------------- fit loop
    def fit(self, dataset):
        train_hist, val_hist = [], []

        train_set, val_set, test_set = torch.utils.data.random_split(
            dataset, [0.7, 0.15, 0.15],
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=self.cfg.batch_size)
        test_loader  = DataLoader(test_set,  batch_size=self.cfg.batch_size)

        model = BaseGNN.get(
            self.cfg.model,
            num_node_features=self.cfg.num_node_features,
            hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)

        opt      = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        loss_fn  = torch.nn.MSELoss()
        best_val, epochs_no_improve = float("inf"), 0

        ckpt_full  = Path(self.cfg.out_dir) / "best.ckpt"
        ckpt_state = Path(self.cfg.out_dir) / "best_state_dict.pt"
        ckpt_full.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, opt, loss_fn,
                                      self.device, amp=self.cfg.amp)
            va_loss = evaluate(model, val_loader, loss_fn, self.device)

            train_hist.append(tr_loss); val_hist.append(va_loss)
            log.info("Epoch %3d  train %.5f | val %.5f", epoch, tr_loss, va_loss)

            if va_loss < best_val:
                best_val, epochs_no_improve = va_loss, 0
                torch.save(model,             ckpt_full)   # full module
                torch.save(model.state_dict(), ckpt_state) # state-dict
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.cfg.patience:
                    log.info("Early-stopping triggered."); break

        # ---------- test with best model -------------------------------------------------
        best_model = torch.load(ckpt_full, map_location=self.device, weights_only=False)
        test_loss  = evaluate(best_model, test_loader, loss_fn, self.device)

        preds, truth = self._gather_outputs(best_model, test_loader)
        reg_metrics  = regression_metrics(preds, truth)

        return dict(train=train_hist,
                    val=val_hist,
                    test_loss=float(test_loss),
                    **reg_metrics)

    # ---------------------------------------------------------------- gather outs
    def _gather_outputs(self, model, loader):
        preds, trues = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds.append(model(batch).cpu())
                trues.append(batch.y.cpu())
        return torch.cat(preds), torch.cat(trues)
# ──────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        log.info("Random seed set to %d", seed)