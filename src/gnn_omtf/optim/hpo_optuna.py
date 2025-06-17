from __future__ import annotations

"""
Optuna-driven hyper-parameter search for any gnn_omtf model.

Usage (Python):

    from gnn_omtf.optim.hpo_optuna import run_study
    best = run_study("graphs.pt", n_trials=50)
    print(best)

CLI wrapper in optim/cli.py exposes the same functionality:
    gnn-omtf-opt hpo --graphs graphs.pt --trials 50
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import optuna
import torch
from torch_geometric.loader import DataLoader

from gnn_omtf.data import OMTFDataset
from gnn_omtf.training import GraphTrainer

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------- #
def _objective(trial: optuna.Trial, ds: OMTFDataset) -> float:
    # ----- search space ------------------------------------------------
    model_name = trial.suggest_categorical("model", ["gat", "gcn", "sage", "mpl"])
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    # ----- model args ---------------------------------------------------
    model_args = dict(
        in_channels=ds.num_node_features,
        hidden_channels=hidden_dim,
        out_channels=1,  # ✅ customize this if needed
        dropout_p=dropout,
    )

    # ----- trainer config -----------------------------------------------
    cfg = SimpleNamespace(
        model        = model_name,
        model_args   = model_args,
        lr           = lr,
        batch_size   = 1024,
        epochs       = 40,
        patience     = 5,
        device       = "cuda" if torch.cuda.is_available() else "cpu",
        seed         = 42 + trial.number,
        amp          = False,
        out_dir      = f"optuna_runs/trial_{trial.number}",
        num_node_features = ds.num_node_features,
        num_edge_features = ds.num_edge_features,
    )

    # ----- split dataset (70/30) ----------------------------------------
    n = len(ds)
    n_train = int(0.7 * n)
    train_loader = DataLoader(ds[:n_train], batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(ds[n_train:], batch_size=cfg.batch_size)

    # ----- train --------------------------------------------------------
    trainer = GraphTrainer(cfg)
    metrics = trainer.fit(ds)
    log.info("trial %s → %.5f", trial.number, metrics["mse"])
    return metrics["mse"]  # Optuna minimizes by default


# --------------------------------------------------------------------- #
# public helper
# --------------------------------------------------------------------- #
def run_study(graphs_pt: str | Path, n_trials: int = 30,
              study_name: str | None = None,
              pruner: optuna.pruners.BasePruner | None = None
              ) -> Dict[str, object]:
    """Run *n_trials* of hyper-parameter optimisation and return best params."""
    ds = OMTFDataset.load_dataset(graphs_pt)

    study = optuna.create_study(
        study_name   = study_name,
        direction    = "minimize",
        pruner       = pruner or optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler      = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: _objective(t, ds), n_trials=n_trials, timeout=None, show_progress_bar=True)
    log.info("best params: %s", study.best_params)
    return study.best_params


def save_best(best_params: Dict[str, object], path: Path):
    path.write_text(json.dumps(best_params, indent=2))
    log.info("saved → %s", path)
