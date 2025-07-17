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
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import optuna
import torch
from torch_geometric.loader import DataLoader

from gnn_omtf.data import OMTFDataset
from gnn_omtf.training import GraphTrainer
from gnn_omtf.viz import plot_regression, plot_classification   # NEW

log = logging.getLogger(__name__)

def model_key(base: str, task: str) -> str:
    """
    Map a base name ('gat', 'gcn', …) + task to the concrete
    class registered in gnn_omtf.models.

        regression   → '<base>_regressor'
        classification→ '<base>_classifier'
    """
    base = base.lower()
    if task.startswith("class"):
        return f"{base}_classifier"
    return f"{base}_regressor"


# --------------------------------------------------------------------- #
# objective
# --------------------------------------------------------------------- #
def _objective(trial: optuna.Trial, ds: OMTFDataset) -> float:
    # ----- search space ------------------------------------------------
    base_name  = trial.suggest_categorical("model", ["sage_simple", "gcn_simple", "sum_gnn_simple"])
    hidden_dim = trial.suggest_int("hidden_dim", 16, 128, step=16)
    dropout    = trial.suggest_float("dropout", 0.0, 0.5)
    lr         = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

    # ----- model args ---------------------------------------------------
    # Pick classifier vs regressor automatically
    model_name = model_key(base_name, ds.task)
    model_args = dict(
        in_channels=ds.num_node_features,
        hidden_channels=hidden_dim,
        out_channels = (ds.num_classes if ds.is_classification and ds.num_classes > 2
                        else 1),
        dropout_p=dropout,
    )

    # ----- trainer config -----------------------------------------------
    cfg = SimpleNamespace(
        model        = model_name,
        model_args   = model_args,
        lr           = lr,
        batch_size   = 1024,
        epochs       = 100,
        patience     = 5,
        device       = "cuda" if torch.cuda.is_available() else "cpu",
        seed         = 42 + trial.number,
        amp          = False,
        out_dir      = f"optuna_runs/trial_{trial.number}",
        num_node_features = ds.num_node_features,
        num_edge_features = ds.num_edge_features,
        task         = ds.task,        # <- for the trainer
    )

    # ----- split dataset (70/30) ----------------------------------------
    n = len(ds)
    n_train = int(0.7 * n)
    train_loader = DataLoader(ds[:n_train], batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(ds[n_train:], batch_size=cfg.batch_size)

    # ----- train --------------------------------------------------------
    trainer = GraphTrainer(cfg)
    metrics = trainer.fit(ds)

    # What to minimise:
    score = (metrics["mse"] if not ds.is_classification
             else 1.0 - metrics["accuracy"])   # higher acc → lower objective

    log.info("trial %s → %.5f", trial.number, score)
    return score
# --------------------------------------------------------------------- #
# public helper
# --------------------------------------------------------------------- #
# --------------------------------------------------------------------- #
# helper: retrain with fixed params & make plots (NEW)
# --------------------------------------------------------------------- #
def _retrain_and_summarise(best_params: dict, ds: OMTFDataset,
                           run_dir: Path, *, epochs: int = 40) -> None:
    """
    Using *best_params* (for one *base* model), retrain on full train set,
    save:  • config.json  • metrics.json  • best.ckpt  • plots/
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- config --------------------------------------------------
    cfg = SimpleNamespace(
        model   = model_key(best_params["model"], ds.task),
        lr      = best_params["lr"],
        batch_size = 1024,
        epochs     = epochs,
        patience   = 8,
        device     = "cuda" if torch.cuda.is_available() else "cpu",
        seed       = 123,
        amp        = False,
        out_dir    = run_dir,
        num_node_features = ds.num_node_features,
        num_edge_features = ds.num_edge_features,
        task   = ds.task,
        model_args = dict(
            in_channels  = ds.num_node_features,
            hidden_channels = best_params["hidden_dim"],
            out_channels = 1,
            dropout_p    = best_params["dropout"],
        ),
    )

    trainer  = GraphTrainer(cfg)
    summary  = trainer.fit(ds)                      # train ‑> metrics & ckpt
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "config.json" ).write_text(json.dumps(best_params, indent=2))

    # ---------- predictions & plots ------------------------------------
    ckpt = run_dir / "best.ckpt"
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch).cpu())
            trues.append(batch.y.cpu())
    preds = torch.cat(preds); trues = torch.cat(trues)

    plot_dir = run_dir / "viz"
    if ds.is_classification:
        plot_classification(preds, trues, plot_dir)
    else:
        plot_regression(preds, trues, plot_dir)


def run_study(graphs_pt: str | Path,
              n_trials: int = 30,
              *,
              config: str | Path | None = None,
              study_name: str | None = None,
              pruner: optuna.pruners.BasePruner | None = None,
              format: str = "list",  # ← NEW: 'list' or 'batched'
              ) -> Dict[str, object]:

    """Run *n_trials* of hyper-parameter optimisation and return best params."""
    load_kwargs = {"config": str(config)} if config else {}

    if format == "list":
        ds = OMTFDataset.load_dataset(graphs_pt, **load_kwargs)
    elif format == "batched":
        ds = OMTFDataset.load_batched_dict(graphs_pt, **load_kwargs)
    else:
        raise ValueError(f"Unknown dataset format: {format}")


    # ---------------------------------------------------------------- #
    # Derive task from YAML (fallback = regression)
    # ---------------------------------------------------------------- #
    task = "regression"
    num_classes = None
    if config:
        with open(config) as f:
            cfg_yaml = yaml.safe_load(f) or {}
        task = cfg_yaml.get("task", "regression").lower()
        if task.startswith("class"):
            num_classes = cfg_yaml.get("num_classes", 2)

    # expose to the downstream code
    ds.task              = task
    ds.is_classification = task.startswith("class")

    study = optuna.create_study(
        study_name   = study_name,
        direction    = "minimize",
        pruner       = pruner or optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler      = optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda t: _objective(t, ds),
                   n_trials=n_trials,
                   timeout=None,
                   show_progress_bar=True)
    
    overall_best = study.best_params
    
    # ------------------------------------------------------------------ #
    # ❷ Best per *base* model
    # ------------------------------------------------------------------ #
    best_per_model: dict[str, dict] = {}
    for base in {"sage_simple", "gcn_simple", "sum_gnn_simple"}:
        candidates = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.params.get("model") == base
        ]
        if not candidates:
            continue
        best_trial = min(
            candidates,
            key=lambda tr: tr.value
        )
        best_per_model[base] = best_trial.params
        
    # ------------------------------------------------------------------ #
    # ❸ Optional: retrain+plot for every best model
    # ------------------------------------------------------------------ #
    summary_root = Path("optuna_runs") / "hpo_summary"
    for base, params in best_per_model.items():
        _retrain_and_summarise(params, ds, summary_root / base)

    # save meta‑summary
    meta = dict(overall_best=overall_best, best_per_model=best_per_model)
    (summary_root / "summary.json").write_text(json.dumps(meta, indent=2))

    log.info("best params (overall): %s", overall_best)
    
    return overall_best


    return study.best_params


def save_best(best_params: Dict[str, object], path: Path):
    path.write_text(json.dumps(best_params, indent=2))
    log.info("saved → %s", path)
