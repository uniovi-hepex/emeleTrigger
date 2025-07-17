from __future__ import annotations
"""
gnn‑omtf ‑‑ quick visualisation helpers
---------------------------------------

$ gnn-omtf-viz features       graphs.pt
$ gnn-omtf-viz regression     ckpt.pt graphs.pt
$ gnn-omtf-viz classification ckpt.pt graphs.pt
$ gnn-omtf-viz losses         run_dir
"""
import json
from pathlib import Path
import typer
import torch
from torch_geometric.loader import DataLoader

from ..models       import BaseGNN
from ..data         import OMTFDataset
from ..viz          import (            # __init__ re‑exports everything
    plot_graph_features,
    plot_regression,
    plot_classification,
    plot_losses,
)

app = typer.Typer(help="Quick plotting helpers")

# ──────────────────────────────────────────────────────────────────────────
# internal loaders
# ──────────────────────────────────────────────────────────────────────────
def _load_graphs(dataset_pt: Path):
    graphs = torch.load(dataset_pt, weights_only=False)
    return OMTFDataset(dataset=graphs)


def _load_model(ckpt: Path):
    """
    Accept **either**
      • a full `nn.Module` pickled via ``torch.save(model)``
      • or a plain *state‑dict* saved via ``model.state_dict()``.
    """
    cfg_path = ckpt.parent / "config.json"        # produced by HPO summary
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    # 1️⃣ full model object
    if isinstance(obj, torch.nn.Module):
        model = obj

    # 2️⃣ state‑dict  →  rebuild then load
    elif isinstance(obj, dict):
        model = BaseGNN.get(
            cfg["model"],
            hidden_channels = cfg.get("hidden_dim"),
            num_node_features = cfg["num_node_features"],
            out_channels = cfg.get("out_channels", 1),
            dropout_p   = cfg.get("dropout", .0),
        )
        model.load_state_dict(obj)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────
# commands
# ──────────────────────────────────────────────────────────────────────────
@app.command("features")
def features(
    dataset: Path = typer.Argument(..., exists=True, help=".pt graphs"),
    out_dir: Path = typer.Option("viz", help="Folder for PNGs"),
):
    """Histogram of node / edge features + target."""
    ds = _load_graphs(dataset)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    # default to regression; irrelevant for hist‑shapes
    plot_graph_features(loader, out_dir, task=getattr(ds, "task", "regression"))


@app.command("regression")
def regression(
    ckpt:    Path = typer.Argument(..., exists=True),
    dataset: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option("viz", help="Folder for PNGs"),
):
    """Regression: truth / pred hist, scatter & residuals."""
    model  = _load_model(ckpt)
    loader = DataLoader(_load_graphs(dataset), batch_size=512, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch).cpu())
            trues.append(batch.y.cpu())

    plot_regression(torch.cat(preds), torch.cat(trues), out_dir)


@app.command("classification")
def classification(
    ckpt:    Path = typer.Argument(..., exists=True),
    dataset: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option("viz", help="Folder for PNGs"),
):
    """Binary ‑ classification plots (CM, ROC, PR, …)."""
    model  = _load_model(ckpt)
    loader = DataLoader(_load_graphs(dataset), batch_size=512, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch).cpu())
            trues.append(batch.y.cpu())

    plot_classification(torch.cat(preds), torch.cat(trues), out_dir)


@app.command("losses")
def losses(
    run_dir: Path = typer.Argument(..., exists=True,
                                   help="folder that contains metrics.json"),
):
    """Plot the train/val loss curves produced by Trainer."""
    hist_path = run_dir / "metrics.json"
    hist = json.loads(hist_path.read_text())
    plot_losses(hist["train"], hist["val"], run_dir / "loss_curve.png")


# -------------------------------------------------------------------------
def _main() -> None:  app()
if __name__ == "__main__":
    _main()
