#  src/gnn_omtf/viz/cli.py
from __future__ import annotations
import json
from pathlib import Path
import typer
import torch
from torch_geometric.loader import DataLoader

from ..models       import BaseGNN
from ..data         import OMTFDataset
from ..viz          import plot_graph_features, plot_regression, plot_losses

app = typer.Typer(help="Quick plotting helpers")

# --------------------------------------------------------------------------- #
# LOADERS
# --------------------------------------------------------------------------- #
def _load_graphs(dataset_pt: Path):
    graphs = torch.load(dataset_pt, weights_only=False)
    return OMTFDataset(dataset=graphs)

def _load_model(ckpt: Path):
    """
    Accept **either**
      • a full `nn.Module` pickled via ``torch.save(model)`` *or*
      • a plain *state-dict* saved via ``model.state_dict()``.
    """
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    # 1️⃣ already a full model object
    if isinstance(obj, torch.nn.Module):
        model = obj

    # 2️⃣ plain state-dict  →  rebuild then load
    elif isinstance(obj, dict):
        model = BaseGNN.get(
            cfg["model"],
            num_node_features=cfg["num_node_features"],
            hidden_dim=cfg["hidden_dim"],
        )
        model.load_state_dict(obj)

    else:                           # should never happen
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    model.eval()
    return model

# --------------------------------------------------------------------------- #
# COMMANDS
# --------------------------------------------------------------------------- #
@app.command("features")
def features(
    dataset: Path = typer.Argument(..., exists=True, help=".pt graphs"),
    out_dir: Path = typer.Option("viz", help="Folder for PNGs"),
):
    ds = _load_graphs(dataset)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    plot_graph_features(loader, out_dir, task="regression")

@app.command("regression")
def regression(
    ckpt:    Path = typer.Argument(..., exists=True),
    dataset: Path = typer.Argument(..., exists=True),
    out_dir: Path = typer.Option("viz", help="Folder for PNGs"),
):
    model  = _load_model(ckpt)
    loader = DataLoader(_load_graphs(dataset), batch_size=512, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch).cpu())
            trues.append(batch.y.cpu())
    plot_regression(torch.cat(preds), torch.cat(trues), out_dir)

@app.command("losses")
def losses(
    run_dir: Path = typer.Argument(..., exists=True,
                                   help="folder that contains metrics.json"),
):
    hist = json.loads((run_dir / "metrics.json").read_text())
    plot_losses(hist["train"], hist["val"], run_dir / "loss_curve.png")

# ----- entry-point --------------------------------------------------------- #
def _main() -> None:  app()

if __name__ == "__main__":
    _main()
