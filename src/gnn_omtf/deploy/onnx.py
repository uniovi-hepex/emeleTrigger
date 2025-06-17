# src/gnn_omtf/deploy/onnx.py
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from ..models import BaseGNN

onnx_cli = typer.Typer(help="Export trained weights to ONNX")


@onnx_cli.command("export")
def export_onnx(
    ckpt: Path = typer.Argument(..., exists=True, help="Checkpoint (.pt / .ckpt)"),
    out: Path = typer.Option("model.onnx", help="Destination ONNX file"),
):
    """
    Export a trained GNN checkpoint to ONNX.

    The folder that contains *ckpt* must also hold a `config.json`
    (automatically written during training) with at least:

    ```json
    {
      "model": "gat",
      "num_node_features": 5,
      "hidden_dim": 32
    }
    ```
    """
    ckpt = Path(ckpt)

    # ── 1. read the saved hyper-parameters ---------------------------------
    cfg_path = ckpt.parent / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path} next to the checkpoint")

    with cfg_path.open() as f:
        cfg = json.load(f)

    model_name  = cfg["model"]
    feat_dim    = cfg["num_node_features"]
    hidden_dim  = cfg["hidden_dim"]

    # ── 2. load checkpoint (state-dict **or** full model) -------------------
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):          # full model was pickled
        model = obj
    else:                                         # plain state-dict
        model = BaseGNN.get(
            name=model_name,
            num_node_features=feat_dim,
            hidden_dim=hidden_dim,
        )
        model.load_state_dict(obj)
    model.eval()

    # ── 3. dummy example graph ---------------------------------------------
    N = 4  # number of nodes for the dummy export graph
    dummy_input = (
        torch.randn(N, feat_dim),             # x
        torch.randint(0, N, (2, N * 3)),      # edge_index
        torch.zeros(N, dtype=torch.long),     # batch
    )

    # ── 4. export -----------------------------------------------------------
    torch.onnx.export(
        model,
        dummy_input,
        out,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x", "edge_index", "batch"],
        output_names=["y"],
        dynamic_axes={"x": {0: "N"}, "y": {0: "N"}},
    )

    typer.echo(f"✓ ONNX written → {out}")
