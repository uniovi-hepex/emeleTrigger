from __future__ import annotations

import json
import shutil
from xml.parsers.expat import model
import typer
from pathlib import Path
import torch
from wandb import config

from .pruning import global_l1_prune, structured_channel_prune, sparsity_stats, size_of_state_dict
from .quantize import dynamic_int8, prepare_qat, fine_tune_qat
from .eval import evaluate_pair
from gnn_omtf.data import OMTFDataset
from torch_geometric.loader import DataLoader
from typing import List, Optional
from ..models import BaseGNN

app = typer.Typer(add_completion=False)


def load_model_with_config(
    ckpt: Path,
    dataset: Path | None = None,
    config: Path | None = None,
) -> torch.nn.Module:
    config_path = config or (ckpt.parent / "config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    if dataset is not None:
        graphs = torch.load(dataset)
        num_node_features = graphs[0].x.size(1)
    else:
        num_node_features = cfg.get("num_node_features", 5)

    model = BaseGNN.get(
        name=cfg["model"],
        num_node_features=num_node_features,
        hidden_dim=cfg["hidden_dim"],
    )

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" not in state and not hasattr(state, "forward"):
        model.load_state_dict(state)
    elif hasattr(state, "forward"):
        model = state
    else:
        raise ValueError("Unsupported checkpoint format")

    return model


def write_model_folder(model: torch.nn.Module, out_dir: Path, config_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Detect quantized model based on presence of quantization keys
    quantized = any("._packed_params" in k for k in dict(model.named_parameters()).keys())
    if quantized:
        torch.save(model, out_dir / "model.pt")  # Save full object
    else:
        torch.save(model.state_dict(), out_dir / "model.pt")

    # 👇 Patch config.json to ensure `num_node_features` is present
    with open(config_path) as f:
        config = json.load(f)

    # Try to infer from model if not already present
    if "num_node_features" not in config:
        try:
            config["num_node_features"] = model.input_dim  # if your model defines it
        except AttributeError:
            config["num_node_features"] = 5  # fallback or raise warning

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    typer.echo(f"✅ saved → {out_dir}")



@app.command()
def prune(
    ckpt: Path = typer.Option(..., exists=True),
    config: Optional[Path] = typer.Option(None, help="Path to override config.json"),
    amount: float = typer.Option(0.5, help="Fraction of weights to prune"),
    structured: bool = typer.Option(False, "--structured/--unstructured"),
    out_dir: Path = typer.Option("run_compress/pruned"),
):
    model = load_model_with_config(ckpt, config=config)
    if structured:
        structured_channel_prune(model, amount)
    else:
        global_l1_prune(model, amount)

    config_path = config or (ckpt.parent / "config.json")
    write_model_folder(model, out_dir, config_path)

    typer.echo(f"sparsity = {sparsity_stats(model)['global_sparsity']:.3f}")
    typer.echo(f"size     = {size_of_state_dict(model)/1024:.1f} kB")


@app.command("dyn-int8")
def dyn_int8(
    ckpt: Path = typer.Option(..., exists=True),
    config: Optional[Path] = typer.Option(None, help="Path to override config.json"),
    out_dir: Path = typer.Option("run_compress/int8"),
):
    model = load_model_with_config(ckpt, config=config)
    qmodel = dynamic_int8(model)
    config_path = config or (ckpt.parent / "config.json")
    write_model_folder(qmodel, out_dir, config_path)


@app.command()
def qat(
    ckpt: Path = typer.Option(..., exists=True),
    config: Optional[Path] = typer.Option(None, help="Path to override config.json"),
    dataset: Path = typer.Option(..., exists=True),
    epochs: int = typer.Option(3),
    out_dir: Path = typer.Option("run_compress/qat_int8"),
):
    model = load_model_with_config(ckpt, dataset=dataset, config=config)
    ds = OMTFDataset.load_dataset(dataset)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    model = prepare_qat(model)
    qmodel = fine_tune_qat(model, loader, epochs=epochs)
    config_path = config or (ckpt.parent / "config.json")
    write_model_folder(qmodel, out_dir, config_path)


@app.command()
def eval(
    baseline: Path = typer.Option(..., exists=True),
    compressed: Path = typer.Option(..., exists=True),
    dataset: Path = typer.Option(..., exists=True),
    out: Path = typer.Option("compress_report.json"),
):
    report = evaluate_pair(baseline, compressed, dataset)
    out.write_text(json.dumps(report, indent=2))
    typer.echo(json.dumps(report, indent=2))
    typer.echo(f"wrote {out}")


@app.command("sweep-precision")
def sweep_precision(
    ckpt: Path = typer.Option(..., exists=True),
    dataset: Path = typer.Option(..., exists=True),
    out: Path = typer.Option("precision_report.json"),
    widths: List[int] = typer.Option([16, 12, 10]),
    ints: List[int] = typer.Option([6, 5, 4]),
):
    """
    Try every combination of --widths W and --ints I (same length or broadcast).
    """
    import itertools
    if len(ints) == 1:
        ints = ints * len(widths)
    precisions = list(zip(widths, ints))
    from .precision_sweep import sweep as _sweep
    res = _sweep(ckpt, dataset, precisions, out_json=out)
    typer.echo(json.dumps(res, indent=2))


@app.command("edge-prune")
def edge_prune_cmd(
    dataset_in: Path = typer.Option(..., exists=True),
    percentile: float = typer.Option(0.2, help="Fraction edges to drop"),
    out: Path = typer.Option("dataset_edgepruned.pt"),
):
    from torch import load, save
    graphs = load(dataset_in)
    from gnn_omtf.compress.pruning import edge_prune_batch, graph_sparsity
    pruned = edge_prune_batch(graphs, percentile)
    save(pruned, out)
    typer.echo(f"saved pruned dataset → {out}")
    typer.echo(graph_sparsity(pruned))


def _main(): app()
if __name__ == "__main__":
    _main()
