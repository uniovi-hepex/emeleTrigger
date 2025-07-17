from __future__ import annotations
import json, logging, pathlib, typer, torch, ast
from pathlib import Path
from types import SimpleNamespace
from .trainer import GraphTrainer
from ..data.dataset import OMTFDataset
from ..data.transforms import NormalizeNodeEdgesAndDropTwoFeatures

app = typer.Typer(add_completion=False)
logging.basicConfig(level=logging.INFO)

def _convert_paths(obj):
    if isinstance(obj, dict):
        return {k: _convert_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_paths(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj

@app.command()
def run(
    graphs:     Path = typer.Option(..., help="*.pt file with saved graphs"),
    out_dir:    Path = typer.Option("runs", help="Output folder"),
    model:      str  = "sage_simple_classifier",
    model_args: str  = typer.Option("{}", help="Extra model arguments as JSON string, e.g. '{\"hidden_dim\": 64, \"heads\": 2}'"),
    batch_size: int  = 512,
    lr:         float= 5e-4,
    epochs:     int  = 50,
    patience:   int  = 5,
    seed:       int  = 1,
    amp:        bool = False,
    device:     str  = typer.Option("auto", help="'cpu', 'cuda' or 'auto'"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = OMTFDataset(
        dataset=torch.load(graphs, weights_only=False),
        pre_transform=NormalizeNodeEdgesAndDropTwoFeatures,
    )

    # Determine dataset feature dims
    num_node = ds[0].x.size(1) if ds[0].x.dim() > 1 else 1
    num_edge = ds[0].edge_attr.size(1) if getattr(ds[0], "edge_attr", None) is not None else 0

    # Parse and normalize model arguments
    model_kwargs = ast.literal_eval(model_args)
    # Legacy: allow "hidden_dim"
    if "hidden_dim" in model_kwargs:
        model_kwargs["hidden_channels"] = model_kwargs.pop("hidden_dim")
    # Inject required channels
    model_kwargs["in_channels"] = num_node
    model_kwargs["edge_dim"] = num_edge

    # Build configuration for trainer
    cfg = dict(
        model=model,
        model_args=model_kwargs,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        patience=patience,
        seed=seed,
        out_dir=out_dir,
        amp=amp,
        device=device,
        num_node_features=num_node,
        num_edge_features=num_edge,
    )

    trainer = GraphTrainer(SimpleNamespace(**cfg))
    metrics = trainer.fit(ds)

    # ---- bookkeeping --------------------------------------------------------
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(json.dumps(_convert_paths(cfg), indent=2))

    print("✅ finished, metrics:", metrics)


def _main():  # pragma: no cover
    app()

if __name__ == "__main__":
    _main()
