from __future__ import annotations
import typer

import json
import logging
from pathlib import Path


from .hpo_optuna import run_study, save_best
from .nas_darts import run_search

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------ #
# HPO command
# ------------------------------------------------------------------ #
@app.command()
def hpo(
    graphs: Path = typer.Option(..., exists=True, help="*.pt dataset"),
    config: Path = typer.Option(
        None, exists=True, help="YAML created at conversion time"
    ),
    trials: int = typer.Option(40),
    out: Path = typer.Option("best_hparams.json"),
):
    """Bayesian hyper-parameter optimisation via Optuna."""
    best = run_study(graphs, n_trials=trials, config=config)
    save_best(best, out)
    typer.echo(json.dumps(best, indent=2))


# ------------------------------------------------------------------ #
# NAS command
# ------------------------------------------------------------------ #
@app.command()
def nas(
    graphs: Path = typer.Option(..., exists=True),
    epochs: int = typer.Option(50),
    out: Path = typer.Option("best_arch.yaml"),
):
    """(Placeholder) Differentiable Architecture Search."""
    best = run_search(graphs, epochs=epochs, out_yaml=out)
    typer.echo(best)


def _main():  # entry-point used by pyproject.toml
    app()


if __name__ == "__main__":  # pragma: no cover
    _main()
