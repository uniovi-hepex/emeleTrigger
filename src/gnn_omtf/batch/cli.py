from __future__ import annotations
import typer
"""
CLI front-end:  `gnn-omtf-batch …`

Two convenience sub-commands:

* **dataset**  – spawn many `gnn-omtf-data convert …` jobs
* **train**    – spawn many `gnn-omtf-train run …`   jobs
"""

import logging
from pathlib import Path
from typing import List, Optional

import yaml

from .condor import render_and_submit

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #
def _chunks(seq: List[Path], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ------------------------------------------------------------------ #
# dataset sub-command
# ------------------------------------------------------------------ #
@app.command()
def dataset(
    root_dir: Path = typer.Option(..., help="Dir with ROOT files or sub-dirs"),
    output_dir: Path = typer.Option(..., help="Where *.pt will be written"),
    config: Path = typer.Option(..., help="YAML config for OMTFDataset"),
    task: str = typer.Option("regression", help="regression | classification"),
    queue: str = typer.Option("workday", help="HTCondor JobFlavour"),
    files_per_job: int = typer.Option(1, help="ROOT files per condor job"),
    dry_run: bool = typer.Option(False, help="Print submit file only"),
):
    """Spawn many *gnn-omtf-data convert* jobs."""
    root_files = sorted(root_dir.glob("*.root"))
    if not root_files:
        typer.echo(f"No *.root in {root_dir}", err=True)
        raise typer.Exit(1)

    jobs = []
    for idx, chunk in enumerate(_chunks(root_files, files_per_job), 1):
        jobs.append(
            dict(
                idx=idx,
                cmd="gnn-omtf-data",
                args=(
                    f"convert --root-dir {' '.join(map(str, chunk))} "
                    f"--output {output_dir / f'ds_{task}_{idx:04d}.pt'} "
                    f"--task {task} --config {config}"
                ),
                queue=queue,
            )
        )

    render_and_submit(jobs, dry_run=dry_run)


# ------------------------------------------------------------------ #
# train sub-command
# ------------------------------------------------------------------ #
@app.command()
def train(
    graphs_dir: Path = typer.Option(..., help="Folder with *.pt graphs"),
    output_dir: Path = typer.Option(..., help="Where to place model / plots"),
    model: str = typer.Option("gat"),
    hidden: int = typer.Option(32),
    batch: int = typer.Option(1024),
    epochs: int = typer.Option(50),
    lr: float = typer.Option(5e-4),
    queue: str = typer.Option("workday"),
    gpus: bool = typer.Option(True, help="request_gpus=1"),
    chunk_pt: int = typer.Option(20, help="*.pt files per job"),
    dry_run: bool = typer.Option(False),
):
    """Spawn *gnn-omtf-train run* jobs."""
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        typer.echo(f"No *.pt in {graphs_dir}", err=True)
        raise typer.Exit(1)

    jobs = []
    for idx, chunk in enumerate(_chunks(pt_files, chunk_pt), 1):
        jobs.append(
            dict(
                idx=idx,
                cmd="gnn-omtf-train",
                args=(
                    f"run --graphs {' '.join(map(str, chunk))} "
                    f"--out-dir {output_dir / f'job_{idx:04d}'} "
                    f"--model {model} --hidden-dim {hidden} "
                    f"--batch-size {batch} --epochs {epochs} --lr {lr}"
                ),
                queue=queue,
            )
        )

    render_and_submit(jobs, dry_run=dry_run)


def _main():
    app()


if __name__ == "__main__":
    _main()
