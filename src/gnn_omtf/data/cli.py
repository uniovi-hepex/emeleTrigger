"""
Command-line utilities for dataset manipulation.

Once the package is installed this exposes:

    $ gnn-omtf-data convert --root-dir /path/to/root --stub-vars stubPhi stubEta ...
    $ gnn-omtf-data plot    --dataset dataset.pt  --idx 0
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer

from .dataset import OMTFDataset

app = typer.Typer(add_completion=False)
log = logging.getLogger("gnn_omtf.data.cli")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _common_dataset_kwargs(
    *,
    root_dir: Path,
    tree_name: str,
    muon_vars: Optional[List[str]],
    omtf_vars: Optional[List[str]],
    stub_vars: Optional[List[str]],
    target_vars: Optional[List[str]],
    task: str,
    max_files: Optional[int],
    max_events: Optional[int],
    debug: bool,
) -> Dict[str, Any]:
    """
    Build the kwargs dictionary that will be forwarded to `OMTFDataset`.

    We *only* insert list-type arguments when the user explicitly provided
    them on the CLI.  This way, when they are absent the YAML file can supply
    its default lists without being overwritten by `[]`.
    """
    cfg: Dict[str, Any] = dict(
        root_dir=str(root_dir),
        tree_name=tree_name,
        task=task,
        max_files=max_files,
        max_events=max_events,
        debug=debug,
    )
    if muon_vars:
        cfg["muon_vars"] = muon_vars
    if omtf_vars:
        cfg["omtf_vars"] = omtf_vars
    if stub_vars:
        cfg["stub_vars"] = stub_vars
    if target_vars:
        cfg["target_vars"] = target_vars
    return cfg


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command("convert")
def convert_root_to_pt(
    root_dir: Path = typer.Option(..., help="Directory or .root file(s)"),
    config: Optional[Path] = typer.Option(
        None, help="Optional YAML config file whose keys override defaults."
    ),
    output: Path = typer.Option(
        "dataset.pt", help="Where to save the resulting torch file"
    ),
    tree_name: str = typer.Option(
        "simOmtfPhase2Digis/OMTFHitsTree", help="TTree path inside ROOT file"
    ),
    # NOTE:  the default is **None**, *not* []  ➜ lets us tell if the user typed it.
    muon_vars: Optional[List[str]] = typer.Option(
        None, help="Muon variables to extract"
    ),
    omtf_vars: Optional[List[str]] = typer.Option(
        None, help="OMTF/global variables"
    ),
    stub_vars: Optional[List[str]] = typer.Option(
        None, help="Stub‐level (node) variables"
    ),
    target_vars: Optional[List[str]] = typer.Option(
        None, help="Regression / classification targets"
    ),
    task: str = typer.Option("regression", help="Task type: regression | classification"),
    max_files: Optional[int] = typer.Option(None, help="Cut after N files"),
    max_events: Optional[int] = typer.Option(None, help="Cut after N events"),
    debug: bool = typer.Option(False, help="Verbose logging"),
):
    """Read ROOT file(s) and serialise the dataset to a *.pt* file."""
    if debug:
        logging.basicConfig(level=logging.INFO)

    ds = OMTFDataset(
        **_common_dataset_kwargs(
            root_dir=root_dir,
            tree_name=tree_name,
            muon_vars=muon_vars,
            omtf_vars=omtf_vars,
            stub_vars=stub_vars,
            target_vars=target_vars,
            task=task,
            max_files=max_files,
            max_events=max_events,
            debug=debug,
        ),
        config=str(config) if config else None,
    )
    ds.save_dataset(output)
    log.info("Saved %s graphs → %s", len(ds), output)


@app.command("plot")
def plot_graph(
    dataset: Path = typer.Option(..., help="*.pt file produced by `convert`"),
    idx: int = typer.Option(0, help="Graph index to draw"),
    out_png: Optional[Path] = typer.Option(None, help="Save PNG instead of showing"),
):
    """Visualise one graph from a saved *.pt dataset."""
    ds = OMTFDataset.load_dataset(dataset)
    ds.plot_graph(idx, filename=str(out_png) if out_png else None)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def _main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    _main()
