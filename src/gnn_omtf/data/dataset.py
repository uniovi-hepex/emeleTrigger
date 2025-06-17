"""OMTF → PyG `Dataset` converter.

Reads ROOT files (via *uproot* / *awkward*), builds per-event graphs and
returns a list-based dataset ready for a `DataLoader`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx

try:
    import uproot
    import awkward as ak
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "`uproot` and `awkward` are required for OMTFDataset. "
        "Install with:  pip install uproot awkward"
    ) from exc

import matplotlib.pyplot as plt
import networkx as nx
import yaml

from .converter_utils import (
    HW_ETA_TO_ETA_FACTOR,
    getEdgesFromLogicLayer,
    get_global_phi,
    get_stub_r,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class OMTFDataset(Dataset):
    """Convert Phase-2 OMTF ROOT ntuples into a list-based PyG dataset.

    Parameters
    ----------
    root_dir:
        Directory **or** single .root file to read.
    task:
        `"regression"` or `"classification"` – influences edge construction.
    pre_transform, transform:
        Standard PyG hooks.
    config:
        Optional YAML file – any key inside overrides constructor defaults.
    **kwargs:
        See attribute list below; all become instance attributes.
    """

    def __init__(self, **kwargs):
        # ------------------------------------------------------------------ #
        # 1. merge YAML config (if provided)
        # ------------------------------------------------------------------ #
        cfg_path = kwargs.get("config")
        if cfg_path:
            with open(cfg_path) as f:
                cfg_from_file = yaml.safe_load(f) or {}
            # CLI kwargs override YAML *after* the load
            kwargs = {**cfg_from_file, **kwargs}      #  ← single-line merge


        # ------------------------------------------------------------------ #
        # 2. store parameters
        # ------------------------------------------------------------------ #
        self.tree_name: str = kwargs.get("tree_name", "simOmtfPhase2Digis/OMTFHitsTree")
        self.muon_vars: Sequence[str] = kwargs.get("muon_vars", [])
        self.omtf_vars: Sequence[str] = kwargs.get("omtf_vars", [])
        self.stub_vars: Sequence[str] = kwargs.get("stub_vars", [])
        self.target_vars: Sequence[str] = kwargs.get("target_vars", [])
        self.task: str = kwargs.get("task", "regression").lower()
        self.max_files: int | None = kwargs.get("max_files")
        self.max_events: int | None = kwargs.get("max_events")
        self.debug: bool = bool(kwargs.get("debug", False))
        self.pre_transform = kwargs.get("pre_transform")
        self.transform = kwargs.get("transform")

        # 3. Load from dataset or ROOT
        if "dataset" in kwargs and kwargs["dataset"] is not None:
            self._data: List[Data] = kwargs["dataset"]
            self.root_dir = "<from .pt file>"
        elif "root_dir" in kwargs:
            self.root_dir: str = kwargs["root_dir"]
            self._data = self._load_from_root()
        else:
            raise ValueError("Must provide either `dataset` or `root_dir`")

        super().__init__(transform=self.transform, pre_transform=self.pre_transform)

    # ------------------------------------------------------------------ #
    # ROOT I/O helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _add_extra_vars(arr) -> Any:
        """Enrich awkward array with derived variables (in-place)."""
        if "stubR" not in arr.fields:
            arr["stubR"] = get_stub_r(
                arr["stubType"],
                arr["stubEta"],
                arr["stubLayer"],
                arr["stubQuality"],
            )
        arr["stubEtaG"] = arr["stubEta"] * HW_ETA_TO_ETA_FACTOR
        arr["stubPhiG"] = get_global_phi(arr["stubPhi"], arr["omtfProcessor"])

        if "inputStubR" not in arr.fields:
            arr["inputStubR"] = get_stub_r(
                arr["inputStubType"],
                arr["inputStubEta"],
                arr["inputStubLayer"],
                arr["inputStubQuality"],
            )
        arr["inputStubEtaG"] = arr["inputStubEta"] * HW_ETA_TO_ETA_FACTOR
        arr["inputStubPhiG"] = get_global_phi(
            arr["inputStubPhi"], arr["omtfProcessor"]
        )

        arr["muonQPt"] = arr["muonCharge"] * arr["muonPt"]
        # safe division: if pt==0 set q/pt to 0 (avoids inf→nan in loss)
        arr["muonQOverPt"] = (
            np.where(arr["muonPt"] != 0,
                     arr["muonCharge"] / arr["muonPt"],
                     0.0))
        return arr

    def _load_from_root(self) -> List[Data]:
        """Read ROOT/awkd files and build a list of `torch_geometric.data.Data`."""
        data_list: List[Data] = []
        files = self._discover_root_files()

        events_seen = 0
        for fname in files:
            log.info("Processing %s", fname)
            tree = uproot.open(fname)[self.tree_name]
            arr = self._add_extra_vars(tree.arrays(library="ak"))

            for evt in ak.to_list(arr):
                if self.max_events and events_seen >= self.max_events:
                    break
                if evt["stubNo"] == 0 or evt["inputStubNo"] == 0:
                    continue  # skip empty events

                node_feats = torch.tensor(
                    [evt[v] for v in self.stub_vars], dtype=torch.float32
                ).T  # shape (N, F)

                target = torch.tensor(
                    [evt[v] for v in self.target_vars], dtype=torch.float32
                )
                target = target.T if target.ndim == 2 else target

                edge_idx, edge_attr = self._build_edges(evt)

                data = Data(
                    x=node_feats,
                    edge_index=edge_idx,
                    edge_attr=edge_attr,
                    y=target,
                )

                data.muon_vars = torch.tensor(
                    [evt[v] for v in self.muon_vars], dtype=torch.float32
                )
                data.omtf_vars = torch.tensor(
                    [evt[v] for v in self.omtf_vars], dtype=torch.float32
                )

                if self.pre_transform:
                    data = self.pre_transform(data)
                if data is not None:
                    data_list.append(data)
                    events_seen += 1

                if events_seen % 500 == 0:
                    log.info("…processed %s events", events_seen)

        return data_list

    # ------------------------------------------------------------------ #
    # Graph construction helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _delta_phi(phi1, phi2):
        dphi = phi1 - phi2
        return (dphi + torch.pi) % (2 * torch.pi) - torch.pi

    @staticmethod
    def _delta_eta(eta1, eta2):
        return eta1 - eta2

    def _build_edges(self, evt, stub_prefix: str = "stub"):
        layer = evt[f"{stub_prefix}Layer"]
        phi = evt[f"{stub_prefix}Phi"]
        eta = evt[f"{stub_prefix}Eta"]

        edges, attrs = [], []
        for id1, lay1 in enumerate(layer):
            for id2, lay2 in enumerate(layer):
                if lay1 == lay2:
                    continue
                if lay2 in getEdgesFromLogicLayer(lay1):
                    dphi = self._delta_phi(phi[id1], phi[id2])
                    deta = self._delta_eta(eta[id1], eta[id2])
                    edges.append([id1, id2])
                    attrs.append([dphi, deta])

        edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float32)
        return edge_index, edge_attr

    # ------------------------------------------------------------------ #
    # PyG Dataset overrides
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:  # noqa: D401  (PyG expects this name)
        return len(self._data)

    def get(self, idx):
        return self._data[idx]

    # Alias for PyG compatibility
    __getitem__ = get

    # ------------------------------------------------------------------ #
    # Pretty printing & quick viz
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:  # noqa: D401
        return f"{self.__class__.__name__}(n_graphs={len(self)})"

    # quick plotting helpers kept for notebook usage
    def plot_graph(self, idx: int, filename: str | None = None, seed: int = 42):
        data = self.get(idx)
        G = to_networkx(data, to_undirected=True)
        labels = {i: int(data.x[i, 3]) for i in range(data.x.size(0))}
        pos = nx.spring_layout(G, seed=seed)

        nx.draw(G, pos, labels=labels, node_color="skyblue", node_size=500)
        plt.title(f"Example graph #{idx}")
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _discover_root_files(self) -> List[Path]:
        path = Path(self.root_dir)
        if path.is_file() and path.suffix == ".root":
            return [path]
        if path.is_dir():
            return sorted(path.glob("*.root"))
        raise ValueError(f"{self.root_dir!s} is neither a .root file nor a directory")
    
    def save_dataset(self, path: str | Path):
        """Serialise the internal list of graphs to disk."""
        torch.save(self._data, path)

    @classmethod
    def load_dataset(cls, path: str | Path) -> "OMTFDataset":
        """
        Factory that builds a **new** OMTFDataset directly from a .pt/.pkl file.

        Parameters
        ----------
        path : str | Path
            File created earlier with `torch.save(dataset, path)`.

        Returns
        -------
        OMTFDataset
            Fresh dataset, ready for a DataLoader.
        """
        graphs = torch.load(path, weights_only=False)
        log.info("✅  Dataset loaded from %s  (graphs = %d)", path, len(graphs))
        return cls(dataset=graphs)

