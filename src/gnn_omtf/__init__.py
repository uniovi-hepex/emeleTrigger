"""
gnn_omtf — Graph‑Neural‑Network toolkit for OMTF studies
"""
from importlib import metadata as _metadata

# ------------------------------------------------------------------ #
# Public version
# ------------------------------------------------------------------ #
try:
    __version__ = _metadata.version("gnn-omtf")
except _metadata.PackageNotFoundError:      # editable install
    __version__ = "0.0.0.dev0"

# ------------------------------------------------------------------ #
# Re‑export the most useful classes so users can do:
#   from gnn_omtf import GATClassifier, BaseGNN, …
# ------------------------------------------------------------------ #
from .models import (
    # base
    BaseGNN,
    # GAT
    GATRegressor,  GATClassifier,
    # GCN
    GCNRegressor,  GCNClassifier,  GCNNodeClassifier,
    # GraphSAGE
    GraphSAGERegressor, GraphSAGEClassifier, GraphSAGESimpleClassifier,
    # MPL
    MPLNNRegressor, MPLNNClassifier,
)

__all__ = [
    # base
    "BaseGNN",
    # GAT
    "GATRegressor", "GATClassifier",
    # GCN
    "GCNRegressor", "GCNClassifier", "GCNNodeClassifier",
    # GraphSAGE
    "GraphSAGERegressor", "GraphSAGEClassifier", "GraphSAGESimpleClassifier",
    # MPL
    "MPLNNRegressor", "MPLNNClassifier",
    # sumGNN
    "SumGNNRegressor", "SumGNNClassifier", "SumGNNNodeClassifier", "SumGNNSimpleClassifier",
    # misc
    "__version__",
]
