"""
gnn_omtf — Graph-Neural-Network toolkit for OMTF studies
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
# Re-export frequently-used classes so users can do:
#   from gnn_omtf import BaseGNN, GATRegressor, …
# ------------------------------------------------------------------ #
from .models import (        # ←  this is now valid
    BaseGNN,
    GATRegressor,
    GCNRegressor,
    GraphSAGEModel,
    MPLNNRegressor,
)

__all__ = [
    "BaseGNN",
    "GATRegressor",
    "GCNRegressor",
    "GraphSAGEModel",
    "MPLNNRegressor",
    "__version__",
]
