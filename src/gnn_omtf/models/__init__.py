from __future__ import annotations

# core
from .gnn_base import BaseGNN

# GAT
from .gat import (
    GATRegressor,
    GATClassifier,
)

# GCN
from .gcn import (
    GCNRegressor,
    GCNClassifier,        # graph‑level
    GCNNodeClassifier,    # node‑level
)

# Graph‑SAGE
from .sage import (
    GraphSAGERegressor,
    GraphSAGEClassifier,
    GraphSAGESimpleClassifier,
)

# MPL
from .mpl import (
    MPLNNRegressor,
    MPLNNClassifier,
)

from .sumGNN import (
    SumGNNRegressor,
    SumGNNClassifier,
    SumGNNNodeClassifier,
    SumGNNSimpleClassifier,
)

__all__ = [
    # base
    "BaseGNN",
    # gat
    "GATRegressor",
    "GATClassifier",
    # gcn
    "GCNRegressor",
    "GCNClassifier",
    "GCNNodeClassifier",
    # sage
    "GraphSAGERegressor",
    "GraphSAGEClassifier",
    "GraphSAGESimpleClassifier",
    # mpl
    "MPLNNRegressor",
    "MPLNNClassifier",
    # sumGNN
    "SumGNNRegressor",
    "SumGNNClassifier",
    "SumGNNNodeClassifier",
    "SumGNNSimpleClassifier",
]
