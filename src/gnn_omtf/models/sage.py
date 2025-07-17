from __future__ import annotations
"""
Graph‑SAGE variants
───────────────────
• sage_regressor          – regression
• sage_classifier         – classification
(aliases graphsage_* are also registered for convenience)
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from .gnn_base import BaseGNN, register_model


# --------------------------------------------------------------------------- #
# Shared 4‑layer backbone
# --------------------------------------------------------------------------- #
class _SAGEBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.convs = nn.ModuleList([
            SAGEConv(in_channels,            hidden_channels * 4),
            SAGEConv(hidden_channels * 4,    hidden_channels * 2),
            SAGEConv(hidden_channels * 2,    hidden_channels * 2),
            SAGEConv(hidden_channels * 2,    hidden_channels * 2),
        ])

    # ------------------------------------------------------------------ #
    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return global_mean_pool(x, batch)           # [n_graphs, hidden*2]


# --------------------------------------------------------------------------- #
# 1) Graph‑level regression head
# --------------------------------------------------------------------------- #
@register_model("sage_regressor")
@register_model("graphsage_regressor")   # alias
class GraphSAGERegressor(BaseGNN):
    """4‑layer GraphSAGE ➜ mean‑pool ➜ MLP regression."""

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int = 1,
        dropout_p:     float = 0.2,
    ):
        super().__init__()
        self.backbone = _SAGEBackbone(in_channels, hidden_channels, dropout_p)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)
        return self.mlp(emb).squeeze(-1)


# --------------------------------------------------------------------------- #
# 2) Graph‑level classification head
# --------------------------------------------------------------------------- #
@register_model("sage_classifier")
@register_model("graphsage_classifier")   # alias
class GraphSAGEClassifier(BaseGNN):
    """4‑layer GraphSAGE ➜ mean‑pool ➜ linear classifier."""

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int = 1,    # 1 = binary, >1 = multi‑class
        dropout_p:     float = 0.2,
    ):
        super().__init__()
        self.backbone = _SAGEBackbone(in_channels, hidden_channels, dropout_p)
        self.fc       = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)
        return self.fc(emb).squeeze(-1 if self.fc.out_features == 1 else 0)

@register_model("sage_simple_classifier")
@register_model("graphsage_simple_classifier")  # alias
class GraphSAGESimpleClassifier(BaseGNN):
    """1-layer GraphSAGE ➜ mean-pool ➜ MLP classifier."""

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        out_channels:    int = 1,    # 1 = binary, >1 = multi-class
        dropout_p:       float = 0.2,
    ):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)

        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1 if self.mlp[-1].out_features == 1 else 0)
