from __future__ import annotations
"""
Graph‑Convolutional‑Network (GCN) models
────────────────────────────────────────
•  gcn_regressor             – graph‑level scalar regression
•  gcn_classifier            – graph‑level classification (binary / multi‑class)
•  gcn_node_classifier       – node‑level logits (kept for convenience)
•  gcn_simple_classifier     – lightweight one‑layer GCN for quick prototyping
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool

from .gnn_base import BaseGNN, register_model


# --------------------------------------------------------------------------- #
# Shared four‑layer backbone
# --------------------------------------------------------------------------- #
class _GCNBackbone(nn.Module):
    """4×GCN ➜ global‑max pool ➜ embedding vector."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.convs = nn.ModuleList([
            GCNConv(in_channels, hidden_channels * 4),
            GCNConv(hidden_channels * 4, hidden_channels * 2),
            GCNConv(hidden_channels * 2, hidden_channels * 2),
            GCNConv(hidden_channels * 2, hidden_channels * 2),
        ])

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = getattr(data, "edge_weight", None)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        return global_max_pool(x, batch)  # [n_graphs, hidden*2]


# --------------------------------------------------------------------------- #
# 1) Graph‑level regression model
# --------------------------------------------------------------------------- #
@register_model("gcn_regressor")
class GCNRegressor(BaseGNN):
    """GCN backbone ➜ MLP ➜ single regression value."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = _GCNBackbone(in_channels, hidden_channels, dropout_p)
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
# 2) Graph‑level classification model
# --------------------------------------------------------------------------- #
@register_model("gcn_classifier")
class GCNClassifier(BaseGNN):
    """GCN backbone ➜ linear head ➜ logits for graph classification."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = _GCNBackbone(in_channels, hidden_channels, dropout_p)
        self.fc = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)
        return self.fc(emb).squeeze(-1 if self.fc.out_features == 1 else 0)


# --------------------------------------------------------------------------- #
# 3) Node‑level classifier (unchanged)
# --------------------------------------------------------------------------- #
@register_model("gcn_node_classifier")
class GCNNodeClassifier(BaseGNN):
    """3‑layer GCN that outputs logits per node (no pooling)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout_p = dropout_p

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.conv3(x, edge_index)


# --------------------------------------------------------------------------- #
# 4) Simple one‑layer GCN (lightweight)
# --------------------------------------------------------------------------- #
@register_model("gcn_simple_classifier")
class GCNSimple(BaseGNN):
    """One‑layer GCN ➜ global‑max pool ➜ linear head.
       Fast, minimal model for quick prototyping."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.dropout_p = dropout_p
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = getattr(data, "edge_weight", None)

        # single GCN layer + activation + dropout
        x = F.relu(self.conv(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # graph readout
        emb = global_max_pool(x, batch)

        # final head
        out = self.fc(emb)
        return out.squeeze(-1 if self.fc.out_features == 1 else 0)
