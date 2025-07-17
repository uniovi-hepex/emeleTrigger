from __future__ import annotations
"""
Sum‑Aggregator Graph Neural Networks (SumGNN)
─────────────────────────────────────────────
• sum_gnn_regressor         – graph‑level scalar regression
• sum_gnn_classifier        – graph‑level classification
• sum_gnn_node_classifier   – node‑level logits (no pooling)
• sum_gnn_simple_classifier – one‑layer SumGNN for quick prototyping
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_add_pool

from .gnn_base import BaseGNN, register_model

__all__ = [
    "SumGNNRegressor",
    "SumGNNClassifier",
    "SumGNNNodeClassifier",
    "SumGNNSimpleClassifier",
]

class SumConv(MessagePassing):
    """Basic message‑passing where messages are simply summed neighbors."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")  # “add” aggregation
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # propagate() aggregates neighbor x_j via message()
        out = self.propagate(edge_index, x=x)
        return self.lin(out)

    def message(self, x_j: torch.Tensor):
        # message is identity: pass neighbor features directly
        return x_j

class _SumGNNBackbone(nn.Module):
    """Two SumConv layers ➜ nonlinearity & dropout ➜ global add‑pooling."""
    def __init__(self, in_channels: int, hidden_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.conv1 = SumConv(in_channels, hidden_channels)
        self.conv2 = SumConv(hidden_channels, hidden_channels)
        self.dropout_p = dropout_p

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        # 1st layer: sum neighbors, linear, ReLU, dropout
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # 2nd layer: same
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # graph‑level readout by summing node embeddings per graph
        return global_add_pool(x, batch)  # [n_graphs, hidden_channels]

@register_model("sum_gnn_regressor")
class SumGNNRegressor(BaseGNN):
    """SumGNN backbone ➜ small MLP ➜ single scalar regression."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = _SumGNNBackbone(in_channels, hidden_channels, dropout_p)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)           # [n_graphs, hidden]
        return self.mlp(emb).squeeze(-1)    # [n_graphs]

@register_model("sum_gnn_classifier")
class SumGNNClassifier(BaseGNN):
    """SumGNN backbone ➜ linear head ➜ logits for graph classification."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = _SumGNNBackbone(in_channels, hidden_channels, dropout_p)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)  # [n_graphs, hidden]
        # squeeze for binary vs. multi‑class
        return self.fc(emb).squeeze(-1 if self.fc.out_features == 1 else 0)

@register_model("sum_gnn_node_classifier")
class SumGNNNodeClassifier(BaseGNN):
    """Two‑layer SumConv ➜ node logits (no pooling)."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = SumConv(in_channels, hidden_channels)
        self.conv2 = SumConv(hidden_channels, out_channels)
        self.dropout_p = dropout_p

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.conv2(x, edge_index)  # [num_nodes, out_channels]

@register_model("sum_gnn_simple_classifier")
class SumGNNSimpleClassifier(BaseGNN):
    """One‑layer SumConv ➜ global add‑pool ➜ linear head.
       Minimal model for very fast prototyping."""
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv = SumConv(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout_p = dropout_p

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        emb = global_add_pool(x, batch)
        out = self.fc(emb)
        return out.squeeze(-1 if self.fc.out_features == 1 else 0)
