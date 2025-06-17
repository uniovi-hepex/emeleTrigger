from __future__ import annotations

"""Graph-Convolutional-Network variants."""



import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool

from .gnn_base import BaseGNN, register_model

# ---------------------------------------------------------------------------- #
# Regression – 4×GCN → 4×MLP
# ---------------------------------------------------------------------------- #


@register_model("gcn")
class GCNRegressor(BaseGNN):
    """Deep GCN regressor with global-max pooling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.output_dim = out_channels
        self.dropout_p = dropout_p

        self.convs = nn.ModuleList(
            [
                GCNConv(in_channels, hidden_channels * 4),
                GCNConv(hidden_channels * 4, hidden_channels * 2),
                GCNConv(hidden_channels * 2, hidden_channels * 2),
                GCNConv(hidden_channels * 2, hidden_channels * 2),
            ]
        )

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
        x, edge_index, edge_weight, batch = (
            data.x.float(),
            data.edge_index,
            getattr(data, "edge_weight", None),
            data.batch,
        )
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = global_max_pool(x, batch)
        return self.mlp(x).squeeze(-1)



# ---------------------------------------------------------------------------- #
# Node-level classifier – 3×GCN, returns logits per node
# ---------------------------------------------------------------------------- #


@register_model("gcn_node_classifier")
class GCNNodeClassifier(BaseGNN):
    """Simple 3-layer GCN for per-node classification (no pooling)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.output_dim = out_channels
        self.dropout_p = dropout_p

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):  # type: ignore[override]
        x, edge_index = data.x.float(), data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.conv3(x, edge_index)  # logits per node
