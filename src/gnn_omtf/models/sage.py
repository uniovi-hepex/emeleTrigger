from __future__ import annotations


"""Graph-SAGE regressor."""


import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from .gnn_base import BaseGNN, register_model


@register_model("sage")
@register_model("graphsage")  # alias
class GraphSAGEModel(BaseGNN):
    """4-layer GraphSAGE + mean-pool → MLP head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.output_dim = out_channels

        self.convs = nn.ModuleList([
            SAGEConv(in_channels, hidden_channels * 4),
            SAGEConv(hidden_channels * 4, hidden_channels * 2),
            SAGEConv(hidden_channels * 2, hidden_channels * 2),
            SAGEConv(hidden_channels * 2, hidden_channels * 2),
        ])

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
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1)
