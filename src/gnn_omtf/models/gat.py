from __future__ import annotations

"""Graph-Attention-Network regressor."""

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from .gnn_base import BaseGNN, register_model

@register_model("gat")
class GATRegressor(BaseGNN):
    """2-layer GAT + global max⊕mean pooling → MLP head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout_p: float = 0.2,
        heads: int = 1,
        edge_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.output_dim = out_channels
        self.dropout_p = dropout_p

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
        )

        self.fc = nn.Linear(hidden_channels * 2, out_channels)

    # ------------------------------------------------------------------ #

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))

        x = torch.cat(
            [global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1
        )
        return self.fc(x).squeeze(-1)
