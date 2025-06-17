from __future__ import annotations

"""Custom Message-Passing architecture (“MPL”)."""



import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    AttentionalAggregation,
    MessagePassing,
)
from torch_geometric.utils import softmax

from .gnn_base import BaseGNN, register_model

# --------------------------------------------------------------------------- #
# Low-level operator
# --------------------------------------------------------------------------- #


class _MPL(MessagePassing):
    """One MPL layer with learnable attention weights."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(aggr="add")
        self.msg_mlp = nn.Linear(in_channels * 2, out_channels)
        self.node_mlp = nn.Linear(in_channels, out_channels)

        self.w_msg = nn.Linear(2 * out_channels, 1)
        self.w_node = nn.Linear(2 * out_channels, 1)

        self.alpha_mlp = nn.Sequential(
            nn.Linear(in_channels, 16),
            nn.Tanh(),
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(out_channels, 16),
            nn.Tanh(),
        )
        self.score = nn.Linear(16, 1)

    # ------------------------------------------------------------------ #

    def forward(self, x, edge_index):  # noqa: D401
        msg = self.propagate(edge_index, x=x)
        x_t = F.relu(self.node_mlp(x))

        w1 = torch.sigmoid(self.w_msg(torch.cat([x_t, msg], dim=1)))
        w2 = torch.sigmoid(self.w_node(torch.cat([x_t, msg], dim=1)))
        return w1 * msg + w2 * x_t

    def message(self, x_i, x_j, edge_index):  # type: ignore[override]
        msg = F.relu(self.msg_mlp(torch.cat([x_i, x_j - x_i], dim=1)))
        w = self.score(
            self.alpha_mlp(x_i) * self.beta_mlp(msg)
        )  # element-wise product
        w = softmax(w, edge_index[0])  # normalize across neighbours
        return msg * w


# --------------------------------------------------------------------------- #
# Full network – registered as “mpl”
# --------------------------------------------------------------------------- #


@register_model("mpl")
class MPLNNRegressor(BaseGNN):
    """Stack 4×MPL, dual attentional pooling + small MLP head."""

    def __init__(self, in_channels: int, dropout_p: float = 0.2) -> None:
        super().__init__()
        self.dropout_p = dropout_p
        self.input_dim = in_channels
        self.hidden_dim = 64
        self.output_dim = 1

        self.conv1 = _MPL(in_channels, 128)
        self.conv2 = _MPL(128, 64)
        self.conv3 = _MPL(64, 64)
        self.conv4 = _MPL(64, 64)

        self.global_att_pool1 = AttentionalAggregation(nn.Linear(64, 1))
        self.global_att_pool2 = AttentionalAggregation(nn.Linear(64, 1))

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    # ------------------------------------------------------------------ #

    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x1 = self.global_att_pool1(x, batch)

        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x2 = self.global_att_pool2(x, batch)

        x = torch.cat([x1, x2], dim=1)  # (B, 128)
        return self.head(x).squeeze(-1)
