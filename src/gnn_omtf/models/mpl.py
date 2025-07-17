from __future__ import annotations
"""
Custom Message‑Passing architecture (“MPL”)
──────────────────────────────────────────
• mpl_regressor  – graph‑level scalar regression
• mpl_classifier – graph‑level classification (binary / multi‑class)
"""
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
# Low‑level operator (unchanged)
# --------------------------------------------------------------------------- #
class _MPL(MessagePassing):
    """Single MPL layer with learnable attention weights."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.msg_mlp  = nn.Linear(in_channels * 2, out_channels)
        self.node_mlp = nn.Linear(in_channels, out_channels)

        self.w_msg  = nn.Linear(2 * out_channels, 1)
        self.w_node = nn.Linear(2 * out_channels, 1)

        self.alpha_mlp = nn.Sequential(nn.Linear(in_channels, 16), nn.Tanh())
        self.beta_mlp  = nn.Sequential(nn.Linear(out_channels, 16), nn.Tanh())
        self.score     = nn.Linear(16, 1)

    # ------------------------------------------------------------------ #
    def forward(self, x, edge_index):  # noqa: D401
        msg  = self.propagate(edge_index, x=x)
        x_t  = F.relu(self.node_mlp(x))

        w1 = torch.sigmoid(self.w_msg(torch.cat([x_t, msg], dim=1)))
        w2 = torch.sigmoid(self.w_node(torch.cat([x_t, msg], dim=1)))
        return w1 * msg + w2 * x_t

    def message(self, x_i, x_j, edge_index):  # type: ignore[override]
        msg = F.relu(self.msg_mlp(torch.cat([x_i, x_j - x_i], dim=1)))
        w   = self.score(self.alpha_mlp(x_i) * self.beta_mlp(msg))          # element‑wise
        w   = softmax(w, edge_index[0])                                     # normalise per‑source
        return msg * w


# --------------------------------------------------------------------------- #
# Shared 4‑layer backbone + dual attentional pooling
# --------------------------------------------------------------------------- #
class _MPLBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout_p: float = 0.2):
        super().__init__()
        self.dropout_p = dropout_p
        self.conv1 = _MPL(in_channels,          hidden_channels * 2)
        self.conv2 = _MPL(hidden_channels * 2,  hidden_channels)
        self.conv3 = _MPL(hidden_channels,      hidden_channels)
        self.conv4 = _MPL(hidden_channels,      hidden_channels)

        self.pool1 = AttentionalAggregation(nn.Linear(hidden_channels, 1))
        self.pool2 = AttentionalAggregation(nn.Linear(hidden_channels, 1))

    # ------------------------------------------------------------------ #
    def forward(self, data):  # type: ignore[override]
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x1 = self.pool1(x, batch)                       # [n_graphs, hidden]

        x = F.relu(self.conv3(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x2 = self.pool2(x, batch)                       # [n_graphs, hidden]

        return torch.cat([x1, x2], dim=1)               # [n_graphs, hidden*2]


# --------------------------------------------------------------------------- #
# 1) Graph‑level regression head
# --------------------------------------------------------------------------- #
@register_model("mpl_regressor")
class MPLNNRegressor(BaseGNN):
    """Stacked MPL ➜ attentional pooling ➜ MLP regression head."""

    def __init__(
        self,
        in_channels:    int,
        hidden_channels: int = 64,
        out_channels:     int = 1,
        dropout_p:     float = 0.2,
    ):
        super().__init__()
        self.backbone = _MPLBackbone(in_channels, hidden_channels, dropout_p)

        self.head = nn.Sequential(
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
        return self.head(emb).squeeze(-1)


# --------------------------------------------------------------------------- #
# 2) Graph‑level classification head
# --------------------------------------------------------------------------- #
@register_model("mpl_classifier")
class MPLNNClassifier(BaseGNN):
    """Stacked MPL ➜ attentional pooling ➜ linear classifier."""

    def __init__(
        self,
        in_channels:    int,
        hidden_channels: int = 64,
        out_channels:     int = 1,   # 1 = binary; >1 = multi‑class
        dropout_p:     float = 0.2,
    ):
        super().__init__()
        self.backbone = _MPLBackbone(in_channels, hidden_channels, dropout_p)
        self.fc       = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):  # type: ignore[override]
        emb = self.backbone(data)
        return self.fc(emb).squeeze(-1 if self.fc.out_features == 1 else 0)
