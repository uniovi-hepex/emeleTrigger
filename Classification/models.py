import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class EdgeGNNClassifier(nn.Module):
    def __init__(self, in_channels, edge_in, hidden_channels):
        super().__init__()
        # Encoder que usa los atributos de aristas
        self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_in, heads=4, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_in, heads=4, concat=False)
        # MLP para clasificar aristas
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels + edge_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Obtener embeddings nodales
        x = nn.functional.relu(self.conv1(x, edge_index, edge_attr))  # -> [num_nodes, hidden_channels]
        x = nn.functional.relu(self.conv2(x, edge_index, edge_attr))
        
        # 2. Recolectar embeddings de origen y destino
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        # 3. Concatenar embeddings + atributos de arista
        edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)

        # 4. Clasificador: logit por arista
        return self.edge_mlp(edge_feat).view(-1)


class EdgeGNNClassifier_OneLayer(nn.Module):
    def __init__(self, in_channels, edge_in, hidden_channels):
        super().__init__()
        # Encoder que usa los atributos de aristas
        self.conv = TransformerConv(in_channels, hidden_channels, edge_dim=edge_in, heads=4, concat=False)
        # MLP para clasificar aristas
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*hidden_channels + edge_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # 1. Obtener embeddings nodales
        x = self.conv(x, edge_index, edge_attr)  # -> [num_nodes, hidden_channels]

        # 2. Recolectar embeddings de origen y destino
        src, dst = edge_index
        x_src = x[src]
        x_dst = x[dst]

        # 3. Concatenar embeddings + atributos de arista
        edge_feat = torch.cat([x_src, edge_attr, x_dst], dim=-1)

        # 4. Clasificador: logit por arista
        return self.edge_mlp(edge_feat).view(-1)
