# OMTFGnnModel.py

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, GINConv

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, gnn_type='GCN'):
        """
        Initializes the GNN model.

        Parameters:
        - input_dim (int): Dimension of input node features.
        - hidden_dim (int): Dimension of hidden layers.
        - output_dim (int): Dimension of output layer.
        - num_layers (int): Number of GNN layers.
        - dropout (float): Dropout rate.
        - gnn_type (str): Type of GNN layer ('GCN', 'GraphSAGE', 'GAT', 'GIN').
        """
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Define GNN layers based on gnn_type
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if hasattr(self, 'batch_norms') else None

        if gnn_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'GraphSAGE':
            self.convs.append(GraphSAGE(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GraphSAGE(hidden_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim))
        elif gnn_type == 'GIN':
            self.convs.append(GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
            for _ in range(num_layers - 1):
                self.convs.append(GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            if hasattr(self, 'batch_norms') and len(self.batch_norms) >= self.convs.index(conv) + 1:
                x = self.batch_norms[self.convs.index(conv)](x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)

        out = self.linear(x)
        return out
