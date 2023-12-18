# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:sphinx
#     text_representation:
#       extension: .py
#       format_name: sphinx
#       format_version: '1.1'
#       jupytext_version: 1.16.0
# ---

from utils import get_test_data, visualize_graph, get_stub_r
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

import awkward as ak
import copy

import os

import h5py

import pandas as pd
from pandas.plotting import scatter_matrix

import numpy as np
from pyntcloud import PyntCloud

# viz
import networkx as nx
import matplotlib.pyplot as plt

# Do we need normalization?
def resize_and_format_data(points, image):
    pass

torch.set_default_dtype(torch.float64)
#torch.set_default_dtype(torch.float32)


viz=True

# Get data
branches = get_test_data('pd')
print(branches.head())

# Create graphs with nodes and edges according to the preset connection map
pg_graphs = convert_to_graphs(branches, viz)


# Create a DataLoader

# Once it's shuffled, we slice the data to split
train_dataset = pg_graphs[:int(121/2)]
test_dataset = pg_graphs[int(121/2):]

#for d in train_dataset:
#    d.to(torch.device("mps"))
#for d in test_dataset:
#    d.to(torch.device("mps"))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=True)


# Define a simple GNN model
import torch.nn.functional as F
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_channels, num_outputs):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_channels)
        self.conv2 = GCNConv(num_channels, num_outputs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.double()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.double()
        return x


# Instantiate the model, loss function, and optimizer
model=None
dotrain=True  # move this to runtime option
if not dotrain:
    model = torch.load('models/model.pth')
    model.eval()
else:
    model = GCN(pg_graphs[0].num_node_features,8,1)#.to(torch.device("mps"))


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100


def train():
    model.train()

    total_loss=0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        #total_loss += float(loss) * data.num_graphs
        total_loss += float(loss)

    return total_loss/len(train_loader.dataset)

def test():
    with torch.no_grad():
        model.eval()

        total_loss = 0
        for data in test_loader:
            out = model(data)
            loss = criterion(out, data.y.unsqueeze(1))
            #total_loss += float(loss) * data.num_graphs
            total_loss += float(loss)
        return total_loss/len(test_loader.dataset)

train_losses=[]
test_losses=[]


if dotrain:
    for epoch in range(num_epochs):
        train_loss = train()
        test_loss = test()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch:02d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
        torch.save(model, 'models/mode.pth')
    epochs = [ x for x in range(num_epochs)]
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, test_losses, label="Test loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss (rescaled)")
    plt.legend(loc='best')
    plt.show()


truth=[]
pred=[]
for data in train_loader:
    out = model(data)
    for o, t in zip(out, data.y):
        pred.append(o.detach().numpy())
        truth.append(t.detach().numpy())

plt.scatter(truth, pred)
plt.xlabel("True muon pT")
plt.ylabel("Predicted muon pT")
plt.show()


#nonsensical for the moment
plt.hist(pred, label="Predicted muon pT")
plt.hist(truth, label="True muon pT")
plt.show()
quit()


