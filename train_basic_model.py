from utils import get_test_data, visualize_graph, get_stub_r
import torch
import torch.nn as nn

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

from torch_geometric.utils.convert import from_networkx, to_networkx

# Do we need normalization?
def resize_and_format_data(points, image):
    pass

torch.set_default_dtype(torch.float64)
#torch.set_default_dtype(torch.float32)

# Get data
branches = get_test_data('pd')
print(branches.head())
#br=get_test_data('pd')
#pd.set_option('display.max_columns', None)
#print(br.head())

# Keep this
#branches = get_test_data('ak')
#generate_hdf5_dataset_with_padding(branches, 'data/point_clouds.hd5')
#dataset = get_training_dataset('data/point_clouds.hd5')



def convert_to_point_cloud(arr):

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubEta'], arr['stubLogicLayer'])
    for index, row in arr.iterrows():
        if not ( len(arr['stubR'][index])==len(arr['stubEta'][index]) and len(arr['stubEta'][index])==len(arr['stubPhi'][index])):
            print('HUGE PROBLEM, sizes not match: R,eta,phi ', len(arr['stubR'][index]), len(arr['stubEta'][index]), len(arr['stubPhi'][index]))
    arr=arr.rename(columns={"stubR": "x", "stubEta": "y", "stubPhi": "z"}, errors="raise")

    keep=['x', 'y', 'z', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']

    sky=[]
    for index, row in arr.iterrows():
        # Here now I need to enrich this with the stublogiclayer etc, for the edges
        pc = {} #{'x': row['stubR'], 'y': row['stubEta'], 'z': row['stubPhi'], 'stubLogicLayer': row['stubLogicLayer']} # down the line maybe convert to actual cartesian coordinates
        for label in keep:
            pc[label] = row[label]
        # Create a PyntCloud object from the DataFrame
        if len(pc['x'])==0:
            continue
        cloud = PyntCloud(pd.DataFrame(pc))
        sky.append(cloud)

    return sky

def convert_to_point_cloud_bad(arr):

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubLogicLayer'])
    
    pc = {'x': arr['stubR'], 'y': arr['stubEta'], 'z': arr['stubPhi']} # down the line maybe convert to actual cartesian coordinates
    print('THECLOUD', pc)
    # Create a PyntCloud object from the DataFrame
    cloud = PyntCloud(pd.DataFrame(pc))
    return cloud


sky = convert_to_point_cloud(branches)

#print(sky[0].points)
#print(sky[0].points.describe())
#sky[0].points.boxplot()
#plt.show()
#scatter_matrix(sky[0].points, diagonal="kde", figsize=(8,8))
#plt.show()

# Create a list to store individual graphs
graphs = []

logicLayersConnectionMap={
    #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
    # Put here catalog of names0
    0: [2,6,7],
    2: [4,6],
    4: [6],
    6: [7,8],
    7: [8],
    8: [],
    9: [7],
}

def getEdgesFromLogicLayer(logicLayer):
    return (logicLayersConnectionMap[logicLayer] if logicLayer<10 else [])


def convert_to_graphs(arr):

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubEta'], arr['stubLogicLayer'])
    
    keep=['stubR', 'stubEta', 'stubPhi', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']

    sky=[]
    for index, row in arr.iterrows():
        # Here now I need to enrich this with the stublogiclayer etc, for the edges
        pc = {} #{'x': row['stubR'], 'y': row['stubEta'], 'z': row['stubPhi'], 'stubLogicLayer': row['stubLogicLayer']} # down the line maybe convert to actual cartesian coordinates
        for label in keep:
            pc[label] = row[label]
        # Create a PyntCloud object from the DataFrame
        if len(pc['stubR'])==0:
            continue
        sky.append(pd.DataFrame(pc))

    return sky

sky = convert_to_graphs(branches)


for index, cloud in enumerate(sky):
    graph = nx.DiGraph()
    nodes = []
    keep=['stubR', 'stubEta', 'stubPhi', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer']
    edges=[]
    for index, row in cloud.iterrows(): # build edges based on stubLayer
        #nodes.append((index, {k: row[k] for k in keep}))
        nodes.append((index, { 'x': [row[k] for k in keep], 'y' : np.float64(row['muonPt']) }))
        dests=getEdgesFromLogicLayer(row['stubLogicLayer'])
        for queriedindex, row in cloud.iterrows():
            if queriedindex in dests:
                edges.append((index,queriedindex))
    # Rename back
    #cp=cloud.points.rename(columns={"x": "stubR", "y": "stubEta", "z": "stubPhi", "muonPt": "y"}, errors="raise")
    graph.add_nodes_from(nodes) # Must be the transpose, it reads by colum instead of by row
    graph.add_edges_from(edges)
    graphs.append(graph)
#for index, cloud in enumerate(sky):
#    graph = nx.DiGraph()
#    edges=[]
#    for index, row in cloud.points.iterrows(): # build edges based on stubLayer
#        dests=getEdgesFromLogicLayer(row['stubLogicLayer'])
#        for queriedindex, row in cloud.points.iterrows():
#            if queriedindex in dests:
#                edges.append((index,queriedindex))
#    # Rename back
#    cp=cloud.points.rename(columns={"x": "stubR", "y": "stubEta", "z": "stubPhi", "muonPt": "y"}, errors="raise")
#    graph.add_nodes_from(cp.T) # Must be the transpose, it reads by colum instead of by row
#    graph.add_edges_from(edges)
#    graphs.append(graph)


viz=True
if viz:
    gmax=None
    nmax=0
    for graph in graphs:
        numnodes = graph.number_of_nodes()
        if numnodes>nmax:
            gmax=copy.deepcopy(graph)
            nmax=numnodes
    nx.draw(gmax, with_labels=True)
    print(gmax.nodes.data(True))
    g=from_networkx(gmax)
    print(g)
    print()
    print(g.y)
    print(g.edge_index)
    plt.show()




# Convert each NetworkX graph to a PyTorch Geometric Data object
pg_graphs=[from_networkx(g) for g in graphs]


pg=None
nmax=0
for g in pg_graphs:
    numnodes=g.num_nodes
    if numnodes>nmax:
        pg=copy.deepcopy(g)
        nmax=numnodes

print('networkx graph')
print(pg)
print(pg.x)
print(pg.y)
print(pg.edge_index)
print('-----------------')
# Gather some statistics about the graph.
print(f'Number of nodes: {pg.num_nodes}') #Number of nodes in the graph
print(f'Number of edges: {pg.num_edges}') #Number of edges in the graph
print(f'Average node degree: {pg.num_edges / pg.num_nodes:.2f}') # Average number of nodes in the graph
print(f'Contains isolated nodes: {pg.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
print(f'Contains self-loops: {pg.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
print(f'Is undirected: {pg.is_undirected()}') #Is the graph an undirected graph
nx.draw(to_networkx(pg), with_labels=True)



#data_list = []
#for graph in pg_graphs:
#    print('pajoerlia', type(graph))
#    # Extract node features and labels
#    #print(graph, list(graph.nodes), list(graph.edges))
#    #print("FEATURES", graph.nodes(data=True))
#    #print("LABELS", graph.nodes(data='muonPt'))
#    #node_features = graph.nodes(data=True)
#    #node_labels = graph.nodes(data='muonPt')
#    keep=['stubR', 'stubEta', 'stubPhi', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']
#    print('mehhhhhh', graph.nodes(data='stubProc'))
#    # Convert to PyTorch tensors
#    x = torch.tensor([graph.nodes(data=feat) for feat in node_features], dtype=torch.float32)
#    y = torch.tensor([graph.nodes(data=label) for label in node_labels], dtype=torch.float32)
#
#    # Assuming your graph has edges
#    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
#
#    # Create a PyTorch Geometric Data object
#    data = Data(x=x, edge_index=edge_index, y=y)
#
#    data_list.append(data)



# Create a DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

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

# I will remove most of what is below when designing a better structure of the GNN and cleaning up the code, next week (17december onwards),
##########################################################################################################################################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
##########################################################################################################################################################################################################################################
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
model.eval()
pred = model(test_loader[0]).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
print("TRAINED")


from torch_geometric.nn import GCNConv
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()



quit()





# Create a DataLoader
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Define a simple GNN model
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleGNN(in_channels=num_features, out_channels=16, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

print("TRAINED")
## Iterate through each event and create a graph
#for index, row in branches.iterrows():
#    # Create a directed graph for each event
#    graph = nx.DiGraph()
#    
#    # Add nodes with attributes
#    node_attributes = row.to_dict()
#    graph.add_node(index, **node_attributes)
#    
#    # Add the graph to the list
#    graphs.append(graph)

####################################
# So, what I do below is basically wrong, in the sense that it runs, but it doesn-t contain the edges structure that would be needed.
# Therefore, here would go the code that creates the edges structure
# A sample code (for the tracker) is here>  https://github.com/CMS-GNN-Tracking-Hackathon-2021/interaction-network/blob/main/graph_construction/build_graph.py ,
# where they define the allowed edges between the various parts of the tracker, in their case
####################


# Assume your graph has node features and labels
# You should replace this with your actual node features and labels
node_features = torch.randn((num_nodes, num_features))  # Replace with your actual features
node_labels = torch.randint(0, num_classes, (num_nodes,))  # Replace with your actual labels

# Convert each NetworkX graph to a PyTorch Geometric Data object
data_list = []
for graph in list_of_networkx_graphs:
    # Extract node features and labels
    node_features = graph.nodes(data='features')
    node_labels = graph.nodes(data='label')

    # Convert to PyTorch tensors
    x = torch.tensor([node['features'] for node in node_features], dtype=torch.float64)
    y = torch.tensor([node['label'] for node in node_labels], dtype=torch.long)

    # Assuming your graph has edges
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y)

    data_list.append(data)

# Create a DataLoader
train_loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Define a simple GNN model
class SimpleGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        self.conv2 = GCNConv(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleGNN(in_channels=num_features, out_channels=16, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

dataset = OMTFDataset("data/", "point_clouds.hd5")


inspectDataset=False

if inspectDataset:
    print(dataset)
    for image, label in tfds.as_numpy(dataset):
        print(type(image), type(label), label)

print('Explore dataset')
print(f'Number of graphs in the dataset: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')


#from torch_geometric.transforms import KNNGraph
#import torch_geometric
#
#dataset.transform = torch_geometric.transforms.Compose([dataset, KNNGraph(k=6)])
#
##Since we have one graph in the dataset, we will select the graph and explore it's properties
#print(dataset.transform)




data = dataset[0]
print('Graph properties')
print('==============================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}') #Number of nodes in the graph
print(f'Number of edges: {data.num_edges}') #Number of edges in the graph
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}') # Average number of nodes in the graph
print(f'Contains isolated nodes: {data.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
print(f'Contains self-loops: {data.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
print(f'Is undirected: {data.is_undirected()}') #Is the graph an undirected graph



from torch_geometric.utils import to_networkx

G = to_networkx(data, to_undirected=True)
visualize_graph(G, color=data.y)


from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_features, 4)
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, dataset.num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

model = GCN()
print(model)




model = GCN()
criterion = torch.nn.CrossEntropyLoss()  #Initialize the CrossEntropyLoss function.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Initialize the Adam optimizer.

def train(data):
    optimizer.zero_grad()  # Clear gradients.
    out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h

for epoch in range(401):
    loss, h = train(data)
    print(f'Epoch: {epoch}, Loss: {loss}')
   


visualize_graph(model, color=data.y)




# What comes below mostly still doesn't make any sense---we don't want to do supervised learning here.




X_train, X_test, y_train, y_test = train_test_split(branches, branches, test_size=0.33)

train_dataset = OMTFDataset(X_train, y_train)
test_dataset = OMTFDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

# A simple neural network, to start with
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    def forward(self, x):
        # Pass data through conv1
        x = self.linear_relu_stack(x)
        return x

def train_loop(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    losses=[]
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X[:,:-3]) # Change this
        #if (all_equal3(pred.detach().numpy())):
        #    print(\"All equal!\")
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    scheduler.step()
    return np.mean(losses)

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    losses=[]
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X[:,:-3]) # Change this
            loss = loss_fn(pred, y).item()
            losses.append(loss)
            test_loss += loss
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return np.mean(losses)
    #test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

epochs = 30
learningRate = 0.01

model = NeuralNetwork()

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


train_losses=[]
test_losses=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss=train_loop(train_dataloader, model, loss_fn, optimizer, scheduler)
    test_loss=test_loop(test_dataloader, model, loss_fn)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    print("Avg train loss", train_loss, ", Avg test loss", test_loss, "Current learning rate", scheduler.get_last_lr())
print("Done!")


plt.plot(train_losses, label="Average training loss")
plt.plot(test_losses, label="Average test loss")
plt.legend(loc="best")
