from dataviz import get_test_data
import torch
import torch.nn as nn

import awkward as ak
import copy

import os

import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np
from pyntcloud import PyntCloud

# viz
import networkx as nx
import matplotlib.pyplot as plt

#from torch.utils.data import Dataset
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader
#import torch.nn.functional as F
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import matplotlib

from itertools import combinations

# Stuff that will go in a library
def visualize_graph(G, color):
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=43), with_labels=False,
                     node_color=color, cmap='Set2')
    plt.show()



# End of stuff that will go in a library

# Convert the dataset into a point cloud dataset
import h5py

# Define a function to pad a list with zeros
def pad_list(lst, max_list_length):
    return lst + [0] * (max_list_length - len(lst))

def pad_branches(df):
    max_list_length = df.applymap(lambda x: len(x) if isinstance(x, list) else 0).max().max()
    max_int_list_length = df.applymap(lambda x: np.max([len(y) if isinstance(y, list) else 0 for y in x].append(0)) if isinstance(x, list) else 0).max().max() # lolazo
    # Apply the padding function to all columns containing lists
    for col in df.columns:
        df[col] = df[col].apply(lambda x: pad_list(x, max_list_length) if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [  pad_list(y, max_int_list_length) if isinstance(y, list) else y for y in x] if isinstance(x, list) else x)
        #for stuff in df[col]:
        #        print("Stuff: ", len(stuff) if isinstance(stuff, list) else type(stuff))
    return df


def convert_to_point_cloud(branches):

    #point_cloud_data = branches.copy()
    #point_cloud_data = copy.deepcopy(branches[['muonPt', 'muonEta', 'muonPhi']])
    point_cloud_data = {'x': branches['muonPt'], 'y': branches['muonEta'], 'z': branches['muonPhi']} # down the line maybe convert to actual cartesian coordinates
    # Create a PyntCloud object from the DataFrame
    cloud = PyntCloud(pd.DataFrame(point_cloud_data))

    return cloud


def generate_hdf5_dataset_with_padding(branches, hdf5_filename, save=False):

    # Padding   
    #padded_branches=np.asarray(pad_branches(branches))
    # Padding and flattening
    #padded_branches=ak.fill_none(ak.pad_none(branches, 2, clip=True), 999)
    padded_branches=ak.to_dataframe(branches)

    #padded_branches=branches.pad(longest, clip=True).fillna(0).regular()
    #for name in padded_branches:
    #    print('name', name)
    #    print('counts', padded_branches[name])
    #    #longest = padded_branches[name].counts.max()
    #    padded_branches[name] = padded_branches[name].pad(longest).fillna(0).regular()
    
    #padded_branches = branches.pad(branches.counts.max()).fillna(0)
    
    point_clouds = convert_to_point_cloud(branches)
    point_cloud_array = point_clouds.points.values

    #print("SHAPE", padded_branches.shape)
    #print("Columns shape")
    #for i in range(39):
    #    if isinstance(padded_branches[0,i], list):
    #        print('List of ', type(padded_branches[0,i][0]))
    #    else:
    #        print(type(padded_branches[0,i]))
                    
    print('Padded branches type', type(padded_branches))

    with h5py.File(hdf5_filename, 'w') as f:
        #f.create_dataset('images', data= np.asarray(padded_branches.values, dtype=np.float64))
        f.create_dataset('images', data=padded_branches, dtype=np.float64)
        f.create_dataset('point_clouds', data=point_cloud_array)


# Do we need normalization?
def resize_and_format_data(points, image):
    pass



#def get_training_dataset(hdf5_path, BATCH_SIZE=128):
#
#    with h5py.File(hdf5_path, 'r') as f:
#        # Assume 'your_dataset' is the name of the dataset in the HDF5 file
#        x_train = f['/point_clouds']
#        y_train = f['/images']
#        # Convert the h5py dataset to a TensorFlow dataset
#        x_train = tf.data.Dataset.from_tensor_slices(x_train)
#        y_train = tf.data.Dataset.from_tensor_slices(y_train)
#        # Zip them to create pairs
#        training_dataset = tf.data.Dataset.zip((x_train,y_train))
#        
#        # Shuffle, prepare batches, etc ...
#        training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
#        training_dataset = training_dataset.batch(BATCH_SIZE)
#        training_dataset = training_dataset.repeat()
#        training_dataset = training_dataset.prefetch(-1)
#        
#        return training_dataset


from torch_geometric.data import Dataset, Data

class OMTFDataset(Dataset):
    def __init__(self, root, filename, transform=None, pre_transform=None, pre_filter=None):
        self.filename=filename
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        return [f'torchdata_{idx}.pt' for idx in range(len(self.raw_paths)) ]

    def download(self):
        pass

    def process(self):
        idx = 0

        for raw_path in self.raw_paths:
            # Read data from `raw_path`.


            # HUGE MISTAKE. I NEED TO HAVE ONE GRAPH PER EVENT.
            datafile= h5py.File(raw_path,'r')
            # Get the point clouds
            node_feats = datafile['point_clouds'][()]
            G = nx.Graph()
            G.add_nodes_from(node_feats)
            # Get the original points
            node_labels = datafile['images'][()]
            print(node_feats)
            edge_index = [list(combinations(x, 2)) for x in node_feats]
            print("There are edges", len(edge_index[0]))
            data = Data(x=node_feats,edge_index=edge_index,y=node_labels)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f'torchdata_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'torchdata_{idx}.pt'))
        return data


    
def get_training_dataset(hdf5_path, BATCH_SIZE=128):


    # Get the point clouds
    x_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/point_clouds')
    # Get the original points
    y_train = tfio.IODataset.from_hdf5(hdf5_path, dataset='/images')
    # Zip them to create pairs
    training_dataset = tf.data.Dataset.zip((x_train,y_train))
    # Apply the data transformations
    #training_dataset = training_dataset.map(resize_and_format_data)
    
    # Shuffle, prepare batches, etc ...
    training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
    training_dataset = training_dataset.batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat()
    training_dataset = training_dataset.prefetch(-1)
    
    # Return dataset
    return training_dataset



# Get data
#branches = get_test_data('pd')
#print(branches.head())

br=get_test_data('pd')
pd.set_option('display.max_columns', None)
print(br.head())
quit()
if False:
    branches = get_test_data('ak')
    generate_hdf5_dataset_with_padding(branches, 'data/point_clouds.hd5')


#dataset = get_training_dataset('data/point_clouds.hd5')



# Create a list to store individual graphs
graphs = []

# Iterate through each event and create a graph
for index, row in branches.iterrows():
    # Create a directed graph for each event
    graph = nx.DiGraph()
    
    # Add nodes with attributes
    node_attributes = row.to_dict()
    graph.add_node(index, **node_attributes)
    
    # Add the graph to the list
    graphs.append(graph)

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
    x = torch.tensor([node['features'] for node in node_features], dtype=torch.float32)
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
