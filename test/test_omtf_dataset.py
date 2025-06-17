import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os, sys

import matplotlib.pyplot as plt

## Create the dataset with OMTF dataset
sys.path.append(os.path.join(os.getcwd(), '.', 'tools','training'))

from OMTFDataset import OMTFDataset,remove_empty_or_nan_graphs

if os.path.exists("/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/MuGun_Displaced/"):
    ROOTDIR = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/MuGun_Displaced/"
else: 
    ROOTDIR = "../../data/MuGun_Displaced_v250409_Dumper_l1omtf_001.root"

print("Creating the dataset")
regression=False

if regression: 
    config_file = "configs/dataset_regression.yml"
    dataset = OMTFDataset(root_dir=ROOTDIR, tree_name="simOmtfPhase2Digis/OMTFHitsTree", config=config_file, max_events=1000,max_files=1, pre_transform=remove_empty_or_nan_graphs, task='regression', debug=True)
else:
    config_file = "configs/dataset_classification.yml"
    dataset = OMTFDataset(root_dir=ROOTDIR, tree_name="simOmtfPhase2Digis/OMTFHitsTree", config=config_file, max_events=1000,max_files=1, pre_transform=remove_empty_or_nan_graphs, task='classification', debug=True)


print("Checking the dataset ")
print("Length of the dataset: ", len(dataset))
print(dataset[1])
print(dataset[1].x)
print(dataset[1].y)
print(dataset[1].edge_attr)
print(dataset[1].edge_index)
print(dataset[1].edge_label)
print(dataset[1].muon_vars)
print(dataset[1].omtf_vars)

print("Saving the dataset")
dataset.save_dataset("test_dataset.pt")

print("Now plotting some graphs")
dataset.plot_graph(idx=0, filename="graph0.png")
dataset.plot_example_graphs(filename="example_graphs.png")

print("Loading the dataset")
dataset2 = OMTFDataset.load_dataset("test_dataset.pt")

print("Checking the dataset2 ")
print("Length of the dataset2: ", len(dataset2))
print(dataset2[0])
print("Node features:  ")
print(dataset2[0].x)
print("Target features: ")
print(dataset2[0].y)
print("Edges attributes: ")
print(dataset2[0].edge_attr)
print("Edges index: ")
print(dataset2[0].edge_index)
print("Edges label: ")
print(dataset2[0].edge_label)
print("Muon vars: ")
print(dataset2[0].muon_vars)
print("OMTF vars: ")
print(dataset2[0].omtf_vars)


print("Checking the loaded dataset2 is identical to the original one")
for i in range(len(dataset)):
    assert torch.all(torch.eq(dataset[i].x, dataset2[i].x)), f"Mismatch in x for index {i}"
    assert torch.all(torch.eq(dataset[i].edge_index, dataset2[i].edge_index)), f"Mismatch in edge_index for index {i}"
    assert torch.all(torch.eq(dataset[i].edge_attr, dataset2[i].edge_attr)), f"Mismatch in edge_attr for index {i}"
    
    assert torch.all(torch.eq(dataset[i].y, dataset2[i].y)), f"Mismatch in y for index {i}"


## Now take a dataset and plot some graphs: features, muon vars, omtf vars, edge_index, edge_attr, target and number of edges / nodes
print("Plotting some graphs from the dataset2")

