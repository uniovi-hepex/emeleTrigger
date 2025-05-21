import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os, sys

import matplotlib.pyplot as plt

## Create the dataset with OMTF dataset
sys.path.append(os.path.join(os.getcwd(), '.', 'tools','training'))

from OMTFDataset import OMTFDataset,remove_empty_or_nan_graphs

if os.path.exists("/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/"):
    ROOTDIR = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/"
else: 
    ROOTDIR = "../../data/MuGun_Displaced_v250409_Dumper_l1omtf_001.root"

print("Creating the dataset")
mu_vars = ['muonQOverPt',"muonQPt",'muonPropEta', 'muonPropPhi','muonDxy']
omtf_vars = ['omtfEta','omtfPhi','omtfPt','omtfUPt']
st_vars =  ['inputStubEtaG', 'inputStubPhiG','inputStubR', 'inputStubLayer','inputStubType']

dataset = OMTFDataset(root_dir=ROOTDIR, tree_name="simOmtfPhase2Digis/OMTFHitsTree", muon_vars=mu_vars, stub_vars=st_vars, omtf_vars=omtf_vars, max_events=1000,max_files=1, pre_transform=remove_empty_or_nan_graphs)

print("Checking the dataset ")
print("Length of the dataset: ", len(dataset))
print(dataset[6])
print(dataset[6].x)
print(dataset[6].y)
print(dataset[6].edge_index)

print("Saving the dataset")
dataset.save_dataset("test_dataset.pt")

print("Now plotting some graphs")
dataset.plot_graph(idx=0, filename="graph0.png")
dataset.plot_example_graphs(filename="example_graphs.png")

print("Loading the dataset")
dataset2 = OMTFDataset.load_dataset("test_dataset.pt")

print("Checking the dataset2 ")
print("Length of the dataset2: ", len(dataset2))
print(dataset2[6])
print(dataset2[6].x)
print(dataset2[6].y)
print(dataset2[6].edge_index)


print("Checking the loaded dataset2 is identical to the original one")
for i in range(len(dataset)):
    assert torch.all(torch.eq(dataset[i].x, dataset2[i].x)), f"Mismatch in x for index {i}"
    assert torch.all(torch.eq(dataset[i].edge_index, dataset2[i].edge_index)), f"Mismatch in edge_index for index {i}"
    assert torch.all(torch.eq(dataset[i].edge_attr, dataset2[i].edge_attr)), f"Mismatch in edge_attr for index {i}"
    
    assert torch.all(torch.eq(dataset[i].y, dataset2[i].y)), f"Mismatch in y for index {i}"


