import h5py
import networkx as nx

import torch
from torch_geometric.data import Dataset, Data

from itertools import combinations

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


    
