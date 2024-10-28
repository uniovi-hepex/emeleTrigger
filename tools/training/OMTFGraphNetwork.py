import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx, to_networkx

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import uproot
from pyntcloud import PyntCloud
import networkx as nx

import copy

class GCN(nn.Module):
    name = "GCN"
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

class OMTFGraphNetwork():
    def __init__(self, model_fn=None):
        torch.set_default_dtype(torch.float64)
        #torch.set_default_dtype(torch.float32)
        self.model_fn= model_fn if model_fn else GCN # Use the default model or a user-provided one. Ideally, after testing, the user-provided one can be hardcoded as a class above.
        self.num_epochs = 100
        self.branches=None
        self.pg_graphs=None

        self.NUM_PROCESSORS = 3
        self.NUM_PHI_BINS = 5400
        self.HW_ETA_TO_ETA_FACTOR=0.010875
        #self.HwEtaToEta conversion: 0.010875
        #self.HwPhiToGlbPhi conversion:  hwPhi* 2. * M_PI / 576
        self.LOGIC_LAYERS_CONNECTION_MAP={
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
        self.LOGIC_LAYERS_LABEL_MAP={
                #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
                # Put here catalog of names0
                0: 'MB1',
                2: 'MB2',
                4: 'MB3',
                6: 'ME1/3',
                7: 'ME2/2',
                8: 'ME3/2',
                9: 'ME1/2',
        }
        self.sky=[]
        
    def load_data(self, path,fraction=1., viz=False, bsize=64):
        branches = uproot.open(path)
        print(branches.show())
        branches=branches.arrays(library='pd')
        print('Downsampling...')
        self.branches=branches.sample(frac=fraction)
        print('Downsampling done')
        # Calculate deltaphis between layers, adding it to a new column
        # this will be our proxy to the magnetic field
        self.branches['stubDPhi'] = self.branches['stubPhi'].apply(lambda x: np.diff(x))
        self.branches['stubDEta'] = self.branches['stubEta'].apply(lambda x: np.diff(x))

        print(self.branches.head())

        # Create graphs with nodes and edges according to the preset connection map
        self.convert_to_graphs(viz)

        print('Graphs created', len(self.pg_graphs))
        # Once it's shuffled, we slice the data to split
        self.train_dataset = self.pg_graphs[:int(121/2)]
        self.test_dataset  = self.pg_graphs[int(121/2):]

        #for d in train_dataset:
        #    d.to(torch.device("mps"))
        #for d in test_dataset:
        #    d.to(torch.device("mps"))
        self.train_loader = DataLoader(self.train_dataset, batch_size=bsize, shuffle=True)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=bsize, shuffle=True)

    def _get_stub_r(self, stubTypes, stubEtas, stubLogicLayers):
        rs=[]
        for stubType, stubEta, stubLogicLayer in zip(stubTypes, stubEtas, stubLogicLayers):
            r=None
            if stubType == 3: # DTs
                if stubLogicLayer==0:
                    r= 431.133
                elif stubLogicLayer==2:
                    r=512.401
                elif stubLogicLayer==4:
                    r=617.946
            elif stubType==9: # CSCs
                if stubLogicLayer==6:
                    z=7
                elif stubLogicLayer==9:
                    z=6.8
                elif stubLogicLayer==7:
                    z=8.2
                elif stubLogicLayer==8:
                    z=9.3
                r= z/np.cos( np.tan(2*np.arctan(np.exp(- stubEta*self.HW_ETA_TO_ETA_FACTOR)))  )
            elif stubType==5: # RPCs, but they will be shut down because they leak poisonous gas
                r=999.
            rs.append(r)

        if len(rs) != len(stubTypes):
            print('Tragic tragedy. R has len', len(rs), ', stubs have len', len(stubTypes))
        return np.array(rs, dtype=object)

    def getEdgesFromLogicLayer(self, logicLayer):
        return (self.LOGIC_LAYERS_CONNECTION_MAP[logicLayer] if logicLayer<10 else [])

    def getLayerNameFromLogicLayer(self, logicLayer):
        return (self.LOGIC_LAYERS_LABEL_MAP[logicLayer] if logicLayer<10 else 'RPC')

    def addEdges(self):
        self.graphs = []
        print('adding edges from sky of size', len(self.sky))
        for index, cloud in enumerate(self.sky):
            graph = nx.DiGraph()
            nodes = []
            keep=['stubR', 'stubEta', 'stubPhi','stubPhiB', 'stubPhiDist', 'stubEtaDist', 'stubQuality', 'stubBx', 'stubType', 'stubTiming', 'stubLogicLayer']
            edges=[]
            #print('Index of sky', index, ', cloud has size', len(cloud))
            for idx, row in cloud.iterrows(): # build edges based on stubLayer
                # The node name must also have the index, otherwise the same node is updated
                nodes.append((idx, { 'x': [row[k] for k in keep], 'y' : np.float64(row['muonPt'])})) 
                #nodes.append((f"{idx}, {self.getLayerNameFromLogicLayer(row['stubLogicLayer'])}", { 'x': [row[k] for k in keep], 'y' : np.float64(row['muonPt'])}))
                dests=self.getEdgesFromLogicLayer(row['stubLogicLayer'])
                for queriedindex, row in cloud.iterrows():
                    if queriedindex in dests:
                        edges.append((idx, queriedindex))
                        #edges.append((f"{idx}, {self.getLayerNameFromLogicLayer(row['stubLogicLayer'])}",f"{queriedindex}, {self.getLayerNameFromLogicLayer(row['stubLogicLayer'])}"))
            #cp=cloud.points.rename(columns={"x": "stubR", "y": "stubEta", "z": "stubPhi", "muonPt": "y"}, errors="raise")
            graph.add_nodes_from(nodes) # Must be the transpose, it reads by colum instead of by row
            graph.add_edges_from(edges)
            self.graphs.append(graph)
        print('edges added')

    def convert_to_graphs(self, viz=False):
        self.get_stub_r = np.vectorize(self._get_stub_r)
        self.branches['stubR'] = self.get_stub_r(self.branches['stubType'], self.branches['stubEta'], self.branches['stubLogicLayer'])
        keep=['stubR', 'stubEta', 'stubPhi','stubPhiB', 'stubPhiDist', 'stubEtaDist', 'stubQuality', 'stubBx', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']
        self.sky=[]
        for index, row in self.branches.iterrows():
            # Here now I need to enrich this with the stublogiclayer etc, for the edges
            pc = {} #{'x': row['stubR'], 'y': row['stubEta'], 'z': row['stubPhi'], 'stubLogicLayer': row['stubLogicLayer']} # down the line maybe convert to actual cartesian coordinates
            for label in keep:
                pc[label] = row[label]
            if len(pc['stubR'])==0:
                continue
            self.sky.append(pd.DataFrame(pc))
        print('Sky has size', len(self.sky))
        self.addEdges()
        print('edges added')
        # Convert each NetworkX graph to a PyTorch Geometric Data object
        self.pg_graphs=[from_networkx(g) for g in self.graphs]
        print('converted to geometric')
        if viz:
            gmax=None
            nmax=0
            for graph in self.graphs:
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
            plt.savefig('samplegraph.png')
            plt.clf()
            pg=None
            nmax=0
            for g in self.pg_graphs:
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
        print('dibujado')
        
    def foldPhi (self, phi):
        if (phi > NUM_PHI_BINS / 2):
            return (phi - NUM_PHI_BINS)
        elif (phi < -NUM_PHI_BINS / 2):
            return (phi + NUM_PHI_BINS)
        return phi
    
    def phiRel(self, phi, processor):
        return phi - foldPhi(NUM_PHI_BINS / NUM_PROCESSORS * (processor) + NUM_PHI_BINS / 24)
        
    def train(self):
        self.model.train()

        total_loss=0
        for data in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.criterion(out, data.y.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                #total_loss += float(loss) * data.num_graphs
                total_loss += float(loss)

        return total_loss/len(self.train_loader.dataset)

    def test(self):
        with torch.no_grad():
            self.model.eval()

            total_loss = 0
            for data in self.test_loader:
                out = self.model(data)
                loss = self.criterion(out, data.y.unsqueeze(1))
                #total_loss += float(loss) * data.num_graphs
                total_loss += float(loss)
            return total_loss/len(self.test_loader.dataset)

    def load_model(self, path):
        self.model=torch.load(path)
        self.model.eval()

    def instantiate_model(self, pars=None):
        if not pars and self.model_fn.name == 'GCN':
            self.model = self.model_fn(self.pg_graphs[0].num_node_features,8,1)#.to(torch.device("mps"))
        else:
            self.model = self.model_fn(pars)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def do_training(self, num_epochs=None):
        if num_epochs:
            print(f'Default number of epochs ({self.num_epochs}) replaced at runtime with {num_epochs}')
            self.num_epochs=num_epochs
        train_losses=[]
        test_losses=[]
        for epoch in range(self.num_epochs):
            train_loss = self.train()
            test_loss = self.test()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f'Epoch: {epoch:02d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
            torch.save(self.model, 'models/mode.pth')
        epochs = [ x for x in range(self.num_epochs)]
        plt.plot(epochs, train_losses, label="Train loss")
        plt.plot(epochs, test_losses, label="Test loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss (rescaled)")
        plt.legend(loc='best')
        plt.savefig('losses.png')
        plt.clf()

    def visualize_prediction(self):
        truth=[]
        pred=[]
        print("Visualizing preditions")
        for data in self.test_loader:
            out = self.model(data)
            for o, t in zip(out, data.y):
                pred.append(o.detach().numpy())
                truth.append(t.detach().numpy())
        print("Acquired arrays")
        plt.scatter(truth, pred)
        plt.xlabel("True muon pT")
        plt.ylabel("Predicted muon pT")
        plt.savefig('pred_2d.png')
        print("scatter created")
        plt.clf()
        print("Meh")
        plt.hist(pred, label="Predicted muon pT")
        plt.hist(truth, label="True muon pT")
        print("wtf")
        plt.xlabel("Muon pT")
        plt.savefig('pred_1d.png')
        print("one d created")
