import os
import uproot
import torch
from torch_geometric.data import Dataset, Data
import networkx as nx
from operator import xor

from torch_geometric.utils.convert import to_networkx

import numpy as np
import matplotlib.pyplot as plt


''' Auxliary functions and variables'''
NUM_PROCESSORS = 3
NUM_PHI_BINS = 5400
HW_ETA_TO_ETA_FACTOR=0.010875
LOGIC_LAYERS_LABEL_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: 'MB1',
            2: 'MB2',
            4: 'MB3',
            6: 'ME1/3',
            7: 'ME2/2',
            8: 'ME3/2',
            9: 'ME1/2',
            10: 'RB1in',
            11: 'RB1out',
            12: 'RB2in',
            13: 'RB2out',
            14: 'RB3',
            15: 'RE1/3',
            16: 'RE2/3',
            17: 'RE3/3'
        }

def get_global_phi(phi, processor):
    p1phiLSB = 2 * np.pi / NUM_PHI_BINS
    if isinstance(phi, list):
        return [(processor * 192 + p + 600) % NUM_PHI_BINS * p1phiLSB for p in phi]
    else:
        return (processor * 192 + phi + 600) % NUM_PHI_BINS * p1phiLSB


def get_stub_r(stubTypes, stubEta, stubLayer, stubQuality):
    rs = []
    for stubType, stubEta, stubLayer, stubQuality in zip(stubTypes, stubEta, stubLayer, stubQuality):
        r = None
        if stubType == 3:  # DTs
            if stubLayer == 0:
                r = 431.133
            elif stubLayer == 2:
                r = 512.401
            elif stubLayer == 4:
                r = 617.946

            # Low-quality stubs are shifted by 23.5/2 cm
            if stubQuality == 2 or stubQuality == 0:
                r = r - 23.5 / 2
            elif stubQuality == 3 or stubQuality == 1:
                r = r + 23.5 / 2

        elif stubType == 9:  # CSCs
            if stubLayer == 6:
                z = 690  # ME1/3
            elif stubLayer == 9:
                z = 700  # M1/2
            elif stubLayer == 7:
                z = 830
            elif stubLayer == 8:
                z = 930
            r = z / np.cos(np.tan(2 * np.arctan(np.exp(-stubEta * HW_ETA_TO_ETA_FACTOR))))
        elif stubType == 5:  # RPCs, but they will be shut down because they leak poisonous gas
            r = 999.
            if stubLayer == 10:
                r = 413.675  # RB1in
            elif stubLayer == 11:
                r = 448.675  # RB1out
            elif stubLayer == 12:
                r = 494.975  # RB2in
            elif stubLayer == 13:
                r = 529.975  # RB2out
            elif stubLayer == 14:
                r = 602.150  # RB3
            elif stubLayer == 15:
                z = 720  # RE1/3
            elif stubLayer == 16:
                z = 790  # RE2/3
            elif stubLayer == 17:
                z = 970  # RE3/3
            if r == 999.:
                r = z / np.cos(np.tan(2 * np.arctan(np.exp(-stubEta * HW_ETA_TO_ETA_FACTOR))))

        rs.append(r)

    if len(rs) != len(stubTypes):
        print('Tragic tragedy. R has len', len(rs), ', stubs have len', len(stubTypes))
    return np.array(rs, dtype=object)
    
def getEtaKey(eta):
    if abs(eta) < 0.92:
        return 1
    elif abs(eta) < 1.1:
        return 2
    elif abs(eta) < 1.15:
        return 3
    elif abs(eta) < 1.19:
        return 4
    else:
        return 5
    
def getListOfConnectedLayers(eta):
    etaKey=getEtaKey(eta)    

    LAYER_ORDER_MAP = {
            1: [10,0,11,12,2,13,14,4,6,15],
            2: [10,0,11,12,2,13,6,15,16,7],
            3: [10,0,11,6,15,16,7,8,17],
            4: [10,0,11,16,7,8,17],
            5: [10,0,9,16,7,8,17],
    }
    return LAYER_ORDER_MAP[etaKey]    

def getEdgesFromLogicLayer(logicLayer,withRPC=True):
    LOGIC_LAYERS_CONNECTION_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: [2,4,6,7,8,9],   #MB1: [MB2, MB3, ME1/3, ME2/2]
            4: [6],             #MB3: [ME1/3]
            2: [4,6,7],         #MB2: [MB3, ME1/3]
            6: [7,8],           #ME1/3: [ME2/2]
            7: [8,9],           #ME2/2: [ME3/2]
            8: [9],             #ME3/2: [RE3/3]
            9: [],              #ME1/2: [RE2/3, ME2/2]
    }
    LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC = {
            0:  [2,4,6,7,8,9,10,11,12,13,14,15,16,17], 
            1:  [2,4,6,7,8,9,10,11,12,13,14,15,16,17], 
            2:  [4,6,7,10,11,12,13,14,15,16],       #MB2: [MB3, ME1/3]
            3:  [4,6,7,10,11,12,13,14,15,16],       #MB2: [MB3, ME1/3]
            4:  [6,10,11,12,13,14,15],         #MB3: [ME1/3]
            5:  [6,10,11,12,13,14,15],         #MB3: [ME1/3]
            6:  [7,8,10,11,12,13,14,15,16,17],         #ME1/3: [ME2/2]
            7:  [8,9,10,11,15,16,17],         #ME2/2: [ME3/2]
            8:  [9,10,11,15,16,17],        #ME3/2: [RE3/3]
            9:  [7,10,16,17],         #ME1/2: [RE2/3, ME2/2]
            10: [11,12,13,14,15,16,17],
            11: [12,13,14,15,16,17],
            12: [13,14,15,16],
            13: [14,15,16],
            14: [15],
            15: [16,17],
            16: [17],
            17: []
    }
        
    if (withRPC):
        return (LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC[logicLayer])
    else:
        if (logicLayer>=10): return []
        return (LOGIC_LAYERS_CONNECTION_MAP[logicLayer])

def remove_empty_or_nan_graphs(data):
    # Verificar si el grafo está vacío
    if data.x.size(0) == 0 or data.edge_index.size(1) == 0:
        return None

    # Verificar si hay valores nan en x, edge_attr o y
    if torch.isnan(data.x).any() or (data.edge_attr is not None and torch.isnan(data.edge_attr).any()) or torch.isnan(data.y).any():
        return None

    return data

class OMTFDataset(Dataset):
    def __init__(self, root_dir=None, tree_name=None, muon_vars=None, stub_vars=None, transform=None, pre_transform=None, dataset=None, max_files=None, max_events=None):
        """
        Args:
            root_dir (str): Directorio que contiene los archivos ROOT.
            tree_name (str): Nombre del árbol dentro de los archivos ROOT.
            muon_vars (list of str, optional): Lista de variables de muones a extraer.
            stub_vars (list of str, optional): Lista de variables de stubs a extraer.
            transform (callable, optional): Una función/transformación que toma un objeto Data y lo devuelve transformado.
            pre_transform (callable, optional): Una función/transformación que se aplica antes de guardar los datos.
        """
        self.max_files = max_files if max_files is not None else None
        self.max_events = max_events
        if dataset is not None:
            self.dataset = dataset
        else:
            self.root_dir = root_dir
            self.tree_name = tree_name
            self.muon_vars = muon_vars if muon_vars is not None else []
            self.stub_vars = stub_vars if stub_vars is not None else []
            self.pre_transform = pre_transform
            self.dataset = self.load_data_from_root()
        
        self.transform = transform

    def add_extra_vars_to_tree(self, tree):
        """
        Add some extra variables....
        """
        df = tree.arrays(library="pd")  # Convertir el árbol a un DataFrame de pandas
        df = df[df.stubNo > 0]
        if 'stubR' not in df.columns:
            df['stubR'] = df.apply(lambda x: get_stub_r(x['stubType'], x['stubEta'], x['stubLayer'], x['stubQuality']), axis=1)
        df['stubPhi'] = df['stubPhi'] + df['stubPhiB']
        df['stubEtaG'] = df['stubEta'] * HW_ETA_TO_ETA_FACTOR
        df = df[df.columns.drop(list(df.filter(regex='inputStub')))]
        df['stubPhiG'] = df.apply(lambda x: get_global_phi(x['stubPhi'], x['omtfProcessor']), axis=1)
        df['muonPropEta'] = df['muonPropEta'].abs()
        df['muonQPt'] = df['muonCharge'] * df['muonPt']
        df['muonQOverPt'] = df['muonCharge'] / df['muonPt']

        return df
    
    def load_data_from_root(self):
        data_list = []
        files_processed = 0
        events_processed = 0

        # Verificar si root_dir es un directorio o un archivo
        if os.path.isdir(self.root_dir):
            root_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".root")]
        elif os.path.isfile(self.root_dir) and self.root_dir.endswith(".root"):
            root_files = [self.root_dir]
        else:
            raise ValueError(f"{self.root_dir} is not a valid directory or ROOT file")

        for root_file in root_files:
            if self.max_files is not None and files_processed >= self.max_files:
                break

            with uproot.open(root_file) as file:
                tree = file[self.tree_name]
                # drop the event if it has no stubs
                
                df = self.add_extra_vars_to_tree(tree)

                for index, row in df.iterrows():
                    if events_processed % 100 == 0:
                        print(f"Processed {events_processed} events")
                    if self.max_events is not None and events_processed >= self.max_events:
                        break
                    layers=row['stubLayer']
                    stub_phi=row['stubPhi']
                    stub_phiB=row['stubPhiB']
                    for layerL in [1,3,5]:
                        try:
                            layers.index(layerL)
                        except ValueError:
                            continue
                        else:
                            index2=layers.index(layerL)                            
                            index1=index2-1
                            print(f'Entry {index}: Layers {layerL-1} - {layerL}   Phi{layerL}: {stub_phi[index1]}  Phi{layerL+1}: {stub_phi[index2]}   PhiB{layerL}: {stub_phiB[index1]}  PhiB{layerL+1}: {stub_phiB[index2]}   DeltaPhi {stub_phi[index1]-stub_phi[index2]}     DeltaPhiB {stub_phiB[index1]-stub_phiB[index2]}')
                            
                    # Create nodes and edges
                    stub_array = np.vstack([row[var] for var in self.stub_vars]).astype(np.float32).T
                    x = torch.tensor(stub_array, dtype=torch.float)
                    edge_index, edge_attr = self.create_edges(row)

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([row[var] for var in self.muon_vars], dtype=torch.float))
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    if data is not None:
                        data_list.append(data)
                    events_processed += 1

            files_processed += 1
        return data_list

    def getDeltaPhi(self,phi1,phi2):
        dphi = phi1 - phi2
        dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
        return dphi

    def getDeltaEta(self,eta1,eta2):
        return eta1-eta2
    
    def create_edges(self, row):
        stubLayer = row['stubLayer']
        stubPhi = row['stubPhi']
        stubEta = row['stubEta']
        edge_index = []
        edge_attr = []
        if xor(0 in stubLayer, 1 in stubLayer):
            print("01 pair")
        if xor(2 in stubLayer, 3 in stubLayer):
            print("23 pair")
        if xor(4 in stubLayer, 5 in stubLayer):
            print("45 pair")
        for extra in [1,3,5]:
            if extra in stubLayer:
                stubLayer.remove(extra)
        for stub1Id,stub1Layer in enumerate(stubLayer):
            for stub2Id, stub2Layer in enumerate(stubLayer):
                #print(f'stubt1Id:{stub1Id}   stub1Layer:{stub1Layer}    stubt2Id:{stub2Id}   stub2Layer:{stub2Layer}')
                if stub1Layer == stub2Layer: continue
                if stub2Layer in getEdgesFromLogicLayer(stub1Layer):
                    dphi = self.getDeltaPhi(stubPhi[stub1Id],stubPhi[stub2Id])
                    deta = self.getDeltaEta(stubEta[stub1Id],stubEta[stub2Id])
                    edge_index.append([stub1Id, stub2Id])
                    edge_attr.append([dphi, deta])
                    edge_index.append([stub2Id, stub1Id]) 
                    edge_attr.append([-dphi, -deta])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return edge_index, edge_attr

    def len(self):
        return len(self.dataset)
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, idx):
        return self.get(idx)
    
    def __repr__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def plot_graph(self, idx, filename=None, seed=42):
        data = self.get(idx)
        
        G = to_networkx(data, to_undirected=True)
        labels = {i: int(data.x[i, 3].item()) for i in range(data.x.shape[0])}

        '''G = nx.Graph()
        
        for i, node_feature in enumerate(data.x):
            G.add_node(i, feature=node_feature.tolist())
        
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i]) #, weight=data.edge_weight[i].item() if 'edge_weight' in data else 1.0)
        '''
        pos = nx.spring_layout(G,seed=seed)
        nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=500)
        #nx.draw_networkx_edge_labels(G, pos) # edge_labels=edge_labels)
        
        plt.title(f'Grafo de ejemplo del índice {idx}')
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()

    def plot_example_graphs(self, filename=None, seed=42):
        """
        Escanea el dataset y muestra grafos de ejemplo con diferentes números de nodos (de 4 a 10).
        Etiqueta los nodos con la cuarta columna de data.x.
        """
        print('Drawing example graphs into ', filename)
        # draw a figure with 6 subplots

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        axs = axs.ravel()
        ax = plt.gca()
        ax.margins(0.08)
    
        num_nodes_list = range(3, 11)
        found_graphs = {num_nodes: False for num_nodes in num_nodes_list}

        for index, data in enumerate(self.dataset):
            num_nodes = data.x.shape[0]
            if num_nodes in num_nodes_list and not found_graphs[num_nodes]:
                #G = to_networkx(data, to_undirected=True)

                G = nx.Graph()
                
                # Añadir nodos con etiquetas de la cuarta columna de data.x
                for i, node_feature in enumerate(data.x):
                    G.add_node(i, label=int(node_feature[3].item()))
                
                # Añadir aristas
                edge_index = data.edge_index.numpy()
                for i in range(edge_index.shape[1]):
                    G.add_edge(edge_index[0, i], edge_index[1, i])
                
                # Dibujar el grafo
                pos = nx.spring_layout(G,seed=seed)
                labels = nx.get_node_attributes(G, 'label')
                nx.draw(G, pos, labels=labels, node_color='skyblue', edge_color='gray', ax=axs[num_nodes-3])
                
                # Dibujar etiquetas de los pesos de las aristas
                #edge_labels = nx.get_edge_attributes(G, 'weight')
                #nx.draw_networkx_edge_labels(G, pos, ax=axs[num_nodes-3])
                
                #plt.title(f'Grafo de ejemplo del índice {index} con {num_nodes} nodos')
                #plt.show()
                axs[num_nodes-3].set_title(f"{G}") 
                axs[num_nodes-3].axis("off")

                found_graphs[num_nodes] = True
                if all(found_graphs.values()):
                    break  # Salir del bucle si se han encontrado grafos para todos los números de nodos

        plt.tight_layout()
        plt.show()
        if filename is not None:
            plt.savefig(filename)
        plt.close()  # Cerrar la gráfica automáticamente


    def save_dataset(self, file_path):
        torch.save(self.dataset, file_path)
        print(f"Dataset guardado en {file_path}")

    @staticmethod
    def load_dataset(file_path):
        dataset = torch.load(file_path)
        print(f"Dataset cargado desde {file_path}")
        return OMTFDataset(dataset=dataset)

def main():
    import argparse
    from torch_geometric.data import DataLoader

    parser = argparse.ArgumentParser(description="Load ROOT files and create a PyTorch Geometric dataset")
    parser.add_argument('--root_dir', type=str, required=True, help='Directory containing the ROOT files')
    parser.add_argument('--tree_name', type=str, default="simOmtfPhase2Digis/OMTFHitsTree", help='Name of the tree inside the ROOT files')
    parser.add_argument('--muon_vars', nargs='+', type=str, required=True, help='List of muon variables to extract')
    parser.add_argument('--stub_vars', nargs='+', type=str, required=True, help='List of stub variables to extract')
    parser.add_argument('--plot_example', action='store_true', help='Plot an example graph')
    parser.add_argument('--save_path', type=str, help='Path to save the dataset')
    parser.add_argument('--load_path', type=str, help='Path to load the dataset')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max_events', type=int, help='Maximum number of events to process')

    args = parser.parse_args()

    if args.load_path:
        dataset = OMTFDataset.load_dataset(args.load_path)
    else:
        pre_transformation = remove_empty_or_nan_graphs
        dataset = OMTFDataset(root_dir=args.root_dir, pre_transform=pre_transformation, tree_name=args.tree_name, muon_vars=args.muon_vars, stub_vars=args.stub_vars, max_files=args.max_files, max_events=args.max_events)
        if args.save_path:
            dataset.save_dataset(args.save_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in dataloader:
        print(data)

    if args.plot_example:
        dataset.plot_example_graph(index=0, num_nodes=5)

if __name__ == "__main__":
    main()
