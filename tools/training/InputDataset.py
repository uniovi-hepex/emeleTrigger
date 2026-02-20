import os
import uproot
import torch
from torch_geometric.data import Dataset, Data
import awkward as ak
import networkx as nx
from operator import xor

from torch_geometric.utils.convert import to_networkx

import numpy as np
import matplotlib.pyplot as plt

import yaml

try:
    from converter import remove_empty_or_nan_graphs
except ImportError:
    def remove_empty_or_nan_graphs(data):
        """Placeholder function if converter is not available"""
        return data if data.x.shape[0] > 0 else None

''' Dataset classes for L1Nano and OMTF data '''

class L1NanoDataset(Dataset):
    def __init__(self, **kwargs):
        """
        PyTorch Dataset para leer eventos de L1Nano ROOT files.
        
        Parámetros:
            root_dir (str): Directorio que contiene los archivos ROOT o ruta a un archivo ROOT.
            tree_name (str): Nombre del árbol (default: "Events").
            dR_threshold (float): Umbral de dR para matching geométrico (default: 0.15).
            max_files (int): Número máximo de archivos a procesar.
            max_events (int): Número máximo de eventos a procesar.
            debug (bool): Modo debug.
            pre_transform (callable): Función de pre-transformación.
            transform (callable): Función de transformación (opcional).
        """
        config_file = kwargs.get("config")
        if config_file is not None:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            for key, value in config.items():
                print(f"Setting {key} from config file: {value}")
                kwargs[key] = value

        self.root_dir      = kwargs.get("root_dir")
        self.tree_name     = kwargs.get("tree_name", "Events")
        
        # Definir features de stubs y GenPart
        self.stub_vars     = kwargs.get("stub_vars", [
            'eta1', 'eta2', 'phi1', 'phi2', 'qual', 'type', 
            'depthregion', 'etaregion', 'phiregion', 'tfLayer', 'etaqual'
        ])
        self.genpart_vars  = kwargs.get("genpart_vars", [
            'pt', 'eta', 'phi', 'mass', 'pdgId', 'dXY', 'lXY'
        ])
        
        self.dR_threshold  = kwargs.get("dR_threshold", 0.15)
        self.task          = kwargs.get("task", "classification")  # classification or regression
        self.max_files     = kwargs.get("max_files", None)
        self.max_events    = kwargs.get("max_events", None)
        self.debug         = kwargs.get("debug", False)
        self.pre_transform = kwargs.get("pre_transform")
        self.transform     = kwargs.get("transform")

        if "dataset" in kwargs and kwargs["dataset"] is not None:
            self.dataset = kwargs["dataset"]
        else:
            self.dataset = self.load_data_from_root()

    def add_extra_vars_to_tree(self, arr):
        """
        Adds extra variables to the L1Nano tree.
        For L1Nano, we mostly work with stub and GenPart info directly.
        """
        return arr
    
    def load_data_from_root(self):
        """
        Load events from L1Nano ROOT files and create graph data.
        """
        data_list = []
        files_processed = 0
        events_processed = 0

        # Check if root_dir is a directory or a file
        if os.path.isdir(self.root_dir):
            root_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".root")]
        elif os.path.isfile(self.root_dir) and self.root_dir.endswith(".root"):
            root_files = [self.root_dir]
        else:
            raise ValueError(f"{self.root_dir} is not a valid directory or ROOT file")

        # Define branches to load
        stub_branches = [f"stub_{var}" for var in self.stub_vars]
        genpart_branches = [f"GenPart_{var}" for var in self.genpart_vars]
        all_branches = stub_branches + genpart_branches + ['GenPart_statusFlags']
        
        for root_file in root_files:
            print(f"Processing file: {root_file}")
            if self.max_files is not None and files_processed >= self.max_files:
                break
            
            try:
                file = uproot.open(root_file)
                tree = file[self.tree_name]
                
                # Load data using arrays with 'how=zip'
                events_data = tree.arrays(
                    filter_name=all_branches,
                    how="zip",
                    library="ak"
                )
                
                # Convert to list for iteration
                num_events = len(events_data)
                print(f"  Found {num_events} events in file")
                
                for event_idx in range(num_events):
                    if self.max_events is not None and events_processed >= self.max_events:
                        break
                    
                    event = events_data[event_idx]
                    
                    # Extract stub features
                    stub_features = self._extract_stub_features(event)
                    
                    # Extract GenPart features (muons only)
                    genpart_features = self._extract_genpart_features(event)
                    
                    # Skip events with no stubs
                    if stub_features.shape[0] == 0:
                        continue
                    
                    # Create edges between stubs (by tfLayer)
                    edge_index, edge_attr = self._create_edges_by_layer(stub_features)
                    
                    # Match stubs to gen muons
                    stub_labels = self._match_stubs_to_genpart(event)
                    
                    # Create graph data
                    graph_data = Data(
                        x=stub_features,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=stub_labels,
                        genpart_features=genpart_features
                    )
                    
                    # Apply pre-transform if provided
                    if self.pre_transform is not None:
                        graph_data = self.pre_transform(graph_data)
                        if graph_data is None:
                            continue
                    
                    data_list.append(graph_data)
                    events_processed += 1
                    
                    if self.debug and events_processed % 100 == 0:
                        print(f"  Processed {events_processed} events")
                
                files_processed += 1
                file.close()
                
            except Exception as e:
                print(f"  Error processing file {root_file}: {e}")
                continue
        
        print(f"\nTotal events loaded: {len(data_list)}")
        return data_list

    def _extract_stub_features(self, event):
        """
        Extract stub features as tensor [num_stubs, num_features].
        """
        # Access stub arrays directly from event.stub
        num_stubs = len(event.stub.eta1) if hasattr(event.stub, 'eta1') else 0
        
        if num_stubs == 0:
            return torch.zeros((0, len(self.stub_vars)), dtype=torch.float32)
        
        features = []
        for feat in self.stub_vars:
            if hasattr(event.stub, feat):
                feat_array = getattr(event.stub, feat)
                feat_tensor = torch.tensor(ak.to_numpy(feat_array), dtype=torch.float32)
            else:
                feat_tensor = torch.zeros(num_stubs, dtype=torch.float32)
            features.append(feat_tensor)
        
        stub_tensor = torch.stack(features, dim=1)
        return stub_tensor
    
    def _extract_genpart_features(self, event):
        """
        Extract GenPart features (muons only) as tensor [num_muons, num_features].
        """
        # Filter for muons: abs(pdgId) == 13 and isLastCopy
        pdgIds = event.GenPart.pdgId if hasattr(event.GenPart, 'pdgId') else []
        
        if len(pdgIds) == 0:
            return torch.zeros((0, len(self.genpart_vars)), dtype=torch.float32)
        
        mask_muons = abs(pdgIds) == 13
        
        # Filter by statusFlags (bit 13 = isLastCopy)
        if hasattr(event.GenPart, 'statusFlags'):
            statusFlags = event.GenPart.statusFlags
            mask_lastcopy = (statusFlags & (1 << 13)) != 0
            mask_muons = mask_muons & mask_lastcopy
        
        if ak.sum(mask_muons) == 0:
            return torch.zeros((0, len(self.genpart_vars)), dtype=torch.float32)
        
        features = []
        for feat in self.genpart_vars:
            if hasattr(event.GenPart, feat):
                feat_array = getattr(event.GenPart, feat)[mask_muons]
                feat_tensor = torch.tensor(ak.to_numpy(feat_array), dtype=torch.float32)
            else:
                num_muons = ak.sum(mask_muons)
                feat_tensor = torch.zeros(num_muons, dtype=torch.float32)
            features.append(feat_tensor)
        
        genpart_tensor = torch.stack(features, dim=1) if features else torch.zeros((ak.sum(mask_muons), 0), dtype=torch.float32)
        return genpart_tensor
    
    def _match_stubs_to_genpart(self, event):
        """
        Perform geometric matching between stubs and gen muons using dR.
        Returns tensor of shape [num_stubs] with muon index (or -1 if no match).
        """
        # Use offeta1/offphi1 if available, otherwise eta1/phi1
        stub_eta = event.stub.offeta1 if hasattr(event.stub, 'offeta1') else event.stub.eta1
        stub_phi = event.stub.offphi1 if hasattr(event.stub, 'offphi1') else event.stub.phi1
        
        num_stubs = len(stub_eta)
        
        # Get gen muons
        pdgIds = event.GenPart.pdgId if hasattr(event.GenPart, 'pdgId') else []
        if len(pdgIds) == 0:
            return torch.full((num_stubs,), -1, dtype=torch.long)
        
        mask_muons = abs(pdgIds) == 13
        if hasattr(event.GenPart, 'statusFlags'):
            statusFlags = event.GenPart.statusFlags
            mask_lastcopy = (statusFlags & (1 << 13)) != 0
            mask_muons = mask_muons & mask_lastcopy
        
        if ak.sum(mask_muons) == 0:
            return torch.full((num_stubs,), -1, dtype=torch.long)
        
        muon_eta = ak.to_numpy(event.GenPart.eta[mask_muons])
        muon_phi = ak.to_numpy(event.GenPart.phi[mask_muons])
        
        stub_eta_np = ak.to_numpy(stub_eta)
        stub_phi_np = ak.to_numpy(stub_phi)
        
        matched_indices = np.full(num_stubs, -1, dtype=np.int64)
        
        for i in range(num_stubs):
            min_dR = float('inf')
            matched_idx = -1
            
            for j in range(len(muon_eta)):
                deta = stub_eta_np[i] - muon_eta[j]
                dphi = stub_phi_np[i] - muon_phi[j]
                
                # Normalize dphi to [-pi, pi]
                while dphi > np.pi:
                    dphi -= 2*np.pi
                while dphi < -np.pi:
                    dphi += 2*np.pi
                
                dR = np.sqrt(deta**2 + dphi**2)
                
                if dR < min_dR and dR < self.dR_threshold:
                    min_dR = dR
                    matched_idx = j
            
            matched_indices[i] = matched_idx
        
        return torch.tensor(matched_indices, dtype=torch.long)
    
    def _create_edges_by_layer(self, stub_features):
        """
        Create edges connecting stubs between consecutive tfLayer.
        stub_features: tensor [num_stubs, num_features]
        tfLayer is feature index 9.
        """
        num_stubs = stub_features.shape[0]
        
        if num_stubs == 0:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 3), dtype=torch.float32)
        
        # Get tfLayer (feature index 9)
        tflayer = stub_features[:, 9].numpy()
        eta = stub_features[:, 0].numpy()
        phi = stub_features[:, 2].numpy()
        
        edge_index = []
        edge_attr = []
        
        unique_layers = np.unique(tflayer)
        
        # Connect stubs between consecutive layers
        for layer_idx in range(len(unique_layers) - 1):
            current_layer = unique_layers[layer_idx]
            next_layer = unique_layers[layer_idx + 1]
            
            current_indices = np.where(tflayer == current_layer)[0]
            next_indices = np.where(tflayer == next_layer)[0]
            
            # Connect stubs within dR_threshold
            for i in current_indices:
                for j in next_indices:
                    deta = eta[i] - eta[j]
                    dphi = phi[i] - phi[j]
                    
                    # Normalize dphi
                    while dphi > np.pi:
                        dphi -= 2 * np.pi
                    while dphi < -np.pi:
                        dphi += 2 * np.pi
                    
                    dR = np.sqrt(deta**2 + dphi**2)
                    
                    if dR < 0.3:  # dR threshold for edges
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # undirected
                        edge_attr.append([deta, dphi, dR])
                        edge_attr.append([deta, dphi, dR])
        
        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
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
        dataset = torch.load(file_path, weights_only=False)
        print(f"Dataset cargado desde {file_path}")
        return L1NanoDataset(dataset=dataset)

def main():
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(description="Load L1Nano ROOT files and create a PyTorch Geometric dataset")
    parser.add_argument('--config', type=str, help='Path to the configuration file with parameters')
    parser.add_argument('--root_dir', type=str, required=True, help='Directory containing the ROOT files or path to a single ROOT file')
    parser.add_argument('--tree_name', type=str, default="Events", help='Name of the tree inside the ROOT files')
    parser.add_argument('--stub_vars', nargs='+', type=str, 
                        default=['eta1', 'eta2', 'phi1', 'phi2', 'qual', 'type', 'depthregion', 'etaregion', 'phiregion', 'tfLayer', 'etaqual'],
                        help='List of stub variables to extract')
    parser.add_argument('--genpart_vars', nargs='+', type=str,
                        default=['pt', 'eta', 'phi', 'mass', 'pdgId', 'dXY', 'lXY'],
                        help='List of GenPart variables to extract')
    parser.add_argument('--dR_threshold', type=float, default=0.15, help='dR threshold for stub-muon matching')
    parser.add_argument('--task', type=str, default='classification', help='Task type (classification or regression)')
    parser.add_argument('--plot_example', action='store_true', help='Plot an example graph')
    parser.add_argument('--save_path', type=str, help='Path to save the dataset')
    parser.add_argument('--load_path', type=str, help='Path to load the dataset')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max_events', type=int, help='Maximum number of events to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.load_path:
        dataset = L1NanoDataset.load_dataset(args.load_path)
    else:
        pre_transformation = remove_empty_or_nan_graphs
        dataset = L1NanoDataset(pre_transform=pre_transformation, **vars(args))

        if args.save_path:
            dataset.save_dataset(args.save_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(f"Dataset loaded with {len(dataset)} events")
    print(f"First event: {dataset[0]}")

    if args.plot_example and len(dataset) > 0:
        dataset.plot_graph(idx=0)

if __name__ == "__main__":
    main()
