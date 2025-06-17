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

from converter import get_stub_r, get_global_phi, HW_ETA_TO_ETA_FACTOR, getEdgesFromLogicLayer, remove_empty_or_nan_graphs
from converter import getEdgesFromLogicLayer, get_layer_order
''' Auxliary functions and variables move to some auxiliary file '''

def add_stubCosPhi(arr, input_field, output_field):
    # Aplica np.cos directamente sobre el arreglo awkwrd
    stub_cos = np.cos(arr[input_field])
    # Agrega el nuevo campo al array
    arr = ak.with_field(arr, stub_cos, output_field)
    return arr

def add_stubSinPhi(arr, input_field, output_field):
    # Aplica np.cos directamente sobre el arreglo awkwrd
    stub_cos = np.sin(arr[input_field])
    # Agrega el nuevo campo al array
    arr = ak.with_field(arr, stub_cos, output_field)
    return arr

def add_layer_order(arr, eta_field, layer_field, output_field):
    """
    Add the layer order to the array based on the eta and layer.
    """
    stub_eta   = np.abs(arr[eta_field])
    stub_layer = np.abs(arr[layer_field])
    arr = ak.with_field(arr, get_layer_order(stub_eta, stub_layer), output_field)
    return arr

class OMTFDataset(Dataset):
    def __init__(self, **kwargs):
        """
        Los parámetros se pueden pasar como keyword arguments:
            root_dir (str): Directorio que contiene los archivos ROOT.
            tree_name (str): Nombre del árbol (default: "simOmtfPhase2Digis/OMTFHitsTree").
            muon_vars (list): Variables de muones a extraer.
            stub_vars (list): Variables de stubs a extraer.
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
            # Combinar: los parámetros ya presentes en kwargs toman prioridad.
            for key, value in config.items():
                print(f"Setting {key} from config file: {value}")
                kwargs[key] = value

        self.root_dir      = kwargs.get("root_dir")
        self.tree_name     = kwargs.get("tree_name", "simOmtfPhase2Digis/OMTFHitsTree")
        self.muon_vars     = kwargs.get("muon_vars", [])
        self.omtf_vars     = kwargs.get("omtf_vars", [])
        self.stub_vars     = kwargs.get("stub_vars", [])
        self.target_vars   = kwargs.get("target_vars", [])
        self.task          = kwargs.get("task", "regression")
        self.max_files     = kwargs.get("max_files", None)  # None means no limit
        self.max_events    = kwargs.get("max_events",None)
        self.debug         = kwargs.get("debug", False)
        self.pre_transform = kwargs.get("pre_transform")
        self.transform     = kwargs.get("transform")

        if "dataset" in kwargs and kwargs["dataset"] is not None:
            self.dataset = kwargs["dataset"]
        else:
            self.dataset = self.load_data_from_root()

    def add_extra_vars_to_tree(self, arr):
        """
        Add some extra variables....
        """
        if not hasattr(arr, "stubR"):
            arr['stubR'] = get_stub_r(arr['stubType'], arr['stubEta'], arr['stubLayer'], arr['stubQuality'])
        arr['stubEtaG'] = arr['stubEta'] * HW_ETA_TO_ETA_FACTOR
        arr['stubPhiG'] = get_global_phi(arr['stubPhi'], arr['omtfProcessor'])  ## need to check this value!! (not sure it is OK)
        arr = add_stubCosPhi(arr, 'stubPhiG', 'stubCosPhi')
        arr = add_stubSinPhi(arr, 'stubPhiG', 'stubSinPhi')
        #arr = add_layer_order(arr, 'stubEtaG', 'stubLayer', 'stubLayerOrder')

        if not hasattr(arr, "inputStubR"):
            arr['inputStubR'] = get_stub_r(arr['inputStubType'], arr['inputStubEta'], arr['inputStubLayer'], arr['inputStubQuality'])
        arr['inputStubEtaG'] = arr['inputStubEta'] * HW_ETA_TO_ETA_FACTOR
        arr['inputStubPhiG'] = get_global_phi(arr['inputStubPhi'], arr['omtfProcessor'])  ## need to check this value!! (not sure it is OK)
        arr = add_stubCosPhi(arr, 'inputStubPhiG', 'inputStubCosPhi')
        arr = add_stubSinPhi(arr, 'inputStubPhiG', 'inputStubSinPhi')
        #arr = add_layer_order(arr, 'inputStubEtaG', 'inputStubLayer', 'inputStubLayerOrder')

        arr['muonQPt'] = arr['muonCharge'] * arr['muonPt']
        arr['muonQOverPt'] = arr['muonCharge'] / arr['muonPt']
        
        return arr
    
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
            print(f"Processing file: {root_file}")
            if self.max_files is not None and files_processed >= self.max_files:
                break
            
            file = uproot.open(root_file)
            tree = file[self.tree_name]
            arr = tree.arrays(library="ak")
            
            arr = self.add_extra_vars_to_tree(arr)

            for event in ak.to_list(arr):
                if events_processed % 500 == 0:
                    print(f"Processed {events_processed} events")
                if self.max_events is not None and events_processed >= self.max_events:
                    break

                # drop the event if it has no stubs
                if (event['stubNo']) == 0 or (event['inputStubNo']) == 0:
                    continue
               

                # Now create nodes and edges: 
                node_features = torch.tensor([event[st] for st in self.stub_vars], dtype=torch.float32).transpose(0,1)
                target_tensor = torch.tensor([event[st] for st in self.target_vars], dtype=torch.float32)
                if target_tensor.dim() == 2:
                    target_features = target_tensor.transpose(0, 1)
                else:
                    target_features = target_tensor
                
                if self.task == 'classification':
                    edge_index, edge_attr, edge_label = self.create_edges(event, 'inputStub')
                    target_features = None               
                elif self.task == 'regression':
                    edge_index, edge_attr, edge_label = self.create_edges(event)               

                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=target_features, edge_label=edge_label, dtype=torch.float)

                # Add extra features to the data object
                data.muon_vars = torch.tensor([event[var] for var in self.muon_vars], dtype=torch.float)
                data.omtf_vars = torch.tensor([event[var] for var in self.omtf_vars], dtype=torch.float)
                
                if self.pre_transform is not None:
                    # Apply pre-transformations to the data
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
    
    def getDeltaR(self, r1, r2):
        """
        Calculate the delta R between two points 
        """
        return r1 - r2
    
    def create_edges(self, row, stubName='stub'):
        stubLayer = row['%sLayer' % stubName]
        stubPhi = row['%sPhi' % stubName]
        stubEta = row['%sEta' % stubName]
        stubR = row['%sR' % stubName]
        stubIsMatched = row['%sIsMatched' % stubName] if '%sIsMatched' % stubName in row else None
        edge_index = []
        edge_attr = []
        edge_label = []  # This is only used for classification tasks, so it can be empty for regression tasks
        for stub1Id,stub1Layer in enumerate(stubLayer):
            for stub2Id, stub2Layer in enumerate(stubLayer):
                #print(f'stubt1Id:{stub1Id}   stub1Layer:{stub1Layer}    stubt2Id:{stub2Id}   stub2Layer:{stub2Layer}')
                if stub1Layer == stub2Layer: continue
                if stub2Layer in getEdgesFromLogicLayer(stub1Layer):
                    dphi = self.getDeltaPhi(stubPhi[stub1Id],stubPhi[stub2Id])
                    deta = self.getDeltaEta(stubEta[stub1Id],stubEta[stub2Id])
                    dr   = self.getDeltaR(stubR[stub1Id], stubR[stub2Id])
                    edge_index.append([stub1Id, stub2Id])
                    edge_attr.append([dphi, deta, dr])
                    edge_label.append(int(stubIsMatched[stub1Id] and stubIsMatched[stub2Id]) if stubIsMatched is not None else 0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return edge_index, edge_attr, edge_label

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
        return OMTFDataset(dataset=dataset)

def main():
    import argparse
    from torch_geometric.loader import DataLoader

    parser = argparse.ArgumentParser(description="Load ROOT files and create a PyTorch Geometric dataset")
    parser.add_argument('--config', type=str, help='Path to the configuration file with parameters')
    parser.add_argument('--root_dir', type=str, required=True, help='Directory containing the ROOT files')
    parser.add_argument('--tree_name', type=str, default="simOmtfPhase2Digis/OMTFHitsTree", help='Name of the tree inside the ROOT files')
    parser.add_argument('--muon_vars', nargs='+', type=str, help='List of muon variables to extract')
    parser.add_argument('--omtf_vars', nargs='+', type=str, help='List of OMTF variables to extract')
    parser.add_argument('--stub_vars', nargs='+', type=str, help='List of stub variables to extract')
    parser.add_argument('--task', type=str, default='regression', help='Task type (classification or regression)')
    parser.add_argument('--plot_example', action='store_true', help='Plot an example graph')
    parser.add_argument('--save_path', type=str, help='Path to save the dataset')
    parser.add_argument('--load_path', type=str, help='Path to load the dataset')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max_events', type=int, help='Maximum number of events to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.load_path:
        dataset = OMTFDataset.load_dataset(args.load_path)
    else:
        pre_transformation = remove_empty_or_nan_graphs
        dataset = OMTFDataset(pre_transform=pre_transformation, **vars(args))

        if args.save_path:
            dataset.save_dataset(args.save_path)


    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    if args.plot_example:
        dataset.plot_example_graph(index=0, num_nodes=5)

if __name__ == "__main__":
    main()
