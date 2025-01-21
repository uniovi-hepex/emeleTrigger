import os
import uproot
import torch
from torch_geometric.data import Dataset, Data
import networkx as nx


import numpy as np
import matplotlib.pyplot as plt


''' Auxliary functions and variables'''
NUM_PROCESSORS = 3
NUM_PHI_BINS = 5400
HW_ETA_TO_ETA_FACTOR=0.010875

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
    

class OMTFDataset(Dataset):
    def __init__(self, root_dir, tree_name, muon_vars=None, stub_vars=None, transform=None, pre_transform=None, dataset=None, max_files=None, max_events=None):
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
            self.transform = transform
            self.pre_transform = pre_transform
            self.dataset = self.load_data_from_root()

    def add_extra_vars_to_tree(self, tree):
        """
        Add some extra variables....
        """
        df = tree.arrays(library="pd")  # Convertir el árbol a un DataFrame de pandas

        df['stubR'] = df.apply(lambda x: get_stub_r(x['stubType'], x['stubEta'], x['stubLayer'], x['stubQuality']), axis=1)
        df['stubPhi'] = df['stubPhi'] + df['stubPhiB']
        df['stubEtaG'] = df['stubEta'] * HW_ETA_TO_ETA_FACTOR
        df['stubPhiG'] = df.apply(lambda x: get_global_phi(x['stubPhi'], x['omtfProcessor']), axis=1)
        df['muonPropEta'] = df['muonPropEta'].abs()
        df['muonQPt'] = df['muonCharge'] * df['muonPt']
        df['muonQOverPt'] = df['muonCharge'] / df['muonPt']

        return df
    
    def load_data_from_root(self):
        data_list = []
        files_processed = 0
        events_processed = 0

        for root_file in os.listdir(self.root_dir):
            if self.max_files is not None and files_processed >= self.max_files:
                break
            if root_file.endswith(".root"):
                file_path = os.path.join(self.root_dir, root_file)
                with uproot.open(file_path) as file:
                    tree = file[self.tree_name]
                    df = self.add_extra_vars_to_tree(tree)  # Convertir el árbol a DataFrame y agregar variables

                    for i in range(len(df)):
                        if self.max_events is not None and events_processed >= self.max_events:
                            break
                        if events_processed % 100 == 0:
                            print(f"Processed {events_processed} events")

                        muon_sample = {var: df[var].iloc[i] for var in self.muon_vars}
                        stub_sample = {var: df[var].iloc[i] for var in self.stub_vars}

                        # Realizar operaciones con las variables y añadir algunas más
                        stub_sample['new_var'] = stub_sample[self.stub_vars[0]] * 2  # Ejemplo de nueva variable

                        # Crear nodos y aristas
                        x = torch.tensor([stub_sample[var] for var in self.stub_vars], dtype=torch.float).view(-1, len(self.stub_vars))
                        edge_index = self.create_edges(stub_sample[self.stub_vars[1]])

                        data = Data(x=x, edge_index=edge_index, y=torch.tensor([muon_sample[var] for var in self.muon_vars], dtype=torch.float))
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                        events_processed += 1

                files_processed += 1

        return data_list

    def create_edges(self, stubLayer):
        edge_index = []
        for i in range(len(stubLayer)):
            for j in range(i + 1, len(stubLayer)):
                edge_index.append([i, j])
                edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def len(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def get(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __repr__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def __str__(self):
        return '({}({})'.format(self.__class__.__name__, len(self))
    
    def plot_example_graph(self, index=0, num_nodes=None):
        data = self.get(index)
        
        G = nx.Graph()
        
        for i, node_feature in enumerate(data.x):
            G.add_node(i, feature=node_feature.tolist())
        
        edge_index = data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i], weight=data.edge_weight[i].item() if 'edge_weight' in data else 1.0)
        
        if num_nodes is not None:
            nodes_to_keep = list(G.nodes)[:num_nodes]
            G = G.subgraph(nodes_to_keep)
        
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title(f'Grafo de ejemplo del índice {index} con {num_nodes if num_nodes is not None else "todos"} nodos')
        plt.show()

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
    parser.add_argument('--tree_name', type=str, required=True, help='Name of the tree inside the ROOT files')
    parser.add_argument('--muon_vars', nargs='+', required=True, help='List of muon variables to extract')
    parser.add_argument('--stub_vars', nargs='+', required=True, help='List of stub variables to extract')
    parser.add_argument('--plot_example', action='store_true', help='Plot an example graph')
    parser.add_argument('--save_path', type=str, help='Path to save the dataset')
    parser.add_argument('--load_path', type=str, help='Path to load the dataset')
    parser.add_argument('--max_files', type=int, help='Maximum number of files to process')
    parser.add_argument('--max_events', type=int, help='Maximum number of events to process')

    args = parser.parse_args()

    if args.load_path:
        dataset = OMTFDataset.load_dataset(args.load_path)
    else:
        dataset = OMTFDataset(root_dir=args.root_dir, tree_name=args.tree_name, muon_vars=args.muon_vars, stub_vars=args.stub_vars, max_files=args.max_files, max_events=args.max_events)
        if args.save_path:
            dataset.save_dataset(args.save_path)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for data in dataloader:
        print(data)

    if args.plot_example:
        dataset.plot_example_graph(index=0, num_nodes=5)

if __name__ == "__main__":
    main()