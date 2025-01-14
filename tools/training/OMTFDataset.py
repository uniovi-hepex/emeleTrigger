import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import BaseTransform


class NormalizeNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x'):
            data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
        return data

class NormalizeEdgeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'edge_attr'):
            data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0)) / data.edge_attr.std(dim=0)
        return data

class NormalizeTargets(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'y'):
            data.y = (data.y - data.y.mean(dim=0)) / data.y.std(dim=0)
        return data
    
class NormalizeSpecificNodeFeatures(BaseTransform):
    def __init__(self, column_indices):
        self.column_indices = column_indices

    def __call__(self, data):
        if hasattr(data, 'x'):
            for column_index in self.column_indices:
                column = data.x[:, column_index]
                mean = column.mean()
                std = column.std()
                data.x[:, column_index] = (column - mean) / std
        return data


class OMTFDataset(Dataset):
    def __init__(self, root_dir, tree_name, muon_vars=None, stub_vars=None, transform=None, pre_transform=None):
        """
        Args:
            root_dir (str): Directorio que contiene los archivos ROOT.
            tree_name (str): Nombre del árbol dentro de los archivos ROOT.
            muon_vars (list of str, optional): Lista de variables de muones a extraer.
            stub_vars (list of str, optional): Lista de variables de stubs a extraer.
            transform (callable, optional): Una función/transformación que toma un objeto Data y lo devuelve transformado.
            pre_transform (callable, optional): Una función/transformación que se aplica antes de guardar los datos.
        """
        self.root_dir = root_dir
        self.tree_name = tree_name
        self.muon_vars = muon_vars if muon_vars is not None else []
        self.stub_vars = stub_vars if stub_vars is not None else []
        self.transform = transform
        self.pre_transform = pre_transform
        self.dataset = self.load_data_from_root()

    def load_data_from_root(self):
        """
        Carga datos desde archivos ROOT utilizando uproot.
        """
        import os
        data_list = []

        for root_file in os.listdir(self.root_dir):
            if root_file.endswith(".root"):
                file_path = os.path.join(self.root_dir, root_file)
                with uproot.open(file_path) as file:
                    tree = file[self.tree_name]
                    muon_data = {var: tree[var].array() for var in self.muon_vars}
                    stub_data = {var: tree[var].array() for var in self.stub_vars}

                    for i in range(len(stub_data[self.stub_vars[0]])):
                        muon_sample = {var: muon_data[var][i] for var in self.muon_vars}
                        stub_sample = {var: stub_data[var][i] for var in self.stub_vars}

                        # Realizar operaciones con las variables y añadir algunas más
                        stub_sample['new_var'] = stub_sample[self.stub_vars[0]] * 2  # Ejemplo de nueva variable

                        # Crear nodos y aristas
                        x = torch.tensor([stub_sample[var] for var in self.stub_vars], dtype=torch.float).view(-1, len(self.stub_vars))
                        edge_index = self.create_edges(stub_sample[self.stub_vars[1]])

                        data = Data(x=x, edge_index=edge_index, y=torch.tensor([muon_sample[var] for var in self.muon_vars], dtype=torch.float))
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)

        return data_list

    def create_edges(self, stubLayer):
        """
        Crea aristas basadas en las capas de los stubs.
        """
        edge_index = []
        for i in range(len(stubLayer)):
            for j in range(i + 1, len(stubLayer)):
                edge_index.append([i, j])
                edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
    
def main():
    import argparse
    from torch_geometric.data import DataLoader
    parser = argparse.ArgumentParser(description="OMTF Dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--muon_vars', type=str, nargs='+', default=['muonQPt','muonPt','muonQOverPt','muonPropEta','muonPropPhi'], help='Muon variables to store in the dataset')
    parser.add_argument('--stub_vars', type=str, nargs='+', default=['stubEtaG', 'stubPhiG','stubR', 'stubLayer','stubType'], help='Stub variables to store in the dataset')
    args = parser.parse_args()

    # Definir transformaciones y filtros si es necesario
    class MyTransform(BaseTransform):
        def __call__(self, data):
            # Implementa la lógica de transformación aquí
            return data

    def my_filter(data):
        # Implementa la lógica de filtrado aquí
        return True

    dataset = OMTFDataset(root=args.data_dir, transform=MyTransform(), pre_filter=my_filter)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for data in loader:
        print(data)

if __name__ == "__main__":
    main()