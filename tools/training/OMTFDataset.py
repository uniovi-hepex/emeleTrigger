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
    def __init__(self, root_dir, muon_vars=None, stub_vars=None, transform=None):
        '''
        Args:
            root_dir (str): Directory with all the .root files  
            transform (BaseTransform, optional): Transformation to be applied to all data. Defaults to None.
            muon_vars (list, optional): muon variables (y) to store in the dataset. Defaults to None.
            stub_vars (list, optional): stub variables (x) to store in the dataset. Defaults to None.
            connection_model (str, optional): Connection model to use. Defaults to 'knn'.
        '''
        super(OMTFDataset, self).__init__(root_dir, transform)
        self.muon_vars = muon_vars
        self.stub_vars = stub_vars
        self.connection_model = connection_model
        self.input_files = os.listdir(root_dir)
        
        self.processed_file = os.path.join(self.processed_dir, 'data.pt')

    @property
    def raw_file_names(self):
        # Devuelve una lista de nombres de archivos en el directorio raw
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pt')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No es necesario descargar nada, ya que los archivos .pt ya están en el directorio raw
        pass

    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            data = torch.load(raw_path)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file)

    def len(self):
        return len(self.processed_paths)

    def get(self, idx):
        data = torch.load(self.processed_file)
        return data[idx]

def main():
    import argparse
    from torch_geometric.data import DataLoader

    parser = argparse.ArgumentParser(description="OMTF Dataset")
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing .pt files')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
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