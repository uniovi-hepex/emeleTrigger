from torch_geometric.transforms import BaseTransform, Compose
import numpy as np

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

# OnlyLayerInfo
class DropLastNodeFeature(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x'):
            data.x = data.x[:, :-1]  # Eliminar la columna
        return data

class DropLastTwoNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x'):
            data.x = data.x[:, :-2]  # Eliminar las dos últimas columnas
        return data

# DropAllLayerInfo
class DropLastThreeNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x'):
            data.x = data.x[:, :-3]  # Eliminar las tres últimas columnas
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
        
# Definir las transformaciones
NormalizeNodeEdgesAndDropOneFeature = Compose([
    NormalizeNodeFeatures(),
    NormalizeEdgeFeatures(),
    DropLastNodeFeature()  # Aplicar la transformación para eliminar la última característica
])

NormalizeNodeEdgesAndDropTwoFeatures = Compose([
    NormalizeNodeFeatures(),
    NormalizeEdgeFeatures(),
    DropLastTwoNodeFeatures()  # Aplicar la transformación para eliminar las dos últimas características
])

NormalizeNodeEdgesAndDropThreeFeatures = Compose([
    NormalizeNodeFeatures(),
    NormalizeEdgeFeatures(),
    DropLastThreeNodeFeatures()  # Aplicar la transformación para eliminar la última característica
])
