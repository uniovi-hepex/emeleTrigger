"""Reusable normalisation / cleaning transforms for PyG graphs."""

from __future__ import annotations

from torch_geometric.transforms import BaseTransform, Compose

from .converter_utils import remove_empty_or_nan_graphs

__all__ = [
    "NormalizeNodeFeatures",
    "NormalizeEdgeFeatures",
    "NormalizeTargets",
    "DropLastTwoNodeFeatures",
    "NormalizeSpecificNodeFeatures",
    "NormalizeNodeEdgesAndDropTwoFeatures",
    "remove_empty_or_nan_graphs",
]


class NormalizeNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, "x"):
            data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
        return data


class NormalizeEdgeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, "edge_attr"):
            data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0)) / data.edge_attr.std(dim=0)
        return data


class NormalizeTargets(BaseTransform):
    def __call__(self, data):
        if hasattr(data, "y"):
            data.y = (data.y - data.y.mean(dim=0)) / data.y.std(dim=0)
        return data


class DropLastTwoNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, "x"):
            data.x = data.x[:, :-2]
        return data


class NormalizeSpecificNodeFeatures(BaseTransform):
    def __init__(self, column_indices):
        self.column_indices = column_indices

    def __call__(self, data):
        if hasattr(data, "x"):
            for idx in self.column_indices:
                col = data.x[:, idx]
                data.x[:, idx] = (col - col.mean()) / col.std()
        return data


# Pre-defined “macro” transform used by some scripts
NormalizeNodeEdgesAndDropTwoFeatures = Compose(
    [
        NormalizeNodeFeatures(),
        NormalizeEdgeFeatures(),
        DropLastTwoNodeFeatures(),
    ]
)
