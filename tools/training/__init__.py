# tools/training/__init__.py

# Código de inicialización
print("Initializing tools.training package")

# Importaciones
from .models import GATRegressor, GraphSAGEModel, MPLNNRegressor, GCNRegressor
from .TrainModelFromGraph import TrainModelFromGraph
from .OMTFDataset import OMTFDataset, remove_empty_or_nan_graphs