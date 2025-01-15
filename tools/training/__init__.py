# tools/training/__init__.py

# Código de inicialización
print("Initializing tools.training package")

# Importaciones
from .models import GATRegressor, GraphSAGEModel, MPLNNRegressor
from .transformations import DropLastTwoNodeFeatures, NormalizeNodeFeatures, NormalizeEdgeFeatures, NormalizeTargets