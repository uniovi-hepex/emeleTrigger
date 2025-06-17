import torch
from torch_geometric.data import Data
from gnn_omtf.models import BaseGNN


def _fake_batch(model_name: str):
    data = Data(
        x=torch.randn(40, 6),
        edge_index=torch.randint(0, 40, (2, 120)),
        batch=torch.zeros(40, dtype=torch.long),
    )

    # ------------------------------------------------------------------ #
    # model-specific kwargs
    # ------------------------------------------------------------------ #
    if model_name == "gat":
        model = BaseGNN.get(model_name, num_node_features=6, hidden_dim=32)
    elif model_name in {"gcn", "sage"}:
        model = BaseGNN.get(model_name, in_channels=6, hidden_channels=16)
    elif model_name == "mpl":
        model = BaseGNN.get(model_name, in_channels=6)          # only this one
    else:  # defensive fallback
        raise ValueError(model_name)

    out = model(data)
    assert out.ndim == 1
    assert out.shape in {torch.Size([1]), torch.Size([40])}


def test_all_architectures():
    for name in ["gat", "gcn", "sage", "mpl"]:
        _fake_batch(name)
