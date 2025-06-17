from __future__ import annotations

import torch
import torch.nn.utils.prune as prune
from torch_geometric.utils import subgraph
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import os


def global_l1_prune(model: torch.nn.Module, amount: float = 0.5) -> torch.nn.Module:
    """
    Prune *amount* (fraction) of weights globally by L1 magnitude.
    """
    parameters_to_prune = [
        (m, "weight")
        for m in model.modules()
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))
    ]
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, _ in parameters_to_prune:
        prune.remove(m, "weight")
    return model


def structured_channel_prune(model: torch.nn.Module, amount: float = 0.3) -> torch.nn.Module:
    """
    Prune entire output channels by their L2 norm (structured).
    Only applied to Linear layers (good for FPGA DSP packing).
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            prune.ln_structured(m, name="weight", amount=amount, n=2, dim=0)
            prune.remove(m, "weight")
    return model


def sparsity_stats(model: torch.nn.Module) -> Dict[str, float]:
    """
    Return global sparsity percentage and per-layer stats.
    """
    total_zeros = total_elems = 0
    per_layer = {}
    for n, p in model.named_parameters():
        zeros = torch.count_nonzero(p == 0).item()
        elems = p.numel()
        per_layer[n] = zeros / elems
        total_zeros += zeros; total_elems += elems
    return dict(global_sparsity=total_zeros / total_elems, per_layer=per_layer)


def size_of_state_dict(model: torch.nn.Module) -> int:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        tmp.flush()
        size = os.path.getsize(tmp.name)
    os.unlink(tmp.name)  # cleanup temp file
    return size


# ========== Edge pruning ========================================== #
def edge_prune_batch(data_list: List, percentile: float = 0.2):
    """
    Drop the lowest `percentile` fraction of edges per-graph.
    Priority:
      1) data.edge_score  (user provides)
      2) data.attn_coeff  (saved by GATConv if return_attention=True)
      3) magnitude of edge_attr[:,0]
    """
    new_list = []
    for data in data_list:
        if hasattr(data, "edge_score"):
            score = data.edge_score
        elif hasattr(data, "attn_coeff"):
            score = data.attn_coeff
        elif data.edge_attr is not None:
            score = data.edge_attr[:, 0].abs()
        else:                              # nothing to rank => keep as is
            new_list.append(data); continue

        k = int((1.0 - percentile) * score.numel())
        if k == 0:                         # prune everything? keep at least 1
            new_list.append(data); continue
        keep_idx = torch.topk(score, k).indices
        edge_index, edge_attr = subgraph(
            keep_idx, data.edge_index, relabel_nodes=False,
            edge_attr=data.edge_attr)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        new_list.append(data)
    return new_list


# ========== Node / channel pruning ================================ #
def node_prune_linear(model: torch.nn.Module, amount: float = 0.3):
    """
    Remove the least-important output channels of Linear layers based on L2 norm.
    """
    import torch.nn.utils.prune as prune
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            prune.ln_structured(mod, name="weight", amount=amount, n=2, dim=0)
            prune.remove(mod, "weight")
    return model


# ========== sparsity on graph level ================================ #
def graph_sparsity(data_list) -> Dict[str, float]:
    e_total = sum(d.edge_index.size(1) for d in data_list)
    v_total = sum(d.num_nodes for d in data_list)
    return {"avg_edges_per_graph": e_total / len(data_list),
            "avg_nodes_per_graph": v_total / len(data_list),
            "edge_density": e_total / v_total}