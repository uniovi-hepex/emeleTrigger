from __future__ import annotations
"""
Compare a *baseline* vs. *compressed* GNN w.r.t. accuracy, size and speed.
"""

from pathlib import Path
from typing import Dict, Union

import json
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_omtf.data import OMTFDataset
from gnn_omtf.metrics import regression_metrics
from gnn_omtf.compress.pruning import size_of_state_dict, sparsity_stats
from gnn_omtf.compress.quantize import bench_throughput       # unchanged
from gnn_omtf.models import BaseGNN                           # factory


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def _rebuild_model_from_cfg(cfg: dict) -> nn.Module:
    """(Re)create an *untrained* model from the saved config dict."""
    return BaseGNN.get(
        name=cfg["model"],
        num_node_features=cfg["num_node_features"],
        hidden_dim=cfg["hidden_dim"],
    )


def _is_quantized(model: nn.Module) -> bool:
    """
    True ⇢ the model contains *any* module whose Python path
    includes '.quantized' (works for both torch.nn.* and torch.ao.nn.*).
    """
    return any(".quantized" in m.__class__.__module__ for m in model.modules())



def _load_dataset_pt(path: Path) -> OMTFDataset:
    """Load a `.pt` file that was created via `torch.save(list_of_graphs)`."""
    graphs = torch.load(path, weights_only=False)
    return OMTFDataset(dataset=graphs)      # relies on constructor fallback

def _run_on_loader(model: nn.Module, loader: DataLoader):
    dev = next(model.parameters(), torch.empty(0)).device
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.clone().to(dev)        # clone → no side-effects
            preds.append(model(batch).cpu())
            trues.append(batch.y.cpu())
    return torch.cat(preds), torch.cat(trues)
# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def try_load_model(ckpt: Path, gpu_device: torch.device) -> nn.Module:
    """
    Load *either* a full `Module` pickled with `torch.save(model)` **or**
    a plain state-dict produced by `model.state_dict()`.

    If the state-dict corresponds to a **dynamic-INT8** model, we quantise
    the freshly rebuilt network *before* `load_state_dict()` so that the
    key-names match (they contain `_packed_params` instead of `weight/bias`).
    """
    try:
        obj: Union[dict, nn.Module] = torch.load(ckpt, map_location="cpu",
                                                 weights_only=False)

        # ── 1. Already a full model object
        if isinstance(obj, nn.Module):
            model = obj

        # ── 2. Plain state-dict: rebuild then load
        elif isinstance(obj, dict):
            cfg_path = ckpt.parent / "config.json"
            with open(cfg_path) as f:
                cfg = json.load(f)

            model = _rebuild_model_from_cfg(cfg)

            # If INT8 dynamic quantisation – detect by packed keys
            if any("._packed_params" in k for k in obj.keys()):
                model = torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )

            model.load_state_dict(obj)

        else:
            raise ValueError("Unsupported checkpoint contents")

        # ── Device placement
        final_device = (torch.device("cpu")
                        if _is_quantized(model)        # INT8 → CPU only
                        else gpu_device)
        return model.to(final_device).eval()

    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {ckpt}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Main public helper                                                          #
# --------------------------------------------------------------------------- #
def evaluate_pair(baseline_ckpt: Path,
                  compressed_ckpt: Path,
                  dataset_pt: Path,
                  batch_size: int = 512) -> Dict[str, float]:

    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline   = try_load_model(baseline_ckpt,   gpu)
    compressed = try_load_model(compressed_ckpt, gpu)

    # 1️⃣ fresh loader for the baseline (may be pushed to GPU)
    ds_base   = _load_dataset_pt(dataset_pt)
    loader_b  = DataLoader(ds_base, batch_size=batch_size, shuffle=False)

    # 2️⃣ a brand-new loader for the (CPU) INT-8 model
    ds_comp   = _load_dataset_pt(dataset_pt)
    loader_c  = DataLoader(ds_comp, batch_size=batch_size, shuffle=False)

    # ---- gather predictions -------------------------------------------------
    pb, tb = _run_on_loader(baseline,   loader_b)
    pc, tc = _run_on_loader(compressed, loader_c)

    mse_b = regression_metrics(pb, tb)["mse"]
    mse_c = regression_metrics(pc, tc)["mse"]

    # ---- throughput (each on its dedicated loader) --------------------------
    tput_b = bench_throughput(baseline,   loader_b)
    tput_c = bench_throughput(compressed, loader_c)

    return dict(
        size_baseline        = size_of_state_dict(baseline)   / 1024.0,  # kB
        size_compressed      = size_of_state_dict(compressed) / 1024.0,
        sparsity             = sparsity_stats(compressed)["global_sparsity"],
        mse_baseline         = mse_b,
        mse_compressed       = mse_c,
        ms_per_ex_baseline   = tput_b,
        ms_per_ex_compressed = tput_c,
    )