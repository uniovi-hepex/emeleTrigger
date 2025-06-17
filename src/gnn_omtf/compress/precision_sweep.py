from __future__ import annotations

"""
Sweep fixed-point precision (W,I) and report MSE + hls4ml resource estimates.
Usage: gnn-omtf-compress sweep-precision --ckpt best.ckpt --dataset ds.pt
"""


from pathlib import Path
from typing import List, Tuple
import json, logging, torch
from torch_geometric.loader import DataLoader

from gnn_omtf.data import OMTFDataset
from gnn_omtf.metrics import regression_metrics
from brevitas.quant import QuantStub, Int8Bias
from brevitas.export.onnx import export_finn_onnx

log = logging.getLogger(__name__)

# ------------------------------------------------------------ #
def quantise_model(model, W: int, I: int):
    """Wrap all Linear layers with Brevitas QuantLinear(Q<W,I>)."""
    from brevitas.nn import QuantLinear
    for name, mod in list(model.named_modules()):
        if isinstance(mod, torch.nn.Linear):
            qmod = QuantLinear(
                in_features=mod.in_features,
                out_features=mod.out_features,
                bias=True,
                weight_bit_width=W,
                weight_integer=W - I,
                bias_quant=Int8Bias)
            qmod.weight.data = mod.weight.data.clone()
            qmod.bias.data = mod.bias.data.clone()
            # replace in parent module
            path = name.split(".")
            parent = model
            for p in path[:-1]:
                parent = getattr(parent, p)
            setattr(parent, path[-1], qmod)
    return model


def evaluate(model, loader):
    device = next(model.parameters()).device
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            preds.append(model(b).cpu())
            trues.append(b.y.cpu())
    return regression_metrics(torch.cat(preds), torch.cat(trues))["mse"]


def sweep(ckpt: Path, dataset: Path,
          precisions: List[Tuple[int, int]],
          out_json: Path = Path("precision_report.json")):
    base = torch.load(ckpt, map_location="cpu", weights_only=False)
    ds = OMTFDataset.load_dataset(dataset)
    loader = DataLoader(ds, batch_size=512)

    results = {}
    for W, I in precisions:
        model_q = quantise_model(copy.deepcopy(base), W, I)
        mse = evaluate(model_q, loader)
        onnx_path = f"tmp_Q{W}_{I}.onnx"
        export_finn_onnx(model_q, input_shape=(1, ds[0].x.shape[1]), onnx_path=onnx_path)
        size_kb = Path(onnx_path).stat().st_size / 1024
        results[f"Q{W}_{I}"] = dict(mse=mse, onnx_kB=size_kb)
        log.info("Q%2d/%2d  mse=%.4e  size=%.1f kB", W, I, mse, size_kb)

    out_json.write_text(json.dumps(results, indent=2))
    return results
