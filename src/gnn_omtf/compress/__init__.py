"""Pruning / quantisation helpers for deployment."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


"""
# 1) prune 70 % of weights (unstructured)
gnn-omtf-compress prune --ckpt runs/best.ckpt --amount 0.7 \
                        --out runs/best_pruned.pt

# 2) optional: dynamic INT8 on pruned model
gnn-omtf-compress dyn-int8 --ckpt runs/best_pruned.pt --out runs/best_int8.pt

# 3) evaluation
gnn-omtf-compress eval \
    --baseline runs/best.ckpt \
    --compressed runs/best_int8.pt \
    --dataset  /eos/.../ds_regression.pt \
    --out compress_report.json

4)The report will look like:
{
  "size_baseline": 412.5,
  "size_compressed": 106.7,
  "sparsity": 0.70,
  "mse_baseline": 0.00042,
  "mse_compressed": 0.00055,
  "ms_per_ex_baseline": 0.48,
  "ms_per_ex_compressed": 0.19
}

# weight + channel pruning
gnn-omtf-compress prune --ckpt best.ckpt --amount 0.5 --structured --out best_cpr.pt

# edge sparsity 50 %
gnn-omtf-compress edge-prune --dataset graphs.pt --percentile 0.5 --out graphs_sparse.pt

# quantise INT8 dynamically
gnn-omtf-compress dyn-int8 --ckpt best_cpr.pt --out best_cpr_int8.pt

# evaluate impact
gnn-omtf-compress eval \
   --baseline best.ckpt \
   --compressed best_cpr_int8.pt \
   --dataset   graphs_sparse.pt

"""