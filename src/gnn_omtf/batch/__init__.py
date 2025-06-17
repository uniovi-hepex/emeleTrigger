"""HTCondor / LXPLUS batch-submission helpers for gnn-omtf."""
from importlib import resources as _r

# allow template access via:  resources.files(...).joinpath(...)
templates = _r.files(__name__) / "templates"

__all__ = ["templates"]

"""

# dry-run a dataset conversion
gnn-omtf-batch dataset \
  --root-dir /eos/…/Dumper_Ntuples \
  --output-dir /eos/…/Graphs \
  --config configs/dataset_regression.yml \
  --task regression \
  --files-per-job 2 \
  --dry-run

# actually submit training jobs
gnn-omtf-batch train \
  --graphs-dir /eos/…/Graphs/HTo2LongLived/ \
  --output-dir /eos/…/Models/HTo2LongLived \
  --model sage --hidden 32 --epochs 50 \
  --queue workday
"""