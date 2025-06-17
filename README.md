# gnn-omtf

_Graph-Neural-Network toolkit for CMS **O**verlap **M**uon **T**rack**F**inder (OMTF)_

> From ROOT → graphs → training & optimisation → pruning / quantisation  
> → FPGA/HLS deployment — all in one modular repository.

---

## Table of contents
1. Quick‑start
2. Source‑tree layout
3. Core pipeline
4. Command‑line reference
5. Development & CI
6. Citing / licence

----------------------------------------------------------------------
## 1  Quick‑start

```bash
# clone and install (editable)
git clone https://gitlab.cern.ch/you/gnn-omtf.git
cd gnn-omtf
pip install -e .[dev,precision,fpga]

# convert ROOT → graphs
gnn-omtf-data convert \
    --config  configs/dataset_regression.yml \
    --root-dir /eos/.../*.root \
    --output   graphs_reg.pt

# train a baseline model
gnn-omtf-train run \
    --graphs  graphs_reg.pt \
    --model   sage \
    --hidden-dim 64 \
    --epochs  50 \
    --out-dir runs/baseline

# hyper‑parameter search
gnn-omtf-opt hpo \
    --graphs graphs_reg.pt --trials 40 \
    --out    arch/best_hparams.json

# prune + INT8 quantise + evaluate
gnn-omtf-compress prune      --ckpt runs/baseline/best.ckpt --amount 0.7
gnn-omtf-compress dyn-int8   --ckpt model_pruned.pt
gnn-omtf-compress eval \
    --baseline   runs/baseline/best.ckpt \
    --compressed model_int8.pt \
    --dataset    graphs_reg.pt

# export to ONNX 
gnn-omtf-deploy onnx --ckpt model_int8.pt --out model.onnx --feature-dim 64
```

----------------------------------------------------------------------
## 2  Source‑tree layout

```
src/gnn_omtf
├── data/       : ROOT → graphs, transforms, CLI
├── models/     : GCN, GAT, SAGE, MPL, BaseGNN registry
├── training/   : Trainer, callbacks, CLI
├── optim/      : Optuna HPO + PC‑DARTS NAS
├── compress/   : pruning, quantisation, precision sweep
├── deploy/     : ONNX + HLS4ML export
├── batch/      : HTCondor wrapper + Jinja2 templates
├── metrics/    : regression / classification KPIs
└── viz/        : Matplotlib helpers
```

----------------------------------------------------------------------
## 3  Core pipeline

```
ROOT  ─► gnn-omtf-data          ─► Graph .pt
graph ─► gnn-omtf-train / optim ─► Baseline ckpt
ckpt  ─► compress               ─► Pruned / INT8 ckpt
both  ─► compress eval          ─► metrics report
ckpt  ─► deploy onnx → hls4ml   ─► Vivado project
```

----------------------------------------------------------------------
## 4  Command‑line reference

| Stage        | Entry‑point                                                                  | Highlights                                       |
|--------------|------------------------------------------------------------------------------|--------------------------------------------------|
| Data         | `gnn-omtf-data convert`                                                      | YAML‑driven ROOT ingest, graph stats             |
| Training     | `gnn-omtf-train run`                                                         | any BaseGNN or `--arch-yaml` from NAS            |
| Optimisation | `gnn-omtf-opt hpo / nas`                                                     | Bayesian or differentiable arch‑search           |
| Compression  | `gnn-omtf-compress prune / dyn-int8 / qat / sweep-precision / eval`          | end‑to‑end size‑vs‑accuracy dashboard            |
| Deployment   | `gnn-omtf-deploy onnx / hls`                                                 | one‑liner to Vivado HLS project                  |
| Batch        | `gnn-omtf-batch condor`                                                      | templated grid sweeps (CERN HTCondor)            |

Run any command with `--help` for full options.

----------------------------------------------------------------------
## 5  Development & CI

* **Tests**   `pytest -q` (coverage in CI)  
* **Formatting / lint**   `black`, `ruff`, `mypy` via *pre‑commit*  
* CI template under `.github/workflows/`

----------------------------------------------------------------------
## 6  Citing / licence


Licensed under the **Apache 2.0** licence (see `LICENSE`).

