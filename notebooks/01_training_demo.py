# %%
# %% -----------------------------
# # gnn-omtf · Quick-start demo
# Build a ROOT → PyG dataset, train a tiny GAT, and plot the results.
# -----------------------------------------

# %% ------------------ imports ------------
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from gnn_omtf.data import OMTFDataset
from gnn_omtf.data.transforms import NormalizeNodeEdgesAndDropTwoFeatures
from gnn_omtf.models import BaseGNN
from gnn_omtf.metrics import regression_metrics
from gnn_omtf.viz import plot_graph_features, plot_regression

# %matplotlib inline  (if running in JupyterLab)

# %% ---------- config & paths -------------
ROOTDIR = Path("/eos/cms/.../Dumper_Ntuples_v240725")
if not ROOTDIR.exists():
    ROOTDIR = Path("../../data/Dumper_Ntuples_v240725")

OUT   = Path("runs/demo")
OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 512
HIDDEN     = 32
EPOCHS     = 20
LR         = 5e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% ------------- load dataset ------------
ds = OMTFDataset(
    root_dir=ROOTDIR,
    muon_vars=["muonQPt", "muonPt", "muonQOverPt"],
    stub_vars=["stubEtaG", "stubPhiG", "stubR", "stubLayer", "stubType"],
    max_files=1,
    max_events=10_000,
    pre_transform=NormalizeNodeEdgesAndDropTwoFeatures,
)

loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
plot_graph_features(loader, OUT / "plots/features", task="regression")

# %% -------- split & data-loaders ----------
n = len(ds)
n_train = int(0.7 * n) // BATCH_SIZE * BATCH_SIZE  # full batches
train_ds = ds[:n_train]
test_ds  = ds[n_train:]

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# %% ------------- build model -------------
model = BaseGNN.get(
    "gat",
    num_node_features=ds[0].x.size(1),
    hidden_dim=HIDDEN,
).to(DEVICE)

optim   = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# %% ------------ train loop ---------------
def train_epoch(model, loader):
    model.train(); tot = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optim.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y.view_as(pred))
        loss.backward(); optim.step()
        tot += loss.item() * batch.num_graphs
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval(); tot = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        pred  = model(batch)
        tot  += loss_fn(pred, batch.y.view_as(pred)).item() * batch.num_graphs
    return tot / len(loader.dataset)

train_hist, val_hist = [], []
for epoch in range(1, EPOCHS + 1):
    tl = train_epoch(model, train_loader)
    vl = eval_epoch(model,  test_loader)
    train_hist.append(tl); val_hist.append(vl)
    print(f"Epoch {epoch:02d} | train {tl:.4f} | val {vl:.4f}")

torch.save(model.state_dict(), OUT / "best.ckpt")

# %% -------- evaluate & plot --------------
@torch.no_grad()
def gather(model, loader):
    model.eval(); p, t = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        p.append(model(batch).cpu()); t.append(batch.y.cpu())
    return torch.cat(p), torch.cat(t)

pred, truth = gather(model, test_loader)
plot_regression(pred.numpy(), truth.numpy(), OUT / "plots/results")
print(regression_metrics(pred, truth))



