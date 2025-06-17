from __future__ import annotations
import logging, yaml, math, copy
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool

from gnn_omtf.data import OMTFDataset

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 1.  Ops in the search space
# ------------------------------------------------------------------ #
class Identity(nn.Module):
    def forward(self, x, edge_index): return x


class Zero(nn.Module):
    def forward(self, x, edge_index):
        return torch.zeros_like(x)


OPS = {
    "gcn":       lambda C: GCNConv(C, C),
    "sage":      lambda C: SAGEConv(C, C),
    "gat":       lambda C: GATConv(C, C, heads=1, concat=False),
    "identity":  lambda C: Identity(),
    "zero":      lambda C: Zero(),
}


# ------------------------------------------------------------------ #
# 2.  MixedOp (softmax over candidates)
# ------------------------------------------------------------------ #
class MixedOp(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self._ops = nn.ModuleDict({k: OPS[k](C) for k in OPS})

    def forward(self, x, edge_index, weights):
        out = 0
        for w, op in zip(weights, self._ops.values()):
            if w.item() != 0:
                out = out + w * op(x, edge_index)
        return out


# ------------------------------------------------------------------ #
# 3.  DARTS cell
# ------------------------------------------------------------------ #
class DartsCell(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.N = 4
        self.edges: nn.ModuleDict = nn.ModuleDict()
        idx = 0
        for i in range(self.N):
            for j in range(i + 2):          # include inputs 0 & 1
                self.edges[f"edge_{idx}"] = MixedOp(C)
                idx += 1

    def forward(self, x, edge_index, alphas):
        states: List[torch.Tensor] = [x, x]  # two input nodes (prev layers)
        offset = 0
        for i in range(self.N):
            s = 0
            for j in range(i + 2):
                weights = F.softmax(alphas[offset + j], dim=-1)
                s = s + self.edges[f"edge_{offset+j}"](states[j], edge_index, weights)
            offset += i + 2
            states.append(s)
        return torch.mean(torch.stack(states[-self.N:], dim=0), 0)  # concat→mean


# ------------------------------------------------------------------ #
# 4.  Full network (2 cells stacked)
# ------------------------------------------------------------------ #
class Network(nn.Module):
    def __init__(self, C_in: int, C_hidden: int, num_classes: int = 1, layers: int = 2):
        super().__init__()
        self.pre = nn.Linear(C_in, C_hidden)
        self.cells = nn.ModuleList([DartsCell(C_hidden) for _ in range(layers)])
        self.classifier = nn.Linear(C_hidden, num_classes)

        # arch parameters – one α vector per edge
        self._arch_params = nn.ParameterList()
        num_edges_cell = sum(i + 2 for i in range(4))
        for _ in range(layers * num_edges_cell):
            self._arch_params.append(nn.Parameter(1e-3 * torch.randn(len(OPS))))

    @property
    def arch_parameters(self): return self._arch_params

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.pre(x))
        offset = 0
        for cell in self.cells:
            edges_in_cell = len(cell.edges)
            x = cell(x, edge_index, self._arch_params[offset: offset + edges_in_cell])
            offset += edges_in_cell
        graph_emb = global_mean_pool(x, batch)
        return self.classifier(graph_emb).squeeze(1)


# ------------------------------------------------------------------ #
# 5.  Search routine
# ------------------------------------------------------------------ #
def run_search(graphs_pt: str | Path,
               hidden_dim: int = 32,
               batch_size: int = 256,
               search_epochs: int = 30,
               lr_w: float = 5e-4,
               lr_alpha: float = 3e-3,
               out_yaml: str | Path = "best_arch.yaml") -> Dict:
    ds = OMTFDataset.load_dataset(graphs_pt)
    n = len(ds); split = int(0.5 * n)
    train_loader = DataLoader(ds[:split], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ds[split:], batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C_in   = ds[0].x.size(1)
    net = Network(C_in, hidden_dim).to(device)

    opt_w     = Adam(net.parameters(),      lr=lr_w,     weight_decay=3e-4)
    opt_alpha = Adam(net.arch_parameters,  lr=lr_alpha, weight_decay=1e-3)

    for epoch in range(search_epochs):
        # >>> train weights
        net.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_w.zero_grad()
            loss = F.mse_loss(net(batch), batch.y.float())
            loss.backward(); opt_w.step()

        # >>> update architecture on val split
        net.eval()
        for batch in val_loader:
            batch = batch.to(device)
            opt_alpha.zero_grad()
            loss = F.mse_loss(net(batch), batch.y.float())
            loss.backward(); opt_alpha.step()
            break   # one mini-b per epoch is enough for α-update

        if epoch % 5 == 0:
            log.info("epoch %02d  train-loss %.4f", epoch, loss.item())

    # ---------------------------------------------------------------- #
    # discretise: pick highest-α op per edge
    genotype = []
    for alpha in net.arch_parameters:
        op_idx = int(torch.argmax(alpha).item())
        genotype.append(list(OPS)[op_idx])
    yaml.safe_dump(genotype, open(out_yaml, "w"))
    log.info("saved genotype → %s", out_yaml)
    return {"genotype": genotype, "hidden_dim": hidden_dim}
