from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import torch
import torch.quantization as tq
from torch_geometric.loader import DataLoader


def dynamic_int8(model: torch.nn.Module) -> torch.nn.Module:
    return tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def prepare_qat(model: torch.nn.Module) -> torch.nn.Module:
    model.train()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(model, inplace=True)
    return model


def fine_tune_qat(model, loader: DataLoader, epochs: int = 3, lr: float = 1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch.y.view_as(out).float())
            loss.backward(); opt.step()
    return tq.convert(model.eval(), inplace=False)


def bench_throughput(model, loader, n_batches: int = 10) -> float:
    """
    Simple wall-clock throughput in ms / example (CPU).
    """
    model.eval(); model.cpu()
    t0 = time.time()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i == n_batches: break
            model(batch)
    dt = (time.time() - t0) * 1000
    total_ex = min(n_batches, len(loader)) * loader.batch_size
    return dt / total_ex
