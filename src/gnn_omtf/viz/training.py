# src/gnn_omtf/viz/training.py
"""Utilities to plot train/val curves after a run."""

from pathlib import Path
import matplotlib.pyplot as plt

def plot_losses(train_hist, val_hist, out: Path):
    plt.figure()
    plt.plot(train_hist, label="train")
    plt.plot(val_hist,   label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()
    print(f"Saved training curves to {out}")