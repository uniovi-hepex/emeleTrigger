# src/gnn_omtf/viz/graph_features.py
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader

__all__ = ["plot_graph_features"]


def _pad(lst, length, prefix):
    """Return *lst* extended to *length* with generic names."""
    return lst + [f"{prefix}{i}" for i in range(len(lst), length)]


def plot_graph_features(loader: DataLoader,
                        out_dir: Path,
                        *,
                        task: str = "regression"):
    """Draw histograms for one batch of a PyG DataLoader (CI-safe)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = next(iter(loader))                       # single batch
    x         = batch.x.cpu().numpy()               # (N_nodes, F_node)
    edge_attr = batch.edge_attr.cpu().numpy()       # (N_edges, F_edge)
    target    = batch.y.cpu().numpy()

    n_node = x.shape[1]
    n_edge = edge_attr.shape[1]
    n_tot  = n_node + n_edge + 1                    # +1 for target

    # ------------------------------------------------------------------ #
    #  build a label for every plotted feature
    # ------------------------------------------------------------------ #
    node_names = ["η", "φ", "R", "Δφ_idx", "Δη_idx"]          # canonical 5
    edge_names = ["Δφ", "Δη"]                                # canonical 2
    target_name = ["Q/pt" if task == "regression" else "matched"]

    node_names = _pad(node_names, n_node, "x")
    edge_names = _pad(edge_names, n_edge, "e")
    feature_names = node_names + edge_names + target_name

    # ------------------------------------------------------------------ #
    #  figure layout
    # ------------------------------------------------------------------ #
    n_cols = 3
    n_rows = ceil(n_tot / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(5 * n_cols, 4 * n_rows),
                            squeeze=False)
    axs = axs.ravel()

    # node-feature histos
    for i in range(n_node):
        axs[i].hist(x[:, i], bins=30, alpha=0.8)
        axs[i].set_title(feature_names[i])

    # edge-feature histos
    for i in range(n_edge):
        idx = n_node + i
        axs[idx].hist(edge_attr[:, i], bins=30, alpha=0.8)
        axs[idx].set_title(feature_names[idx])

    # target
    axs[n_node + n_edge].hist(target, bins=30, alpha=0.8)
    axs[n_node + n_edge].set_title(feature_names[-1])

    # remove unused panels (grid might be larger than needed)
    for j in range(n_tot, len(axs)):
        fig.delaxes(axs[j])

    for ax in axs[:n_tot]:
        ax.set_ylabel("freq")

    fig.tight_layout()
    png = out_dir / f"input_features_{task}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"✓ saved feature histograms → {png}")
