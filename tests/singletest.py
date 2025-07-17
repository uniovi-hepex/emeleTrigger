from collections import Counter
import torch
from torch_geometric.loader import DataLoader
from gnn_omtf.data import OMTFDataset  # adjust import to your project

import torch
import torch
print("Torch build:", torch.__version__, "CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))


# 1) load your graphs (same as you do in training/viz)
ds = OMTFDataset(dataset=torch.load("graphs_evan.pt", weights_only=False))

# 2) extract all labels
#    assumes `data.y` is an integer (0…C‑1) or a one‑element tensor for binary
labels = []
for data in ds:
    y = data.y
    # if y is a 1‑element tensor:
    if torch.is_tensor(y):
        y = y.item()
    labels.append(int(y))

# 3) count and percentage
counts = Counter(labels)
total = len(labels)
print(f"Total graphs: {total}\n")
for cls, cnt in sorted(counts.items()):
    pct = 100 * cnt / total
    print(f"Class {cls:>2}: {cnt:>5} samples   ({pct:5.2f}%)")
