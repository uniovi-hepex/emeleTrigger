from pathlib import Path
import torch
from torch_geometric.data import Data

pt_path = Path("tests/debug_smoke/graphs_reg.pt")   # ← adjust if needed
graphs   = torch.load(pt_path, weights_only=False)  # returns a Python list

print(f"{pt_path.name}: {len(graphs)} graphs")
assert isinstance(graphs, list) and all(isinstance(g, Data) for g in graphs)

g = graphs[0]
print("\n--- first graph ---")
print(f"num nodes     : {g.num_nodes}")
print(f"num edges     : {g.num_edges}")
print(f"node features : {g.x.shape}")    # (N, F)  – should match len(stub_vars)
print(f"edge features : {g.edge_attr.shape}")  # (E, 2)
print(f"target shape  : {g.y.shape}")
print(f"has muon vars : {hasattr(g, 'muon_vars')}  -> {getattr(g,'muon_vars', None)}")
print(f"has omtf vars : {hasattr(g, 'omtf_vars')} -> {getattr(g,'omtf_vars', None)}")

# Optional deep sanity-check
for idx, g in enumerate(graphs):
    assert g.x.ndim == 2 and g.x.size(1) == 5      # 5 stub-vars in YAML
    assert g.edge_index.shape[0] == 2
    assert g.edge_attr.size(0) == g.edge_index.size(1)
print("\nDataset looks consistent ✅")


import uproot, awkward as ak, numpy as np, torch
tree = uproot.open("tests/smoke_stub.root")["simOmtfPhase2Digis/OMTFHitsTree"]
arr  = tree.arrays(["muonCharge", "muonPt"], library="ak")
bad  = (arr["muonPt"] == 0)
print("events with muonPt == 0:", np.count_nonzero(bad))