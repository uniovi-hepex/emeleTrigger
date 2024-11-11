# scripts/inspect_graphs.py

import torch
from torch_geometric.data import Data

def main():
    dataset = torch.load('../data/processed/graphs.pt')
    print(f'Total graphs: {len(dataset)}')
    sample_graph = dataset[0]
    print(sample_graph)
    # Visualize a sample graph
    from torch_geometric.utils import to_networkx
    import matplotlib.pyplot as plt
    import networkx as nx

    G = to_networkx(sample_graph, to_undirected=True)
    nx.draw(G, with_labels=True)
    plt.show()

if __name__ == "__main__":
    main()
