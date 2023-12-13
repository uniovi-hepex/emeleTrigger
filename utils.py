import os
import uproot

def get_test_data(library=None):
    if not os.path.isfile('data/omtfAnalysis2.root'):
        os.system('wget http://www.hep.uniovi.es/vischia/omtfsweetlove/omtfAnalysis2.root -P data/')
        print('File downloaded into data/')
    else:
        print('File already exists in data/')
        
    branches = uproot.open('data/omtfAnalysis2.root:simOmtfPhase2Digis/OMTFHitsTree')
    print(branches.show())
    branches=branches.arrays(library=library) if library else branches.arrays()
    return branches

def visualize_graph(G, color):
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=43), with_labels=False,
                     node_color=color, cmap='Set2')
    plt.show()

