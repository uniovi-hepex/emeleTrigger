import os
import uproot
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

from torch_geometric.utils.convert import from_networkx, to_networkx

NUM_PROCESSORS = 6
NUM_PHI_BINS = 5400
HW_ETA_TO_ETA_FACTOR=0.010875
#HwEtaToEta conversion: 0.010875
    #HwPhiToGlbPhi conversion:  hwPhi* 2. * M_PI / 576
LOGIC_LAYERS_CONNECTION_MAP={
    #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
    # Put here catalog of names0
    0: [2,6,7],
    2: [4,6],
    4: [6],
    6: [7,8],
    7: [8],
    8: [],
    9: [7],
}


def foldPhi (phi):
    if (phi > NUM_PHI_BINS / 2):
        return (phi - NUM_PHI_BINS)
    elif (phi < -NUM_PHI_BINS / 2):
        return (phi + NUM_PHI_BINS)
    return phi

def phiRel(phi, processor):
    return phi - foldPhi(NUM_PHI_BINS / NUM_PROCESSORS * (processor) + NUM_PHI_BINS / 24)


def _get_stub_r(stubTypes, stubDetIds, stubEtas, stubLogicLayers):
    rs=[]
    for stubType, stubDetId, stubEta, stubLogicLayer in zip(stubTypes, stubDetIds, stubEtas, stubLogicLayers):
        r=None
        if stubType == 3: # DTs
            if stubLogicLayer==0:
                r= 431.133
            elif stubLogicLayer==2:
                r=512.401
            elif stubLogicLayer==4:
                r=617.946
        elif stubType==9: # CSCs
            if stubLogicLayer==6:
                z=7
            elif stubLogicLayer==9:
                z=6.8
            elif stubLogicLayer==7:
                z=8.2
            elif stubLogicLayer==8:
                z=9.3
            r= z/np.cos( np.tan(2*np.arctan(np.exp(- stubEta*HW_ETA_TO_ETA_FACTOR)))  )
        elif stubType==5: # RPCs, but they will be shut down because they leak poisonous gas
            r=999.
        rs.append(r)

    if len(rs) != len(stubTypes):
        print('Tragic tragedy. R has len', len(rs), ', stubs have len', len(stubTypes))
    return np.array(rs, dtype=object)
get_stub_r = np.vectorize(_get_stub_r)


def get_test_data(library=None, mode=None):

    if mode=="old" and not os.path.isfile('data/omtfAnalysis2.root'):
        os.system('wget http://www.hep.uniovi.es/vischia/omtfsweetlove/omtfAnalysis2.root -P data/')
        
        print('File omtfAnalysis2.root downloaded into data/')
    else:
        print('File omtfAnalysis2.root already exists in data/')

    if not os.path.isfile('data/Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root'):
        os.system('wget http://www.hep.uniovi.es/vischia/omtfsweetlove/Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root -P data/')
        print('File Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root downloaded into data/')
    else:
        print('File Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root already exists in data/')

    branches=None
    if mode=="old":
        branches = uproot.open('data/omtfAnalysis2.root:simOmtfPhase2Digis/OMTFHitsTree')
    else:
        branches = uproot.open('data/Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root:simOmtfPhase2Digis/OMTFHitsTree')
        #branches = uproot.open('data/SingleMu_GT131X_Extrapolation_GhostBusterTest_FlatPt0To1000Dxy3m_NonDegraded_Stub_v2.root:simOmtfDigis/OMTFHitsTree')
        #branches = uproot.open('data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Ctau5m_Stub.root:simOmtfDigis/OMTFHitsTree')
        
        
    print(branches.show())
    branches=branches.arrays(library=library) if library else branches.arrays()
    print('Downsampling...')
    branches=branches.sample(frac=0.5)
    print('Downsampling done')
    # Calculate deltaphis between layers, adding it to a new column
    # this will be our proxy to the magnetic field
    branches['stubDPhi'] = branches['stubPhi'].apply(lambda x: np.diff(x))

    return branches

def visualize_graph(G, color):
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=43), with_labels=False,
                     node_color=color, cmap='Set2')
    plt.show()


def convert_to_point_cloud(arr):

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubEta'], arr['stubLogicLayer'])
    for index, row in arr.iterrows():
        if not ( len(arr['stubR'][index])==len(arr['stubEta'][index]) and len(arr['stubEta'][index])==len(arr['stubPhi'][index])):
            print('HUGE PROBLEM, sizes not match: R,eta,phi ', len(arr['stubR'][index]), len(arr['stubEta'][index]), len(arr['stubPhi'][index]))
    arr=arr.rename(columns={"stubR": "x", "stubEta": "y", "stubPhi": "z"}, errors="raise")

    keep=['x', 'y', 'z', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']

    sky=[]
    for index, row in arr.iterrows():
        # Here now I need to enrich this with the stublogiclayer etc, for the edges
        pc = {} #{'x': row['stubR'], 'y': row['stubEta'], 'z': row['stubPhi'], 'stubLogicLayer': row['stubLogicLayer']} # down the line maybe convert to actual cartesian coordinates
        for label in keep:
            pc[label] = row[label]
        # Create a PyntCloud object from the DataFrame
        if len(pc['x'])==0:
            continue
        cloud = PyntCloud(pd.DataFrame(pc))
        sky.append(cloud)

    return sky


def convert_to_point_cloud_old(arr):

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubLogicLayer'])
    
    pc = {'x': arr['stubR'], 'y': arr['stubEta'], 'z': arr['stubPhi']} # down the line maybe convert to actual cartesian coordinates
    print('THECLOUD', pc)
    # Create a PyntCloud object from the DataFrame
    cloud = PyntCloud(pd.DataFrame(pc))
    return cloud


def convert_to_point_cloud_vold(branches):

    #point_cloud_data = branches.copy()
    #point_cloud_data = copy.deepcopy(branches[['muonPt', 'muonEta', 'muonPhi']])
    point_cloud_data = {'x': branches['muonPt'], 'y': branches['muonEta'], 'z': branches['muonPhi']} # down the line maybe convert to actual cartesian coordinates
    # Create a PyntCloud object from the DataFrame
    cloud = PyntCloud(pd.DataFrame(point_cloud_data))

    return cloud

def getEdgesFromLogicLayer(logicLayer):
    return (LOGIC_LAYERS_CONNECTION_MAP[logicLayer] if logicLayer<10 else [])


def addEdges(sky):
    graphs = []
    for index, cloud in enumerate(sky):
        graph = nx.DiGraph()
        nodes = []
        keep=['stubR', 'stubEta', 'stubPhi', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer']
        edges=[]
        for index, row in cloud.iterrows(): # build edges based on stubLayer
            #nodes.append((index, {k: row[k] for k in keep}))
            nodes.append((index, { 'x': [row[k] for k in keep], 'y' : np.float64(row['muonPt']) }))
            dests=getEdgesFromLogicLayer(row['stubLogicLayer'])
            for queriedindex, row in cloud.iterrows():
                if queriedindex in dests:
                    edges.append((index,queriedindex))
                # Rename back
        #cp=cloud.points.rename(columns={"x": "stubR", "y": "stubEta", "z": "stubPhi", "muonPt": "y"}, errors="raise")
        graph.add_nodes_from(nodes) # Must be the transpose, it reads by colum instead of by row
        graph.add_edges_from(edges)
        graphs.append(graph)
        #for index, cloud in enumerate(sky):
        #    graph = nx.DiGraph()
        #    edges=[]
        #    for index, row in cloud.points.iterrows(): # build edges based on stubLayer
        #        dests=getEdgesFromLogicLayer(row['stubLogicLayer'])
        #        for queriedindex, row in cloud.points.iterrows():
        #            if queriedindex in dests:
        #                edges.append((index,queriedindex))
        #    # Rename back
        #    cp=cloud.points.rename(columns={"x": "stubR", "y": "stubEta", "z": "stubPhi", "muonPt": "y"}, errors="raise")
        #    graph.add_nodes_from(cp.T) # Must be the transpose, it reads by colum instead of by row
        #    graph.add_edges_from(edges)
        #    graphs.append(graph)
    return graphs


def convert_to_graphs(branches, viz=False):
    arr = convert_to_point_cloud(branches)

    arr['stubR'] = get_stub_r(arr['stubType'], arr['stubDetId'], arr['stubEta'], arr['stubLogicLayer'])
    
    keep=['stubR', 'stubEta', 'stubPhi', 'stubProc', 'stubPhiB', 'stubEtaSigma', 'stubQuality', 'stubBx', 'stubDetId', 'stubType', 'stubTiming', 'stubLogicLayer', 'muonPt']

    sky=[]
    for index, row in arr.iterrows():
        # Here now I need to enrich this with the stublogiclayer etc, for the edges
        pc = {} #{'x': row['stubR'], 'y': row['stubEta'], 'z': row['stubPhi'], 'stubLogicLayer': row['stubLogicLayer']} # down the line maybe convert to actual cartesian coordinates
        for label in keep:
            pc[label] = row[label]
        # Create a PyntCloud object from the DataFrame
        if len(pc['stubR'])==0:
            continue
        sky.append(pd.DataFrame(pc))

    graphs=addEdges(sky)
    if viz:
        gmax=None
        nmax=0
        for graph in graphs:
            numnodes = graph.number_of_nodes()
            if numnodes>nmax:
                gmax=copy.deepcopy(graph)
                nmax=numnodes
        nx.draw(gmax, with_labels=True)
        print(gmax.nodes.data(True))
        g=from_networkx(gmax)
        print(g)
        print()
        print(g.y)
        print(g.edge_index)
        plt.show()
        pg=None
        nmax=0
        for g in pg_graphs:
            numnodes=g.num_nodes
            if numnodes>nmax:
                pg=copy.deepcopy(g)
                nmax=numnodes

        print('networkx graph')
        print(pg)
        print(pg.x)
        print(pg.y)
        print(pg.edge_index)
        print('-----------------')
        # Gather some statistics about the graph.
        print(f'Number of nodes: {pg.num_nodes}') #Number of nodes in the graph
        print(f'Number of edges: {pg.num_edges}') #Number of edges in the graph
        print(f'Average node degree: {pg.num_edges / pg.num_nodes:.2f}') # Average number of nodes in the graph
        print(f'Contains isolated nodes: {pg.has_isolated_nodes()}') #Does the graph contains nodes that are not connected
        print(f'Contains self-loops: {pg.has_self_loops()}') #Does the graph contains nodes that are linked to themselves
        print(f'Is undirected: {pg.is_undirected()}') #Is the graph an undirected graph
        nx.draw(to_networkx(pg), with_labels=True)


    # Convert each NetworkX graph to a PyTorch Geometric Data object
    pg_graphs=[from_networkx(g) for g in graphs]
    return pg_graphs



# Define a function to pad a list with zeros
def pad_list(lst, max_list_length):
    return lst + [0] * (max_list_length - len(lst))

def pad_branches(df):
    max_list_length = df.applymap(lambda x: len(x) if isinstance(x, list) else 0).max().max()
    max_int_list_length = df.applymap(lambda x: np.max([len(y) if isinstance(y, list) else 0 for y in x].append(0)) if isinstance(x, list) else 0).max().max() # lolazo
    # Apply the padding function to all columns containing lists
    for col in df.columns:
        df[col] = df[col].apply(lambda x: pad_list(x, max_list_length) if isinstance(x, list) else x)
        df[col] = df[col].apply(lambda x: [  pad_list(y, max_int_list_length) if isinstance(y, list) else y for y in x] if isinstance(x, list) else x)
        #for stuff in df[col]:
        #        print("Stuff: ", len(stuff) if isinstance(stuff, list) else type(stuff))
    return df

def generate_hdf5_dataset_with_padding(branches, hdf5_filename, save=False):

    # Padding   
    #padded_branches=np.asarray(pad_branches(branches))
    # Padding and flattening
    #padded_branches=ak.fill_none(ak.pad_none(branches, 2, clip=True), 999)
    padded_branches=ak.to_dataframe(branches)

    #padded_branches=branches.pad(longest, clip=True).fillna(0).regular()
    #for name in padded_branches:
    #    print('name', name)
    #    print('counts', padded_branches[name])
    #    #longest = padded_branches[name].counts.max()
    #    padded_branches[name] = padded_branches[name].pad(longest).fillna(0).regular()
    
    #padded_branches = branches.pad(branches.counts.max()).fillna(0)
    
    point_clouds = convert_to_point_cloud(branches)
    point_cloud_array = point_clouds.points.values

    #print("SHAPE", padded_branches.shape)
    #print("Columns shape")
    #for i in range(39):
    #    if isinstance(padded_branches[0,i], list):
    #        print('List of ', type(padded_branches[0,i][0]))
    #    else:
    #        print(type(padded_branches[0,i]))
                    
    print('Padded branches type', type(padded_branches))

    with h5py.File(hdf5_filename, 'w') as f:
        #f.create_dataset('images', data= np.asarray(padded_branches.values, dtype=np.float64))
        f.create_dataset('images', data=padded_branches, dtype=np.float64)
        f.create_dataset('point_clouds', data=point_cloud_array)


def get_training_dataset(hdf5_path, BATCH_SIZE=128):

    with h5py.File(hdf5_path, 'r') as f:
        # Assume 'your_dataset' is the name of the dataset in the HDF5 file
        x_train = f['/point_clouds']
        y_train = f['/images']
        # Convert the h5py dataset to a TensorFlow dataset
        x_train = tf.data.Dataset.from_tensor_slices(x_train)
        y_train = tf.data.Dataset.from_tensor_slices(y_train)
        # Zip them to create pairs
        training_dataset = tf.data.Dataset.zip((x_train,y_train))
        

        # Shuffle, prepare batches, etc ...
        training_dataset = training_dataset.shuffle(100, reshuffle_each_iteration=True)
        training_dataset = training_dataset.batch(BATCH_SIZE)
        training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.prefetch(-1)
        
        return training_dataset

