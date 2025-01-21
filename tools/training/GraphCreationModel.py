import torch
from torch_geometric.utils.convert import from_networkx

import numpy as np
import matplotlib.pyplot as plt

import uproot
import networkx as nx
import copy

import mplhep as hep
f = plt.figure()
plt.close()

plt.style.use(hep.style.CMS)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 10  # Ajustar el tamaño del título de los ejes
plt.rcParams["axes.labelsize"] = 10  # Ajustar el tamaño de las etiquetas de los ejes

class GraphCreationModel(): 
    def __init__(self, data_path, graph_save_paths, model_connectivity):
        self.graph_save_paths = graph_save_paths
        self.data_path = data_path
        self.model_connectivity = model_connectivity

        self.NUM_PROCESSORS = 3
        self.NUM_PHI_BINS = 5400
        self.HW_ETA_TO_ETA_FACTOR=0.010875

        self.keep_branches = ['stubEtaG', 'stubPhiG','stubR', 'stubLayer','stubType','muonQPt','muonQOverPt','muonPropEta','muonPropPhi']
        self.muon_vars = ['muonQPt','muonPt','muonQOverPt','muonPropEta','muonPropPhi']
        self.stub_vars = ['stubEtaG', 'stubPhiG','stubR', 'stubLayer','stubType']
        self.dataset = None
        self.pyg_graphs = None
        self.graphs = None
        if model_connectivity == "all":
            print('All layers are connected to each other')
            self.connectivity = -9
        else:
            print(f'Only {model_connectivity}-neighbours are connected')
            self.connectivity = int(model_connectivity)

    def __str__(self):
        return f"GraphCreationModel(data_path={self.data_path}, graph_save_paths={self.graph_save_paths}, model_connectivity={self.model_connectivity})"

    def set_muon_vars(self, muon_vars):
        self.muon_vars = muon_vars

    def set_stub_vars(self, stub_vars):
        self.stub_vars = stub_vars
    
    ##  load files: 
    def load_data(self, datapath=None, debug=False):
        ## load the dataset
        if datapath is None:
           datapath = self.data_path

        print('Loading data from', datapath)

        self.dataset = uproot.open(datapath)
        self.dataset = self.dataset.arrays(library="pd")

        ## add auxiliary information:  
        self.dataset = self.add_auxiliary_info(self.dataset)

        ##  filter out the dataset:
        self.dataset = self.dataset[self.keep_branches]
        
       
    def add_auxiliary_info(self, dataset):
        dataset['stubR'] = dataset.apply(lambda x: self.get_stub_r(x['stubType'], x['stubEta'], x['stubLayer'], x['stubQuality']), axis=1)
        dataset['stubPhi'] = dataset['stubPhi']+dataset['stubPhiB']
        dataset['stubEtaG'] = dataset['stubEta']*self.HW_ETA_TO_ETA_FACTOR
        dataset['stubPhiG'] = dataset.apply(lambda x: self.get_global_phi(x['stubPhi'], x['omtfProcessor']), axis=1)
        dataset['stubPhiB'] = dataset['stubPhi'] 
        dataset['muonPropEta'] = dataset.apply(lambda x: abs(x['muonPropEta']), axis=1)
        dataset['muonQPt'] = dataset['muonCharge']*dataset['muonPt']
        dataset['muonQOverPt'] = dataset['muonCharge']/dataset['muonPt']
        
        return dataset

    ##  calculate coordinate r for each stub
    def get_stub_r(self, stubTypes, stubEtas, stubLayer, stubQuality):
        rs=[]
        for stubType, stubEta, stubLayer,stubQuality in zip(stubTypes, stubEtas, stubLayer, stubQuality):
            r=None
            if stubType == 3: # DTs
                if stubLayer==0:
                    r=431.133
                elif stubLayer==2:
                    r=512.401
                elif stubLayer==4:
                    r=617.946

                # Low-quality stubs are shifted  by 23.5/2 cm
                if (stubQuality == 2 or stubQuality == 0):
                    r = r - 23.5 / 2 
                elif (stubQuality == 3 or stubQuality == 1):
                    r = r + 23.5 / 2

            elif stubType==9: # CSCs
                if stubLayer==6:
                    z=690  #ME1/3
                elif stubLayer==9:
                    z=700   #M1/2
                elif stubLayer==7:
                    z=830
                elif stubLayer==8:
                    z=930
                r= z/np.cos( np.tan(2*np.arctan(np.exp(- stubEta*self.HW_ETA_TO_ETA_FACTOR)))  )
            elif stubType==5: # RPCs, but they will be shut down because they leak poisonous gas
                r=999.
                if (stubLayer == 10):
                    r = 413.675;  #RB1in
                elif (stubLayer == 11):
                    r = 448.675;  #RB1out
                elif (stubLayer == 12):
                    r = 494.975;  #RB2in
                elif (stubLayer == 13):
                    r = 529.975  #RB2out
                elif (stubLayer == 14):
                    r = 602.150  #RB3
                elif (stubLayer == 15):
                    z = 720 #RE1/3
                elif (stubLayer == 16):
                    z = 790 #RE2/3
                elif (stubLayer == 17):
                    z = 970 #RE3/3
                if (r==999.): 
                    r = z/np.cos( np.tan(2*np.arctan(np.exp(- stubEta*self.HW_ETA_TO_ETA_FACTOR)))  )
                
            rs.append(r)

        if len(rs) != len(stubTypes):
            print('Tragic tragedy. R has len', len(rs), ', stubs have len', len(stubTypes))
        return np.array(rs, dtype=object)
       
    def get_global_phi(self, phi, processor):
        p1phiLSB = 2 * np.pi / self.NUM_PHI_BINS

        if isinstance(phi, list):
            return [(processor * 192 + p + 600) % self.NUM_PHI_BINS * p1phiLSB for p in phi]
        else:
            return (processor * 192 + phi + 600) % self.NUM_PHI_BINS * p1phiLSB

    def getEtaKey(self,eta):
        #eta*=self.HW_ETA_TO_ETA_FACTOR
        if abs(eta) < 0.92:
            return 1
        elif abs(eta) < 1.1:
            return 2
        elif abs(eta) < 1.15:
            return 3
        elif abs(eta) < 1.19:
            return 4
        else:
            return 5
    
    def getListOfConnectedLayers(self,eta):
        etaKey=self.getEtaKey(eta)    

        LAYER_ORDER_MAP = {
            1: [10,0,11,12,2,13,14,4,6,15],
            2: [10,0,11,12,2,13,6,15,16,7],
            3: [10,0,11,6,15,16,7,8,17],
            4: [10,0,11,16,7,8,17],
            5: [10,0,9,16,7,8,17],
        }
        return LAYER_ORDER_MAP[etaKey]    
    

    def getEdgesFromLogicLayer(self,logicLayer,withRPC=True):

        LOGIC_LAYERS_CONNECTION_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: [2,4,6,7,8,9],   #MB1: [MB2, MB3, ME1/3, ME2/2]
            2: [4,6,7],         #MB2: [MB3, ME1/3]
            4: [6],             #MB3: [ME1/3]
            6: [7,8],           #ME1/3: [ME2/2]
            7: [8,9],           #ME2/2: [ME3/2]
            8: [9],             #ME3/2: [RE3/3]
            9: [],              #ME1/2: [RE2/3, ME2/2]
        }
        LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC = {
            0:  [2,4,6,7,8,9,10,11,12,13,14,15,16,17], 
            2:  [4,6,7,10,11,12,13,14,15,16],       #MB2: [MB3, ME1/3]
            4:  [6,10,11,12,13,14,15],         #MB3: [ME1/3]
            6:  [7,8,10,11,12,13,14,15,16,17],         #ME1/3: [ME2/2]
            7:  [8,9,10,11,15,16,17],         #ME2/2: [ME3/2]
            8:  [9,10,11,15,16,17],        #ME3/2: [RE3/3]
            9:  [7,10,16,17],         #ME1/2: [RE2/3, ME2/2]
            10: [11,12,13,14,15,16,17],
            11: [12,13,14,15,16,17],
            12: [13,14,15,16],
            13: [14,15,16],
            14: [15],
            15: [16,17],
            16: [17],
            17: []
        }
        LOGIC_LAYERS_LABEL_MAP={
            #(0,2), (2,4), (0,6), (2,6), (4,6), (6,7), (6,8), (0,7), (0,9), (9,7), (7,8)]
            # Put here catalog of names0
            0: 'MB1',
            2: 'MB2',
            4: 'MB3',
            6: 'ME1/3',
            7: 'ME2/2',
            8: 'ME3/2',
            9: 'ME1/2',
            10: 'RB1in',
            11: 'RB1out',
            12: 'RB2in',
            13: 'RB2out',
            14: 'RB3',
            15: 'RE1/3',
            16: 'RE2/3',
            17: 'RE3/3'
        }
        if (withRPC):
            return (LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC[logicLayer])
        else:
            if (logicLayer>=10): return []
            return (LOGIC_LAYERS_CONNECTION_MAP[logicLayer])
    
    def getNodeWeight(self,stubType):
        if stubType == 3:
            return 3
        elif stubType == 9:
            return 2
        else:
            return 1
        
    def getEdgeWeight(self,type1,type2):
        return self.getNodeWeight(type1) + self.getNodeWeight(type2)

    def getDeltaPhi(self,phi1,phi2):
        dphi = phi1 - phi2
        dphi = (dphi + torch.pi) % (2 * torch.pi) - torch.pi
        return dphi

    def getDeltaEta(self,eta1,eta2):
        return eta1-eta2
    
    ### TODO:  generate graphs from dataset, without using networkx...
    def generate_graphs_from_dataset(self):
        self.graphs = []
        for index, row in self.dataset.iterrows():
            for stubId,stubLayer_label in enumerate(row['stubLayer']):
                x = [row[k][stubId] for k in self.stub_vars]
                y = [row[k] for k in self.muon_vars]
                edge_index = []


    def convert_to_graph(self):

        self.graphs = []
        for index, row in self.dataset.iterrows():
            G = nx.Graph()
            for stubId,stubLayer_label in enumerate(row['stubLayer']):
                # Usar stubLayer como etiqueta del nodo
                G.add_node(stubLayer_label, x=[row[k][stubId] for k in self.stub_vars], y=[row[k] for k in self.muon_vars])
                ### Connected to all possible layers
                if self.connectivity < 0:
                    for target_node_layer in self.getEdgesFromLogicLayer(stubLayer_label):
                        # Asegurarse de que el target_node_layer ya existe como nodo en el grafo
                        if target_node_layer == stubLayer_label: continue
                        if target_node_layer in row['stubLayer']:
                            # get index of target node
                            target_node_index = row['stubLayer'].index(target_node_layer)

                            # Añadir arista usando etiquetas de stubLayer
                            G.add_edge(stubLayer_label, target_node_layer, deltaPhi=self.getDeltaPhi(row['stubPhiG'][stubId],row['stubPhiG'][target_node_index]), deltaEta=self.getDeltaEta(row['stubEtaG'][stubId],row['stubEtaG'][target_node_index]), weight=self.getEdgeWeight(row['stubType'][stubId],row['stubType'][target_node_index]))
                else: ##  connected to k-neighbours
                    connected_layers = self.getListOfConnectedLayers(row['stubEtaG'][stubId])
                    source_node_layer_index = connected_layers.index(stubLayer_label)
                    for idx, target_node_layer in enumerate(connected_layers):
                        if target_node_layer == stubLayer_label: continue
                        if target_node_layer not in row['stubLayer']: continue 
                        if abs(source_node_layer_index - idx) > self.connectivity: continue

                        # Asegurarse de que el target_node_layer ya existe como nodo en el grafo
                        target_node_index = row['stubLayer'].index(target_node_layer)
                                            
                        # Añadir arista usando etiquetas de stubLayer
                        G.add_edge(stubLayer_label, target_node_layer, deltaPhi=self.getDeltaPhi(row['stubPhiG'][stubId],row['stubPhiG'][target_node_index]), deltaEta=self.getDeltaEta(row['stubEtaG'][stubId],row['stubEtaG'][target_node_index]), weight=self.getEdgeWeight(row['stubType'][stubId],row['stubType'][target_node_index]))
            
            self.graphs.append(G)
        print('Graphs created and stored')

    def getGraphWithNodes(self,nodes):
        for graph in self.graphs:
            numnodes = graph.number_of_nodes()
            if numnodes==nodes and nx.is_connected(graph):
                return copy.deepcopy(graph)
    
        return gmax

    def draw_graph(self, G, savefig="graph.png"):
        print('Drawing graph into ', savefig)
        pos = nx.spring_layout(G)
        nx.draw(G, pos)
        nx.draw_networkx_labels(G, pos)
        plt.title(f"{G}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig(savefig)
        plt.close()  # Cerrar la gráfica automáticamente


    def draw_example_graphs(self,savefig="graph.png",seed=42):
        print('Drawing example graphs into ', savefig)
        # draw a figure with 6 subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()
        ax = plt.gca()
        ax.margins(0.08)

        for i in range(6):
            example = self.getGraphWithNodes(i+4)
            if example is None:
                continue
            pos = nx.spring_layout(example, seed=seed)
            nx.draw(example, pos, ax=axs[i])
            nx.draw_networkx_labels(example, pos,ax=axs[i])
            axs[i].set_title(f"{example}") 

            axs[i].axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig(savefig)
        plt.close()  # Cerrar la gráfica automáticamente

    def verifyGraphs(self):
        print('Verifying graphs, checking connectivity')
        for G in self.graphs:
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            graph_density = nx.density(G) if num_nodes > 1 else 0  # La densidad solo tiene sentido si hay al menos 2 nodos
            
            
            # Solo verificar la conectividad si el grafo no es nulo
            if num_nodes > 0:
                is_connected = nx.is_connected(G)
                
                # Iterar sobre los nodos para imprimir los valores de x y y
                if not is_connected: 
                    print(f"Graph with {num_nodes} nodes and {num_edges} edges is {'connected' if is_connected else 'not connected'}")
                    for node in G.nodes:
                        # Asumiendo que los atributos x y y están almacenados en cada nodo
                        x = G.nodes[node].get('x', 'No definido')
                        y = G.nodes[node].get('y', 'No definido')
                        print(f"Nodo {node}: x = {x}, y = {y}")
                    self.draw_graph(G,f'unconnected_graph_{num_nodes}n_{num_edges}e.png')
                    return None     
        print('All graphs are connected')

    def saveTorchDataset(self, save_path=None):
        if save_path is None:
            save_path = self.graph_save_paths

        self.pyg_graphs = [from_networkx(g) for g in self.graphs if (g.number_of_nodes() > 0)]
        torch.save(self.pyg_graphs, save_path)

    def loadTorchDataset(self, load_path=None):
        if load_path is None:
            load_path = self.graph_save_paths
        self.pyg_graphs = torch.load(load_path)
        return self.pyg_graphs

    ## draw nodes properties:
    def draw_node_properties(self, property_name, savefig="node_properties.png"):
        
    
        print(f'Drawing node properties {property_name} into ', savefig)
        


def main():
    
    import argparse
    parser = argparse.ArgumentParser(description="Graph Creation model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data file")
    parser.add_argument("--graph_save_paths", type=str, default=None, help="Path to save the graph")
    parser.add_argument("--model_connectivity", type=str, default="all", help="Model connectivity")
    parser.add_argument("--muon_vars", type=str, default="muonQPt", help="Muon variables")
    parser.add_argument("--stub_vars", type=str, default="stubEtaG", help="Stub variables")
    # add parser for validation
    parser.add_argument('--validate', action='store_true', help='Validate the model')
                        
    args = parser.parse_args()

    graphs = GraphCreationModel(args.data_path, args.graph_save_paths, args.model_connectivity)
    graphs.set_muon_vars([args.muon_vars])
    graphs.load_data()
    graphs.convert_to_graph()
    if args.validate:
        graphs.draw_example_graphs("graph_example_ALLlayers.png")
        graphs.verifyGraphs()
    graphs.saveTorchDataset()

    '''
    import sys, os
    FOLDER = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v240725/"
    GRAPH_FOLDER = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725/"

    # do a check if the folder exists and ls the files
    # if not, exit
    if not os.path.exists(FOLDER):
        print("Folder does not exist")
        sys.exit()
    list_of_files = os.listdir(FOLDER)

    print("List of files in the folder: ", list_of_files)


    i = 1
    for file in list_of_files:
        print("Processing file: ", file)
        graphs = GraphCreationModel("%s/%s:simOmtfPhase2Digis/OMTFHitsTree" %(FOLDER,file), "vix_graph_ALL_layers_15Oct_onlypt_%03d.pkl" %(i), "all")
        graphs.set_muon_vars(['muonQPt'])
        graphs.load_data()
        graphs.convert_to_graph()
        #graphs.draw_example_graphs("graph_example_ALLlayers.png")
        #graphs.verifyGraphs()
        graphs.saveTorchDataset()

        graphs_3layer = GraphCreationModel("%s/%s:simOmtfPhase2Digis/OMTFHitsTree" %(FOLDER,file), "vix_graph_3_layers_15Oct_onlypt_%03d.pkl" %(i), "3")
        graphs_3layer.set_muon_vars(['muonQPt'])
        graphs_3layer.load_data()
        graphs_3layer.convert_to_graph()
        #graphs_3layer.draw_example_graphs("graph_example_3_layers.png")
        #graphs_3layer.verifyGraphs()
        graphs_3layer.saveTorchDataset()
        i += 1
    '''
    
if __name__ == "__main__":
    main()
   

