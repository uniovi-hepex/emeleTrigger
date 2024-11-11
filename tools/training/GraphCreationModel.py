# data_processing.py

import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import numpy as np
import pandas as pd
import uproot

from joblib import Parallel, delayed
from tqdm import tqdm

import matplotlib.pyplot as plt
#import mplhep as hep
import networkx as nx
import gc
import logging

# Configure Matplotlib for consistent styling
#plt.style.use(hep.style.CMS)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 10


class GraphCreationModel:
    def __init__(self, raw_data_dir, graph_save_path, model_connectivity):
        """
        Initializes the GraphCreationModel with directories and connectivity settings.

        Parameters:
        - raw_data_dir (str): Directory containing raw ROOT files.
        - graph_save_path (str): Path to save the processed graph dataset.
        - model_connectivity (str or int): 'all' for full connectivity or integer for k-neighbors.
        """
        self.graph_save_path = graph_save_path
        self.model_connectivity = model_connectivity
        self.raw_data_dir = raw_data_dir
        self.NUM_PROCESSORS = 3
        self.NUM_PHI_BINS = 5400
        self.HW_ETA_TO_ETA_FACTOR = 0.010875

        self.keep_branches = [
            'stubEtaG', 'stubPhiG', 'stubR', 'stubLayer',
            'stubType', 'muonQPt', 'muonQOverPt', 'muonPropEta', 'muonPropPhi'
        ]
        self.muon_vars = ['muonQPt', 'muonPt', 'muonQOverPt', 'muonPropEta', 'muonPropPhi']
        self.stub_vars = ['stubEtaG', 'stubPhiG', 'stubR', 'stubLayer', 'stubType']
        self.dataset = []       # Initialize as empty list
        self.pyg_graphs = []    # Initialize as empty list

        if model_connectivity == "all":
            logging.info('All layers are connected to each other')
            self.connectivity = -9
        else:
            logging.info(f'Only {model_connectivity}-neighbours are connected')
            self.connectivity = int(model_connectivity)

    def __str__(self):
        return (f"GraphCreationModel(raw_data_dir={self.raw_data_dir}, "
                f"graph_save_path={self.graph_save_path}, "
                f"model_connectivity={self.model_connectivity})")

    def set_muon_vars(self, muon_vars):
        """
        Sets the muon variables for the dataset.

        Parameters:
        - muon_vars (list of str): List of muon variable names.
        """
        self.muon_vars = muon_vars
        logging.info(f"Muon variables set to: {self.muon_vars}")

    def set_stub_vars(self, stub_vars):
        """
        Sets the stub variables for the dataset.

        Parameters:
        - stub_vars (list of str): List of stub variable names.
        """
        self.stub_vars = stub_vars
        logging.info(f"Stub variables set to: {self.stub_vars}")

    def load_data(self, debug=False):
        """
        Loads and preprocesses ROOT files from the raw_data_dir.

        Parameters:
        - debug (bool): If True, loads only a subset of data for debugging purposes.
        """
        logging.info(f'Loading data from {self.raw_data_dir}')
        root_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.root')]
        if debug:
            root_files = root_files[:5]  # Load only first 5 files for debugging
            logging.info("Debug mode enabled: Loading only first 5 ROOT files.")

        # Retrieve the tree name from the configuration and ensure it's a string
        tree_name = self.config['data_download'].get('tree_name', 'simOmtfPhase2Digis/OMTFHitsTree')

        # Updated required branches to include all dependencies for auxiliary info
        required_branches = [
            'stubPhiB', 'stubEta', 'muonCharge', 'muonPt', 
            'stubQuality', 'omtfProcessor', 'stubType', 'stubLayer', 'stubPhi',
            'muonPropEta', 'muonPropPhi'  # Added these
        ]

        # Define stub-related columns to explode
        stub_columns = ['stubType', 'stubEta', 'stubLayer', 'stubQuality', 'stubPhi', 'stubPhiB', 'muonPropEta', 'muonPropPhi']

        for file_name in tqdm(root_files, desc="Processing ROOT files"):
            file_path = os.path.join(self.raw_data_dir, file_name)
            try:
                with uproot.open(file_path) as file:
                    # Identify the exact tree key
                    tree_keys = [key for key in file.keys() if key.startswith(tree_name)]

                    if not tree_keys:
                        logging.error(f"Tree '{tree_name}' not found in {file_name}. Skipping file.")
                        continue

                    # Assuming only one tree matches
                    tree = file[tree_keys[0]]
                    data = tree.arrays(library="pd")

                # Ensure required branches are present
                missing_branches = [b for b in required_branches if b not in data.columns]
                if missing_branches:
                    logging.error(f"Missing branches {missing_branches} in {file_name}. Skipping file.")
                    continue

                # Check if stub columns are list-like (i.e., have multiple stubs per event)
                is_list_like = data[stub_columns].applymap(lambda x: isinstance(x, (list, np.ndarray))).all(axis=1)
                if is_list_like.any():
                    logging.info(f"Exploding stub-related columns in {file_name}")
                    
                    # **Step 1: Verify Consistent List Lengths**
                    # Calculate the length of lists in each stub column
                    list_lengths = data[stub_columns].applymap(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1)

                    # Check if all stub columns have the same list length for each row
                    consistent_lengths = list_lengths.nunique(axis=1) == 1
                    inconsistent_rows = ~consistent_lengths

                    num_inconsistent = inconsistent_rows.sum()
                    if num_inconsistent > 0:
                        logging.warning(f"{num_inconsistent} rows have inconsistent stub column lengths and will be dropped.")

                        # **Step 2: Log Details of Inconsistent Rows**
                        # Extract the inconsistent rows
                        inconsistent_data = data[inconsistent_rows]
                        inconsistent_lengths = list_lengths[inconsistent_rows]

                        # Log the list lengths of each stub column for the first few inconsistent rows
                        logging.debug(f"Inconsistent rows list lengths (first 5 rows):\n{inconsistent_lengths.head()}")

                        # Optionally, save inconsistent rows to a CSV for external analysis
                        inconsistent_lengths.to_csv(f'inconsistent_rows_{file_name}.csv', index=False)
                        logging.debug(f"Saved inconsistent rows to 'inconsistent_rows_{file_name}.csv' for further analysis.")

                        # **Step 3: Drop Inconsistent Rows**
                        data = data[consistent_lengths]

                    # Now, proceed to explode
                    data = data.explode(stub_columns)

                    # After exploding, some rows may have NaN if some stub attributes were missing
                    data = data.dropna(subset=stub_columns)
                    logging.info(f"Exploded data has {len(data)} rows")

                # Inspect DataFrame dtypes
                logging.debug(f"DataFrame dtypes after explosion:\n{data.dtypes}")

                # Convert AwkwardExtensionArray to standard types if necessary
                for col in required_branches:
                    if isinstance(data[col].dtype, awkward_pandas.array.AwkwardExtensionArray):
                        logging.debug(f"Converting column '{col}' from AwkwardExtensionArray to standard type.")
                        data[col] = data[col].apply(lambda x: x if isinstance(x, (int, float)) else np.nan)

                # Verify conversion
                for col in required_branches:
                    if isinstance(data[col].dtype, awkward_pandas.array.AwkwardExtensionArray):
                        logging.error(f"Column '{col}' is still of type AwkwardExtensionArray after conversion.")
                        raise TypeError(f"Column '{col}' is of unsupported type.")

                # Add auxiliary information
                data = self.add_auxiliary_info(data)

                # Select only the columns you need
                data = data[self.keep_branches]
                self.dataset.append(data)
                logging.info(f"Processed {file_name}")
            except KeyError as ke:
                logging.error(f"Key error while processing {file_name}: {ke}")
            except AttributeError as ae:
                logging.error(f"Attribute error while processing {file_name}: {ae}")
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")

    # Concatenate all DataFrames into a single DataFrame
    if self.dataset:
        try:
            self.dataset = pd.concat(self.dataset, ignore_index=True)
            logging.info(f"Total records after concatenation: {len(self.dataset)}")
        except ValueError as ve:
            logging.error(f"Error during concatenation: {ve}")
            # Optionally, handle partial concatenation or other strategies
    else:
        logging.warning("No data loaded. Check the ROOT files and branches.")


    def add_auxiliary_info(self, dataset):
        """
        Adds auxiliary computed columns to the dataset.

        Parameters:
        - dataset (pd.DataFrame): Original dataset.

        Returns:
        - pd.DataFrame: Dataset with added auxiliary columns.
        """
        logging.info("Adding auxiliary information...")

        # Check if necessary columns exist
        required_columns = ['stubPhiB', 'stubEta', 'muonCharge', 'muonPt', 'stubQuality', 'omtfProcessor']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            logging.error(f"Missing required columns for auxiliary computations: {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Vectorized computation for 'stubPhiG'
        dataset['stubPhiG'] += dataset['stubPhiB']

        # Vectorized computation for 'stubEtaG'
        dataset['stubEtaG'] = dataset['stubEta'] * self.HW_ETA_TO_ETA_FACTOR

        # Vectorized computation for 'muonPropEta'
        dataset['muonPropEta'] = dataset['muonPropEta'].abs()

        # Vectorized computation for 'muonQPt'
        dataset['muonQPt'] = dataset['muonCharge'] * dataset['muonPt']

        # Vectorized computation for 'muonQOverPt'
        dataset['muonQOverPt'] = dataset['muonCharge'] / dataset['muonPt']

        # Vectorized computation for 'stubR'
        dataset['stubR'] = self.compute_stub_r_vectorized(
            dataset['stubType'].values,
            dataset['stubEta'].values,
            dataset['stubLayer'].values,
            dataset['stubQuality'].values
        )

        # Vectorized computation for 'stubPhiG'
        dataset['stubPhiG'] = self.compute_global_phi_vectorized(
            dataset['stubPhiG'].values,
            dataset['omtfProcessor'].values
        )

        return dataset

    def compute_stub_r_vectorized(self, stubTypes, stubEtas, stubLayers, stubQualities):
        """
        Calculates coordinate r for each stub in a vectorized manner.

        Parameters:
        - stubTypes (np.array): Array of stub types.
        - stubEtas (np.array): Array of stub eta values.
        - stubLayers (np.array): Array of stub layer indices.
        - stubQualities (np.array): Array of stub quality indices.

        Returns:
        - np.array: Calculated r values.
        """
        logging.info("Computing stubR vectorized...")
        r = np.full_like(stubTypes, fill_value=np.nan, dtype=np.float64)

        # DTs (stubType == 3)
        dt_mask = stubTypes == 3
        dt_layers = stubLayers[dt_mask]
        dt_r = np.full_like(dt_layers, fill_value=np.nan, dtype=np.float64)

        # Assign base r based on stubLayer
        dt_r = np.where(
            dt_layers == 0, 431.133,
            np.where(
                dt_layers == 2, 512.401,
                np.where(dt_layers == 4, 617.946, np.nan)
            )
        )

        # Adjust for stubQuality
        low_quality_mask = (stubQualities[dt_mask] == 2) | (stubQualities[dt_mask] == 0)
        high_quality_mask = (stubQualities[dt_mask] == 3) | (stubQualities[dt_mask] == 1)
        dt_r[low_quality_mask] -= 23.5 / 2
        dt_r[high_quality_mask] += 23.5 / 2

        r[dt_mask] = dt_r

        # CSCs (stubType == 9)
        csc_mask = stubTypes == 9
        csc_layers = stubLayers[csc_mask]
        eta_factor = self.HW_ETA_TO_ETA_FACTOR
        z_csc = np.select(
            [
                csc_layers == 6,
                csc_layers == 9,
                csc_layers == 7,
                csc_layers == 8
            ],
            [
                690, 700, 830, 930
            ],
            default=np.nan
        )
        r[csc_mask] = z_csc / np.cos(
            np.tan(2 * np.arctan(np.exp(-stubEtas[csc_mask] * eta_factor)))
        )

        # RPCs (stubType == 5)
        rpc_mask = stubTypes == 5
        rpc_layers = stubLayers[rpc_mask]
        r_rpc = np.full_like(rpc_layers, fill_value=999.0, dtype=np.float64)

        # Assign specific r based on stubLayer
        z_rpc = np.select(
            [
                rpc_layers == 10,
                rpc_layers == 11,
                rpc_layers == 12,
                rpc_layers == 13,
                rpc_layers == 14,
                rpc_layers == 15,
                rpc_layers == 16,
                rpc_layers == 17
            ],
            [
                413.675,  # RB1in
                448.675,  # RB1out
                494.975,  # RB2in
                529.975,  # RB2out
                602.150,  # RB3
                720 / np.cos(np.tan(2 * np.arctan(np.exp(-stubEtas[rpc_mask] * eta_factor)))),
                790 / np.cos(np.tan(2 * np.arctan(np.exp(-stubEtas[rpc_mask] * eta_factor)))),
                970 / np.cos(np.tan(2 * np.arctan(np.exp(-stubEtas[rpc_mask] * eta_factor))))
            ],
            default=999.0
        )
        # Replace 999 with calculated z/cos(...)
        r_rpc = np.where(r_rpc == 999.0, z_rpc, r_rpc)

        r[rpc_mask] = r_rpc

        # Handle any remaining NaNs if necessary
        r = np.nan_to_num(r, nan=999.0)  # Replace NaNs with a default value if desired

        return r

    def compute_global_phi_vectorized(self, phi, processor):
        """
        Calculates global phi in a vectorized manner.

        Parameters:
        - phi (np.array): Array of local phi values.
        - processor (np.array): Array of processor indices.

        Returns:
        - np.array: Calculated global phi values.
        """
        logging.info("Computing global phi vectorized...")
        p1phiLSB = 2 * np.pi / self.NUM_PHI_BINS
        global_phi = ((processor * 192 + phi + 600) % self.NUM_PHI_BINS) * p1phiLSB
        return global_phi

    def getEtaKey(self, eta):
        """
        Determines the eta key based on the absolute eta value.

        Parameters:
        - eta (float): Eta value.

        Returns:
        - int: Eta key.
        """
        abs_eta = np.abs(eta)
        if abs_eta < 0.92:
            return 1
        elif abs_eta < 1.1:
            return 2
        elif abs_eta < 1.15:
            return 3
        elif abs_eta < 1.19:
            return 4
        else:
            return 5

    def getListOfConnectedLayers(self, eta):
        """
        Retrieves a list of connected layers based on eta.

        Parameters:
        - eta (float): Eta value.

        Returns:
        - list of int: Connected layers.
        """
        etaKey = self.getEtaKey(eta)

        LAYER_ORDER_MAP = {
            1: [10, 0, 11, 12, 2, 13, 14, 4, 6, 15],
            2: [10, 0, 11, 12, 2, 13, 6, 15, 16, 7],
            3: [10, 0, 11, 6, 15, 16, 7, 8, 17],
            4: [10, 0, 11, 16, 7, 8, 17],
            5: [10, 0, 9, 16, 7, 8, 17],
        }
        return LAYER_ORDER_MAP.get(etaKey, [])

    def getEdgesFromLogicLayer(self, logicLayer, withRPC=True):
        """
        Retrieves connected layers based on logicLayer and RPC inclusion.

        Parameters:
        - logicLayer (int): Logic layer index.
        - withRPC (bool): Whether to include RPC layers.

        Returns:
        - list of int: Connected layers.
        """
        LOGIC_LAYERS_CONNECTION_MAP = {
            0: [2, 4, 6, 7, 8, 9],   # MB1: [MB2, MB3, ME1/3, ME2/2]
            2: [4, 6, 7],             # MB2: [MB3, ME1/3]
            4: [6],                   # MB3: [ME1/3]
            6: [7, 8],                # ME1/3: [ME2/2, ME3/2]
            7: [8, 9],                # ME2/2: [ME3/2, RE1/3]
            8: [9],                   # ME3/2: [RE3/3]
            9: [],                    # ME1/2: []
        }
        LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC = {
            0: [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            2: [4, 6, 7, 10, 11, 12, 13, 14, 15, 16],
            4: [6, 10, 11, 12, 13, 14, 15],
            6: [7, 8, 10, 11, 12, 13, 14, 15, 16, 17],
            7: [8, 9, 10, 11, 15, 16, 17],
            8: [9, 10, 11, 15, 16, 17],
            9: [7, 10, 16, 17],
            10: [11, 12, 13, 14, 15, 16, 17],
            11: [12, 13, 14, 15, 16, 17],
            12: [13, 14, 15, 16],
            13: [14, 15, 16],
            14: [15],
            15: [16, 17],
            16: [17],
            17: []
        }

        if withRPC:
            return LOGIC_LAYERS_CONNECTION_MAP_WITH_RPC.get(logicLayer, [])
        else:
            if logicLayer >= 10:
                return []
            return LOGIC_LAYERS_CONNECTION_MAP.get(logicLayer, [])

    def getDeltaPhi(self, phi1, phi2):
        """
        Calculates the delta phi between two angles, ensuring the result is within [-pi, pi].

        Parameters:
        - phi1 (float or np.array): First phi value(s).
        - phi2 (float or np.array): Second phi value(s).

        Returns:
        - float or np.array: Delta phi value(s).
        """
        # Ensure inputs are numpy arrays for vectorization
        phi1 = np.array(phi1)
        phi2 = np.array(phi2)
        dphi = phi1 - phi2
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        return dphi

    def getDeltaEta(self, eta1, eta2):
        """
        Calculates the delta eta between two eta values.

        Parameters:
        - eta1 (float or np.array): First eta value(s).
        - eta2 (float or np.array): Second eta value(s).

        Returns:
        - float or np.array: Delta eta value(s).
        """
        # Vectorized delta eta
        return eta1 - eta2

    def convert_to_graph(self, num_workers=4, device="cuda"):
        """
        Converts the loaded dataset into PyTorch Geometric graph objects in parallel.

        Parameters:
        - num_workers (int): Number of parallel workers.
        - device (str): Device to run computations on ('cuda' or 'cpu').
        """
        logging.info("Converting dataset to PyTorch Geometric graphs...")

        def process_row(row):
            if row["muonQPt"] == 0:
                return None

            stub_layers = row['stubLayer']
            num_stubs = len(stub_layers)
            if num_stubs == 0:
                return None

            # Node features: stub_vars
            x = np.array([[row[var][i] for var in self.stub_vars] for i in range(num_stubs)], dtype=np.float32)

            # Edge indices and edge attributes
            edge_index = []
            edge_attr = []

            for i in range(num_stubs):
                source_layer = stub_layers[i]
                if self.connectivity < 0:
                    connected_layers = self.getEdgesFromLogicLayer(source_layer)
                else:
                    connected_layers = self.getListOfConnectedLayers(row['stubEtaG'][i])

                for target_layer in connected_layers:
                    if target_layer == source_layer:
                        continue
                    if target_layer not in stub_layers:
                        continue
                    j = stub_layers.index(target_layer)
                    if self.connectivity < 0:
                        # All-layer connectivity
                        deltaPhi = self.getDeltaPhi(row['stubPhiG'][i], row['stubPhiG'][j])
                        deltaEta = self.getDeltaEta(row['stubEtaG'][i], row['stubEtaG'][j])
                        edge_index.append([i, j])
                        edge_attr.append([deltaPhi, deltaEta])
                    else:
                        # K-neighbor connectivity
                        connected_layers_list = self.getListOfConnectedLayers(row['stubEtaG'][i])
                        try:
                            source_idx = connected_layers_list.index(source_layer)
                            target_idx = connected_layers_list.index(target_layer)
                        except ValueError:
                            continue  # Skip if layer not found
                        if abs(source_idx - target_idx) > self.connectivity:
                            continue
                        deltaPhi = self.getDeltaPhi(row['stubPhiG'][i], row['stubPhiG'][j])
                        deltaEta = self.getDeltaEta(row['stubEtaG'][i], row['stubEtaG'][j])
                        edge_index.append([i, j])
                        edge_attr.append([deltaPhi, deltaEta])

            if not edge_index:
                return None

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)

            # Assuming target labels are one-hot encoded; adjust as per your task
            y = torch.tensor([row[var] for var in self.muon_vars], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            return data

        # Parallel processing with progress bar
        try:
            results = Parallel(n_jobs=num_workers)(
                delayed(process_row)(row) for _, row in tqdm(self.dataset.iterrows(), total=self.dataset.shape[0], desc="Converting DataFrames")
            )
        except AttributeError as ae:
            logging.error(f"Attribute error during graph conversion: {ae}")
            return

        # Filter out None results
        self.pyg_graphs = [data for data in results if data is not None]
        logging.info(f'Graphs created and stored: {len(self.pyg_graphs)} graphs')

    def draw_graph(self, data, savefig="graph.png"):
        """
        Visualizes a single graph and saves it as an image.

        Parameters:
        - data (torch_geometric.data.Data): Graph data.
        - savefig (str): Filename to save the graph visualization.
        """
        logging.info(f'Drawing graph into {savefig}')
        G = to_networkx(data, to_undirected=True)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=300, node_color='skyblue', edge_color='gray')
        plt.title(f"Graph with {G.number_of_nodes()} nodes")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
        logging.info(f'Graph saved as {savefig}')

    def draw_example_graphs(self, savefig="graph_examples.png", seed=42, num_examples=6):
        """
        Draws and saves a specified number of example graphs.

        Parameters:
        - savefig (str): Filename to save the example graph visualizations.
        - seed (int): Seed for layout reproducibility.
        - num_examples (int): Number of example graphs to visualize.
        """
        logging.info('Drawing example graphs...')
        if not self.pyg_graphs:
            logging.warning("No graphs to draw.")
            return

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        for i in range(num_examples):
            if i >= len(self.pyg_graphs):
                break
            data = self.pyg_graphs[i]
            G = to_networkx(data, to_undirected=True)
            pos = nx.spring_layout(G, seed=seed)
            nx.draw(G, pos, ax=axs[i], with_labels=True, node_size=300, node_color='skyblue', edge_color='gray')
            axs[i].set_title(f"Graph {i+1} with {G.number_of_nodes()} nodes")
            axs[i].axis("off")

        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
        logging.info(f'Example graphs saved as {savefig}')

    def verifyGraphs(self):
        """
        Verifies that all graphs are connected. Saves disconnected graphs as images.
        """
        logging.info('Verifying graphs for connectivity...')
        disconnected_graphs = []
        for idx, data in enumerate(self.pyg_graphs):
            G = to_networkx(data, to_undirected=True)
            if not nx.is_connected(G):
                disconnected_graphs.append((idx, G))

        if disconnected_graphs:
            logging.warning(f"Found {len(disconnected_graphs)} disconnected graphs.")
            for idx, G in disconnected_graphs:
                logging.warning(f"Graph {idx} is not connected. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                self.draw_graph(data=Data.from_networkx(G), savefig=f'unconnected_graph_{idx}.png')
        else:
            logging.info('All graphs are connected.')

    def saveTorchDataset(self, save_path=None):
        """
        Saves the PyTorch Geometric dataset to disk.

        Parameters:
        - save_path (str, optional): Path to save the dataset. Defaults to self.graph_save_path.
        """
        if save_path is None:
            save_path = self.graph_save_path

        torch.save(self.pyg_graphs, save_path)
        logging.info(f'PyTorch Geometric dataset saved to {save_path}')

    def loadTorchDataset(self, load_path=None):
        """
        Loads a PyTorch Geometric dataset from disk.

        Parameters:
        - load_path (str, optional): Path to load the dataset from. Defaults to self.graph_save_path.

        Returns:
        - list of torch_geometric.data.Data: Loaded graph dataset.
        """
        if load_path is None:
            load_path = self.graph_save_path

        self.pyg_graphs = torch.load(load_path)
        logging.info(f'PyTorch Geometric dataset loaded from {load_path}')
        return self.pyg_graphs

    def draw_node_properties(self, property_name, savefig="node_properties.png"):
        """
        Placeholder for a method to draw node properties.

        Parameters:
        - property_name (str): Name of the property to visualize.
        - savefig (str): Filename to save the visualization.
        """
        logging.info(f'Drawing node properties {property_name} into {savefig}')
        # Implement visualization based on the property_name
        pass

    def clear_memory(self):
        """
        Clears the dataset and graph data from memory.
        """
        logging.info("Clearing memory...")
        self.dataset = []
        self.pyg_graphs = []
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Memory cleared.")
