# data_processing.py

import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

import numpy as np
import pandas as pd
import awkward as ak
import uproot
from awkward_pandas.array import AwkwardExtensionArray

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
    def __init__(self, raw_data_dir, graph_save_path, model_connectivity, config):

        self.graph_save_paths = graph_save_path
        self.raw_data_dir = raw_data_dir
        self.model_connectivity = model_connectivity

        self.NUM_PROCESSORS = 3
        self.NUM_PHI_BINS = 5400
        self.HW_ETA_TO_ETA_FACTOR = 0.010875

        # Define scalar and list columns separately
        self.scalar_columns = ['eventNum', 'muonQPt', 'muonPropEta', 'muonPropPhi']
        self.list_columns = [
            'stubLayer', 'stubQuality', 'stubPhi', 'stubEtaG',
            'stubR', 'stubType', 'inputStubLogicLayer', 'inputStubProc',
            'inputStubPhi', 'inputStubPhiB', 'inputStubEta',
            'inputStubQuality', 'inputStubBx', 'inputStubTiming',
            'inputStubDetId', 'inputStubType', 'inputStubIsMatched',
            'inputStubDeltaPhi0', 'inputStubDeltaPhi1', 'inputStubDeltaPhi2',
            'inputStubDeltaPhi3', 'inputStubDeltaPhi4', 'inputStubDeltaPhi5',
            'inputStubDeltaPhi6', 'inputStubDeltaPhi7',
            'inputStubDeltaEta0', 'inputStubDeltaEta1',
            'inputStubDeltaEta2', 'inputStubDeltaEta3',
            'inputStubDeltaEta4', 'inputStubDeltaEta5',
            'inputStubDeltaEta6', 'inputStubDeltaEta7'
        ]

        # Define the required branches for loading
        self.required_branches = [
            'stubPhiB', 'stubEta', 'muonCharge', 'muonPt', 
            'stubQuality', 'omtfProcessor', 'stubType', 'stubLayer', 'stubPhi',
            'muonPropEta', 'muonPropPhi'
        ]
        
        self.list_fields_to_vectorize = [
            'stubEta', 'stubQuality', 'stubType', 'stubLayer', 'stubPhi'
        ]

        self.keep_branches = ['stubEtaG', 'stubPhiG','stubR', 'stubLayer','stubType','muonQPt','muonQOverPt','muonPropEta','muonPropPhi']
        self.muon_vars = ['muonQPt', 'muonPt', 'muonQOverPt', 'muonPropEta', 'muonPropPhi']
        self.stub_vars = ['stubEtaG', 'stubPhiG', 'stubR', 'stubLayer', 'stubType']
        self.dataset = []       # Initialize as empty list
        self.pyg_graphs = []    # Initialize as empty list
        self.config = config    # Store config as instance 

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
        



    def vectorize_columns(self, df: pd.DataFrame, fixed_size: int = 18, default_value: float = 0.0) -> pd.DataFrame:
        """
        Vectorizes list-type columns by converting Awkward Arrays to fixed-size numpy float arrays.

        Parameters:
        - df (pd.DataFrame): The original DataFrame.
        - fixed_size (int): The fixed size for the output vectors.
        - default_value (float): The default value to use for padding.

        Returns:
        - pd.DataFrame: DataFrame with vectorized float array columns.
        """
        logging.info("Starting vectorization of list-type columns...")
        
        for field in self.list_fields_to_vectorize:
            if field not in df.columns:
                logging.warning(f"Field '{field}' not found in DataFrame. Skipping vectorization.")
                continue

            # Prepare a dictionary to hold the vectorized columns
            vectorized_data = {}
            
            # Vectorize each entry
            field_values = df[field].apply(lambda x: np.array(ak.to_numpy(x) if isinstance(x, ak.Array) else x))
            for i in range(fixed_size):
                vectorized_data[f"{field}_vec_{i}"] = [
                    np.pad(vals, (0, max(0, fixed_size - len(vals))), constant_values=default_value)[i]
                    if isinstance(vals, np.ndarray) else default_value
                    for vals in field_values
                ]
            
            # Concatenate vectorized columns to the original DataFrame
            df = pd.concat([df, pd.DataFrame(vectorized_data, index=df.index)], axis=1)

        # Drop the original list-type columns
        df = df.drop(columns=self.list_fields_to_vectorize, errors='ignore')
        logging.info("Vectorization completed successfully.")
        return df






    def load_data(self, debug=True):
        """
        Loads and preprocesses ROOT files from the raw_data_dir.

        Parameters:
        - debug (bool): If True, loads only a subset of data for debugging purposes.
        """
        logging.info(f'Loading data from {self.raw_data_dir}')
        root_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.root')]
        if debug:
            root_files = root_files[:2]  # Load only first 5 files for debugging
            logging.info("Debug mode enabled: Loading only first 5 ROOT files.")

        # Retrieve the tree name from the configuration
        tree_name = self.config['data_download'].get('tree_name', 'simOmtfPhase2Digis/OMTFHitsTree')
        logging.info(f"Using tree name: {tree_name}")

        for file_name in tqdm(root_files, desc="Processing ROOT files"):
            file_path = os.path.join(self.raw_data_dir, file_name)
            try:
                with uproot.open(file_path) as file:
                    # Identify the exact tree key
                    tree_keys = [key for key in file.keys() if key.startswith(tree_name)]
                    if not tree_keys:
                        logging.error(f"Tree '{tree_name}' not found in {file_name}. Skipping file.")
                        continue

                    # Load the tree
                    tree = file[tree_keys[0]]
                    logging.info(f"Loading data from tree: {tree_keys[0]} in {file_name}")

                    # Load required branches into an awkward array
                    branches_to_load = self.required_branches
                    data_awkward = tree.arrays(branches_to_load, library="ak", entry_stop=None if not debug else 10000)
                    logging.info(f"Loaded data from {file_name} with {len(data_awkward)} entries.")

                    # Convert awkward array to pandas DataFrame
                    data = ak.to_dataframe(data_awkward)

                # Ensure required branches are present
                missing_branches = [b for b in self.required_branches if b not in data.columns]
                if missing_branches:
                    logging.error(f"Missing branches {missing_branches} in {file_name}. Skipping file.")
                    continue

                # Vectorize list-like (jagged) columns
                if self.list_fields_to_vectorize:
                    logging.info(f"Vectorizing list-like columns in {file_name}")
                    data = self.vectorize_columns(data, fixed_size=18, default_value=0.0)

                # Add auxiliary information
                data = self.add_auxiliary_info(data)
                                # Expand keep_branches to include all vectorized columns for list-type fields
                expanded_keep_branches = []
                for col in self.keep_branches:
                    if col in ['stubEtaG', 'stubPhiG', 'stubR', 'stubLayer', 'stubType']:
                        expanded_keep_branches.extend([f"{col}_vec_{i}" for i in range(18)])
                    else:
                        expanded_keep_branches.append(col)

                # Select only the necessary columns after expanding keep_branches
                try:
                    data = data[expanded_keep_branches]
                except KeyError as e:
                    missing_columns = [col for col in expanded_keep_branches if col not in data.columns]
                    logging.error(f"Missing columns after expansion: {missing_columns}")
                    raise KeyError(f"Missing columns after expansion: {missing_columns}")

                # Select only the necessary columns
                #data = data[self.keep_branches]
                self.dataset.append(data)
                #log  how data looks like
                logging.info(data.head())
                #log the shape of the data
                logging.info(data.shape)
                #log the columns of the data
                logging.info(data.columns)
                #log the data types of the columns
                logging.info(data.dtypes)
                logging.info(f"Processed {file_name} successfully.")

            except KeyError as ke:
                logging.error(f"Key error while processing {file_name}: {ke}", exc_info=True)
            except AttributeError as ae:
                logging.error(f"Attribute error while processing {file_name}: {ae}", exc_info=True)
            except TypeError as te:
                logging.error(f"Failed to process {file_name}: {te}", exc_info=True)
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}", exc_info=True)

        # Concatenate all DataFrames into a single DataFrame
        if self.dataset:
            try:
                self.dataset = pd.concat(self.dataset, ignore_index=True)
                logging.info(f"Total records after concatenation: {len(self.dataset)}")
            except ValueError as ve:
                logging.error(f"Error during concatenation: {ve}", exc_info=True)
        else:
            logging.warning("No data loaded. Check the ROOT files and branches.")


    def add_auxiliary_info(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Adds auxiliary computed columns to the dataset in a vectorized manner.

        Parameters:
        - dataset (pd.DataFrame): Original dataset.

        Returns:
        - pd.DataFrame: Dataset with added auxiliary columns.
        """
        logging.info("Adding auxiliary information...")

        # Define required vectorized columns
        required_columns = [
            'stubPhiB', 'muonCharge', 'muonPt', 'omtfProcessor', 'muonPropEta', 'muonPropPhi'
        ]
        stub_vectorized_columns = ['stubEta_vec', 'stubQuality_vec', 'stubType_vec', 'stubLayer_vec', 'stubPhi_vec']
        required_columns += [f"{col}_{i}" for col in stub_vectorized_columns for i in range(18)]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        if missing_columns:
            logging.error(f"Missing required columns for auxiliary computations: {missing_columns}")
            raise KeyError(f"Missing required columns: {missing_columns}")

        # Update 'stubPhi' by adding 'stubPhiB'
        dataset['stubPhiB'] += dataset['stubPhiB']

        # Compute 'stubEtaG' as a scaled version of 'stubEta'
        for i in range(18):
            dataset[f'stubEtaG_vec_{i}'] = dataset[f'stubEta_vec_{i}'] * self.HW_ETA_TO_ETA_FACTOR

        # Compute 'stubPhiG' using vectorized global phi
        for i in range(18):
            dataset[f'stubPhiG_vec_{i}'] = self.compute_global_phi_vectorized(
                phi=dataset[f'stubPhi_vec_{i}'].values,
                processor=dataset['omtfProcessor'].values
            )

        # Compute 'muonPropEta' as the absolute value of 'muonPropEta' (scalar)
        dataset['muonPropEta'] = dataset['muonPropEta'].abs()

        # Compute 'muonQPt' and 'muonQOverPt'
        dataset['muonQPt'] = dataset['muonCharge'] * dataset['muonPt']
        dataset['muonQOverPt'] = dataset['muonCharge'] / dataset['muonPt'].replace(0, np.nan)
        dataset['muonQOverPt'] = dataset['muonQOverPt'].fillna(0)

        # Compute 'stubR' using vectorized helper function
        stubType_vals = dataset[[f'stubType_vec_{i}' for i in range(18)]].values
        stubEta_vals = dataset[[f'stubEta_vec_{i}' for i in range(18)]].values
        stubLayer_vals = dataset[[f'stubLayer_vec_{i}' for i in range(18)]].values
        stubQuality_vals = dataset[[f'stubQuality_vec_{i}' for i in range(18)]].values

        # Compute stubR in a vectorized way
        stubR_vals = np.full_like(stubType_vals, np.nan, dtype=np.float64)
        for idx in range(stubType_vals.shape[0]):
            stubR_vals[idx] = self.compute_stub_r_vectorized(
                stubTypes=stubType_vals[idx],
                stubEtas=stubEta_vals[idx],
                stubLayers=stubLayer_vals[idx],
                stubQualities=stubQuality_vals[idx]
            )

        # Assign computed stubR vectors back to the DataFrame
        for i in range(18):
            dataset[f'stubR_vec_{i}'] = stubR_vals[:, i]

        # Ensure 'stubPhiB' is updated
        dataset['stubPhiB'] = dataset['stubPhiB']

        logging.info("Auxiliary information added successfully.")

        # Verify that all auxiliary columns are present
        auxiliary_columns = ['muonQPt', 'muonQOverPt'] + [f'stubEtaG_vec_{i}' for i in range(18)] + [f'stubPhiG_vec_{i}' for i in range(18)] + [f'stubR_vec_{i}' for i in range(18)]
        missing_auxiliary = [col for col in auxiliary_columns if col not in dataset.columns]
        if missing_auxiliary:
            logging.error(f"Missing auxiliary columns after computation: {missing_auxiliary}")
            raise KeyError(f"Missing auxiliary columns: {missing_auxiliary}")

        # Optional: Handle NaN values
        for col in auxiliary_columns:
            if dataset[col].isnull().any():
                logging.warning(f"Null values found in auxiliary column: {col}")
                dataset[col] = dataset[col].fillna(0)

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
        logging.debug("Computing stubR vectorized for a single event...")
        r = np.full_like(stubTypes, fill_value=np.nan, dtype=np.float64)

        # DTs (stubType == 3)
        dt_mask = stubTypes == 3
        dt_layers = stubLayers[dt_mask]
        dt_qualities = stubQualities[dt_mask]

        # Assign base r based on stubLayer
        dt_r = np.where(
            dt_layers == 0, 431.133,
            np.where(
                dt_layers == 2, 512.401,
                np.where(dt_layers == 4, 617.946, np.nan)
            )
        )

        # Adjust for stubQuality
        low_quality_mask = (dt_qualities == 2) | (dt_qualities == 0)
        high_quality_mask = (dt_qualities == 3) | (dt_qualities == 1)
        dt_r = np.where(low_quality_mask, dt_r - 23.5 / 2, dt_r)
        dt_r = np.where(high_quality_mask, dt_r + 23.5 / 2, dt_r)

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
        angles = 2 * np.arctan(np.exp(-stubEtas[csc_mask] * eta_factor))
        cos_angles = np.cos(np.tan(angles))
        # Handle potential division by zero
        cos_angles = np.where(cos_angles == 0, 1e-6, cos_angles)
        r[csc_mask] = z_csc / cos_angles

        # RPCs (stubType == 5)
        rpc_mask = stubTypes == 5
        rpc_layers = stubLayers[rpc_mask]
        eta_rpc = stubEtas[rpc_mask] * eta_factor

        # Compute angles for RPCs
        angles_rpc = 2 * np.arctan(np.exp(-eta_rpc))
        cos_angles_rpc = np.cos(np.tan(angles_rpc))
        cos_angles_rpc = np.where(cos_angles_rpc == 0, 1e-6, cos_angles_rpc)

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
                720 / cos_angles_rpc,  # Adjusted
                790 / cos_angles_rpc,  # Adjusted
                970 / cos_angles_rpc   # Adjusted
            ],
            default=999.0
        )

        # Replace 999 with calculated z/cos(...)
        r_rpc = np.where(z_rpc == 999.0, z_rpc, z_rpc)
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
        logging.debug("Computing global phi vectorized...")
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

        if not isinstance(self.dataset, pd.DataFrame):
            logging.error("Dataset is not a pandas DataFrame. Conversion aborted.")
            return

        if self.dataset.empty:
            logging.error("Dataset is empty. No graphs to convert.")
            return


        def process_row(row):
            if row["muonQPt"] == 0:
                return None

            # Extract vectorized stub features
            stub_layers = row[[f'stubLayer_vec_{i}' for i in range(18)]].values
            stub_phi_g = row[[f'stubPhiG_vec_{i}' for i in range(18)]].values
            stub_eta_g = row[[f'stubEtaG_vec_{i}' for i in range(18)]].values
            stub_r = row[[f'stubR_vec_{i}' for i in range(18)]].values
            stub_type = row[[f'stubType_vec_{i}' for i in range(18)]].values

            # Identify valid stubs (where stubLayer is not the default value, e.g., 0)
            valid_mask = stub_layers != 0  # Assuming 0 is the default/padding value
            if not valid_mask.any():
                return None

            # Extract valid stub features
            valid_stub_layers = stub_layers[valid_mask]
            valid_stub_phi_g = stub_phi_g[valid_mask]
            valid_stub_eta_g = stub_eta_g[valid_mask]
            valid_stub_r = stub_r[valid_mask]
            valid_stub_type = stub_type[valid_mask]
            num_stubs = len(valid_stub_layers)

            if num_stubs == 0:
                return None

            # Node features: stub_vars
            x = np.stack([
                valid_stub_eta_g,
                valid_stub_phi_g,
                valid_stub_r,
                valid_stub_layers,
                valid_stub_type
            ], axis=1).astype(np.float32)

            # Edge indices and edge attributes
            edge_index = []
            edge_attr = []

            for i in range(num_stubs):
                source_layer = valid_stub_layers[i]
                if self.connectivity < 0:
                    connected_layers = self.getEdgesFromLogicLayer(source_layer)
                else:
                    connected_layers = self.getListOfConnectedLayers(valid_stub_eta_g[i])

                for target_layer in connected_layers:
                    if target_layer == source_layer:
                        continue
                    if target_layer not in valid_stub_layers:
                        continue
                    # Find all indices where target_layer matches
                    target_indices = np.where(valid_stub_layers == target_layer)[0]
                    for j in target_indices:
                        if i == j:
                            continue
                        deltaPhi = self.getDeltaPhi(valid_stub_phi_g[i], valid_stub_phi_g[j])
                        deltaEta = self.getDeltaEta(valid_stub_eta_g[i], valid_stub_eta_g[j])
                        edge_index.append([i, j])
                        edge_attr.append([deltaPhi, deltaEta])

            if not edge_index:
                return None

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            x = torch.tensor(x, dtype=torch.float)

            # Assuming target labels are [muonQPt, muonPt, muonQOverPt, muonPropEta, muonPropPhi]
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
