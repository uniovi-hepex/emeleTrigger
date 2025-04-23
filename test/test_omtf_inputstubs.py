import os, sys
import uproot

import numpy as np
import matplotlib.pyplot as plt

ROOTFILE = "../../data/Dumper_MuGun_FullEta_v250409_001.root"

print("Creating the dataset")
mu_vars = ['muonPt',"muonEta","muonPhi", "muonPropPhi", "muonPropEta"]
st_vars =  [#'stubEta', 'stubPhi','stubR', 'stubLayer','stubType',
            'inputStubEta', 'inputStubPhi','inputStubR', 'inputStubLayer','inputStubType','inputStubIsMatched',
            'inputStubProc','inputStubPhiB', 'inputStubQuality']

root_file = ROOTFILE
tree_name = "simOmtfPhase2Digis/OMTFHitsTree"
events_processed = 0

def is_empty_list(x):
    return isinstance(x, list) and len(x) == 0

with uproot.open(root_file) as file:
    tree = file[tree_name]
    df = tree.arrays(library="pd")

    # print the columns of the dataframe
    print("Columns in the dataframe:")
    print(df.columns)
    # print the first 5 rows of the dataframe
    print("First 5 rows of ALL dataframe:")
    print(df.head())

    df = df[df["muonEvent"] == 0]
    # Drop rows with NaN or epsilon values in the inputStub and stub variables
    common_st_vars = [col for col in st_vars if col in df.columns]
    if not common_st_vars:
        raise ValueError("No se encontraron columnas de stub en el dataframe.")

    # Convertir las columnas a cadena y eliminar filas que tengan valores vacíos
    df = df[~(df[common_st_vars].map(is_empty_list).any(axis=1))]
    print("First 5 rows of the dataframe:")
    print(df.head())

    ## NOW THE FUN BEGINS...
    print("Dimensiones del dataframe:", df.shape)

    for index, row in df.iterrows():
        print(index, row)
        # Check if the row contains empty or NaN values in the stub variables
        if any(is_empty_list(row[var]) for var in common_st_vars):
            print(f"Row {index} contains empty or NaN values in stub variables. Skipping...")
            continue    

        if events_processed % 100 == 0:
            print(f"Processed {events_processed} events")

'''                    # Create nodes and edges
                    stub_array = np.vstack([row[var] for var in self.stub_vars]).astype(np.float32).T
                    x = torch.tensor(stub_array, dtype=torch.float)
                    edge_index = self.create_edges(row['stubLayer'])

                    data = Data(x=x, edge_index=edge_index, y=torch.tensor([row[var] for var in self.muon_vars], dtype=torch.float))
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    if data is not None:
                        data_list.append(data)
                    events_processed += 1
'''
