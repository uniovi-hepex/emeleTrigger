import uproot

file_path = "/home/pleguina/work/emeleTrigger/tools/training_v2/data/raw/Dumper_l1omtf_001.root"

desired_tree = "simOmtfPhase2Digis/OMTFHitsTree"

with uproot.open(file_path) as file:
    print("Available keys:", file.keys())
    # Remove .encode() to keep desired_tree as a string
    tree_keys = [key for key in file.keys() if key.startswith(desired_tree)]
    if not tree_keys:
        print(f"Tree '{desired_tree}' not found in {file_path}.")
    else:
        tree = file[tree_keys[0]]
        print("Available branches in the tree:", tree.keys())
        # Optionally, load a small portion of the tree
        data = tree.arrays(library="pd", entry_stop=10000)
        print 
        print(data)
        
        
        #print all the columns of the first 10 rows
        
        for i in range(1): 
            #and their values
            for column in data.columns:
                print(f"{column}: {data[column].iloc[i]}")  
                #print the type of the column
                print(f"{column}: {data[column].dtype}")
                
