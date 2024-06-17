###Format:   ([List of files], IsData?)
phedexPath = "./"

files_SingleMu_FlatPt0To1000Dxy3m = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_FlatPt0To1000Dxy3m_NonDegraded_Stub.root", "simOmtfDigis/OMTFHitsTree") # 
files_SingleMu_XTo2LLP4Mu_Ctau5m  = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Ctau5m_Stub.root", "simOmtfDigis/OMTFHitsTree") # Muo
files_SingleMu_XTo2LLP4Mu_Ctau5m_DUMP  = (phedexPath + "data/SingleMu_l1omtf_Dump.root", "simOmtfPhase2Digis/OMTFHitsTree") # Muo
files_SingleMu_XTo2LLP4Mu_Ctau3m  = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Stub.root","simOmtfDigis/OMTFHitsTree")
files_SingleMu_PromptSampleFlatPt = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_PromptSampleFlatPt_Stub.root","simOmtfDigis/OMTFHitsTree")

datasets = {
    'MuGun_FlatPt0To1000Dxy3m'  : {'name'       : 'MuGun_FlatPt0To1000Dxy3m', # Name of the dataset
                                   'samples'    : files_SingleMu_FlatPt0To1000Dxy3m[0], #  FILE
                                   'treename'   : files_SingleMu_FlatPt0To1000Dxy3m[1], # Is prompt?
                                   'color'      : 'black', # Plotting color
                                   'label'      : "Muon Gun ($d_{xy}=3m$)", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },   
    'MuGun_XTo2LLP4Mu_Ctau5m'  : {'name'       : 'MuGun_XTo2LLP4Mu_Ctau5m', # Name of the dataset
                                  'samples'    : files_SingleMu_XTo2LLP4Mu_Ctau5m[0], #  FILE
                                  'treename'  : files_SingleMu_XTo2LLP4Mu_Ctau5m[1], # Is prompt?
                                  'color'     : 'black', # Plotting color
                                  'label'     : "X to $4\mu (ctau=5m)$", # Legend label
                                  'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
    'MuGun_XTo2LLP4Mu_Ctau5m_DUMP'  : {'name'       : 'MuGun_XTo2LLP4Mu_Ctau5m_DUMP', # Name of the dataset
                                  'samples'    : files_SingleMu_XTo2LLP4Mu_Ctau5m_DUMP[0], #  FILE
                                  'treename'  : files_SingleMu_XTo2LLP4Mu_Ctau5m_DUMP[1], # Is prompt?
                                  'color'     : 'black', # Plotting color
                                  'label'     : "X to $4\mu (ctau=5m)$", # Legend label
                                  'selection'  : {'acceptance' : '',  #Only take muons inside acceptance
                                 } 
                                },
    'MuGun_XTo2LLP4Mu_Ctau3m'  : {'name'       : 'MuGun_XTo2LLP4Mu_Ctau3m', # Name of the dataset
                                  'samples'    : files_SingleMu_XTo2LLP4Mu_Ctau3m[0], #  FILE
                                   'treename'  : files_SingleMu_XTo2LLP4Mu_Ctau3m[1], # Is prompt?
                                   'color'     : 'black', # Plotting color
                                   'label'     :  "X to $4\mu (ctau=3m)$", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
    'MuGun_PromptSampleFlatPt' : {'name'       : 'MuGun_PromptSampleFlatPt', # Name of the dataset
                                  'samples'    : files_SingleMu_PromptSampleFlatPt[0], #  FILE
                                   'treename'    : files_SingleMu_PromptSampleFlatPt[1], # Is prompt?
                                   'color'     : 'black', # Plotting color
                                   'label'     : "Muon Gun (Prompt)", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
}
