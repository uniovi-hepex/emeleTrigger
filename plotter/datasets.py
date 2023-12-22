###Format:   ([List of files], IsData?)
phedexPath = "./"

files_SingleMu_FlatPt0To1000Dxy3m = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_FlatPt0To1000Dxy3m_NonDegraded_Stub.root", "simOmtfDigis/OMTFHitsTree") # 
files_SingleMu_XTo2LLP4Mu_Ctau5m  = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Ctau5m_Stub.root", "simOmtfDigis/OMTFHitsTree") # Muo
files_SingleMu_XTo2LLP4Mu_Ctau3m  = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Stub.root","simOmtfDigis/OMTFHitsTree")
files_SingleMu_PromptSampleFlatPt = (phedexPath + "data/SingleMu_GT131X_Extrapolation_GhostBusterTest_PromptSampleFlatPt_Stub.root","simOmtfDigis/OMTFHitsTree")

datasets = {
    'MuGun_FlatPt0To1000Dxy3m'  : {'samples'    : files_SingleMu_FlatPt0To1000Dxy3m[0], #  FILE
                                   'treename'   : files_SingleMu_FlatPt0To1000Dxy3m[1], # Is prompt?
                                   'color'      : 'black', # Plotting color
                                   'label'      : "Muon Gun ($d_{xy}=3m$)", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },   
    'MuGun_XTo2LLP4Mu_Ctau5m'  : {'samples'    : files_SingleMu_XTo2LLP4Mu_Ctau5m[0], #  FILE
                                   'treename'  : files_SingleMu_XTo2LLP4Mu_Ctau5m[1], # Is prompt?
                                   'color'     : 'black', # Plotting color
                                   'label'     : "$X \rightarrow 4\mu (c_{\tau}=5m)$", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
    'MuGun_XTo2LLP4Mu_Ctau3m'  : {'samples'    : files_SingleMu_XTo2LLP4Mu_Ctau3m[0], #  FILE
                                   'treename'  : files_SingleMu_XTo2LLP4Mu_Ctau3m[1], # Is prompt?
                                   'color'     : 'black', # Plotting color
                                   'label'     : "$X \rightarrow 4\mu (c_{\tau}=3m)$", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
    'MuGun_PromptSampleFlatPt' : {'samples'    : files_SingleMu_PromptSampleFlatPt[0], #  FILE
                                   'prompt'    : files_SingleMu_PromptSampleFlatPt[1], # Is prompt?
                                   'color'     : 'black', # Plotting color
                                   'label'     : "Muon Gun (Prompt)", # Legend label
                                   'selection'  : {'acceptance' : 'muonPropEta!=0 & muonPropPhi!=0',  #Only take muons inside acceptance
                                 } 
                                },
}
