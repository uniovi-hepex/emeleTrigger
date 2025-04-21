 ###Format:   ([List of files], IsData?)
phedexPath = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250409/"

files_HTo2LongLivedTo2mu2jets = (phedexPath + "HTo2LongLivedTo2mu2jets/*root", "simOmtfPhase2Digis/OMTFHitsTree")
files_MuGun_Displaced = (phedexPath + "MuGun_Displaced/*root", "simOmtfPhase2Digis/OMTFHitsTree")
files_MuGun_FullEta_FlatPt1to1000 = (phedexPath + "MuGun_FullEta_FlatPt1to1000/*root", "simOmtfPhase2Digis/OMTFHitsTree")
files_MuGun_FullEta_OneOverPt_1to100 = (phedexPath + "MuGun_FullEta_OneOverPt_1to100/*root", "simOmtfPhase2Digis/OMTFHitsTree")


files_SingleMu_Prompt_OneOverPt = (phedexPath + "/Dumper_l1omtf_001_july24.root","simOmtfPhase2Digis/OMTFHitsTree")
files_test = ("../data/Dumper_MuGun_FullEta_v250409_001.root", "simOmtfPhase2Digis/OMTFHitsTree")
datasets = {
    'SingleMu_Prompt_OneOverPt' : {'name'       : 'SingleMu_Prompt_OneOverPt', # Name of the dataset
                                  'samples'    : files_SingleMu_Prompt_OneOverPt[0], #  FILE
                                  'treename'    : files_SingleMu_Prompt_OneOverPt[1], # Is prompt?
                                  'color'     : 'black', # Plotting color
                                  'label'     : "Muon Gun (Prompt)", # Legend label
                                  'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                 } 
                                },
    'test' : {'name'       : 'test', # Name of the dataset
                                    'samples'    : files_test[0], #  FILE
                                    'treename'    : files_test[1], # Is prompt?
                                    'color'     : 'black', # Plotting color
                                    'label'     : "test", # Legend label
                                    'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                     } 
                                    },
}