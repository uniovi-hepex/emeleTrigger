import glob

 ###Format:   ([List of files], IsData?)
phedexPath = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/"

files_HTo2LongLivedTo2mu2jets = (glob.glob(phedexPath + "HTo2LongLivedTo2mu2jets/*root"))
files_MuGun_Displaced = (glob.glob(phedexPath + "MuGun_Displaced/*root"))
files_MuGun_FullEta_FlatPt1to1000 = (glob.glob(phedexPath + "MuGun_FullEta_FlatPt1to1000/*root"))
files_MuGun_FullEta_OneOverPt_1to100 = (glob.glob(phedexPath + "MuGun_FullEta_OneOverPt_1to100/*root"))

files_SingleMu_Prompt_OneOverPt = (phedexPath + "/Dumper_l1omtf_001_july24.root")
files_test = ("../data/Dumper_MuGun_FullEta_v250409_001.root")

##convert files_HTo2LongLivedTo2mu2jets to a list of files

datasets = {
    'HTo2LongLivedTo2mu2jets' : {'name'       : 'HTo2LongLivedTo2mu2jets', # Name of the dataset
                                  'samples'    : files_HTo2LongLivedTo2mu2jets, #  FILE
                                  'treename'    : "simOmtfPhase2Digis/OMTFHitsTree", # Is prompt?
                                  'color'     : 'black', # Plotting color
                                  'label'     : "HTo2LongLivedTo2mu2jets", # Legend label
                                  'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                        }
                                },
    'MuGun_Displaced' : {'name'       : 'MuGun_Displaced', # Name of the dataset
                         'samples'    : files_MuGun_Displaced, #  FILE
                         'treename'    : "simOmtfPhase2Digis/OMTFHitsTree", # Is prompt?
                         'color'     : 'black', # Plotting color
                         'label'     : "Muon Gun (Displaced)", # Legend label
                         'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                            }
                        },
    'MuGun_FullEta_OneOverPt_1to100' : {'name'       : 'MuGun_FullEta_OneOverPt_1to100', # Name of the dataset
                                    'samples'    : files_MuGun_FullEta_OneOverPt_1to100, #  FILE 
                                    'treename'    : "simOmtfPhase2Digis/OMTFHitsTree", # Is prompt?
                                    'color'     : 'black', # Plotting color
                                    'label'     : "Muon Gun (OneOverPt_1to100)", # Legend label
                                    'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                        }
                                },
}


''''MuGun_FullEta_FlatPt1to1000' : {'name'       : 'MuGun_FullEta_FlatPt1to1000', # Name of the dataset
                                    'samples'    : files_MuGun_FullEta_FlatPt1to1000, #  FILE
                                    'treename'    : "simOmtfPhase2Digis/OMTFHitsTree", # Is prompt?
                                    'color'     : 'black', # Plotting color
                                    'label'     : "Muon Gun (FlatPt1to1000)", # Legend label
                                    'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                        }   
                                },
     '''
'''    'test' : {'name'       : 'test', # Name of the dataset
                                    'samples'    : files_test[0], #  FILE
                                    'treename'    : files_test[1], # Is prompt?
                                    'color'     : 'black', # Plotting color
                                    'label'     : "test", # Legend label
                                    'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                     } 
                                    },
'''
