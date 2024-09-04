 ###Format:   ([List of files], IsData?)
phedexPath = "./../../data"


files_SingleMu_Prompt_OneOverPt = (phedexPath + "/Dumper_l1omtf_001_july24.root","simOmtfPhase2Digis/OMTFHitsTree")
files_test = ("./data/TestEvents__t22__Patterns_ExtraplMB1nadMB2DTQualAndEtaFixedP_ValueP1Scale_t20_v1_SingleMu_iPt_and_OneOverPt_classProb17_recalib2_minDP0.root", "simOmtfPhase2Digis/OMTFHitsTree")
datasets = {
    'SingleMu_Prompt_OneOverPt' : {'name'       : 'SingleMu_Prompt_OneOverPt', # Name of the dataset
                                  'samples'    : files_SingleMu_Prompt_OneOverPt[0], #  FILE
                                  'treename'    : files_SingleMu_Prompt_OneOverPt[1], # Is prompt?
                                  'color'     : 'black', # Plotting color
                                  'label'     : "Muon Gun (Prompt)", # Legend label
                                  'selection'  : {'acceptance' : '(muonPropEta!=0) & (muonPropPhi!=0)',  #Only take muons inside acceptance
                                 } 
                                },
}