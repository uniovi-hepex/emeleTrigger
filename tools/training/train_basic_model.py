# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
# ---

# %matplotlib inline

# +
import matplotlib.pyplot as plt

from OMTFGraphNetwork import OMTFGraphNetwork

# Instantiate the network
g=OMTFGraphNetwork() # Optionally pass a class name. The class must inherit from nn.Module. DO NOT PASS THE CONSTRUCTOR (the constructor is called via g.instantiate_model() below)

#g.load_data('data/omtfAnalysis2.root:simOmtfPhase2Digis/OMTFHitsTree', fraction=0.05, viz=True)
#g.load_data('data/Displaced_cTau5m_XTo2LLTo4Mu_condPhase2_realistic_l1omtf_12.root:simOmtfPhase2Digis/OMTFHitsTree', fraction=1., viz=True)
#g.load_data('data/SingleMu_GT131X_Extrapolation_GhostBusterTest_FlatPt0To1000Dxy3m_NonDegraded_Stub_v2.root:simOmtfDigis/OMTFHitsTree', fraction=1., viz=True)
g.load_data('data/SingleMu_GT131X_Extrapolation_GhostBusterTest_XTo2LLP4Mu_Ctau5m_Stub.root:simOmtfDigis/OMTFHitsTree', fraction=0.01, viz=True)

        
# Train the model or load a pretrained model
dotrain=True
if not dotrain:
    g.load_model('models/model.pth')
else:
    g.instantiate_model() # pass parameters of the model here, if needed
    g.do_training(10)

# Look at output plots
g.visualize_prediction()


