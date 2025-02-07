# Trigger with machine learning

## Installation

To generate the proper environment: 
```
virtualenv pyenv  --python=3.11
source <name_of_venv>/bin/activate
pip install .
```
If you have python 3.11 already installed in your system, you can also run `python3.11 -m venv pyenv` instead of the `virtualenv` command above. The list of dependencies can be seen in requirements.txt and installed by runing: `pip install -r requirements.txt`. In Lxplus, this line seems to have all the required dependencies: 
```
source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh
```

## Graph / Dataset creation (please ignore it for now)

An example of its utilization can be seen in `test/test_omtf_dataset.py` or you can also run it interactively: 

```
python tools/training/OMTFDataset.py --root_dir /eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v240725/Dumper_l1omtf_001.root --tree_name simOmtfPhase2Digis/OMTFHitsTree --muon_vars "[muonQOverPt]" --stub_vars "['stubEtaG', 'stubPhiG', 'stubR', 'stubLayer', 'stubType']" --save_path /eos/cms/store/user/folguera/L1TMuon/INTREPID/GraphsDataset_v240725_250128//OmtfDataset_Jan27_muonQOverPt_001.pt
```

### Using lxplus batch system
```
python test/submitJobs_DatasetCreation.py 
```
## Training

An example of use can be seen in `test/test_omtf_training.py` or you can also run it interactively: 

```
python tools/training/TrainModelFromGraph.py --model_type SAGE --hidden_dim 32 --normalization NodesAndEdgesAndOnlySpatial --graph_path /eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_250115/ --out_model_path ./Test_Model_wGraphs_v240725_250115/ --do_train --save_tag SAGE_NodesAndEdgesAndOnlySpatial_Bsize64_lr5e-4_241106_20files_3neighbours --batch_size 1024 --learning_rate 0.001 --num_files 20 --graph_name vix_graph_6Nov_all_muonQOverPt --epochs 10
```
### Using lxplus batch system

This script, ``test/submitJobs_training.py`` needs some pre-configuration on the types of models, input and output folder and graph names. 

```########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_250115/" # list with all the file directories
queue = "workday"  (8h)  # Other options can be checked in: https://batchdocs.web.cern.ch/local/submit.html#job-flavours
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Model_Graphsv240725_QOverPtRegression_250115/"
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/TrainingModel/"
ModelTypes = ['GCN', 'SAGE', 'MPNN']
NormalizationTypes = ['NodesAndEdgesAndOnlySpatial']
InputGraphs = [""] #"3neighbours_muonQOverPt/", "all_connections_muonQOverPt/"]
GraphName = "vix_graph_6Nov_all_muonQOverPt" 

Epochs = 50
########   customization end   #########
```

Once configured, it will automatically generate a tmp workdir to submit all the jobs to the batch system: 

``
python test/submitJobs_training.py 
``