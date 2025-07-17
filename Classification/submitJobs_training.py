#!/usr/bin/env python
import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250514_250625/MuGun_Displaced/" # list with all the file directories
queue = "workday" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw
OutputDir = "/eos/user/e/eallergu/L1TMuon/INTREPID/Classification/Model_Graphs_v250514_250625/Try"
WORKDIR = "/afs/cern.ch/user/e/eallergu/workdir/INTREPID/tmp/TrainingClassification"
ModelTypes = ['EdgeClassifier']
NormalizationTypes = ['NodesAndEdgesAndLayerInfo']
InputGraphs = [""] #"3neighbours_muonQOverPt/", "all_connections_muonQOverPt/"]
GraphName = "OmtfDataset_Jun17_classification" 

NumFiles = 20
Epochs = 50
HiddenDim = 64
BatchSize = 1024
EdgeAttr = "Yes"
########   customization end   #########

path = os.getcwd()
print('do not worry about folder creation:\n')
os.system("rm -rf %s" %(WORKDIR))
os.system("mkdir %s" %(WORKDIR))
os.system("mkdir %s/exec" %(WORKDIR))
os.system("mkdir %s/batchlogs" %(WORKDIR))

if not os.path.exists(OutputDir):
    print("OutputDir %s does not exist" %(OutputDir))
    os.system("mkdir %s" %(OutputDir))
else :
    print("Warning: OutputDir already exists. It will be overwritten\n")
    print("OutputDir: %s" %(OutputDir))


## print info
print("InputFolder: %s" %(InputFolder))
print("OutputDir: %s" %(OutputDir))

##### creating job #####
file_count = 0
for model in ModelTypes:
    for normalization in NormalizationTypes: 
        for input_graph in InputGraphs:
            file_count += 1
            print("Creating job for model %s with normalization %s and input graphs %s" %(model, normalization, input_graph))
            SaveTag = model + "_" + normalization + "_Bsize64_lr5e-4_20files_"
            if "all" in input_graph:
                SaveTag = SaveTag + "allConnections"
            else:
                SaveTag = SaveTag + "3neighbours"
    
            with open('%s/exec/job_train_model_%02d.sh' %(WORKDIR, file_count), 'w') as fout:
                fout.write("#!/bin/sh\n")
                fout.write("echo\n")
                fout.write("echo\n")
                fout.write("echo 'START---------------'\n")
                fout.write("echo 'WORKDIR ' ${PWD}\n")
                fout.write("cd "+str(path)+"\n")
                fout.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
                fout.write("echo 'Saving Model in  %s' \n" %(OutputDir))
                fout.write("python3 TrainModelFromGraph.py --model_type %s --hidden_dim %d --normalization %s --graph_path %s --out_model_path %s --do_train --save_tag %s --batch_size %d --learning_rate 0.001 --num_files %d --graph_name %s --epochs %d --edge_attr %s\n" %(model, HiddenDim, normalization, InputFolder+input_graph, OutputDir, SaveTag, BatchSize, NumFiles, GraphName, Epochs, EdgeAttr))  
                fout.write("echo 'STOP---------------'\n")
                fout.write("echo\n")
                fout.write("echo\n")
            os.system("chmod 755 %s/exec/job_train_model_%02d.sh" %(WORKDIR, file_count))

###### create submit.sub file ####
with open('%s/submit.sub' %(WORKDIR), 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(ClusterId)$(ProcId)\n")
    fout.write("output                  = %s/batchlogs/$(ClusterId).$(ProcId).out\n" %(WORKDIR))
    fout.write("error                   = %s/batchlogs/$(ClusterId).$(ProcId).err\n"    %(WORKDIR))
    fout.write("log                     = %s/batchlogs/$(ClusterId).log\n"             %(WORKDIR))
    fout.write("request_gpus            = 1\n")
    fout.write('+JobFlavour = "%s"\n' %(queue))
    fout.write("\n")
    fout.write("queue filename matching (%s/exec/job_*sh)\n" %(WORKDIR))

###### sends bjobs ######
print()
print("to submit all jobs do: ")
print("....................................................................")
print("cd %s" %(WORKDIR))
print("cat submit.sub")
print("condor_submit submit.sub")
print("cd -")

print()
print("### CHECK your jobs:")
print("condor_q")
print()
print("....................................................................")
print('END')
print()
