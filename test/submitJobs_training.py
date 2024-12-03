#!/usr/bin/env python
import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_241106/3neighbours_muonQOverPt/" # list with all the file directories
queue = "workday" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Model_v240725_Bsize64_lr5e-4_NOnormNodes_GAT_241106/"
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/TrainingModel/"
ModelTypes = ['SAGE', 'MPNN']
NormalizationTypes = ['DropLastTwoNodeFeatures', 'NodesAndEdgesAndOnlySpatial']
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
file_count = 1
for model in ModelTypes:
    for normalization in NormalizationTypes: 
        SaveTag = model + "_" + normalization + "_Bsize1024_lr1e-3_241203"
        with open('%s/exec/job_train_model_%02d.sh' %(WORKDIR, file_count), 'w') as fout:
            fout.write("#!/bin/sh\n")
            fout.write("echo\n")
            fout.write("echo\n")
            fout.write("echo 'START---------------'\n")
            fout.write("echo 'WORKDIR ' ${PWD}\n")
            fout.write("cd "+str(path)+"\n")
            fout.write("source pyenv/bin/activate\n")
            fout.write("echo 'Saving Model in  %s' \n" %(OutputDir))
            fout.write("python tools/training/TrainModelFromGraph.py --model_type %s --hidden_dim 32 --normalization %s --graph_path %s --out_path %s --do_train --save_tag %s --batch_size 1024 --learning_rate 0.001\n" %(model, normalization, InputFolder, OutputDir, SaveTag))  
            fout.write("echo 'STOP---------------'\n")
            fout.write("echo\n")
            fout.write("echo\n")
        os.system("chmod 755 %s/exec/job_train_model_%02d.sh" %(WORKDIR, file_count))

###### create submit.sub file ####
with open('submit.sub', 'w') as fout:
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
os.system("echo submit.sub")
#os.system("condor_submit submit.sub")

print()
print("your jobs:")
os.system("condor_q")
print()
print('END')
print()
