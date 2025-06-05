#!/usr/bin/env python
import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250514/" 
Datasets = ["HTo2LongLivedTo2mu2jets","MuGun_Displaced","MuGun_FullEta_OneOverPt_1to100"] 
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250514_250530/"

queue = "longlunch" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw

ConfigFile = ["configs/dataset_classification.yml","configs/dataset_regression.yml"]
Tasks = ['classification','regression']
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/DatasetCreation/"
GraphFileName = "OmtfDataset_May30"
########   customization end   #########

path = os.getcwd()
print('do not worry about folder creation:\n')
if os.path.exists(WORKDIR):
    os.system(f"rm -rf {WORKDIR}")
os.makedirs(WORKDIR)
os.makedirs(os.path.join(WORKDIR, "exec"))
os.makedirs(os.path.join(WORKDIR, "batchlogs"))

if not os.path.exists(OutputDir):
    print(f"OutputDir {OutputDir} does not exist")
    os.makedirs(OutputDir)
    for dataset in Datasets:
        os.makedirs(os.path.join(OutputDir, dataset))
        print(f"OutputDir: {OutputDir}/{dataset}")
    print("OutputDir created\n")
elif os.path.exists(OutputDir) and not os.path.exists(os.path.join(OutputDir, Datasets[0])):
    print(f"OutputDir {OutputDir} exists but not the dataset folder")
    for dataset in Datasets:
        os.makedirs(os.path.join(OutputDir, dataset))
        print(f"OutputDir: {OutputDir}/{dataset}")
    print("OutputDir created\n")
else:
    print("Warning: OutputDir already exists. It will be overwritten\n")
    print(f"OutputDir: {OutputDir}")

##### loop for creating and sending jobs #####
for dataset in Datasets:
    i = 1
    ## create list of files
    idir = InputFolder+dataset+"/"
    odir = OutputDir+dataset+"/"
    if not os.path.exists(idir):
        print("InputFolder %s does not exist" %(idir))
        sys.exit()
    list_of_files = os.listdir(idir)

    ## print info
    print("InputFolder: %s" %(idir))
    print("OutputDir: %s" %(odir))
    print("Number of files: %d" %(len(list_of_files)))

    for ifile in list_of_files:
        ##### creates jobs #######
        with open('%s/exec/job_%s_%03d.sh' %(WORKDIR,dataset,i), 'w') as fout:
            fout.write("#!/bin/sh\n")
            fout.write("echo\n")
            fout.write("echo\n")
            fout.write("echo 'START---------------'\n")
            fout.write("echo 'WORKDIR ' ${PWD}\n")
            fout.write("cd "+str(path)+"\n")
            fout.write("source %s/pyenv/bin/activate\n" %(path))
            for idx, task in enumerate(Tasks):
                config_file = os.path.join(path, ConfigFile[idx])
                fout.write("echo 'Running With Task: %s' \n" %(task))
                output_graph_name = "%s/%s_%s_%03d.pt" %(odir, GraphFileName, task, i)
                fout.write("echo 'Saving graphs in %s' \n" %(output_graph_name))
                fout.write("python tools/training/OMTFDataset.py --root_dir %s --config %s --save_path %s --task %s \n" %(idir+ifile, config_file, output_graph_name, task))  
            fout.write("echo 'STOP---------------'\n")
            fout.write("echo\n")
            fout.write("echo\n")
        os.system("chmod 755 %s/exec/job_%s_%03d.sh" %(WORKDIR,dataset,i))
        i+=1

###### create submit.sub file ####
with open('%s/submit.sub' %(WORKDIR), 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(ClusterId)$(ProcId)\n")
    fout.write("output                  = %s/batchlogs/$(ClusterId).$(ProcId).out\n" %(WORKDIR))
    fout.write("error                   = %s/batchlogs/$(ClusterId).$(ProcId).err\n"    %(WORKDIR))
    fout.write("log                     = %s/batchlogs/$(ClusterId).log\n"             %(WORKDIR))
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
