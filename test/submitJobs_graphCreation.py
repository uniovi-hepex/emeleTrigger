#!/usr/bin/env python
import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v240725/" # list with all the file directories
queue = "microcentury" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_250115/"
Connectivity = ["all"]
MuonVars = ["muonQOverPt"]
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/GraphCreation/"
GraphFileName = "vix_graph_6Nov"
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

## create list of files
if not os.path.exists(InputFolder):
    print("InputFolder does not exist")
    sys.exit()
list_of_files = os.listdir(InputFolder)

## print info
print("InputFolder: %s" %(InputFolder))
print("OutputDir: %s" %(OutputDir))
print("Connectivity: %s" %(Connectivity))
print("Number of files: %d" %(len(list_of_files)))

##### loop for creating and sending jobs #####
i=1
for ifile in list_of_files:
    ##### creates jobs #######
    with open('%s/exec/job_%03d.sh' %(WORKDIR,i), 'w') as fout:
        fout.write("#!/bin/sh\n")
        fout.write("echo\n")
        fout.write("echo\n")
        fout.write("echo 'START---------------'\n")
        fout.write("echo 'WORKDIR ' ${PWD}\n")
        fout.write("cd "+str(path)+"\n")
        fout.write("source pyenv/bin/activate\n")
        for connection in Connectivity:
            fout.write("echo 'Running Connectivity: %s' \n" %(connection))
            for muvars in MuonVars:
                fout.write("echo 'Running With MuonVar: %s' \n" %(muvars))
                output_graph_name = "%s/%s_%s_%s_%03d.pt" %(OutputDir, GraphFileName, connection, muvars, i)
                fout.write("echo 'Saving graphs in %s' \n" %(output_graph_name))
                fout.write("python tools/training/GraphCreationModel.py --data_path %s:simOmtfPhase2Digis/OMTFHitsTree --muon_vars %s --graph_save_paths %s --model_connectivity %s\n" %(InputFolder+ifile, muvars, output_graph_name, connection))  
        fout.write("echo 'STOP---------------'\n")
        fout.write("echo\n")
        fout.write("echo\n")
    os.system("chmod 755 %s/exec/job_%03d.sh" %(WORKDIR,i))
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
os.system("cd %s" %(WORKDIR))
os.system("cat submit.sub")
os.system("condor_submit submit.sub")
os.system("cd -")

print()
print("your jobs:")
os.system("condor_q")
print()
print('END')
print()
