#!/usr/bin/env python
import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
InputFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Dumper_Ntuples_v250312/SingleMu_FlatPt1to1000_FullEta_Apr04_125X/" # list with all the file directories
queue = "microcentury" # give bsub queue -- 8nm (8 minutes), 1nh (1 hour), 8nh, 1nd (1day), 2nd, 1nw (1 week), 2nw
OutputDir = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250312_250314/SingleMu_FlatPt1to1000/"
MuonVars = ["muonQOverPt", "muonQPt"]
StubVars = ["stubEtaG", "stubPhiG", "stubR", "stubLayer", "stubType"]
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/DatasetCreation/"
GraphFileName = "OmtfDataset_Mar14"
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
else:
    print("Warning: OutputDir already exists. It will be overwritten\n")
    print(f"OutputDir: {OutputDir}")

## create list of files
if not os.path.exists(InputFolder):
    print("InputFolder %s does not exist" %(InputFolder))
    sys.exit()
list_of_files = os.listdir(InputFolder)

## print info
print("InputFolder: %s" %(InputFolder))
print("OutputDir: %s" %(OutputDir))
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
        fout.write("source %s/pyenv/bin/activate\n" %(path))
        for muvars in MuonVars:
            fout.write("echo 'Running With MuonVar: %s' \n" %(muvars))
            output_graph_name = "%s/%s_%s_%03d.pt" %(OutputDir, GraphFileName, muvars, i)
            fout.write("echo 'Saving graphs in %s' \n" %(output_graph_name))
            fout.write("python tools/training/OMTFDataset.py --root_dir %s --tree_name simOmtfPhase2Digis/OMTFHitsTree --muon_vars [%s] --stub_vars %s --save_path %s \n" %(InputFolder+ifile, muvars, StubVars, output_graph_name))  
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
