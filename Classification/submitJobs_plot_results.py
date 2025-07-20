import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
GraphFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250514_250625/MuGun_Displaced/" # list with all the file directories
ModelFolder = "/eos/user/e/eallergu/L1TMuon/INTREPID/Classification/Model_Graphs_v250514_250625/Try"
ModelTypes = ['EdgeClassifier']
NormalizationTypes = ['NodesAndEdgesAndLayerInfo']
InputGraphs = [""]
GraphName = "OmtfDataset_Jun17_classification"

NumFiles = 20
Epochs = 50
HiddenDim = 64
BatchSize = 1024

OutputDir = "/eos/user/e/eallergu/L1TMuon/INTREPID/Classification/2025_07_16_GNN_Classification/Try/"
JustPrint = False

queue = "workday"
WORKDIR = "/afs/cern.ch/user/e/eallergu/workdir/INTREPID/tmp/PlotClassification"

AllMetrics = "AllMetrics"
########   customization end   #########


if JustPrint:
    print("##########################")
    print("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")

    if not os.path.exists(OutputDir):
        print("OutputDir %s does not exist" %(OutputDir))
        os.system("mkdir %s" %(OutputDir))
        
    for model in ModelTypes:
        for normalization in NormalizationTypes: 
            for input_graph in InputGraphs:
                SaveTag = model + "_" + normalization + "_Bsize64_lr5e-4_20files_"
                if "all" in input_graph:
                    SaveTag = SaveTag + "allConnections"
                else:
                    SaveTag = SaveTag + "3neighbours"
                ModelFile = f'model_{model}_{HiddenDim}dim_{Epochs}epochs_{SaveTag}.pth'

                print("python3 TrainModelFromGraph.py --model_type %s --hidden_dim %d --normalization %s --graph_path %s --output_dir %s --do_validation --save_tag %s --batch_size %d --learning_rate 0.001 --num_files %d --graph_name %s --epochs %d --model_path %s/%s --all_metrics %s&\n" %(model, HiddenDim, normalization, GraphFolder+input_graph, OutputDir, SaveTag, BatchSize, NumFiles, GraphName, Epochs, ModelFolder,ModelFile, AllMetrics))



    print("##########################")
    sys.exit()

### NOW SUBMIT THE JOBS

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


file_count = 0
for model in ModelTypes:
    for normalization in NormalizationTypes: 
        for input_graph in InputGraphs:
            file_count += 1
            SaveTag = model + "_" + normalization + "_Bsize64_lr5e-4_20files_"
            if "all" in input_graph:
                SaveTag = SaveTag + "allConnections"
            else:
                SaveTag = SaveTag + "3neighbours"
            ModelFile = f'model_{model}_{HiddenDim}dim_{Epochs}epochs_{SaveTag}.pth'

            with open('%s/exec/job_plot_model_%02d.sh' %(WORKDIR, file_count), 'w') as fout:
                fout.write("#!/bin/sh\n")
                fout.write("echo\n")
                fout.write("echo\n")
                fout.write("echo 'START---------------'\n")
                fout.write("echo 'WORKDIR ' ${PWD}\n")
                fout.write("cd "+str(path)+"\n")
                fout.write("source /cvmfs/sft.cern.ch/lcg/views/LCG_106_cuda/x86_64-el9-gcc11-opt/setup.sh\n")
                fout.write("echo 'Saving Model in  %s' \n" %(OutputDir))
                fout.write("python3 TrainModelFromGraph.py --model_type %s --hidden_dim %d --normalization %s --graph_path %s --output_dir %s --plot_graph_features --do_validation --save_tag %s --batch_size %d --learning_rate 0.001 --num_files %d --graph_name %s --epochs %d --model_path %s/%s --metrics %s\n" %(model, HiddenDim, normalization, GraphFolder+input_graph, OutputDir, SaveTag, BatchSize, NumFiles, GraphName, Epochs, ModelFolder,ModelFile, AllMetrics))
                fout.write("echo 'STOP---------------'\n")
                fout.write("echo\n")
                fout.write("echo\n")
            os.system("chmod 755 %s/exec/job_plot_model_%02d.sh" %(WORKDIR, file_count))

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
