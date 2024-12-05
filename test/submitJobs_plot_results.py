import os,sys

print('START\n')
########   YOU ONLY NEED TO FILL THE AREA BELOW   #########
########   customization  area #########
GraphFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v240725_241106/" # list with all the file directories
ModelFolder = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Model_Graphsv240725_QOverPtRegression_241203/"
ModelTypes = ['SAGE', 'MPNN']
NormalizationTypes = ['DropLastTwoNodeFeatures', 'NodesAndEdgesAndOnlySpatial']
InputGraphs = ["3neighbours_muonQOverPt/", "all_connections_muonQOverPt/"]
GraphName = "vix_graph_6Nov"
Epochs = 50
OutputDir = "/eos/user/f/folguera/www/INTREPID/2024_12_04_GNN_QOverPtRegression/"
JustPrint = True
########   customization end   #########


if JustPrint:
    print("##########################")
    print("source pyenv/bin/activate\n")

    if not os.path.exists(OutputDir):
        print("OutputDir %s does not exist" %(OutputDir))
        os.system("mkdir %s" %(OutputDir))
        
    for model in ModelTypes:
        for normalization in NormalizationTypes: 
            for input_graph in InputGraphs:
                SaveTag = model + "_" + normalization + "_Bsize64_lr5e-4_241106_20files_"
                if "all" in input_graph:
                    SaveTag = SaveTag + "allConnections"
                else:
                    SaveTag = SaveTag + "3neighbours"
                ModelFile = f'model_{model}_32dim_50epochs_{SaveTag}.pth'

                print("python tools/training/TrainModelFromGraph.py --model_type %s --hidden_dim 32 --normalization %s --graph_path %s --output_dir %s --do_validation --save_tag %s --batch_size 1024 --learning_rate 0.001 --num_files 5 --graph_name %s --epochs %d --model_path %s/%s &\n" %(model, normalization, GraphFolder+input_graph, OutputDir, SaveTag, GraphName, Epochs, ModelFolder,ModelFile))



    print("##########################")
    sys.exit()

### NOW SUBMIT THE JOBS
queue = "espresso"
WORKDIR = "/afs/cern.ch/user/f/folguera/workdir/INTREPID/tmp/PlotModel/"

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
            SaveTag = model + "_" + normalization + "_Bsize64_lr5e-4_241106_20files_"
            if "all" in input_graph:
                SaveTag = SaveTag + "allConnections"
            else:
                SaveTag = SaveTag + "3neighbours"
            ModelFile = f'model_{model}_32dim_50epochs_{SaveTag}.pth'

            with open('%s/exec/job_plot_model_%02d.sh' %(WORKDIR, file_count), 'w') as fout:
                fout.write("#!/bin/sh\n")
                fout.write("echo\n")
                fout.write("echo\n")
                fout.write("echo 'START---------------'\n")
                fout.write("echo 'WORKDIR ' ${PWD}\n")
                fout.write("cd "+str(path)+"\n")
                fout.write("source pyenv/bin/activate\n")
                fout.write("echo 'Saving Model in  %s' \n" %(OutputDir))
                fout.write("python tools/training/TrainModelFromGraph.py --model_type %s --hidden_dim 32 --normalization %s --graph_path %s --output_dir %s --plot_graph_features --do_validation --save_tag %s --batch_size 1024 --learning_rate 0.001 --num_files 5 --graph_name %s --epochs %d --model_path %s/%s\n" %(model, normalization, GraphFolder+input_graph, OutputDir, SaveTag, GraphName, Epochs, ModelFolder,ModelFile))
                fout.write("echo 'STOP---------------'\n")
                fout.write("echo\n")
                fout.write("echo\n")
            os.system("chmod 755 %s/exec/job_plot_model_%02d.sh" %(WORKDIR, file_count))

###### create submit.sub file ####
with open('submit.sub', 'w') as fout:
    fout.write("executable              = $(filename)\n")
    fout.write("arguments               = $(ClusterId)$(ProcId)\n")
    fout.write("output                  = %s/batchlogs/$(ClusterId).$(ProcId).out\n" %(WORKDIR))
    fout.write("error                   = %s/batchlogs/$(ClusterId).$(ProcId).err\n"    %(WORKDIR))
    fout.write("log                     = %s/batchlogs/$(ClusterId).log\n"             %(WORKDIR))
    fout.write('+JobFlavour = "%s"\n' %(queue))
    fout.write("\n")
    fout.write("queue filename matching (%s/exec/job_*sh)\n" %(WORKDIR))

###### sends bjobs ######
os.system("echo submit.sub")
os.system("condor_submit submit.sub")

print()
print("your jobs:")
os.system("condor_q")
print()
print('END')
print()
