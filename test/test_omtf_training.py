import os, sys

import matplotlib.pyplot as plt

## Create the dataset with OMTF dataset
sys.path.append(os.path.join(os.getcwd(), '.', 'tools', 'training'))

from TrainModelFromGraph import TrainModelFromGraph
from validation import plot_graph_features, plot_prediction_results, evaluate_model
from models import GraphSAGEModel

config_file = "./configs/training_classification.yml"
graph_path = "/eos/cms/store/user/folguera/L1TMuon/INTREPID/Graphs_v250514_250530/HTo2LongLivedTo2mu2jets/"
output_dir = "./test_HTo2LongLivedTo2mu2jet/"
save_tag = "SAGE_NodesAndEdgesAndOnlySpatial_Bsize64_lr5e-4_250603_allConnections"
graph_name = "OmtfDataset_May30_classification"

trainer = TrainModelFromGraph(config=config_file, graph_path=graph_path, out_model_path=output_dir, save_tag=save_tag, graph_name=graph_name, plot_graph_features=True,task='classification')

print("Using model type:", trainer.model_type)
print("Loading model...")
trainer.load_data()
print("Data loaded. Initializing model...")
trainer.initialize_model()
print("Model initialized. Preparing for training...")

## Now for validation of the graph features: 
from validation import plot_graph_features
plot_graph_features(trainer.train_loader, output_dir=output_dir,label=trainer.save_tag)

## Now for training the model
trainer.Training_loop()

## Once it is trained need some validation: 
trainer.load_trained_model()
from validation import plot_prediction_results, evaluate_model
regression,prediction = evaluate_model(trainer.model, trainer.test_loader, trainer.device)
plot_prediction_results(regression, prediction, output_dir=output_dir,model=trainer.model_type, label=trainer.save_tag)


