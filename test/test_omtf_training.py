import os, sys

import matplotlib.pyplot as plt

## Create the dataset with OMTF dataset
sys.path.append(os.path.join(os.getcwd(), '.', 'tools', 'training'))

from TrainModelFromGraph import TrainModelFromGraph
from validation import plot_graph_feature_histograms, plot_prediction_results, evaluate_model
from models import GraphSAGEModel

model_type = "SAGE"
hidden_dim = 32
normalization = "NodesAndEdgesAndOnlySpatial"
graph_path = "/eos/user/a/acardini/INTREPID/Graphs_v250514/MuGun_Displaced/"
output_dir = "./test_out_model/"

trainer = TrainModelFromGraph(model_type=model_type, hidden_dim=hidden_dim, normalization=normalization, graph_path=graph_path, out_model_path="./test_out_model/", save_tag="SAGE_NodesAndEdgesAndOnlySpatial_Bsize64_lr5e-4_250115_allConnections", batch_size=1024, learning_rate=0.001, num_files=20, graph_name="OmtfDataset_Apr23_muonQPt", epochs=50)

trainer.load_data()
trainer.initialize_model()

## Now for validation of the graph features: 
from validation import plot_graph_feature_histograms
plot_graph_feature_histograms(trainer.train_loader, output_dir=output_dir,label=trainer.save_tag)

## Now for training the model
trainer.Training_loop()

## Once it is trained need some validation: 
trainer.load_trained_model()
from validation import plot_prediction_results, evaluate_model
regression,prediction = evaluate_model(trainer.model, trainer.test_loader, trainer.device)
plot_prediction_results(regression, prediction, output_dir=output_dir,model=trainer.model_type, label=trainer.save_tag)


