import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn as nn
import uproot
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import time
import os, sys

import argparse
import matplotlib.pyplot as plt
from models import GATRegressor,GATv2Regressor

class TrainModelFromGraph:
    def __init__(self, **kwargs):
        self.graph_path = kwargs.get('graph_path', 'graph_folder')
        self.out_path = kwargs.get('out_path', 'Bsize_gmp_64_lr5e-4_v3')
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 0.0005)
        self.epochs = kwargs.get('epochs', 1000)
        self.model_path = kwargs.get('model_path', None)
        self.output_dir = kwargs.get('output_dir', None)
        self.train = kwargs.get('train', False)
        self.evaluate = kwargs.get('evaluate', False)
        self.model = kwargs.get('model', 'GAT')
        self.normalize_features = kwargs.get('normalize_features', False)
        self.normalize_targets = kwargs.get('normalize_targets', False)
        self.normalize_edge_features = kwargs.get('normalize_edge_features', False)
        
        # Inicializa otros atributos y carga los datos aquí
        self.train_loader = None
        self.test_loader = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss()
    
    def load_data(self):
        # Loading data from graph and convert it to DataLoader
        Allgraphs = []
        all_files = os.listdir(self.graph_path)

        # Filter for .pkl files
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        print(f"Using files: {pkl_files}")
        if not pkl_files:
            print("No .pkl files found in the directory.")
            return []
        for pkl_file in pkl_files:
            file_path = os.path.join(self.graph_path, pkl_file)
            with open(file_path, 'rb') as file:
                graphfile = torch.load(file)
                Allgraphs.append(graphfile)

        Graphs_for_training = sum(Allgraphs, [])
        Graphs_for_training_reduced = Graphs_for_training
        # remove extra dimenson in y
        print(f"Total Graphs: {len(Graphs_for_training)}")
        for i in range(0, len(Graphs_for_training)):
            Graphs_for_training_reduced[i].y = Graphs_for_training[i].y.mean(dim=0)

        # Filter out graphs with no nodes
        Graphs_for_training_filtered = [g for g in Graphs_for_training_reduced if g.edge_index.size(1) > 0]  # remove empty graphs

        # Train and test split:
        events = len(Graphs_for_training_filtered)
        ntrain = int((events * 0.7) / self.BatchSize) * self.BatchSize  # to have full batches
        print(f"Training events: {ntrain}")

        # put deltaPhi and deltaEta in the data object as edge_attr
        for i in range(0, len(Graphs_for_training_filtered)):
            Graphs_for_training_filtered[i].edge_attr = torch.stack([Graphs_for_training_filtered[i].deltaPhi.float(), Graphs_for_training_filtered[i].deltaEta.float()], dim=1)

        train_dataset = Graphs_for_training_reduced[:ntrain]
        test_dataset = Graphs_for_training_reduced[ntrain:ntrain * 2]

        print("====================================")
        print("Example of data:")
        print(train_dataset[0].x)
        print(train_dataset[0].edge_index)
        print(train_dataset[0].edge_attr)
        print(train_dataset[0].deltaPhi)
        print(train_dataset[0].deltaEta)
        print(train_dataset[0].y)
        print(train_dataset[0].batch)
        print("====================================")

        # Load data
        self.train_loader = DataLoader(train_dataset, batch_size=self.BatchSize, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.BatchSize, shuffle=False)
        
    def plot_graph_features(self, data_loader):
        feature_names = ["eta", "phi", "R", "layer", "Type"]
        for batch in data_loader:
            features = batch.x.numpy()
            regression = batch.y.numpy()
            deltaphi = batch.deltaPhi.numpy() 
            deltaeta = batch.deltaEta.numpy()
            num_features = features.shape[1]

            fig, axs = plt.subplots(3, 3, figsize=(15, 15))
            axs = axs.flatten()
                
            # Plot node features
            for i in range(num_features):
                nbins = 18 if i==3 else 30
                axs[i].hist(features[:, i], bins=nbins, alpha=0.75)
                axs[i].set_title(f'Feature {feature_names[i]} Histogram')
                axs[i].set_xlabel(f'Feature {feature_names[i]} Value')
                axs[i].set_ylabel('Frequency')
                
            #calculate the average number of edges, dividing by the number of nodes
            num_edges = batch.edge_index.size(1)/batch.x.size(0)
            axs[num_features].hist(num_edges, bins=30, alpha=0.75)
            axs[num_features].set_title('Number of Edges/Node')
            axs[num_features].set_ylabel('Count')
                
            # Plot edge features
            axs[num_features + 1].hist(deltaphi, bins=30, alpha=0.75)
            axs[num_features + 1].set_title(f'Edge Feature deltaPhi Histogram')
            axs[num_features + 1].set_xlabel(f'Edge Feature deltaPhi Value')
            axs[num_features + 1].set_ylabel('Frequency')
                
            axs[num_features + 2].hist(deltaeta, bins=30, alpha=0.75)
            axs[num_features + 2].set_title(f'Edge Feature deltaEta Histogram')
            axs[num_features + 2].set_xlabel(f'Edge Feature deltaEta Value')
            axs[num_features + 2].set_ylabel('Frequency')

            # Plot regression target
            axs[num_features + 3].hist(regression, bins=30, alpha=0.75)
            axs[num_features + 3].set_title('Regression Target Histogram')
            axs[num_features + 3].set_xlabel('Regression Target Value')
            axs[num_features + 3].set_ylabel('Frequency')
                
            plt.tight_layout()
            plt.show()
            break  # Only draw the first batch

    def initialize_model(self):
        num_node_features = 5
        num_edge_features = 2
        hidden_dim = self.BatchSize
        output_dim = 3
        print(f"Using device: {self.device}")
        if self.model == 'GAT':
            self.model = GATRegressor(num_node_features, hidden_dim, output_dim).to(self.device)
        elif self.model == 'GATv2':
            self.model = GATv2Regressor(num_node_features, hidden_dim, output_dim).to(self.device)
        elif self.model == 'GATwithDropout':
            self.model = GATRegressorDO(num_node_features, hidden_dim, output_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LearningRate, weight_decay=0.75)
        print("Model initialized")
        print(self.model)

    def train(self):
        self.model.train()
        total_loss = 0
        i = 0
        for data in self.train_loader:
            data = data.to(self.device)  # Mueve los datos al dispositivo
            self.optimizer.zero_grad()
            out = self.model(data)
            # loss = self.loss_fn(out, data.y.reshape(self.BatchSize,3))
            loss = self.loss_fn(out, data.y.view(out.size()))
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss)

        return total_loss / len(self.train_loader.dataset)

    def test(self):
        with torch.no_grad():
            self.model.eval()
            total_loss = 0
            for data in self.test_loader:
                data = data.to(self.device)
                out = self.model(data)
                # loss = self.loss_fn(out, data.y.reshape(self.BatchSize,3))
                loss = self.loss_fn(out, data.y.view(out.size()))
                total_loss += float(loss)
        return total_loss / len(self.test_loader.dataset)

    def Training_loop(self):
        train_losses = []
        test_losses = []
        print("Start training...")
        for epoch in range(self.Epochs):
            train_loss = self.train()
            test_loss = self.test()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            path = self.out_path
            if not os.path.exists(path):
                os.makedirs(path)
            if epoch == 0:
                torch.save(test_loss, f"{path}/testloss_{epoch + 1}.pt")
                torch.save(train_loss, f"{path}/trainloss_{epoch + 1}.pt")
            elif (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1:02d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
                torch.save(self.model, f"{path}/model_{epoch + 1}.pth")
                torch.save(test_loss, f"{path}/testloss_{epoch + 1}.pt")
                torch.save(train_loss, f"{path}/trainloss_{epoch + 1}.pt")

                plt.plot(train_losses, "b", label="Train loss")
                plt.plot(test_losses, "k", label="Test loss")
                plt.yscale('log')
                plt.savefig(f"{path}/loss_plot.png")
    


'''class PlotRegresson:
    def __init__(self, model, test_loader, batch_size):
        self.model = model
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pt_pred_arr = []
        self.eta_pred_arr = []
        self.phi_pred_arr = []
        self.pt_truth_arr = []
        self.eta_truth_arr = []
        self.phi_truth_arr = []

    def evaluate(self):
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                batch = data.batch
                pred = self.model(data).reshape(self.batch_size, 3)

                for item in range(0, self.batch_size):
                    vector_pred = pred[item]
                    vector_real = data[item].y

                    self.pt_pred_arr.append(vector_pred[0].item())
                    self.eta_pred_arr.append(vector_pred[1].item())
                    self.phi_pred_arr.append(vector_pred[2].item())

                    self.pt_truth_arr.append(vector_real[0].item())
                    self.eta_truth_arr.append(vector_real[1].item())
                    self.phi_truth_arr.append(vector_real[2].item())

    def plot_regression(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.clf()
        plt.hist(self.pt_truth_arr, bins=100, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.pt_pred_arr, bins=100, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression.png"))
        plt.clf()

        plt.hist(self.eta_truth_arr, bins=50, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.eta_pred_arr, bins=50, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "eta_regression.png"))
        plt.clf()

        plt.hist(self.phi_truth_arr, bins=50, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.phi_pred_arr, bins=50, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "phi_regression.png"))
        plt.clf()'''

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate GAT model")
    parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
    parser.add_argument('--out_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default='GAT', help='Model to use for training')
    parser.add_argument('--plot_graph_features', action='store_true', help='Plot the graph features')
    parser.add_argument('--normalize_features', action='store_true', help='Normalize the input features')
    parser.add_argument('--normalize_targets', action='store_true ', help='Normalize the target values')
    parser.add_argument('--normalize_edge_features', action='store_true', help='Normalize the edge features')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output directory for evaluation results')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    # For training:
    trainer = TrainModelFromGraph(**vars(args))
    trainer.load_data()
    if args.plot_graph_features: 
        trainer.plot_graph_features(trainer.train_loader)
    trainer.initialize_model()

    if args.train:
        trainer.Training_loop()

    # For evaluating:
'''    if args.evaluate:    
        trainer.load_data()
        test_loader = trainer.test_loader
        model = torch.load(args.model_path)
            
        evaluator = PlotRegresson(model, test_loader, batch_size=args.batch_size)
        evaluator.evaluate()
        evaluator.plot_regression(output_dir=args.output_dir)'''

if __name__ == "__main__":
    main()