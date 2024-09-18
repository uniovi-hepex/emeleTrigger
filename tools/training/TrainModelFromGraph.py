import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.nn as nn
import uproot
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import time
import os, sys

import matplotlib.pyplot as plt


class GATRegressor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim=3):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, data):
        # load nodel attributes: x, and edge attributes: deltaPhi and deltaEta
        x, edge_index, deltaPhi, deltaEta, batch = data.x.float(), data.edge_index, data.deltaPhi.float(), data.deltaEta.float(), data.batch

        # Combine deltaPhi and deltaEta into edge_attr
        edge_attr = torch.stack([deltaPhi, deltaEta], dim=1)
        # Apply graph convolutions
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)  # Using GAT as it allow to use edge attributes
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        # Global mean pooling to get graph-level output
        # x = gmp(x, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Fully connected layers for regression
        x = self.fc1(x)
        return x


class TrainModelFromGraph:
    def __init__(self, Graph_path, Out_path, BatchSize, LearningRate, Epochs):
        self.Graph_path = Graph_path
        self.Out_path = Out_path
        self.BatchSize = BatchSize
        self.LearningRate = LearningRate
        self.Epochs = Epochs

        self.train_loader = None
        self.test_loader = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self):
        # Loading data from graph and convert it to DataLoader
        Allgraphs = []
        all_files = os.listdir(self.Graph_path)

        # Filter for .pkl files
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        print(f"Using files: {pkl_files}")
        if not pkl_files:
            print("No .pkl files found in the directory.")
            return []
        for pkl_file in pkl_files:
            file_path = os.path.join(self.Graph_path, pkl_file)
            with open(file_path, 'rb') as file:
                graphfile = torch.load(file)
                Allgraphs.append(graphfile)

        Graphs_for_training = sum(Allgraphs, [])
        Graphs_for_training_reduced = Graphs_for_training
        # remove extra dimenson in y
        print(f"Total Graphs: {len(Graphs_for_training)}")
        for i in range(0, len(Graphs_for_training)):
            Graphs_for_training_reduced[i].y = Graphs_for_training[i].y.mean(dim=0)

        # Train and test split:
        events = len(Graphs_for_training_reduced)
        ntrain = int((events * 0.7) / self.BatchSize) * self.BatchSize  # to have full batches
        print(f"Training events: {ntrain}")
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

    def initialize_model(self):
        num_node_features = 5
        num_edge_features = 2
        hidden_dim = self.BatchSize
        output_dim = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = GATRegressor(num_node_features, num_edge_features, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LearningRate, weight_decay=0.75)
        print("Model initialized")
        print(self.model)

    def train(self):
        self.model.train()
        total_loss = 0
        i = 0
        for data in self.train_loader:
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
                out = self.model(data)
                # loss = self.loss_fn(out, data.y.reshape(self.BatchSize,3))
                loss = self.loss_fn(out, data.y.view(out.size()))
                total_loss += float(loss)
        return total_loss / len(self.test_loader.dataset)

    def Training_loop(self):
        self.load_data()
        self.initialize_model()
        train_losses = []
        test_losses = []
        print("Start training...")
        for epoch in range(self.Epochs):
            train_loss = self.train()
            test_loss = self.test()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            path = self.Out_path
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


class PlotRegresson:
    def __init__(self, model, test_loader, batch_size):
        self.model = model
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.pt_pred_arr = []
        self.eta_pred_arr = []
        self.phi_pred_arr = []
        self.pt_truth_arr = []
        self.eta_truth_arr = []
        self.phi_truth_arr = []

    def evaluate(self):
        with torch.no_grad():
            for data in self.test_loader:
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
        plt.clf()

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate GAT model")
    parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
    parser.add_argument('--out_path', type=str, default='Bsize_gmp_64_lr5e-4_v2', help='Output path for the results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v2', help='Output directory for evaluation results')
    parser.add_argument('--train', action='store_true', help='Train the model')

    args = parser.parse_args()


    # For training:
    trainer = TrainModelFromGraph(Graph_path=args.graph_path, Out_path=args.out_path, BatchSize=args.batch_size, LearningRate=args.learning_rate, Epochs=args.epochs)
    if args.train:
        trainer.Training_loop()

    # For evaluating:
    trainer.load_data()
    test_loader = trainer.test_loader
    model = torch.load(args.model_path)
        
    evaluator = PlotRegresson(model, test_loader, batch_size=args.batch_size)
    evaluator.evaluate()
    evaluator.plot_regression(output_dir=args.output_dir)

if __name__ == "__main__":
    main()