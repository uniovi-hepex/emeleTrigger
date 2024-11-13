import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

import os

import argparse
import matplotlib.pyplot as plt
from models import GATRegressor,GATv2Regressor

import time

import itertools

#import torch._dynamo
#torch._dynamo.config.capture_scalar_outputs = True

class NormalizeNodeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'x'):
            data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
        return data

class NormalizeEdgeFeatures(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'edge_attr'):
            data.edge_attr = (data.edge_attr - data.edge_attr.mean(dim=0)) / data.edge_attr.std(dim=0)
        return data

class NormalizeTargets(BaseTransform):
    def __call__(self, data):
        if hasattr(data, 'y'):
            data.y = (data.y - data.y.mean(dim=0)) / data.y.std(dim=0)
        return data
    
class NormalizeSpecificNodeFeatures(BaseTransform):
    def __init__(self, column_indices):
        self.column_indices = column_indices

    def __call__(self, data):
        if hasattr(data, 'x'):
            for column_index in self.column_indices:
                column = data.x[:, column_index]
                mean = column.mean()
                std = column.std()
                data.x[:, column_index] = (column - mean) / std
        return data



class TrainModelFromGraph:
    @staticmethod
    def add_args(parser):
        parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
        parser.add_argument('--out_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training')
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
        parser.add_argument('--model_path', type=str, default=None, help='Path to the saved model for evaluation')
        parser.add_argument('--output_dir', type=str, default=None, help='Output directory for evaluation results')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
        parser.add_argument('--model_type', type=str, default='GAT', help='Model to use for training')
        parser.add_argument('--normalize_features', action='store_true', help='Normalize node features')
        parser.add_argument('--normalize_targets', action='store_true', help='Normalize target features')
        parser.add_argument('--normalize_edge_features', action='store_true', help='Normalize edge features')
        parser.add_argument('--normalize_specific_features', type=int, nargs='+', default=None, help='Normalize specific node feature columns')
        parser.add_argument('--num_files', type=int, default=None, help='Number of graph files to load')
        return parser

    def __init__(self, **kwargs):
        self.graph_path = kwargs.get('graph_path', 'graph_folder')
        self.out_path = kwargs.get('out_path', 'Bsize_gmp_64_lr5e-4_v3')
        self.batch_size = kwargs.get('batch_size', 64)
        self.learning_rate = kwargs.get('learning_rate', 0.0005)
        self.epochs = kwargs.get('epochs', 100)
        self.model_path = kwargs.get('model_path', None)
        self.output_dir = kwargs.get('output_dir', None)
        self.evaluate = kwargs.get('evaluate', False)
        self.model_type = kwargs.get('model_type', 'GAT')
        self.normalize_features = kwargs.get('normalize_features', False)
        self.normalize_targets = kwargs.get('normalize_targets', False)
        self.normalize_edge_features = kwargs.get('normalize_edge_features', False)
        self.normalize_specific_features = kwargs.get('normalize_specific_features', None)
        self.num_files = kwargs.get('num_files', None)  # Número de archivos a cargar
        self.device = kwargs.get('device', 'cpu')
        

        # Initialize other attributes
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = torch.nn.MSELoss().to(self.device)
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.device == 'cuda') else 'cpu')

        # Apply transformations if necessary
        self.transforms = []
        if self.normalize_features:
            self.transforms.append(NormalizeNodeFeatures())
        if self.normalize_edge_features:
            self.transforms.append(NormalizeEdgeFeatures())
        if self.normalize_targets:
            self.transforms.append(NormalizeTargets())
        if self.normalize_specific_features is not None:
            self.transforms.append(NormalizeSpecificNodeFeatures(self.normalize_specific_features))

        # For evaluation:
        self.trained_model = None
        self.pt_pred_arr = []
        self.pt_truth_arr = []

    def load_data(self):
        # Loading data from graph and convert it to DataLoader
        graphs = []
        all_files = os.listdir(self.graph_path)

        # Filter for .pkl files
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        if not pkl_files:
            print("No .pkl files found in the directory.")
            return []
        
        if self.num_files is not None:
            print(f"Loading {self.num_files} files")
            pkl_files = pkl_files[:self.num_files]

        for pkl_file in pkl_files:
            file_path = os.path.join(self.graph_path, pkl_file)
            with open(file_path, 'rb') as file:
                graphfile = torch.load(file)
                graphs.append(graphfile)

        Graphs_for_training = list(itertools.chain.from_iterable(graphs))
        print(f"Total Graphs: {len(Graphs_for_training)}")

        # Filter out graphs with no nodes
        Graphs_for_training_filtered = [g for g in Graphs_for_training if g.edge_index.size(1) > 0]  # remove empty graphs
        
        # remove extra dimension in y and put deltaPhi and deltaEta in the data object as edge_attr
        for i in range(0, len(Graphs_for_training_filtered)):
            Graphs_for_training_filtered[i].y = Graphs_for_training_filtered[i].y.mean(dim=0)
            Graphs_for_training_filtered[i].edge_attr = torch.stack([Graphs_for_training_filtered[i].deltaPhi.float(), Graphs_for_training_filtered[i].deltaEta.float()], dim=1)

        # Apply transformations to the load data... 
        if self.transforms:
            for transform in self.transforms:
                Graphs_for_training_filtered = [transform(data) for data in Graphs_for_training_filtered]

        # Train and test split:
        events = len(Graphs_for_training_filtered)
        ntrain = int((events * 0.7) / self.batch_size) * self.batch_size  # to have full batches
        print(f"Training events: {ntrain}")

        train_dataset = Graphs_for_training_filtered[:ntrain]
        test_dataset = Graphs_for_training_filtered[ntrain:ntrain * 2]

        print("====================================")
        print("Example of data:")
        print(train_dataset[0].x)
        print(train_dataset[0].edge_index)
        print(train_dataset[0].edge_attr)
        print(train_dataset[0].y)
        print("====================================")

        # Load data
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
    def plot_graph_features(self, data_loader):
        feature_names = ["eta", "phi", "R", "layer", "Type"]
        for batch in data_loader:
            features = batch.x.numpy()
            regression = batch.y.numpy()
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
            for i in range(batch.edge_attr.shape[1]):
                axs[i+num_features].hist(batch.edge_attr[:, i], bins=30, alpha=0.75)
                axs[i+num_features].set_title(f'Edge Feature {i} Histogram')
                axs[i+num_features].set_xlabel(f'Edge Feature {i} Value')
                axs[i+num_features].set_ylabel('Frequency')

            #calculate the average number of edges, dividing by the number of nodes
            num_edges = batch.edge_index.size(1)/batch.x.size(0)
            axs[num_features+2].hist(num_edges, bins=30, alpha=0.75)
            axs[num_features+2].set_title('Number of Edges/Node')
            axs[num_features+2].set_ylabel('Count')
            
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
        hidden_dim = self.batch_size
        output_dim = 1 ## ONE FEATURE ONLY!!!
        print(f"Using device: {self.device}")
        if self.model_type == 'GAT':
            self.model = GATRegressor(num_node_features, hidden_dim, output_dim).to(self.device)
        elif self.model_type == 'GATv2':
            self.model = GATv2Regressor(num_node_features, hidden_dim, output_dim).to(self.device)
        elif self.model_type == 'GATwithDropout':
            self.model = GATRegressorDO(num_node_features, hidden_dim, output_dim).to(self.device)
        
        #self.model = torch_geometric.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.75)
        print("Model initialized")
        print(self.model)

    def train(self):
        self.model.train()
        for data in self.train_loader:       
            data = data.to(self.device)  # Mueve los datos al dispositivo
            out = self.model(data)  # Se llama al modelo con los datos
            loss = self.loss_fn(out, data.y.view(out.size()))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    @torch.no_grad()
    def test(self, loader):
        self.model.eval()
        
        total_loss = 0
        total_accuracy = 0
        for data in loader:
            data = data.to(self.device)
            out = self.model(data)
            loss = self.loss_fn(out, data.y.view(out.size()))
            total_loss += float(loss)
            total_accuracy += self.accuracy(out, data.y)
        return total_loss / len(loader.dataset), total_accuracy / len(loader.dataset)

    def accuracy(self, predictions, targets, threshold=0.10):
        # Calcular la diferencia relativa
        relative_diff = torch.abs(predictions - targets) / torch.abs(targets)
        
        # Comparar con el porcentaje dado
        ok = relative_diff < (threshold)
        
        # Calcular la precisión
        acc = ok.sum()
        
        return int(acc)

    def Training_loop(self):
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        print(f"Saving results in {self.out_path}")
        path = self.out_path
        if not os.path.exists(path):
            os.makedirs(path)
        
        print("Start training...")
        for epoch in range(self.epochs):
            start = time.time()

            self.train()
            train_loss, train_accuracy = self.test(self.train_loader)
            test_loss, test_accuracy = self.test(self.test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            if epoch == 0:
                torch.save(test_loss, f"{path}/testloss_{epoch + 1}.pt")
                torch.save(train_loss, f"{path}/trainloss_{epoch + 1}.pt")
            elif (epoch + 1) % 10 == 0:
                print(f'Time per epoch: {time.time() - start:.2f}s')
                print(f'Epoch: {epoch + 1:02d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}')
                torch.save(self.model, f"{path}/model_{epoch + 1}.pth")
                torch.save(test_loss, f"{path}/testloss_{epoch + 1}.pt")
                torch.save(train_loss, f"{path}/trainloss_{epoch + 1}.pt")

                plt.plot(train_losses, "b", label="Train loss")
                plt.plot(test_losses, "k", label="Test loss")
                plt.plot(train_accuracies, "g", label="Train accuracy")
                plt.plot(test_accuracies, "r", label="Test accuracy")
                plt.yscale('log')
                plt.savefig(f"{path}/loss_accuracy_plot.png")
                plt.close()
    
    def load_trained_model(self):
        print(f"Loading model from {self.model_path}")
        self.trained_model = torch.load(self.model_path, map_location=torch.device('cpu'))

    def evaluate(self):
        with torch.no_grad():
            for data in self.test_loader:
                out = self.trained_model(data)
                for item in range(0, out.size(0)):
                    vector_pred = out[item]
                    vector_real = data[item].y
                    self.pt_pred_arr.append(vector_pred.item())
                    self.pt_truth_arr.append(vector_real.item())
    
    def plot_regression(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.clf()
        print(f"Plotting regression in {output_dir}")
        plt.hist(self.pt_truth_arr, bins=100, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.pt_pred_arr, bins=100, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression.png"))
        plt.clf()

        print(f"Plotting scatter in {output_dir}")
        plt.plot(self.pt_truth_arr, self.pt_pred_arr, 'o')
        plt.xlabel("Truth")
        plt.ylabel("Prediction")
        plt.savefig(os.path.join(output_dir, "pt_regression_scatter.png"))
        plt.clf()

        print(f"Plotting difference in {output_dir}")
        # plot difference between truth and prediction
        diff = [x - y for x, y in zip(self.pt_truth_arr, self.pt_pred_arr)]
        plt.hist(diff, bins=100, color='r', alpha=0.5, label="difference")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression_diff.png"))
        plt.clf()

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate GAT model")
    parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
    parser.add_argument('--out_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='GAT', help='Model to use for training')
    parser.add_argument('--plot_graph_features', action='store_true', help='Plot the graph features')
    parser.add_argument('--normalize_features', action='store_true', help='Normalize node features')
    parser.add_argument('--normalize_targets', action='store_true', help='Normalize target features')
    parser.add_argument('--normalize_edge_features', action='store_true', help='Normalize edge features')
    parser.add_argument('--normalize_specific_features', type=int, nargs='+', default=None, help='Normalize specific node feature columns')
    parser.add_argument('--num_files', type=int, default=None, help='Number of graph files to load')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output directory for evaluation results')
    parser.add_argument('--do_train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training')

    args = parser.parse_args()

    # For training:
    trainer = TrainModelFromGraph(**vars(args))
    trainer.load_data()
    if args.plot_graph_features: 
        trainer.plot_graph_features(trainer.train_loader)

    if args.do_train:
        trainer.initialize_model()
        trainer.Training_loop()

    if args.evaluate:
        trainer.load_trained_model()
        trainer.evaluate()
        trainer.plot_regression(output_dir=args.output_dir)


if __name__ == "__main__":
    main()