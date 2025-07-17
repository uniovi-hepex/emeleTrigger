import torch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import Data

import random
import os,sys

import argparse
import matplotlib.pyplot as plt
from models import EdgeGNNClassifier, EdgeGNNClassifier_OneLayer
from transformations import DropLastNodeFeature,DropLastTwoNodeFeatures,DropLastThreeNodeFeatures,NormalizeNodeFeatures,NormalizeEdgeFeatures,NormalizeTargets,NormalizeSpecificNodeFeatures
import pickle

import itertools

class TrainModelFromGraph:
    @staticmethod
    def add_args(parser):
        parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
        parser.add_argument('--graph_name', type=str, default='vix_graph_13Nov_3_muonQOverPt', help='Name of the graph data')
        parser.add_argument('--out_model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
        parser.add_argument('--save_tag', type=str, default='vix_graph_13Nov_3_muonQOverPt', help='Tag for saving the model')
        parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
        parser.add_argument('--earlystop', type=int, default=3, help='Number of epochs before closing the training if there is no improvement in the loss')
        parser.add_argument('--model_path', type=str, default=None, help='Path to the saved model for evaluation')
        parser.add_argument('--do_validation', action='store_true', help='Evaluate the model')
        parser.add_argument('--model_type', type=str, default='EdgeClassifier', help='Model to use for training')
        parser.add_argument('--do_train', action='store_true', help='Train the model')
        parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for the model')
        parser.add_argument('--normalization', type=str, default='NodesAndEdgesAndLayerInfo', help='Type of normalization to apply')
        parser.add_argument('--num_files', type=int, default=None, help='Number of graph files to load')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--edge_attr', type=str, default='Yes', help='Does it have edge_attr?')
        parser.add_argument('--metrics', type=str, default="AllMetrics", help='Metric Labels')
        return parser

    def __init__(self, **kwargs):
        self.graph_path = kwargs.get('graph_path', 'graph_folder')
        self.graph_name = kwargs.get('graph_name', 'vix_graph_13Nov_3_muonQOverPt')
        self.out_model_path = kwargs.get('out_model_path', 'Bsize_gmp_64_lr5e-4_v3')
        self.save_tag = kwargs.get('save_tag', 'vix_graph_13Nov_3_muonQOverPt')
        self.batch_size = kwargs.get('batch_size', 128)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epochs = kwargs.get('epochs', 100)
        self.earlystop = kwargs.get('earlystop', 3)
        self.do_validation = kwargs.get('evaluate', False)
        self.do_train = kwargs.get('do_train', False)
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.model_type = kwargs.get('model_type', 'EdgeClassifier')
        self.model_path = kwargs.get('model_path', f"{self.out_model_path}/model_{self.model_type}_{self.hidden_dim}dim_{self.epochs}epochs_{self.save_tag}.pth")
        self.normalization = kwargs.get('normalization', 'NodesAndEdgesAndLayerInfo')
        self.num_files = kwargs.get('num_files', None)  # Número de archivos a cargar
        self.device = kwargs.get('device', 'cuda')
        self.edge_attr = kwargs.get('edge_attr', 'Yes')
        self.metrics = kwargs.get('metrics', "AllMetrics")

        # Initialize other attributes
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.loss_fn = None

        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.device == 'cuda') else 'cpu')

        # Apply transformations if necessary
        self.transform = None
        if self.normalization == 'NodesAndEdgesAndLayerInfo':
            self.transform = Compose([NormalizeNodeFeatures(),NormalizeEdgeFeatures(),DropLastNodeFeature()])
        elif self.normalization == 'NodesAndEdgesAndOnlySpatial':
            self.transform = Compose([NormalizeNodeFeatures(),NormalizeEdgeFeatures(),DropLastTwoNodeFeatures()])
        elif self.normalization == 'NodesAndEdges':
            self.transform = Compose([NormalizeNodeFeatures(),NormalizeEdgeFeatures()])
        elif self.normalization == 'Nodes':
            self.transform = NormalizeNodeFeatures()
        elif self.normalization == 'Edges':
            self.transform = NormalizeEdgeFeatures()
        elif self.normalization == 'Targets':
            self.transform = NormalizeTargets()
        elif self.normalization == 'DropLastNodeFeature':
            self.transform = DropLastNodeFeature()
        elif self.normalization == 'DropLastTwoNodeFeatures':
            self.transform = DropLastTwoNodeFeatures()
        elif self.normalization == 'DropLastThreeNodeFeatures':
            self.transform = DropLastThreeNodeFeatures()
        elif self.normalization == 'None':
            print("No normalization applied")
            self.transform = None
        else: 
            print(self.normalization)
            print("Unknown normalization type, exiting...")
            sys.exit(1)
    ### Add setter functions for all parameters: 
    def set_graph_path(self, path):
        self.graph_path = path
    def set_graph_name(self, name):
        self.graph_name = name
    def set_out_model_path(self, path):
        self.out_model_path = path
    def set_save_tag(self, tag):
        self.save_tag = tag
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
    def set_epochs(self, epochs):
        self.epochs = epochs
    def set_model_path(self, model_path):
        self.model_path = model_path
    def set_do_validation(self, do_validation):
        self.do_validation = do_validation
    def set_do_train(self, do_train):
        self.do_train = do_train
    def set_hidden_dim(self, hidden_dim):
        self.hidden_dim = hidden_dim
    def set_model_type(self, model_type):
        self.model_type = model_type
    def set_normalization(self, normalization):
        self.normalization = normalization
    def set_num_files(self, num_files):
        self.num_files = num_files
    def set_device(self, device):
        self.device = device
    def set_edge_attr(self, edge_attr):
        self.edge_attr = edge_attr
    def set_metrics(self, metrics):
        self.metrics = metrics

    def load_data(self):
        # Loading data from graph and convert it to DataLoader
        graphs = []
        all_files = os.listdir(self.graph_path)

        # Filter for .pkl files
        graph_files = [f for f in all_files if (f.endswith('.pkl') or f.endswith('.pt')) and self.graph_name in f]
        if not graph_files:
            print("No .pkl/.pt files found in the directory.")
            return []
        
        if self.num_files is not None:
            print(f"Loading {self.num_files} files")
            graph_files = random.sample(graph_files, self.num_files)

        for graph_file in graph_files:
            file_path = os.path.join(self.graph_path, graph_file)
            if graph_file.endswith('.pt'):
                graph = torch.load(file_path)
            elif graph_file.endswith('.pkl'):
                with open(file_path, 'rb') as file:
                    graph = torch.load(file)
            graphs.append(graph)

        Graphs_for_training = sum(graphs, [])
        print(f"Total Graphs: {len(Graphs_for_training)}")

        ### NOW FILTER EMPTY GRAPHS... 
        Graphs_for_training_reduced = Graphs_for_training
        Graphs_for_training_filtered = [
            g for g in Graphs_for_training_reduced
            if not (torch.isnan(g.y).any() or torch.isnan(g.x).any())  and g.edge_index.size(1) > 0
        ]

        if self.edge_attr == "None":
            # remove extra dimension in y and put deltaPhi and deltaEta in the data object as edge_attr
            for i in range(0, len(Graphs_for_training_filtered)):
                Graphs_for_training_filtered[i].y = Graphs_for_training_filtered[i].y.mean(dim=0)
                Graphs_for_training_filtered[i].edge_attr = torch.stack([Graphs_for_training_filtered[i].deltaPhi.float(), Graphs_for_training_filtered[i].deltaEta.float()], dim=1)
        else:
            # remove extra dimension in y
            for i in range(0, len(Graphs_for_training_filtered)):
                Graphs_for_training_filtered[i].y = Graphs_for_training_filtered[i].y.mean(dim=0)      


        Graphs_for_training_filtered = [
            g for g in Graphs_for_training_filtered
            if not (torch.isnan(g.x).any() or torch.isnan(g.edge_attr).any() or torch.isnan(g.y).any())
        ]
        
        print(f"Total Graphs: {len(Graphs_for_training)}")
        print(f"Filtered Graphs: {len(Graphs_for_training_filtered)}")

        # Apply transformations to the load data... 
        if self.transform is not None:
            Graphs_for_training_filtered = [self.transform(data) for data in Graphs_for_training_filtered]

        Graphs_for_training_filtered = [
            g for g in Graphs_for_training_filtered
            if not (torch.isnan(g.x).any() or torch.isnan(g.edge_attr).any() or torch.isnan(g.y).any())
        ]

        # Train and test split:
        random.shuffle(Graphs_for_training_filtered)
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

    def initialize_model(self):
        node_dim = 7   ## NUMBER OF NODE FEATURES!!!
        edge_dim = 3   # Number of edge_attr
        hidden_dim = self.hidden_dim
        print(f"Using device: {self.device}")
        if self.model_type == 'EdgeClassifier':
            self.model = EdgeGNNClassifier(in_channels=node_dim, edge_in=edge_dim, hidden_channels=hidden_dim).to(self.device)
        elif self.model_type == 'EdgeClassifierOneLayer':
            self.model = EdgeGNNClassifier_OneLayer(in_channels=node_dim, edge_in=edge_dim, hidden_channels=hidden_dim).to(self.device)
        #self.model = torch_geometric.compile(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        print("Model initialized")
        print(self.model)

    def train_model(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:       
            data = data.to(self.device)  # Move data to the device
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            loss = self.loss_fn(logits, data.edge_label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def test_model(self, loader):
        self.model.eval()
        
        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            logits = self.model(data.x, data.edge_index, data.edge_attr)
            loss = self.loss_fn(logits, data.edge_label)
            total_loss += loss.item()
        return total_loss / len(loader)

    def Training_loop(self):

        print(f"Saving results in {self.out_model_path}")
        if not os.path.exists(self.out_model_path):
            os.makedirs(self.out_model_path)

        #window = self.earlystop
        best_loss = float('inf')
        best_epoch = int(0)
        counter = int(0)
        
        print("Start training...")
        for epoch in range(self.epochs):
            train_loss = self.train_model(self.train_loader)
            test_loss = self.test_model(self.test_loader)
            torch.save({'train_loss': train_loss, 'test_loss': test_loss}, f"{self.out_model_path}/loss_{self.model_type}_{self.hidden_dim}dim_{epoch+1}epochs_{self.save_tag}.pt")

            if test_loss < best_loss: #found better loss
                best_loss = test_loss
                best_epoch = epoch
                counter = 0
                print(f'Epoch: {epoch + 1:02d}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
                torch.save(self.model.state_dict(), f"{self.out_model_path}/model_{self.model_type}_{self.hidden_dim}dim_{self.epochs}epochs_{self.save_tag}.pth")
            else:
                counter += 1 #increment counter
            """
            #Stop training if more than X epochs have passed without improvements to the test loss
            if counter == window:
                print(f'Early stop reached at epoch {best_epoch + 1}, training closing at epoch {epoch + 1}\n Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}')
                break
            """

    def set_model_path(self, path):
        self.model_path = path
        
    def load_trained_model(self):
        print(f"Loading model from {self.model_path}")
        # load the model, first try state_dict then the model itself
        try:
            print("Loading state_dict")
            print(self.model_path)
            print(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except:
            self.model = torch.load(self.model_path, map_location=self.device)

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate GNN model")
    parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
    parser.add_argument('--graph_name', type=str, default='vix_graph_13Nov_3_muonQOverPt', help='Name of the graph data')
    parser.add_argument('--save_tag', type=str, default='vix_graph_13Nov_3_muonQOverPt', help='Tag for saving the model')
    parser.add_argument('--out_model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='SAGE', help='Model to use for training')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for the model')
    parser.add_argument('--plot_graph_features', action='store_true', help='Plot the graph features')
    parser.add_argument('--normalization', type=str, default='NodesAndEdgesAndLayerInfo', help='Type of normalization to apply')
    parser.add_argument('--num_files', type=int, default=None, help='Number of graph files to load')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--earlystop', type=int, default=3, help='Number of epochs before closing the training if there is no improvement in the loss')
    parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output directory for evaluation results')
    parser.add_argument('--do_train', action='store_true', help='Train the model')
    parser.add_argument('--do_validation', action='store_true', help='Evaluate the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--edge_attr', type=str, default='Yes', help='Does it have edge_attr?')
    parser.add_argument('--metrics', type=str, default="AllMetrics", help='Metric Labels')

    args = parser.parse_args()

    # For training:
    trainer = TrainModelFromGraph(**vars(args))
    trainer.load_data()
    trainer.initialize_model()

    if args.plot_graph_features: 
        from validation import plot_node_feature_histograms, plot_edge_attr_histograms, plot_Nodes_and_Edges_histograms
        plot_node_feature_histograms(trainer.train_loader, output_dir=args.output_dir,label=trainer.save_tag)
        plot_edge_attr_histograms(trainer.train_loader, output_dir=args.output_dir,label=trainer.save_tag)
        plot_Nodes_and_Edges_histograms(trainer.train_loader, output_dir=args.output_dir,label=trainer.save_tag)

    if args.do_train:
        trainer.Training_loop()

    if args.do_validation:
        trainer.load_trained_model()
        from validation import evaluate_model, metric_scores_file, plot_ROC_curve, plot_prec_vs_rec, plot_predicted_results, plot_prec_per_class, evaluate_per_event, summarize_by_variable, summarize_by_variable_efficiency, plot_auc_vs, plot_roc_by_bins, plot_GINI_vs, plot_efficiency_vs, plot_muon_pT
        y_pred, y_true = evaluate_model(trainer.model, trainer.test_loader, trainer.device)
        metric_scores_file(y_pred, y_true, metrics=args.metrics, output_dir=args.output_dir)
        plot_predicted_results(y_pred, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
        results = evaluate_per_event(trainer.model, trainer.train_loader, trainer.device)
        plot_muon_pT(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
        if args.metrics == "AllMetrics":
            plot_ROC_curve(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_auc_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_GINI_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_prec_vs_rec(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_prec_per_class(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_efficiency_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
        if args.metrics == "ROC_AUC_GINI":
            plot_ROC_curve(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_auc_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_GINI_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
        if args.metrics == "Recall_Precision_AP":
            plot_prec_vs_rec(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
            plot_prec_per_class(y_pred, y_true, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)
        if args.metrics == "Efficiency":
            plot_efficiency_vs(results, output_dir=args.output_dir, model=trainer.model_type, label=trainer.save_tag)


if __name__ == "__main__":
    main()
