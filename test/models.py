# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

class GATRegressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        # Verificar la presencia de deltaPhi y deltaEta
        if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
            deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
            edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
        else:
            edge_attr = None

        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.fc1(x)
        return x
    
class PlotRegression:
    def __init__(self, model, test_loader, batch_size):
        self.model = model
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pt_pred_arr = []
        self.pt_truth_arr = []

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

                    self.pt_truth_arr.append(vector_real[0].item())

    def plot_regression(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.clf()
        plt.hist(self.pt_truth_arr, bins=100, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.pt_pred_arr, bins=100, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression.png"))
        plt.clf()
