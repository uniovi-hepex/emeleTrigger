# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_max_pool, global_mean_pool

class GATRegressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim,add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim,add_self_loops=False)
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

class GATRegressorDO(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super(GATRegressorDO, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim,add_self_loops=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim,add_self_loops=False)
        self.fc1 = torch.nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        # Verificar la presencia de deltaPhi y deltaEta
        if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
            deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
            edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
        else:
            edge_attr = None

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.fc1(x)
        return x
    

class GATv2Regressor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim=1):
        super(GATv2Regressor, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_dim,add_self_loops=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim,add_self_loops=False)
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


## THIS IS THE BEST MODEL UP-TO-DATE (28/11/24):
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_max_pool, global_mean_pool, AttentionalAggregation
from torch_geometric.utils import add_self_loops, degree, softmax

class MPL(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPL, self).__init__(aggr='add')
        self.mlp1 = torch.nn.Linear(in_channels*2, out_channels)
        self.mlp2 = torch.nn.Linear(in_channels, out_channels)
        self.mlp3 = torch.nn.Linear(2*out_channels, 1)
        self.mlp4 = torch.nn.Linear(2*out_channels, 1)
        self.mlp5 = torch.nn.Linear(in_channels,16)
        self.mlp6 = torch.nn.Linear(out_channels,16)
        self.mlp7 = torch.nn.Linear(16,1)

    def forward(self, x, edge_index):
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        msg = self.propagate(edge_index, x=x.float())
        x = F.relu(self.mlp2(x))
        w1 = F.sigmoid(self.mlp3(torch.cat([x,msg], dim=1)))
        w2 = F.sigmoid(self.mlp4(torch.cat([x,msg], dim=1)))
        out = w1*msg + w2*x
        
        return out

    def message(self, x_i, x_j, edge_index):
        msg = F.relu(self.mlp1(torch.cat([x_i, x_j-x_i], dim=1)))
        w1 = F.tanh(self.mlp5(x_i))
        w2 = F.tanh(self.mlp6(msg))
        w = self.mlp7(w1*w2)
        w = softmax(w, edge_index[0])
        return msg*w

class MPLNNRegressor(torch.nn.Module):
    def __init__(self,in_channels):
        super(MPLNNRegressor, self).__init__()
        self.conv1 = MPL(in_channels,128 )
        self.conv2 = MPL(128,64)
        self.conv3 = MPL(64,64 )
        self.conv4 = MPL(64,64 )
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 16)
        self.lin3 = torch.nn.Linear(16, 16)
        self.lin4 = torch.nn.Linear(16, 1)
        self.lin5 = torch.nn.Linear(128, 128)
        self.lin6 = torch.nn.Linear(128, 16)
        self.lin7 = torch.nn.Linear(16, 16)
        self.lin8 = torch.nn.Linear(16, 1)
        self.global_att_pool1 = AttentionalAggregation(torch.nn.Sequential(torch.nn.Linear(64, 1)))
        self.global_att_pool2 = AttentionalAggregation(torch.nn.Sequential(torch.nn.Linear(64, 1)))
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x1 = self.global_att_pool1(x, batch)
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x2 = self.global_att_pool2(x, batch)
        x_out = torch.cat([x1, x2], dim=1)
        x = F.relu(self.lin1(x_out))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        x = self.lin4(x).squeeze(1)

        return x
