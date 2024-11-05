# models.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_max_pool, global_mean_pool


class GATRegressor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim=1):
        super(GATRegressor, self).__init__()
        self.conv1 = GATConv(num_node_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        '''        # Verificar la presencia de deltaPhi y deltaEta
                if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
                    deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
                    edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
                else:
                    edge_attr = None
        '''
        x = F.relu(x)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.fc1(x)
        return x
    
    def fit(self,data, epochs): 
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.forward(data)
            loss = self.loss_fn(out, data.y.view(out.size()))
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item()}")

class GATv2Regressor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, output_dim=1):
        super(GATv2Regressor, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, hidden_dim)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim*2, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        # Verificar la presencia de deltaPhi y deltaEta
        '''        if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
                    deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
                    edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
                else:
                    edge_attr = None
        '''
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
        '''        if hasattr(data, 'deltaPhi') and hasattr(data, 'deltaEta'):
                    deltaPhi, deltaEta = data.deltaPhi.float(), data.deltaEta.float()
                    edge_attr = torch.stack([deltaPhi, deltaEta], dim=1).float()
                else:
                    edge_attr = None
        '''
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.fc1(x)
        return x
