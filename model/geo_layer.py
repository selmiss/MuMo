import torch
from torch.nn import Linear
from model.module import MPNN
import torch.nn.functional as F

class GeoGraphLyaerMPNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels, angle_dim=1, geo_operation=True, brics=True, cl=False):

        super(GeoGraphLyaerMPNN, self).__init__()
        self.brics = brics
        self.geo_operation = geo_operation
        self.first_graph_embedder = (node_dim != hidden_channels) or (edge_dim != hidden_channels)
        
        if self.first_graph_embedder:
            self.lin_a = Linear(node_dim, hidden_channels)
            self.lin_b = Linear(edge_dim, hidden_channels)
            self.lin_c = Linear(angle_dim, hidden_channels)
            
        self.lin_gate = Linear(3 * hidden_channels, hidden_channels)
        self.geo_gate = Linear(3 * hidden_channels, hidden_channels)
        if self.brics:
            self.fusion_gate = Linear(2 * hidden_channels, hidden_channels)
            
        self.graph_conv = MPNN(hidden_channels, hidden_channels, hidden_channels)
        if self.geo_operation:
            self.geo_conv = MPNN(hidden_channels, hidden_channels, hidden_channels)
        
    def forward(self, data):
        
        x = data.x.float()
        edge_index = data.edge_index
        
        if edge_index.shape[1] == 0:
            if self.first_graph_embedder:
                x = F.relu(self.lin_a(x))
            data.x = x
            return data
        
        edge_attr = data.edge_attr.float()
        geo_operation = self.geo_operation
        if geo_operation:
            ba_edge_attr = data.ba_edge_attr
            ba_edge_index = data.ba_edge_index
            if ba_edge_index.shape[1] == 0:
                geo_operation = False

        batch = data.batch
        
        
        if self.first_graph_embedder:
            x = F.relu(self.lin_a(x))  # (N, 9) -> (N, hidden_channels)
            edge_attr = F.relu(self.lin_b(edge_attr))  # (N, 3) -> (N, hidden_channels)
            if geo_operation:
                ba_edge_attr = F.relu(self.lin_c(ba_edge_attr)) # (N, 1) -> (N, hidden_channels)

        
        if geo_operation:
            hd_edge_attr = F.relu(self.geo_conv(edge_attr, ba_edge_index, ba_edge_attr))
            beta = self.geo_gate(torch.cat([edge_attr, hd_edge_attr, edge_attr - hd_edge_attr], 1)).sigmoid()
            edge_attr = beta * edge_attr + (1 - beta) * hd_edge_attr

        
        hidden_states = F.relu(self.graph_conv(x, edge_index, edge_attr))
        beta = self.lin_gate(torch.cat([x, hidden_states, x - hidden_states], 1)).sigmoid()
        x = beta * x + (1 - beta) * hidden_states

        data.edge_attr = edge_attr
        
        
        if geo_operation:
            data.ba_edge_attr = ba_edge_attr
        
        
        if self.brics:
            fra_x = data.x.float()
            fra_edge_index = data.fra_edge_index
            if fra_edge_index.shape[1] != 0:
                fra_edge_attr = data.fra_edge_attr.float()
                if geo_operation:
                    bafra_edge_attr = data.bafra_edge_attr
                    bafra_edge_index = data.bafra_edge_index
                    if bafra_edge_index.shape[1] == 0:
                        geo_operation = False
                cluster = data.cluster_idx
                
                if self.first_graph_embedder:
                    fra_x = F.relu(self.lin_a(fra_x))  # (N, 9) -> (N, hidden_channels)
                    fra_edge_attr = F.relu(self.lin_b(fra_edge_attr))  # (N, 3) -> (N, hidden_channels)
                    if geo_operation:

                        bafra_edge_attr = F.relu(self.lin_c(bafra_edge_attr)) # (N, 1) -> (N, hidden_channels)
                
                
                if geo_operation:
                    hd_fra_edge_attr = F.relu(self.geo_conv(fra_edge_attr, bafra_edge_index, bafra_edge_attr))
                    beta = self.geo_gate(torch.cat([fra_edge_attr, hd_fra_edge_attr, fra_edge_attr - hd_fra_edge_attr], 1)).sigmoid()
                    fra_edge_attr = beta * fra_edge_attr + (1 - beta) * hd_fra_edge_attr
                
                
                fra_hidden_states = F.relu(self.graph_conv(fra_x, fra_edge_index, fra_edge_attr))
                beta = self.lin_gate(torch.cat([fra_x, fra_hidden_states, fra_x - fra_hidden_states], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * fra_hidden_states
                
                data.fra_x = fra_x
                data.fra_edge_attr = fra_edge_attr
                if geo_operation:
                    data.bafra_edge_attr = bafra_edge_attr
            else:
                if self.first_graph_embedder:
                    fra_x = F.relu(self.lin_a(fra_x))
                data.fra_x = fra_x
        
        
            # Fusion Option.
            beta = self.fusion_gate(torch.cat([x, fra_x], 1)).sigmoid()
            final_x =  beta * x + (1 - beta) * fra_x
            data.x = final_x
        else:
            data.x = x
            
        return data
    