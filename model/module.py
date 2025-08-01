import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


def graph2batch_sequence(graph_hidden_states, batch):
    device = graph_hidden_states.device

    num_graphs = batch.max().item() + 1
    max_nodes = max((batch == i).sum().item() for i in range(batch.max().item() + 1))

    padded_features = torch.zeros(
        num_graphs, max_nodes, graph_hidden_states.size(1), device=device
    )
    attention_mask = torch.zeros(num_graphs, max_nodes, device=device)

    for i in range(num_graphs):
        graph_node_features = graph_hidden_states[batch == i]
        padded_features[i, : graph_node_features.size(0), :] = graph_node_features
        attention_mask[i, : graph_node_features.size(0)] = 1

    return padded_features, attention_mask


def sequence2batch_graph(padded_features, attention_mask):
    device = padded_features.device
    valid_mask = attention_mask.bool()

    flat_features = padded_features.view(-1, padded_features.size(-1))
    flat_mask = valid_mask.view(-1)

    graph_hidden_states = flat_features[flat_mask]

    batch_size = attention_mask.size(0)
    max_nodes = attention_mask.size(1)
    graph_indices = torch.arange(batch_size, device=device).repeat_interleave(max_nodes)
    batch = graph_indices[flat_mask]

    return graph_hidden_states


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class MPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        """
        Initialize the MPNN layer.
        Args:
            node_dim (int): Dimension of node features.
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Dimension of hidden layers.
        """
        super(MPNN, self).__init__(
            aggr="add"
        )  # Aggregation method: 'add', 'mean', or 'max'
        self.message_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                2 * node_dim + edge_dim, hidden_dim
            ),  # Input: concatenated node and edge features
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = torch.nn.Sequential(
            torch.nn.Linear(
                node_dim + hidden_dim, hidden_dim
            ),  # Input: concatenated current node and aggregated message
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the MPNN layer.
        Args:
            x (Tensor): Node feature matrix [num_nodes, node_dim].
            edge_index (Tensor): Edge index matrix [2, num_edges].
            edge_attr (Tensor): Edge feature matrix [num_edges, edge_dim].
        Returns:
            Tensor: Updated node feature matrix [num_nodes, hidden_dim].
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """
        Compute messages for each edge.
        Args:
            x_i (Tensor): Features of target nodes [num_edges, node_dim].
            x_j (Tensor): Features of source nodes [num_edges, node_dim].
            edge_attr (Tensor): Edge features [num_edges, edge_dim].
        Returns:
            Tensor: Messages [num_edges, hidden_dim].
        """
        msg_input = torch.cat(
            [x_i, x_j, edge_attr], dim=-1
        )  # Concatenate target, source, and edge features
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        """
        Update node states with aggregated messages.
        Args:
            aggr_out (Tensor): Aggregated messages [num_nodes, hidden_dim].
            x (Tensor): Current node features [num_nodes, node_dim].
        Returns:
            Tensor: Updated node features [num_nodes, hidden_dim].
        """
        update_input = torch.cat(
            [x, aggr_out], dim=-1
        )  # Concatenate current node features and aggregated messages
        return self.update_mlp(update_input)


class MultiLayerMPNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers):
        """
        Multi-layer MPNN model with configurable number of layers.
        Args:
            node_dim (int): Dimension of node features.
            edge_dim (int): Dimension of edge features.
            hidden_dim (int): Dimension of hidden layers.
            out_dim (int): Dimension of output layer.
            num_layers (int): Number of MPNN layers.
        """
        super(MultiLayerMPNN, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                MPNN(node_dim if i == 0 else hidden_dim, edge_dim, hidden_dim)
                for i in range(num_layers)
            ]
        )
        # self.fc = torch.nn.Linear(hidden_dim, out_dim)  # Final output layer

    def forward(self, data):

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = x.float()
        edge_attr = edge_attr.float()

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)  # Activation function
        x = global_mean_pool(x, batch)  # Global pooling for graph-level representation
        return x  # Output layer
