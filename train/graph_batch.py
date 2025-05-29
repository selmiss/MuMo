from torch_geometric.data import Batch
import torch

class TriBatch(Batch):
    def __init__(self, batch=None, **kwargs):
        super().__init__(batch, **kwargs)

    @staticmethod
    def from_data_list(data_list):
        
        batch = TriBatch()
        current_node_offset = 0
        current_edge_offset = 0
        current_fra_edge_offset = 0
        current_cluster_offset = 0
        for data in data_list:
            for key, value in data:
                if key in ['edge_index', 'fra_edge_index']:
                    value = value + current_node_offset
                elif key in ['ba_edge_index']:
                    value = value + current_edge_offset
                elif key in ['bafra_edge_index']:
                    value = value + current_fra_edge_offset
                elif key in ['cluster_idx']:
                    value = value + current_cluster_offset
                
                if key in batch and type(value) == torch.Tensor:
                    batch[key] = torch.cat([batch[key], value], dim=batch.__cat_dim__(key, data[key]) or 0)
                elif type(value) == torch.Tensor:
                    batch[key] = value
            current_node_offset += data.num_nodes
            current_edge_offset += data.num_edges
            current_cluster_offset += torch.max(data.cluster_idx) + 1
            # Handle case where fra_edge_attr might not exist
            if hasattr(data, 'fra_edge_attr') and data.fra_edge_attr is not None:
                current_fra_edge_offset += data.fra_edge_attr.shape[0]
            else:
                current_fra_edge_offset += 0
            
        batch.batch = torch.cat(
            [torch.full((data.num_nodes,), i, dtype=torch.long) for i, data in enumerate(data_list)]
        )
        return batch