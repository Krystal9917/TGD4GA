import torch
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x