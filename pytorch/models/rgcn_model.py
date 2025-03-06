import torch
from torch_geometric.nn import RGCNConv
from .rgcn_conv import MaskRGCNConv


class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases=30):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x

class MaskRGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases=10):
        super().__init__()
        self.mlp_w = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.conv1 = MaskRGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = MaskRGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.mlp_w(x)
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x

