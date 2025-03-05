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
    def __init__(self, numerical_dim, categorical_dim, text_dim, input_dim,
                 hidden_dim, output_dim, num_relations, num_bases=10):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.categorical_dim = categorical_dim
        self.text_dim = text_dim
        self.mlp_n = torch.nn.Linear(numerical_dim, numerical_dim, bias=False)
        self.mlp_c = torch.nn.Linear(categorical_dim, categorical_dim, bias=False)
        self.mlp_t = torch.nn.Linear(text_dim, text_dim, bias=False)
        self.conv1 = MaskRGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = MaskRGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x_n = self.mlp_n(x[:, :self.numerical_dim])
        x_c = self.mlp_c(x[:, self.numerical_dim: (self.numerical_dim + self.categorical_dim)])
        x_t = self.mlp_t(x[:, (self.numerical_dim + self.categorical_dim):])
        x = torch.concat([x_n, x_c, x_t], dim=1)
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x

