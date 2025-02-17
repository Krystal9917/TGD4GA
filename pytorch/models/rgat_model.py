import torch
from torch_geometric.nn import RGATConv


class RGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_relations, mod='addictive'):
        super().__init__()
        self.conv1 = RGATConv(input_dim, hidden_dim, heads=num_heads, num_relations=num_relations, mod=mod)
        self.conv2 = RGATConv(hidden_dim*num_heads, output_dim, heads=num_heads, num_relations=num_relations, mod=mod)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x