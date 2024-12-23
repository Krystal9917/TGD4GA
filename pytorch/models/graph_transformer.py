import torch
from torch_geometric.nn import TransformerConv, HGTConv


class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads):
        super().__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=heads, bias=True)
        self.conv2 = TransformerConv(hidden_dim, output_dim, heads=heads, bias=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class HeteroGraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, metadata, heads):
        super().__init__()
        self.conv1 = HGTConv(input_dim, hidden_dim, metadata=metadata, heads=heads)
        self.conv2 = HGTConv(hidden_dim, output_dim, metadata=metadata, heads=heads)

    def forward(self, x_dict, edge_index_dict):
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        return x