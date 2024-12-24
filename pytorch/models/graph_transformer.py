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
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, heads):
        super().__init__()
        self.conv1 = HGTConv(in_channels, hidden_channels, metadata=metadata, heads=heads)
        self.conv2 = HGTConv(hidden_channels, out_channels, metadata=metadata, heads=heads)

    def forward(self, x_dict, edge_index_dict):
        update_x_dict = self.conv1(x_dict, edge_index_dict)
        out = self.conv2(update_x_dict, edge_index_dict)
        return out['uin']