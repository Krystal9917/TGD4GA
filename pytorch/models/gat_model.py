import os
import sys
import torch
from torch_geometric.nn import GATConv, Linear, to_hetero
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_example_test import generate_data_batch


class HeteroGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False, heads=heads, concat=True)
        self.lin1 = Linear(-1, hidden_channels*heads)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False, heads=heads, concat=True)
        self.lin2 = Linear(-1, out_channels*heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x


if __name__ == '__main__':
    batch = generate_data_batch()
    input_dim = batch['uin'].x.shape[1]
    hidden_dim = 32
    output_dim = input_dim
    model = HeteroGAT(hidden_dim, output_dim, heads=2)
    model = to_hetero(model, batch.metadata(), aggr='sum')

    # Forward pass
    out = model(batch.x_dict, batch.edge_index_dict)
    print(out)