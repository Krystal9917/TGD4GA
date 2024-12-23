import os
import sys
import torch
from torch_geometric.nn import RGATConv
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_example_test import generate_data_batch


class RGAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, num_relations=10):
        super().__init__()
        self.conv1 = RGATConv(input_channels, hidden_channels, num_relations=num_relations, num_bases=10)
        self.conv2 = RGATConv(hidden_channels, out_channels, num_relations=num_relations, num_bases=10)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x


if __name__ == '__main__':
    batch = generate_data_batch()
    input_dim = batch['uin'].x.shape[1]
    hidden_dim = 32
    output_dim = input_dim
    model = RGAT(input_dim, hidden_dim, output_dim, num_relations=5)

    # Forward pass
    out = model(batch.x_dict, batch.edge_index_dict)
    print(out)