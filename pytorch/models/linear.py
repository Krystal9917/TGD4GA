import torch
from torch_geometric.nn import Linear


class MultiLinear(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=3):
        super().__init__()
        self.layer_num = layer_num
        self.lin = torch.nn.ModuleList()
        for i in range(layer_num):
            if i == 0:
                self.lin.append(Linear(input_dim, hidden_dim))
            elif i == layer_num - 1:
                self.lin.append(Linear(hidden_dim // 2, output_dim))
            elif i == layer_num - 2:
                self.lin.append(Linear(hidden_dim, hidden_dim // 2))
            else:
                self.lin.append(Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        x_out = []
        for i in range(self.layer_num):
            x = self.lin[i](x)
            if i != self.layer_num - 1:
                x = torch.relu(x)
            else:
                x = torch.softmax(x, dim=1)
            x_out.append(x)
        return x_out
