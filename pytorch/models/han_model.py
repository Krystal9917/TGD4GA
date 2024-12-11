import torch
from torch_geometric.nn import HANConv


class HAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata, heads=4):
        super().__init__()
        self.conv1 = HANConv(in_channels, out_channels, metadata, dropout=0.2, heads=heads)
        self.conv2 = HANConv(out_channels, out_channels, metadata, dropout=0.2, heads=heads)
        # self.w = torch.nn.Linear(out_channels * 2, out_channels, bias=False)

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = self.conv2(out, edge_index)
        # alpha = self.w(torch.concat([x['uin'], out['uin']], dim=1)).sigmoid()
        # out = out['uin'] + alpha * x['uin']
        return out['uin']