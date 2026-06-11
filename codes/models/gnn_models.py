import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.2):
        super(GCN, self).__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            # 只有一层，直接从输入到输出
            self.convs.append(
                GCNConv(in_channels, out_channels)
            )
        else:
            # 第一层
            self.convs.append(
                GCNConv(in_channels, hidden_channels)
            )
            # 中间层（如果有）
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels)
                )
            # 最后一层
            self.convs.append(
                GCNConv(hidden_channels, out_channels)
            )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, heads=4, dropout=0.2):
        super(GAT, self).__init__()
        assert num_layers >= 1, "num_layers must be at least 1"

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        if num_layers == 1:
            # 只有一层，直接从输入到输出
            self.convs.append(
                GATConv(in_channels, out_channels, heads=heads, concat=False, dropout=dropout)
            )
        else:
            # 第一层
            self.convs.append(
                GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
            )
            # 中间层（如果有）
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
                )
            # 最后一层，concat=False 输出类别数
            self.convs.append(
                GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
            )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
        return x