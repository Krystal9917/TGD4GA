import torch
from torch_geometric.nn import RGCNConv
from .rgcn_conv import MaskRGCNConv
from .attn_rgcn_conv import AttnRGCNConv


class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, num_bases=30):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x


class MaskRGCNLayer(torch.nn.Module):
    def __init__(self, mlp_n_in_dim, mlp_c_in_dim, input_dim, output_dim, num_relations,
                 mlp_t_in_dim=None, is_dropout=None, attn_weight_type='split', adapter_type=None,
                 metric_learning='similarity', num_bases=10, is_embed=False):
        super().__init__()
        self.is_embed = is_embed
        if self.is_embed:
            self.mlp_w = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.conv = MaskRGCNConv(mlp_n_in_dim, mlp_c_in_dim, input_dim, output_dim, num_relations,
                                 mlp_t_in_channels=mlp_t_in_dim, is_dropout=is_dropout,
                                 attn_weight_type=attn_weight_type, adapter_type=adapter_type,
                                 metric_learning=metric_learning, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        if self.is_embed:
            x = self.mlp_w(x)
        x = self.conv(x, edge_index, edge_type)
        return x


class MaskRGCN(torch.nn.Module):
    def __init__(self, mlp_n_in_dim, mlp_c_in_dim, input_dim, hidden_dim, output_dim,
                 num_relations, mlp_t_in_dim=None, is_dropout=False, adapter_type=None,
                 metric_learning='similarity', num_bases=10, is_embed=False):
        super().__init__()
        self.conv1 = MaskRGCNLayer(mlp_n_in_dim, mlp_c_in_dim, input_dim, hidden_dim,
                                   num_relations, mlp_t_in_dim=mlp_t_in_dim,
                                   adapter_type=adapter_type, attn_weight_type='split',
                                   is_dropout=is_dropout, metric_learning=metric_learning,
                                   num_bases=num_bases, is_embed=is_embed)
        self.conv2 = MaskRGCNLayer(mlp_n_in_dim, mlp_c_in_dim, hidden_dim, output_dim,
                                   num_relations, mlp_t_in_dim=mlp_t_in_dim,
                                   adapter_type=adapter_type, attn_weight_type='all',
                                   is_dropout=is_dropout, metric_learning=metric_learning,
                                   num_bases=num_bases, is_embed=is_embed)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x


class AttnRGCN(torch.nn.Module):
    def __init__(self, input_dim, mlp_n_in_dim, mlp_c_in_dim, hidden_dim, output_dim, num_relations, num_bases=10,
                 is_embed=False):
        super().__init__()
        self.is_embed = is_embed
        if self.is_embed:
            self.mlp_w = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.conv1 = AttnRGCNConv(input_dim, mlp_n_in_dim, mlp_c_in_dim, hidden_dim, num_relations, num_bases=num_bases)
        self.conv2 = AttnRGCNConv(hidden_dim, mlp_n_in_dim, mlp_c_in_dim, output_dim, num_relations,
                                  num_bases=num_bases)

    def forward(self, x, edge_index, edge_type):
        if self.is_embed:
            x = self.mlp_w(x)
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        return x
