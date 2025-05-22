import torch
from torch_geometric.nn import RGCNConv
from .mask_rgcn_conv import MaskRGCNConv
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
    def __init__(self, input_dim, output_dim, num_relations, mlp_n_in_dim, mlp_c_in_dim,
                 mlp_t_in_dim=None, degree_param= 0.0001, is_dropout=None, attn_weight_type='split',
                 adapter_type=None, metric_learning='similarity', num_bases=10, is_embed=False):
        super().__init__()
        self.is_embed = is_embed
        if self.is_embed:
            self.mlp_w = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.conv = MaskRGCNConv(input_dim, output_dim, num_relations, mlp_n_in_dim, mlp_c_in_dim,
                                 mlp_t_in_channels=mlp_t_in_dim, degree_param=degree_param,
                                 is_dropout=is_dropout, attn_weight_type=attn_weight_type,
                                 adapter_type=adapter_type, metric_learning=metric_learning,
                                 num_bases=num_bases)

    def forward(self, x, edge_index, edge_type, score, deg):
        if self.is_embed:
            x = self.mlp_w(x)
        x = self.conv(x, edge_index, edge_type, score, deg)
        return x


class MaskRGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, mlp_n_in_dim,
                 mlp_c_in_dim, mlp_t_in_dim=None, degree_param= 0.0001, is_dropout=False,
                 adapter_type=None, metric_learning='similarity', num_bases=10, is_embed=False,
                 layer=2, first_attn_weight='split'):
        super().__init__()
        self.layer = layer
        for i in range(self.layer):
            if i == 0:
                self.conv1 = MaskRGCNLayer(input_dim, hidden_dim, num_relations, mlp_n_in_dim,
                                           mlp_c_in_dim, mlp_t_in_dim=mlp_t_in_dim, degree_param=degree_param,
                                           adapter_type=adapter_type, attn_weight_type=first_attn_weight,
                                           is_dropout=is_dropout, metric_learning=metric_learning,
                                           num_bases=num_bases, is_embed=is_embed)
            else:
                self.conv2 = MaskRGCNLayer(hidden_dim, output_dim, num_relations, mlp_n_in_dim,
                                           mlp_c_in_dim, mlp_t_in_dim=mlp_t_in_dim,
                                           degree_param=degree_param, adapter_type=adapter_type,
                                           attn_weight_type='all', is_dropout=is_dropout,
                                           metric_learning=metric_learning,
                                           num_bases=num_bases, is_embed=is_embed)

    def forward(self, x, edge_index, edge_type, score=None, deg=None):
        x_out = []
        x = self.conv1(x, edge_index, edge_type, score, deg)
        x_out.append(x)
        if self.layer == 2:
            x = self.conv2(x, edge_index, edge_type, score, deg)
            x_out.append(x)
        return x_out


class AttnRGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_n_in_dim,
                 mlp_c_in_dim, mlp_t_in_dim, num_relations, num_bases=10, layer=2):
        super().__init__()
        self.layer = layer
        self.conv_list = torch.nn.ModuleList()
        for i in range(self.layer):
            if i == 0:
                self.conv_list.append(AttnRGCNConv(input_dim, hidden_dim, num_relations, mlp_n_in_dim,
                                                   mlp_c_in_dim, mlp_t_in_dim, num_bases=num_bases,
                                                   attn_type='split'))
            elif i == self.layer - 1:
                self.conv_list.append(AttnRGCNConv(hidden_dim, output_dim, num_relations, mlp_n_in_dim,
                                                   mlp_c_in_dim, mlp_t_in_dim, num_bases=num_bases,
                                                   attn_type='all'))
            else:
                self.conv_list.append(AttnRGCNConv(hidden_dim, hidden_dim, num_relations, mlp_n_in_dim,
                                                   mlp_c_in_dim, mlp_t_in_dim, num_bases=num_bases,
                                                   attn_type='all'))

    def forward(self, x, edge_index, edge_type):
        x_out = []
        for i in range(self.layer):
            x = self.conv_list[i](x, edge_index, edge_type)
            x_out.append(x)
        return x_out
