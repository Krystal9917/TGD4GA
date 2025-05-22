from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

import torch_geometric.backend
import torch_geometric.typing
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric import is_compiling
from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptTensor,
    SparseTensor,
    pyg_lib,
    torch_sparse,
)
from torch_geometric.utils import index_sort, one_hot, scatter, spmm


def masked_edge_index(edge_index: Adj, edge_mask: Tensor) -> Adj:
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    return torch_sparse.masked_select_nnz(edge_index, edge_mask, layout='coo')


class Adapter(torch.nn.Module):
    def __init__(self, hidden_dim, reduction=4):
        super().__init__()
        self.down_proj = torch.nn.Linear(hidden_dim, hidden_dim // reduction)
        self.up_proj = torch.nn.Linear(hidden_dim // reduction, hidden_dim)

    def forward(self, x):
        return x + self.up_proj(F.relu(self.down_proj(x)))


class MaskRGCNConv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            num_relations: int,
            mlp_n_in_channels: Optional[int] = None,
            mlp_c_in_channels: Optional[int] = None,
            mlp_t_in_channels: Optional[int] = None,
            degree_param: Optional[float] = 0.0001,
            is_dropout: bool = False,
            attn_weight_type: str = 'split',
            adapter_type: str = 'post_relation',
            metric_learning: str = 'similarity',
            num_bases: Optional[int] = None,
            num_blocks: Optional[int] = None,
            aggr: str = 'mean',
            root_weight: bool = True,
            is_sorted: bool = False,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.mlp_n_in_channels = mlp_n_in_channels
        self.mlp_c_in_channels = mlp_c_in_channels
        self.mlp_t_in_channels = mlp_t_in_channels
        self.degree_param = degree_param
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted
        self.is_dropout = is_dropout
        self.attn_weight_type = attn_weight_type
        self.adapter_type = adapter_type
        if self.adapter_type == 'post_relation':
            self.adapter = torch.nn.ModuleList([
                Adapter(in_channels, reduction=2)
                for i in range(num_relations)
            ])
        elif self.adapter_type == 'post_aggregation':
            self.adapter = Adapter(out_channels, reduction=2)
        self.metric_learning = metric_learning
        if self.attn_weight_type == 'split':
            if self.metric_learning == 'similarity':
                if self.mlp_t_in_channels is None:
                    self.gamma_1 = torch.nn.Parameter(torch.ones(num_relations))
                    self.mask_n_generators = torch.exp
                if self.mlp_c_in_channels is None:
                    self.gamma_2 = torch.nn.Parameter(torch.ones(num_relations) * 0.5)
                    self.mask_c_generators = torch.exp
                if self.mlp_t_in_channels is not None:
                    self.gamma_3 = torch.nn.Parameter(torch.ones(num_relations) * 0.5)
                    self.mask_t_generators = torch.exp
            else:
                if self.mlp_n_in_channels is not None:
                    self.mask_n_generators = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(mlp_n_in_channels, mlp_n_in_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(mlp_n_in_channels, 1),
                            torch.nn.ReLU()
                        ) for _ in range(num_relations)
                    ])
                if self.mlp_c_in_channels is not None:
                    self.mask_c_generators = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(mlp_c_in_channels, mlp_c_in_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(mlp_c_in_channels, 1),
                            torch.nn.ReLU()
                        ) for _ in range(num_relations)
                    ])
                if self.mlp_t_in_channels is not None:
                    self.mask_t_generators = torch.nn.ModuleList([
                        torch.nn.Sequential(
                            torch.nn.Linear(mlp_t_in_channels, mlp_t_in_channels),
                            torch.nn.ReLU(),
                            torch.nn.Linear(mlp_t_in_channels, 1),
                            torch.nn.ReLU()
                        ) for _ in range(num_relations)
                    ])
        else:
            if self.metric_learning == 'similarity':
                self.gamma = torch.nn.Parameter(torch.ones(num_relations))
                self.mask_generators = torch.exp
            else:
                self.mask_generators = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, in_channels),
                        torch.nn.ReLU(),
                        torch.nn.Linear(in_channels, 1),
                        torch.nn.ReLU()
                    ) for _ in range(num_relations)
                ])
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self._use_segment_matmul_heuristic_output: Optional[bool] = None

        if num_bases is not None:
            self.weight = Parameter(
                torch.empty(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.empty(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.empty(num_relations, num_blocks,
                            in_channels[0] // num_blocks,
                            out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.empty(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Parameter(torch.empty(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None,
                score: OptTensor=None, deg: OptTensor=None):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        if self.metric_learning == 'similarity':
            if self.attn_weight_type == 'split':
                if self.mlp_n_in_channels is not None:
                    self.gamma_1 = self.gamma_1.to(x.device)
                if self.mlp_c_in_channels is not None:
                    self.gamma_2 = self.gamma_2.to(x.device)
                if self.mlp_t_in_channels is not None:
                    self.gamma_3 = self.gamma_3.to(x.device)
            else:
                self.gamma = self.gamma.to(x.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))
        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor, edge_type_ptr: OptTensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if not torch.is_floating_point(
                    x_r) and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, edge_type_ptr=None, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out = out + h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================

            use_segment_matmul = torch_geometric.backend.use_segment_matmul
            # If `use_segment_matmul` is not specified, use a simple heuristic
            # to determine whether `segment_matmul` can speed up computation
            # given the observed input sizes:
            if use_segment_matmul is None:
                segment_count = scatter(torch.ones_like(edge_type), edge_type,
                                        dim_size=self.num_relations)

                self._use_segment_matmul_heuristic_output = (
                    torch_geometric.backend.use_segment_matmul_heuristic(
                        num_segments=self.num_relations,
                        max_segment_size=int(segment_count.max()),
                        in_channels=self.weight.size(1),
                        out_channels=self.weight.size(2),
                    ))

                assert self._use_segment_matmul_heuristic_output is not None
                use_segment_matmul = self._use_segment_matmul_heuristic_output

            if (use_segment_matmul and torch_geometric.typing.WITH_SEGMM
                    and not is_compiling() and self.num_bases is None
                    and x_l.is_floating_point()
                    and isinstance(edge_index, Tensor)):

                if not self.is_sorted:
                    if (edge_type[1:] < edge_type[:-1]).any():
                        edge_type, perm = index_sort(
                            edge_type, max_value=self.num_relations)
                        edge_index = edge_index[:, perm]
                edge_type_ptr = index2ptr(edge_type, self.num_relations)
                out = self.propagate(edge_index, x=x_l,
                                     edge_type_ptr=edge_type_ptr, size=size)
            else:
                if self.adapter_type == 'post_relation':
                    for i in range(self.num_relations):
                        tmp = masked_edge_index(edge_index, edge_type == i)

                        if not torch.is_floating_point(x_r):
                            out = out + self.propagate(
                                tmp,
                                x=weight[i, x_l],
                                edge_type_ptr=None,
                                size=size,
                            )
                        else:
                            h = self.propagate(tmp, x=x_l, edge_type=i, size=size)
                            h = self.adapter[i](h)
                            out = out + (h @ weight[i])
                else:
                    for i in range(self.num_relations):
                        tmp = masked_edge_index(edge_index, edge_type == i)

                        if not torch.is_floating_point(x_r):
                            out = out + self.propagate(
                                tmp,
                                x=weight[i, x_l],
                                edge_type_ptr=None,
                                size=size)
                        else:
                            if i == 0:
                                h = x_l
                            else:
                                h = self.propagate(tmp, x=x_l, edge_type=i, size=size,
                                                   score=score, deg=deg)
                            out = out + (h @ weight[i])
                    if self.adapter_type == 'post_aggregation':
                        out = self.adapter(out)

        root = self.root
        if root is not None:
            if not torch.is_floating_point(x_r):
                out = out + root[x_r]
            else:
                out = out + x_r @ root

        if self.bias is not None:
            out = out + self.bias

        return out

    def propagate(self, edge_index: Adj, x: Tensor, edge_type: int,
                  edge_type_ptr: Optional[Tensor] = None,
                  size: Optional[Tensor] = None,
                  score: Optional[Tensor] = None,
                  deg: Optional[Tensor] = None) -> Tensor:
        x_out, x_in = x[edge_index[0, :]], x[edge_index[1, :]]
        # x_pair = x_in - x_out
        x_pair = x_out - x_in
        if self.attn_weight_type == 'split':
            x_n_pair = x_pair[:, :self.mlp_n_in_channels]
            x_c_pair = x_pair[:, self.mlp_n_in_channels:self.mlp_n_in_channels + self.mlp_c_in_channels]
            if self.metric_learning == 'similarity':
                x_n_pair_mean = x_n_pair.abs().mean(dim=1)
                x_c_pair_mean = x_c_pair.abs().mean(dim=1)
                if self.mlp_t_in_channels is not None:
                    x_t_pair = x_pair[:, self.mlp_n_in_channels + self.mlp_c_in_channels:]
                    x_t_pair_mean = x_t_pair.abs().mean(dim=1)
                    n_weight = self.mask_n_generators(-self.gamma_1[edge_type] * x_n_pair_mean)
                    c_weight = self.mask_c_generators(-self.gamma_2[edge_type] * x_c_pair_mean)
                    t_weight = self.mask_t_generators(-self.gamma_2[edge_type] * x_t_pair_mean)
                    passing_weight = (n_weight + c_weight + t_weight).unsqueeze(1)
                else:
                    n_weight = self.mask_n_generators(-self.gamma_1[edge_type] * x_n_pair_mean)
                    c_weight = self.mask_c_generators(-self.gamma_2[edge_type] * x_c_pair_mean)
                    passing_weight = (n_weight + c_weight).unsqueeze(1)
            else:
                if self.mlp_t_in_channels is not None:
                    x_t_pair = x_pair[:, self.mlp_n_in_channels + self.mlp_c_in_channels:]
                    n_weight = self.mask_n_generators[edge_type](x_n_pair)
                    c_weight = self.mask_c_generators[edge_type](x_c_pair)
                    t_weight = self.mask_t_generators[edge_type](x_t_pair)
                    passing_weight = n_weight + c_weight + t_weight
                else:
                    n_weight = self.mask_n_generators[edge_type](x_n_pair)
                    c_weight = self.mask_c_generators[edge_type](x_c_pair)
                    passing_weight = n_weight + c_weight
        else:
            if self.metric_learning == 'similarity':
                x_pair_mean = x_pair.abs().mean(dim=1)
                passing_weight = self.mask_generators(-self.gamma[edge_type] * x_pair_mean)
            else:
                passing_weight = self.mask_generators[edge_type](x_pair)
        if score is not None and deg is not None:
            s_weight = score[edge_index[0, :]] * score[edge_index[1, :]] * (-torch.log((score[edge_index[0, :]] - score[edge_index[1, :]]).abs() + 0.1))
            d_weight = self.degree_param * deg[edge_index[0, :]] * deg[edge_index[1, :]] * (-torch.log(((deg[edge_index[0, :]] - deg[edge_index[1, :]]).abs() + 0.1) /
                                                                                           (deg[edge_index[0, :]] + deg[edge_index[1, :]])))
            s_weight = s_weight.unsqueeze(1)
            d_weight = d_weight.unsqueeze(1)
            passing_weight += s_weight + d_weight
        weighted_x_out = passing_weight * x_out
        if self.is_dropout:
            weighted_x_out = F.dropout(weighted_x_out, p=0.3, training=self.training)
        h_out = scatter(weighted_x_out, edge_index[1, :], dim=0, reduce='sum')
        h = torch.zeros_like(x, device=x.device)
        h_out_index = edge_index[1, :].unique()
        h[h_out_index] = h_out[h_out_index]
        return h

    def message(self, x_j: Tensor, edge_type_ptr: OptTensor) -> Tensor:
        if (torch_geometric.typing.WITH_SEGMM and not is_compiling()
                and edge_type_ptr is not None):
            # TODO Re-weight according to edge type degree for `aggr=mean`.
            return pyg_lib.ops.segment_matmul(x_j, edge_type_ptr, self.weight)

        return x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None)
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')
