from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter

import torch_geometric.backend
import torch_geometric.typing
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


class MaskRGCNConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
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
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.is_sorted = is_sorted
        self.mask_generators = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(2 * in_channels, in_channels),
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
                edge_index: Adj, edge_type: OptTensor = None):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

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
                        out = out + (h @ weight[i])

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
                  size: Optional[Tensor] = None) -> Tensor:
        x_out, x_in = x[edge_index[0, :]], x[edge_index[1, :]]
        x_pair = torch.concat([x_out, x_in], dim=1)
        passing_weight = self.mask_generators[edge_type](x_pair)
        weighted_x_out = passing_weight * x_out
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

