import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class PCARepresentationAlignment(nn.Module):
    """
    PCA-based anomaly score estimation module.

    Implements the formula:
        S_i^{(v)} = f_Φ([ PCA(X_i^{(v)}) || PCA(X_i^{(v)}) - (1/|N(v)|) * Σ_{u∈N(v)} PCA(X_i^{(u)}) ])

    This replaces the original ano_estimate MLP. For each node v, this module:
      1. Reduces input features via PCA (using torch.pca_lowrank).
      2. Computes the neighbor-mean of PCA-reduced features.
      3. Concatenates [PCA(x_v) || PCA(x_v) - mean_neighbor_PCA].
      4. Applies a learnable transformation f_Φ (MLP + Sigmoid) to produce an anomaly score.
    """

    def __init__(self, pca_components, hidden_dim, dropout=0.2):
        """
        Args:
            pca_components: Number of PCA components (k in PCA). The input features
                            will be reduced to this dimension before score computation.
            hidden_dim: Hidden dimension of the learnable transformation MLP.
            dropout: Dropout rate for the MLP.
        """
        super(PCARepresentationAlignment, self).__init__()
        self.pca_components = pca_components
        # f_Φ: a 2-layer MLP that maps the concatenated representation to an anomaly score
        # Input dimension: 2 * pca_components (concatenation of [PCA || PCA - neighbor_mean_PCA])
        # Output dimension: 1 (anomaly score), with Sigmoid activation
        self.f_phi = nn.Sequential(
            nn.Linear(2 * pca_components, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def compute_pca(self, x):
        """
        Compute PCA on the input features using torch.pca_lowrank.

        Args:
            x: Node feature matrix of shape (N, D)

        Returns:
            pca_x: PCA-reduced features of shape (N, pca_components)
        """
        # Center the data
        mean = x.mean(dim=0, keepdim=True)
        x_centered = x - mean

        # Compute PCA via low-rank SVD
        # U: (N, q), S: (q,), V: (D, q) where q = pca_components
        U, S, V = torch.pca_lowrank(x_centered, q=self.pca_components)

        # Project data onto principal components: X_centered @ V
        pca_x = x_centered @ V
        return pca_x

    def compute_neighbor_mean_pca(self, pca_x, edge_index, num_nodes):
        """
        Compute the neighbor-mean of PCA-reduced features for each node.

        Args:
            pca_x: PCA-reduced features of shape (N, pca_components)
            edge_index: Graph edge index of shape (2, E)
            num_nodes: Number of nodes in the graph

        Returns:
            neighbor_mean: Mean PCA features of neighbors for each node,
                           shape (N, pca_components)
        """
        src, dst = edge_index[0], edge_index[1]

        # Gather PCA features of source nodes (neighbors pointing to dst)
        neighbor_features = pca_x[src]  # (E, pca_components)

        # Scatter mean: for each destination node, average its incoming neighbor features
        neighbor_mean = scatter_mean(
            neighbor_features, dst, dim=0, dim_size=num_nodes
        )  # (N, pca_components)

        # Handle isolated nodes: if a node has no neighbors, its neighbor_mean
        # will be all zeros (scatter_mean default). We leave it as-is.
        return neighbor_mean

    def forward(self, x, edge_index, num_nodes=None):
        """
        Forward pass of the PCA-based anomaly score estimation.

        Args:
            x: Node feature matrix of shape (N, D)
            edge_index: Graph edge index of shape (2, E)
            num_nodes: Number of nodes. If None, inferred from x.

        Returns:
            scores: Anomaly scores of shape (N, 1)
        """
        if num_nodes is None:
            num_nodes = x.shape[0]

        # Step 1: PCA reduction
        pca_x = self.compute_pca(x)  # (N, pca_components)

        # Step 2: Neighbor-mean of PCA features
        neighbor_mean_pca = self.compute_neighbor_mean_pca(
            pca_x, edge_index, num_nodes
        )  # (N, pca_components)

        # Step 3: Concatenate [PCA(x_v) || PCA(x_v) - mean_neighbor_PCA]
        residual = pca_x - neighbor_mean_pca  # (N, pca_components)
        concat_features = torch.cat([pca_x, residual], dim=1)  # (N, 2 * pca_components)

        # Step 4: Learnable transformation f_Φ → anomaly score
        scores = self.f_phi(concat_features)  # (N, 1)

        return scores


def apply_pca_alignment_to_batch(batch_x, batch_edge_index, batch_ptr, pca_align_module):
    """
    Apply PCA-based anomaly score estimation to a batched graph,
    processing each subgraph independently.

    This function handles the batched PyG Data object, computing anomaly scores
    for each individual graph within the batch via the PCA alignment module.

    Args:
        batch_x: Node features for the entire batch, shape (total_N, D)
        batch_edge_index: Edge index for the entire batch, shape (2, total_E)
        batch_ptr: Pointer tensor indicating graph boundaries, shape (num_graphs + 1,)
        pca_align_module: Instance of PCARepresentationAlignment

    Returns:
        batch_scores: Anomaly scores for the entire batch, shape (total_N,)
    """
    score_list = []
    num_graphs = batch_ptr.shape[0] - 1

    for g in range(num_graphs):
        start = batch_ptr[g].item()
        end = batch_ptr[g + 1].item()

        # Extract edges within this subgraph (edges where both endpoints are in [start, end))
        edge_mask = (
            (batch_edge_index[0] >= start)
            & (batch_edge_index[0] < end)
            & (batch_edge_index[1] >= start)
            & (batch_edge_index[1] < end)
        )
        g_edge_index = batch_edge_index[:, edge_mask] - start  # relabel to 0-based
        g_x = batch_x[start:end]
        g_num_nodes = end - start

        g_scores = pca_align_module(g_x, g_edge_index, num_nodes=g_num_nodes)  # (g_num_nodes, 1)
        score_list.append(g_scores.squeeze(-1))  # (g_num_nodes,)

    batch_scores = torch.cat(score_list, dim=0)  # (total_N,)
    return batch_scores
