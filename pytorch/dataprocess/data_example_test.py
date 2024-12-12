import os
import sys
import torch
import numpy as np
import networkx as nx
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.data import HeteroData, Batch

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.models.han_model import HAN


def random_drop_nodes(graph_data, aug_ratio=0.1):
    r"""Contrastive model corruption: dropping nodes"""
    node_num = graph_data['uin'].num_nodes
    drop_num = int(node_num * aug_ratio)
    idx_keep = np.random.choice(node_num, (node_num - drop_num), replace=False)
    mask = torch.ones(node_num)
    mask[idx_keep] = 0
    graph_data['uin'].mask = mask
    for edge_type in graph_data.edge_types:
        edge_mask = torch.zeros(graph_data[edge_type].num_edges)
        drop_edge_idx = [index for index, value in enumerate(graph_data[edge_type].edge_index[0, :].tolist()) if
                         value not in idx_keep]
        drop_edge_idx.extend([index for index, value in enumerate(graph_data[edge_type].edge_index[1, :].tolist()) if
                              value not in idx_keep])
        drop_edge_idx = sorted(set(drop_edge_idx))
        edge_mask[drop_edge_idx] = 1
        graph_data[edge_type].mask = edge_mask
    return graph_data


def drop_edges(graph_data, aug_ratio=0.1):
    r"""Contrastive model corruption: permuting edges"""
    edge_types = graph_data.edge_types
    for edge_type in edge_types:
        edge_num = graph_data[edge_type].edge_index.shape[1]
        permute_num = int(edge_num * aug_ratio)
        idx_keep = np.random.choice(edge_num, (edge_num - permute_num), replace=False)
        mask = torch.zeros(edge_num)
        mask[idx_keep] = 1
        # new_edge_index = graph_data[edge_type].edge_index[:, idx_keep]
        graph_data[edge_type].mask = mask
    return graph_data


def generate_data_sample(node_num=10, feature_dim=20):
    pyg_data = HeteroData()
    pyg_data['uin'].x = torch.randn(node_num, feature_dim)
    pyg_data['uin'].y = torch.randint(0, 2, (node_num,))
    pyg_data['uin', 'room', 'uin'].edge_index = torch.randint(0, node_num, (2, 30))
    pyg_data['uin', 'wifi', 'uin'].edge_index = torch.randint(0, node_num, (2, 15))
    pyg_data['uin', 'friend', 'uin'].edge_index = torch.randint(0, node_num, (2, 8))
    return pyg_data


def generate_data_batch(num_graphs=3, node_num=10, feature_dim=20):
    data_list = []
    for i in range(num_graphs):
        data_list.append(generate_data_sample(node_num, feature_dim))
    return Batch.from_data_list(data_list)


def load_batch_from_file():
    path = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/processed/test_samples.pt'
    batch = torch.load(path)
    return batch


def generate_positive_samples_by_fraudar(graph_data, return_nodes_num=False):
    G = nx.DiGraph()
    # 添加节点
    for i in range(graph_data['uin'].num_nodes):
        G.add_node(i, evil_score=graph_data['uin'].score[i].item())

    edge_index = [graph_data[edge_type].edge_index for edge_type in graph_data.edge_types]
    edge_index = torch.concat(edge_index, dim=1)
    unique_edge_index, indices = torch.unique(edge_index, dim=1, return_inverse=True)
    adj = to_dense_adj(edge_index, max_num_nodes=graph_data['uin'].score.shape[0]).squeeze()

    # 添加边
    for i in range(unique_edge_index.shape[1]):
        srt = unique_edge_index[0, i].item()
        dst = unique_edge_index[1, i].item()
        G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
    best_graph, best_density, best_weight = fraudar(G)
    if return_nodes_num:
        return len(best_graph.nodes)
    else:
        idx = torch.zeros(graph_data['uin'].num_nodes, dtype=torch.int)
        flag = torch.tensor([0], dtype=torch.int)
        # 筛选掉fraudar生成子图节点数小于5的样本
        if len(best_graph.nodes) >= 5:
            idx[sorted(best_graph.nodes)] = 1
            flag = torch.tensor([1], dtype=torch.int)
        graph_data['uin'].idx = idx
        graph_data['uin'].flag = flag
        return graph_data


def load_model(model_name, sample_type):
    if model_name == 'RGCN':
        model = RGCN(input_dim=846,
                     hidden_dim=1024,
                     output_dim=846,
                     num_relations=10)
        if sample_type == 'fraudar':
            model_file = f'/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/saved_model/GNN_models/pretraining_filter_subgraph_cl/RGCN_sample_fraudar_filter_5_lr_0.001/uin_gangs_RGCN_model_best_loss.pth'
        elif sample_type == 'random':
            model_file = f'/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/saved_model/GNN_models/pretraining_filter_subgraph_cl/sample_random_filter_5_lr_0.001/uin_gangs_RGCN_model_epoch_100.pth'
    elif model_name == 'HAN':
        metadata = (['uin'],
                    [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                     ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                     ('uin', 'payee', 'uin'), ('uin', 'payer', 'uin'), ('uin', 'bankcard', 'uin'),
                     ('uin', 'download_app', 'uin')])
        model = HAN(in_channels=846,
                    hidden_channels=1024,
                    out_channels=846,
                    metadata=metadata,
                    heads=2)
        model_file = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/saved_model/GNN_models/pretraining_filter_subgraph_cl/HAN_sample_fraudar_filter_5_lr_0.001/uin_gangs_HAN_model_best_loss.pth'
    model_weight = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(model_weight)
    model.eval()
    return model


def get_edge_info(batch):
    edge_index = [batch[edge_type].edge_index for edge_type in batch.edge_types]
    edge_index = torch.concat(edge_index, dim=1)
    edge_counts = [batch[edge_type].num_edges for edge_type in batch.edge_types]
    edge_type = torch.concat([torch.ones(edge_counts[i]) * i for i in range(len(batch.edge_types))])
    return edge_index, edge_type.long()


def extract_single_subgraph(batch, node_indices):
    x = batch['uin'].x[node_indices].clone()
    score = batch['uin'].score[node_indices].clone()
    graph_data = HeteroData()
    graph_data['uin'].x = x
    graph_data['uin'].score = score
    max_node_idx = node_indices.max()
    for edge_type in batch.edge_types:
        try:
            current_max_node_idx = batch[edge_type].edge_index.max()
            max_node_idx = min(max_node_idx, current_max_node_idx)
            if max_node_idx < max_node_idx:
                node_indices = node_indices[node_indices <= max_node_idx]
            edge_index, _ = subgraph(node_indices, batch[edge_type].edge_index, relabel_nodes=True)
        except Exception as e:
            print(f"Extract subgraph error: <{e}>, "
                  f"edge type: {edge_type}, "
                  f"node_indices: {node_indices}")
        else:
            graph_data[edge_type].edge_index = edge_index
    return graph_data


def extract_batch_subgraphs(batch, subgraph_node_indices=None):
    if subgraph_node_indices is None:
        node_indices = batch['uin'].idx
        subgraph_node_indices = (node_indices == 1).nonzero().squeeze()
    x = torch.zeros_like(batch['uin'].x).to(batch['uin'].x.device)
    x[subgraph_node_indices] = batch['uin'].x[subgraph_node_indices].clone()
    new_batch = batch.clone()
    new_batch['uin'].x = x
    subgraph_max_node_idx = subgraph_node_indices.max()
    for edge_type in batch.edge_types:
        try:
            current_max_node_idx = batch[edge_type].edge_index.max()
            max_node_idx = min(subgraph_max_node_idx, current_max_node_idx)
            if max_node_idx < subgraph_max_node_idx:
                subgraph_node_indices = subgraph_node_indices[subgraph_node_indices <= max_node_idx]
            edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index)
        except Exception as e:
            print(f"Extract subgraph error: <{e}>, "
                  f"edge type: {edge_type}, "
                  f"subgraph: {node_indices}")
            del new_batch[edge_type]
        else:
            # no such type of edges
            if edge_index.shape[1] != 0:
                new_batch[edge_type].edge_index = edge_index
            else:
                del new_batch[edge_type]
    return new_batch

def cosine_similarity(h1, h2):
    h1_abs = h1.norm(dim=1)
    h2_abs = h2.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', h1_abs, h2_abs)
    return sim_matrix


if __name__ == '__main__':
    model_name = 'RGCN'
    sampling = 'fraudar'
    batch = load_batch_from_file()
    model = load_model(model_name, sampling)
    if model_name == 'RGCN':
        batch_edge_index, batch_edge_types = get_edge_info(batch)
        batch_h = model(batch['uin'].x, batch_edge_index, batch_edge_types)
    elif model_name == 'HAN':
        batch_h = model(batch.x_dict, batch.edge_index_dict)
    batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
    normal_idx = (batch['uin'].gang_label == 0).nonzero().squeeze().detach().cpu().tolist()
    gang_idx = (batch['uin'].gang_label == 1).nonzero().squeeze().detach().cpu().tolist()
    gang_batch_h_g = batch_h_g[gang_idx]
    normal_batch_h_g = batch_h_g[normal_idx]
    if sampling == 'fraudar':
        fraudar_data_list = []
        for i in range(batch['uin'].gang_label.shape[0]):
            if batch['uin'].gang_label[i] == 1:
                node_indices = (batch['uin'].batch == i).nonzero().squeeze()
                pyg_data = extract_single_subgraph(batch, node_indices)
                fraudar_data = generate_positive_samples_by_fraudar(pyg_data)
                fraudar_data_list.append(fraudar_data)
        fraudar_batch = Batch.from_data_list(fraudar_data_list)

        if model_name == 'RGCN':
            fraudar_batch_edge_index, fraudar_batch_edge_types = get_edge_info(fraudar_batch)
            fraudar_batch_h = model(fraudar_batch['uin'].x, fraudar_batch_edge_index, fraudar_batch_edge_types)
        elif model_name == 'HAN':
            fraudar_batch_h = model(fraudar_batch.x_dict, fraudar_batch.edge_index_dict)
        fraudar_batch_h_g = scatter_mean(fraudar_batch_h, fraudar_batch['uin'].batch, dim=0)
        print(torch.cosine_similarity(gang_batch_h_g, fraudar_batch_h_g).mean())


    print(cosine_similarity(gang_batch_h_g, gang_batch_h_g).mean())
    print(cosine_similarity(normal_batch_h_g, normal_batch_h_g).mean())
    print(cosine_similarity(gang_batch_h_g, normal_batch_h_g).mean())
