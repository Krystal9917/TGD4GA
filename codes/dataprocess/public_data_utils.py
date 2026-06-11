import os
import dgl
import json
import torch
import random
import numpy as np
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from codes.dataprocess.fraudar import fraudar
from torch_geometric.utils import k_hop_subgraph, to_dense_adj
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def load_pt_dataset(dataset_name):
    dataset_file = dataset_name + '.pt'
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir,
                                             'data', 'public_dataset', dataset_file))

    data = torch.load(data_path)
    return data

def load_dgl_dataset(dataset_name):
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir,
                                             'data', 'public_dataset', dataset_name))
    graph, label_dict = load_graphs(data_path)
    return graph, label_dict


def load_mat_dataset(dataset_name):
    dataset_file = dataset_name + '.mat'
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir,
                                             'data', 'public_dataset', dataset_file))
    data = loadmat(data_path)
    return data


def plot_k_hop_graph(edge_index, label):
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G = nx.Graph()
    G.add_edges_from(edges)

    for node in G.nodes():
        G.nodes[node]['category'] = label[node]

    # 计算布局
    pos = nx.spring_layout(G)

    # 根据类别分组节点
    nodes_cat0 = [n for n, attr in G.nodes(data=True) if attr['category'] == 0]
    nodes_cat1 = [n for n, attr in G.nodes(data=True) if attr['category'] == 1]

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, nodelist=nodes_cat0, node_color='green', label='Normal', node_size=60)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_cat1, node_color='red', label='Anomalies', node_size=60)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.legend()
    plt.axis('off')
    plt.show()


def extract_subgraph_by_fraudar(graph_data):
    G = nx.DiGraph()
    # 添加节点
    for i in range(graph_data.num_nodes):
        G.add_node(i, evil_score=graph_data.score[i].item())

    edge_index = graph_data.edge_index
    adj = to_dense_adj(edge_index, max_num_nodes=graph_data.num_nodes).squeeze()

    # 添加边
    for i in range(edge_index.shape[1]):
        srt = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
    best_graph, best_density, best_weight = fraudar(G)
    idx = torch.zeros(graph_data.num_nodes, dtype=torch.int)
    # 筛选掉fraudar生成子图节点数小于3的样本
    if len(best_graph.nodes) >= 3:
        idx[sorted(best_graph.nodes)] = 1
        flag = torch.tensor([1], dtype=torch.int)
    else:
        flag = torch.tensor([0], dtype=torch.int)
    graph_data.idx = idx
    graph_data.flag = flag
    return graph_data

# 存储划分数据集的索引
def split_data(fold, data, k_shot, pt_split, tn_split):
    data = data[pt_split:]
    perm = list(range(len(data)))
    random.seed(fold)
    random.shuffle(perm)
    test_perm = perm[:tn_split]
    tune_perm = perm[tn_split: tn_split+k_shot]
    return tune_perm, test_perm

def split_data_save_index(dataset_name, data, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if dataset_name == 'weibo':
        pt_split = 3135
        tn_split = 1100
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    elif dataset_name == 'fb':
        pt_split = 303
        tn_split = 200
        start_k_shot = 1
        end_k_shot = 21
        step_k_shot = 1
    elif dataset_name == 'Amazon':
        pt_split = 1019
        tn_split = 300
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    else:
        pt_split = 8064
        tn_split = 1700
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    index_dict = {}
    for i in range(1, 6):
        index_dict[f'{str(i)}-fold'] = {}
        for shot in range(start_k_shot, end_k_shot, step_k_shot):
            tune_index, test_index = split_data(i, data, shot, pt_split, tn_split)
            index_dict[f'{str(i)}-fold'][f'{str(shot)}-shot'] = {'tune_index': tune_index, 'test_index': test_index}
    path = os.path.join(save_dir, f'{dataset_name}_index.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(index_dict, f, ensure_ascii=False, indent=4)
    return index_dict

def batch_loss(batch, y_prob, y_true):
    batch_idx = (batch.unique().max() + 1).detach().cpu().item()
    loss = []
    pos_N = []
    for i in range(batch_idx):
        i_y_prob = y_prob[batch == i]
        n = i_y_prob.shape[0]
        i_y_true = y_true[batch == i]
        pos_i_y_idx = (i_y_true == 1).nonzero().squeeze()
        if pos_i_y_idx.shape == torch.Size([]):
            pos_i_y_idx = torch.tensor([pos_i_y_idx], device=batch.device)
        pos_n = pos_i_y_idx.shape[0]
        neg_i_y_idx = (i_y_true == 0).nonzero().squeeze()
        if neg_i_y_idx.shape == torch.Size([]):
            neg_i_y_idx = torch.tensor([neg_i_y_idx], device=batch.device)
        neg_n = neg_i_y_idx.shape[0]
        if pos_n > 0 and neg_n > 0:
            pos_N.append(pos_n)
            pos_i_y_prob = i_y_prob[pos_i_y_idx]
            neg_i_y_prob = i_y_prob[neg_i_y_idx]
            # 同一个子图里的正负节点权重取决于正负节点数量的差异
            pos_weight = torch.tensor(2 * neg_n / n).to(batch.device)
            neg_weight = torch.tensor(pos_n / n).to(batch.device)
            # 同类型的节点权重相同
            pos_part_loss = -torch.log(pos_i_y_prob + 1e-4) * pos_weight
            neg_part_loss = -torch.log(1 - neg_i_y_prob + 1e-4) * neg_weight
            sub_loss = torch.concat([pos_part_loss, neg_part_loss], dim=0).mean()
            loss.append(sub_loss * n)
        elif pos_n > 0 and neg_n == 0:
            pos_N.append(pos_n)
            pos_i_y_prob = i_y_prob[pos_i_y_idx]
            sub_loss = -torch.log(pos_i_y_prob + 1e-4)
            loss.append(sub_loss.sum())
        else:
            pos_N.append(1e3)
            neg_i_y_prob = i_y_prob[neg_i_y_idx]
            sub_loss = -torch.log(1 - neg_i_y_prob + 1e-4)
            loss.append(sub_loss.sum())
    sub_weight = (torch.e / torch.tensor(pos_N)).to(batch.device)
    return (sub_weight * torch.stack(loss, dim=0)).sum().to(batch.device)


def visualize_subgraph_nx(extracted_graph, title="Subgraph Visualization"):
    nx_graph = extracted_graph.to_networkx()
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_graph, seed=42)
    labels = extracted_graph.dstdata["label"].numpy()
    node_colors = ["red" if label == 1 else "green" for label in labels]
    nx.draw_networkx_nodes(nx_graph, pos, node_size=200, node_color=node_colors, alpha=0.6)
    nx.draw_networkx_edges(nx_graph, pos, width=1, alpha=0.6, edge_color='black')
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def multi_layer_sampling(initial_graph, seed_nodes, fanouts):
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

    dataloader = dgl.dataloading.DataLoader(
        initial_graph,
        seed_nodes,
        sampler,
        batch_size=len(seed_nodes),
        shuffle=True,
        drop_last=False
    )

    for input_nodes, output_nodes, blocks in dataloader:
        return blocks

    return None

def dbscan(x, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    y_pred = dbscan.fit_predict(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    unique_true = np.unique(y)
    colors_true = ["green", "red"]
    for i, label in enumerate(unique_true):
        mask = y == label
        ax1.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                    c=[colors_true[i]], label=f'True Class {label}',
                    s=50, alpha=0.6, edgecolors='w', linewidth=0.5)

    unique_pred = np.unique(y_pred)
    colors_pred = ["black", "blue", "orange", "grey"]
    for i, label in enumerate(unique_pred):
        mask = y_pred == label
        if label == -1:
            ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                        c='black', marker='x', label='Noise',
                        s=60, alpha=0.8, linewidth=1.2)
        else:
            ax2.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                        c=[colors_pred[i]], label=f'Cluster {label}',
                        s=50, alpha=0.6, edgecolors='w', linewidth=0.5)

    ari_score = adjusted_rand_score(y, y_pred)
    nmi_score = normalized_mutual_info_score(y, y_pred)

    ax1.set_title('Ground Truth Distribution')
    ax2.set_title(f'DBSCAN Cluster Result\nARI: {ari_score:.4f}, NMI: {nmi_score:.4f}')

    for ax in [ax1, ax2]:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    name = 'tsocial'
    k = 2
    if name in ['tfinance', 'tsocial']:
        dataset, lab_dict = load_dgl_dataset(name)
        graph = dataset[0]
        y = graph.ndata['label']
        if name == 'tfinance':
            y = y.argmax(dim=1)
        normal_idx = (y == 0).nonzero().squeeze()
        anomaly_idx = (y == 1).nonzero().squeeze()
        for idx in anomaly_idx:
            # subgraph, mapping = dgl.khop_out_subgraph(graph, idx.item(), 1)
            # extract_y = y[subgraph.ndata[dgl.NID]]
            # src, dst = subgraph.edges()
            # sub_edge_index = torch.stack([src, dst], dim=0)
            # plot_k_hop_graph(sub_edge_index, extract_y)
            blocks = multi_layer_sampling(graph, idx.unsqueeze(0), [10, 10])
            node_ids = torch.concat([blocks[0].srcdata['_ID'], blocks[0].dstdata['_ID'], blocks[1].srcdata['_ID'], blocks[1].dstdata['_ID']]).unique()
            extracted_graph = dgl.node_subgraph(graph, node_ids)
            visualize_subgraph_nx(extracted_graph, f"Sampling from node={idx.item()}")
            dbscan(extracted_graph.ndata["feature"].numpy(), extracted_graph.ndata["label"].numpy())
        # for idx in normal_idx:
        #     blocks = multi_layer_sampling(graph, idx.unsqueeze(0), [10, 10])
        #     node_ids = torch.concat([blocks[0].srcdata['_ID'], blocks[0].dstdata['_ID'], blocks[1].srcdata['_ID'],
        #                              blocks[1].dstdata['_ID']]).unique()
        #     extracted_graph = dgl.node_subgraph(graph, node_ids)
        #     visualize_subgraph_nx(extracted_graph, f"Sampling from node={idx.item()}")
        #     dbscan(extracted_graph.ndata["feature"].numpy(), extracted_graph.ndata["label"].numpy())
    elif name in ['weibo', 'reddit']:
        dataset = load_pt_dataset(name)
        anomaly_idx = (dataset.y == 1).nonzero().squeeze()
        for idx in anomaly_idx:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(idx.item(), k, dataset.edge_index,
                                                                        relabel_nodes=True)
            extract_y = dataset.y[subset]
            plot_k_hop_graph(sub_edge_index, extract_y)
    else:
        dataset = load_mat_dataset(name)
        if name == 'Amazon':
            adj = dataset['homo']
            y = dataset['label'].squeeze()
        else:
            adj = dataset['A']
            y = dataset['Label'].squeeze()
        row, col = np.nonzero(adj)
        edge_index = np.vstack((row, col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        anomaly_idx = (y == 1).nonzero()[0]
        for idx in anomaly_idx:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(int(idx), k, edge_index, relabel_nodes=True)
            extract_y = y[subset]
            plot_k_hop_graph(sub_edge_index, extract_y)