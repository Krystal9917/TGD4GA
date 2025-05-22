import os
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from transformers import AutoTokenizer, BertModel
from sklearn.feature_extraction import FeatureHasher
from data_statistics import process_json_to_pyg, pyg_to_nx_and_plot

def plot_density_figure(homo_list, hetero_list, tag='node', gang_tag='Anomalies'):
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")
    sns.histplot(homo_list, bins=30, color='blue', stat='frequency', label='Homophily', alpha=0.5)
    sns.histplot(hetero_list, bins=30, color='orange', stat='frequency', label='Heterophily', alpha=0.5)
    if tag == 'node':
        plt.title(f'{gang_tag} Homophily and Heterophily Distributions')
    else:
        plt.title(f'Anomalous Subgraph Homophily and Heterophily Distributions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

def compute_node_homo_hetero_value(pyg_data, gang_label=1):
    gang_idx = (pyg_data['uin'].gang_mem == gang_label).nonzero().squeeze().tolist()
    total_edge_index = torch.concat([pyg_data[edge_type].edge_index for edge_type in pyg_data.edge_types], dim=1)
    adj = to_dense_adj(total_edge_index, max_num_nodes=pyg_data['uin'].num_nodes).squeeze()
    node_homo_list = []
    node_hetero_list = []
    if type(gang_idx) is int:
        gang_idx = [gang_idx]
    for idx in gang_idx:
        edge_idx = (adj[idx, :] != 0).nonzero().squeeze().tolist()
        if type(edge_idx) is int:
            edge_idx = [edge_idx]
        same_class_idx = list(set(edge_idx) & set(gang_idx))
        homo_edges = adj[idx, same_class_idx].sum()
        total_edges = adj[idx, :].sum()
        homo_value = homo_edges / total_edges
        hetero_value = 1 - homo_value
        node_homo_list.append(homo_value)
        node_hetero_list.append(hetero_value)
    node_homo_list = torch.stack(node_homo_list, dim=0)
    node_hetero_list = torch.stack(node_hetero_list, dim=0)
    return node_homo_list, node_hetero_list, node_homo_list.mean(), node_hetero_list.mean()


def compute_subgraph_homo_hetero_value(file_name):
    minirbt_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
                    'projects/mmgog_long_term_sequence_model/minirbt-h256')
    minirbt_tokenizer = AutoTokenizer.from_pretrained(minirbt_path)
    minirbt_model = BertModel.from_pretrained(minirbt_path)
    hasher = FeatureHasher(n_features=256, input_type='string')
    node_homo_list = []
    node_hetero_list = []
    subgraph_homo_list = []
    subgraph_hetero_list = []
    normal_node_homo_list = []
    normal_node_hetero_list = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pyg_data = process_json_to_pyg(data, hasher, minirbt_tokenizer, minirbt_model)
            node_homo, node_hetero, _, _ = compute_node_homo_hetero_value(pyg_data)
            normal_node_homo, normal_node_hetero, _, _ = compute_node_homo_hetero_value(pyg_data, gang_label=0)
            node_homo_list.append(node_homo)
            node_hetero_list.append(node_hetero)
            normal_node_homo_list.append(normal_node_homo)
            normal_node_hetero_list.append(normal_node_hetero)
            # subgraph_homo_list.append(subgraph_homo)
            # subgraph_hetero_list.append(subgraph_hetero)
    node_homo_list = torch.concat(node_homo_list)
    node_hetero_list = torch.concat(node_hetero_list)
    normal_node_homo_list = torch.concat(normal_node_homo_list)
    normal_node_hetero_list = torch.concat(normal_node_hetero_list)
    # subgraph_homo_list = torch.stack(subgraph_homo_list, dim=0)
    # subgraph_hetero_list = torch.stack(subgraph_hetero_list, dim=0)
    plot_density_figure(node_homo_list, node_hetero_list, gang_tag='Anomalies')
    plot_density_figure(normal_node_homo_list, normal_node_hetero_list, gang_tag='Normal Nodes')
    # plot_density_figure(subgraph_homo_list, subgraph_hetero_list, tag='subgraph')


def process_file(file_name):
    minirbt_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
                    'projects/mmgog_long_term_sequence_model/minirbt-h256')
    minirbt_tokenizer = AutoTokenizer.from_pretrained(minirbt_path)
    minirbt_model = BertModel.from_pretrained(minirbt_path)
    hasher = FeatureHasher(n_features=256, input_type='string')
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            pyg_data = process_json_to_pyg(data, hasher, minirbt_tokenizer, minirbt_model)
            if len(pyg_data.edge_types) != 0:
                total_edge_index = torch.concat([pyg_data[edge_type].edge_index for edge_type in pyg_data.edge_types], dim=1)
                pyg_to_nx_and_plot(pyg_data, total_edge_index)

def check_few_shot_data(shot_num):
    shot_file_dir = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model'
                     f'/data/uin_gangs_full_graph_dataset/valid/k_shot_processed/{shot_num}_shot_processed')
    for fold in range(1, 6):
        few_shot_file = os.path.join(shot_file_dir, f'split_{fold}', 'uin_gangs_supervise_full_graph_dataset_train_202503241700.txt')
        process_file(few_shot_file)

if __name__ == '__main__':
    file_name = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/'
                 'data/uin_gangs_full_graph_dataset/valid/raw/uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_599.txt')
    compute_subgraph_homo_hetero_value(file_name)