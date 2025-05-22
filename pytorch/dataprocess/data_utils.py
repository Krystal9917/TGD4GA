import json
import os.path
import random
import torch
import numpy as np
import networkx as nx
from fraudar import fraudar
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj

def read_file(lines):
    while_count = 0
    abnormal_count = 0
    for i, line in enumerate(lines):
        data = json.loads(line)
        if '正常' in data['original_label']:
            while_count += 1
        elif '团伙' in data['original_label']:
            abnormal_count += 1
    print(f"White samples: {while_count}, Abnormal samples: {abnormal_count}")

def split_data(time, in_dir, infile_name, out_dir, train_file_name, test_file_name, exclude=100, k_shot=None):

    with open(in_dir + infile_name, 'r') as f:
        lines = f.readlines()
    total_lines = len(lines)
    if k_shot is not None:
        random.seed(time)
        random.shuffle(lines)
        train_lines = lines[:exclude][:k_shot]
        test_lines = lines[exclude:]
    else:
        random.seed(42)
        random.shuffle(lines)
        print(f"{time}-All: {total_lines}")
        test_len = int(total_lines * 0.2)
        test_lines = lines[(time - 1) * test_len: time * test_len]
        if time == 1:
            train_lines = lines[time * test_len:]
        elif time == 5:
            train_lines = lines[:(time - 1) * test_len]
        else:
            train_lines = lines[:(time - 1) * test_len] + lines[time * test_len:]

    read_file(train_lines)
    print(f"{time}-Train: {len(train_lines)}")
    read_file(test_lines)
    print(f"{time}-Test: {len(test_lines)}")

    with open(out_dir + train_file_name, 'w') as train_f:
        for line in train_lines:
            train_f.write(line)

    with open(out_dir + test_file_name, 'w') as test_f:
        for line in test_lines:
            test_f.write(line)

def process_json_to_pyg(json_data, undirected_edge_types):
    graph_data = HeteroData()
    # edge information
    graph_schema = json_data['graph_schema']
    edge_type_sets = graph_schema["edge_sets"].keys()
    all_edge_type_list = []
    for combined_edges in edge_type_sets:
        edges = combined_edges.split('|')
        all_edge_type_list.extend(edges)
    all_edge_type_list = list(set(all_edge_type_list))
    edge_index_set = {}
    for edge_type in all_edge_type_list:
        edge_index_set[edge_type] = []
    for combined_edges in edge_type_sets:
        edge_types = combined_edges.split('|')
        for edge_type in edge_types:
            for edge_info in graph_schema["edge_sets"][combined_edges]["edges"]:
                try:
                    edge_index_set[edge_type].append(
                        (int(edge_info["src_nodeid"]), int(edge_info["dst_nodeid"])))
                    if edge_type in undirected_edge_types:
                        edge_index_set[edge_type].append(
                            (int(edge_info["dst_nodeid"]), int(edge_info["src_nodeid"])))
                except KeyError:
                    print("Key error")
    # add self loops
    # graph_data[('uin', 'self_loop', 'uin')].edge_index = (
    #     torch.concat([torch.tensor([[i], [i]]) for i in range(uin_acs_numberical_feat.shape[0])], dim=1))
    # other edge types
    for edge_type in all_edge_type_list:
        edge_index_set[edge_type] = list(set(edge_index_set[edge_type]))
        edge_index = [[src, dst] for (src, dst) in edge_index_set[edge_type]]
        edge_index = torch.tensor(edge_index)
        graph_data[('uin', edge_type, 'uin')].edge_index = edge_index.T
        # obtain anomaly score
        graph_data['uin'].score = torch.from_numpy(np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_evil_score"]["float_list"], dtype=np.float32)).float()
        if 'gangs_label' in json_data.keys():
            if "异常" in json_data['gangs_label'].strip():
                graph_data['uin'].gang_label = 1
            else:
                graph_data['uin'].gang_label = 0
        else:
            graph_data['uin'].gang_label = 0
        graph_data['uin'].gang_mem = torch.zeros(graph_data['uin'].score.shape[0])
        if 'uin_gangs_mem_list' in json_data.keys():
            gang_mem_list = json_data['uin_gangs_mem_list'].split(',')
            map_dict = graph_schema['uin2nodeid_map']
            for uin_gang_mem in gang_mem_list:
                try:
                    nodeid = int(map_dict[uin_gang_mem])
                except KeyError:
                    print(f"Process Node Error: Node id={uin_gang_mem} dose not exist in node map.")
                    continue
                else:
                    graph_data['uin'].gang_mem[nodeid] = 1
    return graph_data


def filter_by_fraudar(graph_data, return_nodes_num=False):
    G = nx.DiGraph()
    # 添加节点
    for i in range(graph_data['uin'].score.shape[0]):
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
        idx = torch.zeros(graph_data['uin'].score.shape[0], dtype=torch.int)
        flag = torch.tensor([0], dtype=torch.int)
        # 筛选掉fraudar生成子图节点数小于3的样本
        if len(best_graph.nodes) >= 3:
            idx[sorted(best_graph.nodes)] = 1
            flag = torch.tensor([1], dtype=torch.int)
        graph_data['uin'].idx = idx
        graph_data['uin'].flag = flag
        return graph_data

def jaccard(set_a, set_b):
    intersection = torch.sum(set_a & set_b)
    union = torch.sum(set_a | set_b)
    return intersection / union if union != torch.tensor(0) else torch.tensor(0.0)

# yanghao detection by fraudar
def filter_yanghao_by_fraudar(in_dir, file_name, train_name, test_name):
    train_list = []
    test_list = []
    undirected_edge_types = ['idcardid', 'bankcard', 'device', 'wifi', 'ipv6', 'room']
    with open(in_dir + file_name, 'r') as yh_file:
        for i, line in enumerate(yh_file):
            if len(line) == 0:
                print("Warning: line is empty")
                continue
            json_data = json.loads(line.strip())
            graph_data = process_json_to_pyg(json_data, undirected_edge_types)
            if graph_data is not None:
                graph_data = filter_by_fraudar(graph_data)
                true_y = graph_data['uin'].gang_mem.long()
                pred_y = graph_data['uin'].idx
                score = jaccard(true_y, pred_y)
                if score >= 0.5:
                    train_list.append(line)
                else:
                    test_list.append(line)
    print(f"Train Len:{len(train_list)}, Test Len: {len(test_list)}")
    with open(in_dir + train_name, 'w') as train_yh_file:
        for line in train_list:
            train_yh_file.write(line)

    with open(in_dir + test_name, 'w') as test_yh_file:
        for line in test_list:
            test_yh_file.write(line)

if __name__ == '__main__':
    parent_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/'
    in_dir = 'valid/raw/'
    in_filename = 'uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_599.txt'

    # with open(parent_dir + in_dir + in_filename, 'r') as f:
    #     lines = f.readlines()
    #     read_file(lines)
    for shot_num in np.arange(10, 110, 10):
        out_dir = f'valid/{shot_num}_shot_processed/'
        out_train_name = 'uin_gangs_supervise_full_graph_dataset_train_202503241700.txt'
        out_eval_name = 'uin_gangs_supervise_full_graph_dataset_eval_202503241700.txt'

        for idx in range(1, 6):
            out_path = parent_dir + out_dir + f'split_{idx}/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            split_data(idx, parent_dir + in_dir, in_filename, out_path, out_train_name, out_eval_name, k_shot=shot_num)