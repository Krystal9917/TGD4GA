import os
import json
import logging
import time

import torch
import yaml
import random
import numpy as np
import networkx as nx
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset
from torch_geometric.data import HeteroData, Batch
from torch_geometric.utils import to_dense_adj
from sklearn.feature_extraction import FeatureHasher
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'


class UinGangsDataIterablePyGDDP(IterableDataset):
    """
    数据迭代加载类
    """

    def __init__(self, args_dict, file_path, rank, world_size):
        super(UinGangsDataIterablePyGDDP, self).__init__()
        self.args_dict = args_dict
        self.file_path = file_path
        self.rank = rank
        self.world_size = world_size
        # 读取数据配置文件
        with open(self.args_dict["uin_gangs_enum_yaml_path"], 'r', encoding='utf-8') as file:
            self.uin_gangs_enum = yaml.safe_load(file)

        # 获取分类标签
        self.class_label_enums_dict = self.uin_gangs_enum['class_label_enums']

        # 获取每个类别标签的数量上限
        self.label_class_amount_upper_limit_dict = self.uin_gangs_enum["label_class_amount_upper_limit"]

        # 获取每个类别标签的样本权重
        self.label_class_weight_dict = self.uin_gangs_enum["label_class_weight"]
        self.label_class_weight = torch.tensor(
            [self.label_class_weight_dict[key] for key in range(len(self.label_class_weight_dict))])

        # 读取所有行号并随机打乱
        with open(self.file_path, 'r', encoding="utf-8") as file:
            self.line_indices = list(range(self.rank, sum(1 for _ in file), self.world_size))
            random.shuffle(self.line_indices)
        print(f"Rank: {self.rank}, lines: {len(self.line_indices)}")
        self.control_node_num = self.args_dict["filter_node_num"]
        self.minirbt_tokenizer = AutoTokenizer.from_pretrained(self.args_dict["minirbt_path"])
        self.undirected_edge_types = ['idcardid', 'bankcard', 'device', 'wifi', 'ipv6', 'room']
        self.hasher = FeatureHasher(n_features=300, input_type='string')

    def __len__(self):
        return len(self.line_indices)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # print(worker_info)
        if worker_info is None:  # 单进程
            line_indices = self.line_indices
        else:  # 多进程
            # 为每个进程分配打乱后的行号
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            line_indices = self.line_indices[worker_id::total_workers]

        return self.iterator(line_indices)

    def iterator(self, line_indices):
        # 定义缓冲区, 缓冲区要尽可能比 batch_size 大
        buffer = []
        with open(self.file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in line_indices:
                    line = line.strip()
                    if len(line) == 0:
                        print("Warning: line is empty")
                        continue
                    pre_process_data = self.pre_process(line)
                    if pre_process_data is not None:
                        buffer.append(pre_process_data)
                    if len(buffer) >= self.args_dict["data_buffer_size"]:
                        while buffer:
                            yield buffer.pop()
        # 处理缓冲区中剩余的数据
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def pre_process(self, line):
        data = {}
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            error_position = e.pos
            print("Error context:")
            print(line[max(0, error_position - 50):error_position + 50])

        if data["original_label"] in self.class_label_enums_dict:
            data["label"] = self.class_label_enums_dict[data["original_label"]]
            if random.random() < self.label_class_amount_upper_limit_dict[data["label"]]:
                try:
                    pyg_data = self.process_json_to_pyg(data, control_edge_number=self.control_node_num)
                except Exception as e:
                    print(f"Error: <{e}>")
                    return None
                else:
                    return pyg_data
            else:
                return None
        else:
            return None

    def filter_subgraph(self, line):
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            error_position = e.pos
            print("Error context:")
            print(line[max(0, error_position - 50):error_position + 50])
            return False
        else:
            if data["original_label"] in self.class_label_enums_dict:
                label = self.class_label_enums_dict[data["original_label"]]
                if label == 0:
                    pyg_data = self.process_json_to_pyg(data, control_edge_number=self.control_node_num)
                    if pyg_data is not None:
                        gang_mem_num = self.generate_positive_samples_by_fraudar(pyg_data, return_nodes_num=True)
                        if gang_mem_num < 3:
                            return True
                        else:
                            return False
                    else:
                        return False
                else:
                    return False
            else:
                return False

    def process_json_to_pyg(self, json_data, control_node_number=5, control_edge_number=5):
        graph_data = HeteroData()
        # edge information
        graph_schema = json_data['graph_schema']
        edge_type_sets = graph_schema["edge_sets"].keys()
        if len(edge_type_sets) > control_edge_number:
            uin_acs_numberical_feat = torch.from_numpy(np.array(
                graph_schema["node_sets"]["uin"]["data"]["uin_acs_numberical_feat"]["float_list"],
                dtype=np.float32)).float()
            if uin_acs_numberical_feat.shape[0] > control_node_number:
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
                                if edge_type in self.undirected_edge_types:
                                    edge_index_set[edge_type].append(
                                        (int(edge_info["dst_nodeid"]), int(edge_info["src_nodeid"])))
                            except KeyError:
                                print("Key error")
                for edge_type in all_edge_type_list:
                    edge_index_set[edge_type] = list(set(edge_index_set[edge_type]))
                    edge_index = [[src, dst] for (src, dst) in edge_index_set[edge_type]]
                    edge_index = torch.tensor(edge_index)
                    graph_data[('uin', edge_type, 'uin')].edge_index = edge_index.T

                uin_acs_categorical_feat = torch.from_numpy(self.hasher.transform(np.array(
                    graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"][
                        "string_list"])).toarray()).float()
                graph_data['uin'].x = torch.concat([uin_acs_numberical_feat, uin_acs_categorical_feat], dim=1)
                text_list = np.array(
                    graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
                text_input = self.minirbt_tokenizer(text_list, max_length=256, padding="max_length",
                                                    truncation=True, return_tensors="pt")
                uin_acs_text_feat_input_ids = text_input["input_ids"]
                uin_acs_text_feat_attention_mask = text_input["attention_mask"]
                graph_data['uin'].text_feat_input_ids = uin_acs_text_feat_input_ids
                graph_data['uin'].text_feat_attention_mask = uin_acs_text_feat_attention_mask
                # obtain anomaly score
                graph_data['uin'].score = torch.from_numpy(np.array(
                    graph_schema["node_sets"]["uin"]["data"]["uin_evil_score"]["float_list"], dtype=np.float32)).float()
                if json_data['original_label'].strip() in self.class_label_enums_dict.keys():
                    graph_data['uin'].y = self.class_label_enums_dict[json_data['original_label'].strip()]
                else:
                    graph_data['uin'].y = 1
                if 'gangs_label' in json_data.keys():
                    if "存在异常团伙" in json_data['gangs_label'].strip():
                        graph_data['uin'].gang_label = 1
                    else:
                        graph_data['uin'].gang_label = 0
                else:
                    graph_data['uin'].gang_label = 0
                graph_data['uin'].gang_mem = torch.zeros(graph_data['uin'].x.shape[0])
                if 'uin_gangs_mem_list' in json_data.keys():
                    gang_mem_list = json_data['uin_gangs_mem_list'].split(',')
                    map_dict = graph_schema['uin2nodeid_map']
                    for uin_gang_mem in gang_mem_list:
                        try:
                            nodeid = int(map_dict[uin_gang_mem])
                        except KeyError:
                            print(f"node {uin_gang_mem} dose not exist in this subgraph")
                        else:
                            graph_data['uin'].gang_mem[nodeid] = 1
            else:
                graph_data = None
        else:
            graph_data = None

        return graph_data

    def collate_fn(self, batch_list):
        return Batch.from_data_list(batch_list)

    def pos_collate_fn_for_random(self, batch_list):
        positive_data_list = []
        for pyg_data in batch_list:
            positive_sample = self.random_drop_nodes(pyg_data, self.args_dict["drop_ratio"])
            positive_data_list.append(positive_sample)
        return Batch.from_data_list(positive_data_list)

    def pos_collate_fn_for_fraudar(self, batch_list):
        positive_data_list = []
        for pyg_data in batch_list:
            positive_sample = self.generate_positive_samples_by_fraudar(pyg_data)
            positive_data_list.append(positive_sample)
        return Batch.from_data_list(positive_data_list)

    def drop_nodes_according_to_degree(self, graph_data, aug_ratio=0.1):
        r"""Contrastive model corruption: dropping nodes"""
        node_num = graph_data['uin'].num_nodes
        drop_num = int(node_num * aug_ratio)
        # randomly drop nodes
        # idx_perm = np.random.permutation(node_num)
        # according to the sort of node degree to drop nodes
        edge_index = torch.concat([graph_data[edge_type].edge_index for edge_type in graph_data.edge_types], dim=1)
        adj = to_dense_adj(edge_index, max_num_nodes=node_num).squeeze()
        node_degree = torch.concat([adj.sum(dim=0).unsqueeze(0), adj.sum(dim=1).unsqueeze(0)], dim=0)
        node_degree = node_degree.max(dim=0).values
        degree_set = {i: node_degree[i].item() for i in range(node_degree.shape[0])}
        degree_set = {key: value for (key, value) in sorted(degree_set.items(), key=lambda x: x[1], reverse=False)}
        idx_perm = list(degree_set.keys())
        degree_less_2 = len([deg for deg in list(degree_set.values()) if deg <= 1])
        if degree_less_2 < drop_num:
            drop_num = degree_less_2

        idx_drop = idx_perm[:drop_num]
        idx_non_drop = idx_perm[drop_num:]
        idx_non_drop.sort()
        idx_dict = {idx_non_drop[n]: n for n in list(range(len(idx_non_drop)))}
        new_graph_data = HeteroData()
        new_x = graph_data['uin'].x[idx_non_drop]
        new_graph_data['uin'].x = new_x
        for edge_type in graph_data.edge_types:
            edge_num = graph_data[edge_type].edge_index.shape[1]
            edge_index = graph_data[edge_type].edge_index.detach().cpu().numpy()
            new_edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                              (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
            try:
                new_edge_index = torch.tensor(new_edge_index).transpose_(0, 1)
                new_graph_data[edge_type].edge_index = new_edge_index
            except Exception as e:
                print(f"Drop nodes error: <{e}>")
                new_graph_data[edge_type].edge_index = graph_data[edge_type].edge_index

        return new_graph_data

    def random_drop_nodes(self, graph_data, aug_ratio=0.1):
        r"""Contrastive model corruption: dropping nodes"""
        node_num = graph_data['uin'].num_nodes
        drop_num = int(node_num * aug_ratio)
        idx_keep = np.random.choice(node_num, (node_num - drop_num), replace=False)
        mask = torch.ones(node_num)
        mask[idx_keep] = 0
        graph_data['uin'].mask = mask
        # for edge_type in graph_data.edge_types:
        #     edge_mask = torch.zeros(graph_data[edge_type].num_edges)
        #     drop_edge_idx = [index for index, value in enumerate(graph_data[edge_type].edge_index[0, :].tolist()) if
        #                      value not in idx_keep]
        #     drop_edge_idx.extend(
        #         [index for index, value in enumerate(graph_data[edge_type].edge_index[1, :].tolist()) if
        #          value not in idx_keep])
        #     drop_edge_idx = sorted(set(drop_edge_idx))
        #     edge_mask[drop_edge_idx] = 1
        #     graph_data[edge_type].mask = edge_mask
        return graph_data

    def random_drop_edges(self, graph_data, aug_ratio=0.1):
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

    def generate_positive_samples_by_random_drop(self, graph_data, aug_ratio=0.15):
        new_graph_data = self.random_drop_nodes(graph_data, aug_ratio)
        return new_graph_data

    def generate_positive_samples_by_fraudar(self, graph_data, return_nodes_num=False):
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
            if len(best_graph.nodes) >= self.control_node_num:
                idx[sorted(best_graph.nodes)] = 1
                flag = torch.tensor([1], dtype=torch.int)
            graph_data['uin'].idx = idx
            graph_data['uin'].flag = flag
            return graph_data
