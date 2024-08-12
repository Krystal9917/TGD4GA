import json
import os

import math
import torch
from torch.utils.data import IterableDataset
import dgl


class DataIterable(IterableDataset):
    """
    数据迭代加载类
    """

    def __init__(self, args_dict, folder_path):
        super(DataIterable, self).__init__()
        self.args_dict = args_dict
        self.folder_path = folder_path
        self.file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            file_paths = self.file_paths
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            file_paths = self.file_paths[worker_id * per_worker:(worker_id + 1) * per_worker]

        for file_path in file_paths:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # print("data", data)
                yield data


class DataProcess:
    """
    数据预处理类
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict

    def uin_gangs_collate_fn(self, batch):
        print("batch", len(batch))
        batch_data = {}
        batch_sample = []
        batch_label = []
        batch_graph = []
        # batch_data["batch_sample"] = []
        # batch_data["batch_label"] = []
        # batch_data["batch_graph"] = []
        for data in batch:
            # 提取 label
            sample = {}
            sample["label"] = data["label"]
            # 提取 subgraph_data
            graph_schema = data["graph_schema"]

            # 提取节点信息
            uin_src_node = []
            uin_dst_node = []
            for edge in graph_schema["edge_sets"]["uin-spread-uin"]["edges"]:
                uin_src_node.append(int(edge["src_nodeid"]))
                uin_dst_node.append(int(edge["dst_nodeid"]))
            uin_src_node = torch.tensor(uin_src_node, dtype=torch.long)
            uin_dst_node = torch.tensor(uin_dst_node, dtype=torch.long)
            # 创建异构图
            graph_data = {
                ('uin', 'uin-spread-uin', 'uin'): (uin_src_node, uin_dst_node),
            }
            g = dgl.heterograph(graph_data)
            # 提取节点特征
            g.nodes['uin'].data['uin_number_feat'] = torch.tensor(
                graph_schema["node_sets"]["uin"]["data"]["uin_number_feat"]["float_list"], dtype=torch.float32)

            sample["uin_number_feat_size"] = g.nodes['uin'].data['uin_number_feat'].size(1)
            # print("sample[uin_number_feat_size]", sample["uin_number_feat_size"])

            sample["uin_node_num"] = g.num_nodes("uin")
            # print("sample[uin_node_num]", sample["uin_node_num"])

            sample["subgraph_data"] = g
            batch_sample.append(sample)
            batch_label.append(int(data["label"]))
            batch_graph.append(g)

            batch_data["batch_label"] = torch.tensor(batch_label, dtype=torch.long)
            batch_data["batch_graph"] = dgl.batch(batch_graph)

        # print(batch_data["batch_label"])
        print("dgl.batch(batch_graph)", dgl.batch(batch_graph))

        return batch_data
