import json
import logging
import os
import random
import time

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'
import numpy as np
import torch
import yaml
from sklearn.feature_extraction import FeatureHasher
from torch.utils.data import IterableDataset
import dgl
from transformers import AutoTokenizer, BertModel


class UinGangsDataIterable(IterableDataset):
    """
    数据迭代加载类
    """

    def __init__(self, args_dict, file_path):
        super(UinGangsDataIterable, self).__init__()
        self.args_dict = args_dict
        self.file_path = file_path
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
            self.line_indices = list(range(sum(1 for _ in file)))
            random.shuffle(self.line_indices)
        # print("line_indices: %s", len(self.line_indices))

        # # 预训练的文本embedding模型的分词工具
        # if torch.cuda.is_available() and self.args_dict["device"] == "gpu":
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")
        self.minirbt_tokenizer = AutoTokenizer.from_pretrained(self.args_dict["minirbt_path"])
        # self.minirbt_model = BertModel.from_pretrained(self.args_dict["minirbt_path"]).to(self.device)
        # # 冻结文本模型的参数
        # minirbt_model_params_size = 0
        # for param in self.minirbt_model.parameters():
        #     param.requires_grad = False
        #     minirbt_model_params_size += param.numel()
        # logger.info("minirbt_model params size: %s", minirbt_model_params_size)

        # self.sum = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
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
                        # print("pre_process_data")
                        buffer.append(pre_process_data)
                    if len(buffer) >= self.args_dict["data_buffer_size"]:
                        random.shuffle(buffer)
                        while buffer:
                            # print("data buffer size: ", len(buffer))
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
                return self.generate_dgl_graph(data)
            else:
                return None
        else:
            # self.sum += 1
            # print("not label data sum = ", self.sum, data["original_label"], data["root_uin"])
            return None

    def generate_dgl_graph(self, data):
        start_time = time.time()
        # 提取 label
        sample = {}
        sample["label"] = data["label"]
        # 提取 subgraph_data
        graph_schema = data["graph_schema"]

        # 提取节点信息
        uin_src_node = []
        uin_dst_node = []
        for edge_type in graph_schema["edge_sets"].keys():
            for edge in graph_schema["edge_sets"][edge_type]["edges"]:
                try:
                    uin_src_node.append(int(edge["src_nodeid"]))
                    uin_dst_node.append(int(edge["dst_nodeid"]))
                except KeyError:
                    print("Key error")
                    return None

        uin_src_node = torch.tensor(uin_src_node, dtype=torch.long)
        uin_dst_node = torch.tensor(uin_dst_node, dtype=torch.long)
        # 创建异构图
        graph_data = {
            ('uin', 'uin-spread-uin', 'uin'): (uin_src_node, uin_dst_node),
        }
        # 节点数
        num_nodes_dict = {"uin": len(graph_schema["uin2nodeid_map"])}
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
        # print("g.num_nodes(uin)", g.num_nodes("uin"))
        # print("node_feat_shape", np.array(
        #     graph_schema["node_sets"]["uin"]["data"]["uin_number_feat"]["float_list"], dtype=np.float32).shape)

        # 数值特征
        g.nodes['uin'].data['uin_acs_numberical_feat'] = torch.from_numpy(np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_numberical_feat"]["float_list"],
            dtype=np.float32)).float()

        # 类别特征
        hasher = FeatureHasher(n_features=self.args_dict["uin_acs_categorical_feat_hasher_dim"], input_type='string')
        g.nodes['uin'].data['uin_acs_categorical_feat'] = torch.from_numpy(hasher.transform(np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"]["string_list"])).toarray()).float()
        # print("g.nodes['uin'].data['uin_acs_categorical_feat']", g.nodes['uin'].data['uin_acs_categorical_feat'])

        # # 文本特征
        text_list = np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
        text_input = self.minirbt_tokenizer(text_list, max_length=256, padding="max_length",
                                            truncation=True, return_tensors="pt")
        # text_input = {k: v.to(self.device) for k, v in text_input.items()}
        # with torch.no_grad():
        #     uin_acs_text_feat = self.minirbt_model(text_input["input_ids"], text_input[
        #         "attention_mask"]).pooler_output
        #
        # g.nodes['uin'].data['uin_acs_text_feat'] = uin_acs_text_feat.to("cpu")
        # print("g.nodes['uin'].data['uin_acs_text_feat']", g.nodes['uin'].data['uin_acs_text_feat'])

        sample["uin_acs_text_feat_input_ids"] = text_input["input_ids"]
        sample["uin_acs_text_feat_attention_mask"] = text_input["attention_mask"]

        sample["uin_acs_numberical_feat_size"] = g.nodes['uin'].data['uin_acs_numberical_feat'].size(1)
        # print("sample[uin_number_feat_size]", sample["uin_acs_numberical_feat_size"])

        sample["uin_node_num"] = g.num_nodes("uin")
        # print("sample[uin_node_num]", sample["uin_node_num"])

        sample["subgraph_data"] = g

        end_time = time.time()
        # logger.info(f"The generate heterograph took {end_time - start_time} seconds to complete.")

        return sample

    def uin_gangs_collate_fn(self, batch):
        start_time = time.time()
        # print("batch", len(batch))
        batch_data = {}
        batch_label = []
        batch_graph = []
        batch_uin_acs_text_feat_input_ids = []
        batch_uin_acs_text_feat_attention_mask = []

        for sample in batch:
            batch_label.append(int(sample["label"]))
            batch_graph.append(sample["subgraph_data"])
            batch_uin_acs_text_feat_input_ids.append(sample["uin_acs_text_feat_input_ids"])
            batch_uin_acs_text_feat_attention_mask.append(sample["uin_acs_text_feat_attention_mask"])

        batch_data["batch_label"] = torch.tensor(batch_label, dtype=torch.long)
        batch_data["batch_graph"] = dgl.batch(batch_graph)
        batch_data["batch_uin_acs_text_feat_input_ids"] = batch_uin_acs_text_feat_input_ids
        batch_data["batch_uin_acs_text_feat_attention_mask"] = batch_uin_acs_text_feat_attention_mask

        # batch_data["batch_uin_acs_text_feat"] = batch_uin_acs_text_feat

        # print("Number of subgraphs in the batch:", len(batch_data["batch_graph"].batch_num_nodes()))

        end_time = time.time()
        # logger.info(f"The batch collate_fn took {end_time - start_time} seconds to complete.")
        # print("batch_data", batch_data["batch_uin_acs_text_feat"])

        return batch_data
