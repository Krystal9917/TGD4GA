import json
import os
import sys

import numpy as np
from sklearn.feature_extraction import FeatureHasher

os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
from transformers import BertModel, AutoTokenizer

from mmgog_long_term_sequence_model.pytorch.models.uin_gangs_model import UinGangsModel


class UinGangsModelInfer:
    def __init__(self, args_dict):
        self.args_dict = args_dict

        self.model = UinGangsModel(  # device=self.device,
            uin_acs_numberical_feat_dim=self.args_dict["uin_acs_numberical_feat_dim"],
            uin_acs_text_feat_dim=self.args_dict["uin_acs_text_feat_dim"],
            uin_in_size=self.args_dict["uin_in_size"],
            uin_out_size=self.args_dict["uin_out_size"],
            drop_rate=self.args_dict["drop_rate"],
            out_size=self.args_dict["out_size"],
        )

    def init_model(self):
        self.model.load_state_dict(
            torch.load(self.args_dict["model_states_path"], map_location=self.args_dict["infer_device"]))
        self.model.to(self.args_dict["infer_device"])
        self.model.eval()

        # 预训练的文本embedding模型
        self.minirbt_tokenizer = AutoTokenizer.from_pretrained(self.args_dict["minirbt_path"])
        self.minirbt_model = BertModel.from_pretrained(self.args_dict["minirbt_path"]).to(
            self.args_dict["infer_device"])

        # 冻结文本模型的参数
        for param in self.minirbt_model.parameters():
            param.requires_grad = False

    def infer(self, line):
        sample = self.infer_data_process(line)
        with torch.no_grad():
            pre_graph = self.model(sample["subgraph_data"])
            output = pre_graph.nodes["uin"].data["classify_out"][0]
            return torch.sigmoid(output)

    def infer_data_process(self, line):
        data = {}
        sample = {}
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            error_position = e.pos
            print("Error context:")
            print(line[max(0, error_position - 50):error_position + 50])

        # 提取 subgraph_data
        graph_schema = data["graph_schema"]

        # 提取节点信息
        uin_src_node = []
        uin_dst_node = []

        uin_src_node = torch.tensor(uin_src_node, dtype=torch.long)
        uin_dst_node = torch.tensor(uin_dst_node, dtype=torch.long)
        # 创建异构图
        graph_data = {
            ('uin', 'uin-spread-uin', 'uin'): (uin_src_node, uin_dst_node),
        }
        # 节点数
        num_nodes_dict = {"uin": len(graph_schema["uin2nodeid_map"])}
        g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

        # 数值特征
        g.nodes['uin'].data['uin_acs_numberical_feat'] = torch.from_numpy(np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_numberical_feat"]["float_list"],
            dtype=np.float32)).float()

        # 类别特征
        hasher = FeatureHasher(n_features=self.args_dict["uin_acs_categorical_feat_hasher_dim"],
                               input_type='string')
        g.nodes['uin'].data['uin_acs_categorical_feat'] = torch.from_numpy(hasher.transform(np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"]["string_list"])).toarray()).float()
        print("g.nodes['uin'].data['uin_acs_categorical_feat']", g.nodes['uin'].data['uin_acs_categorical_feat'])

        # # 文本特征
        text_list = np.array(
            graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
        text_input = self.minirbt_tokenizer(text_list, max_length=256, padding="max_length",
                                            truncation=True, return_tensors="pt")

        with torch.no_grad():
            uin_acs_text_feat = self.minirbt_model(text_input["input_ids"], text_input[
                "attention_mask"]).pooler_output

        g.nodes['uin'].data['uin_acs_text_feat'] = uin_acs_text_feat
        print("g.nodes['uin'].data['uin_acs_text_feat']", g.nodes['uin'].data['uin_acs_text_feat'])

        sample["subgraph_data"] = g

        return sample
