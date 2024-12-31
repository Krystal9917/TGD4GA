import json
import os.path

import yaml
import torch
import numpy as np
from transformers import AutoTokenizer
from torch_geometric.data import HeteroData
from sklearn.feature_extraction import FeatureHasher


def process_json_to_pyg(json_data):
    graph_data = HeteroData()
    # edge information
    graph_schema = json_data['graph_schema']
    edge_type_sets = graph_schema["edge_sets"].keys()
    uin_acs_numberical_feat = torch.from_numpy(np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_acs_numberical_feat"]["float_list"],
        dtype=np.float32)).float()
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
    for edge_type in all_edge_type_list:
        edge_index_set[edge_type] = list(set(edge_index_set[edge_type]))
        edge_index = [[src, dst] for (src, dst) in edge_index_set[edge_type]]
        edge_index = torch.tensor(edge_index)
        graph_data[('uin', edge_type, 'uin')].edge_index = edge_index.T

    uin_acs_categorical_feat = torch.from_numpy(hasher.transform(np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"][
            "string_list"])).toarray()).float()
    graph_data['uin'].x = torch.concat([uin_acs_numberical_feat, uin_acs_categorical_feat], dim=1)
    text_list = np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
    text_input = minirbt_tokenizer(text_list, max_length=256, padding="max_length",
                                   truncation=True, return_tensors="pt")
    uin_acs_text_feat_input_ids = text_input["input_ids"]
    uin_acs_text_feat_attention_mask = text_input["attention_mask"]
    graph_data['uin'].text_feat_input_ids = uin_acs_text_feat_input_ids
    graph_data['uin'].text_feat_attention_mask = uin_acs_text_feat_attention_mask
    # obtain anomaly score
    graph_data['uin'].score = torch.from_numpy(np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_evil_score"]["float_list"], dtype=np.float32)).float()
    if json_data['original_label'].strip() in class_label_enums_dict.keys():
        graph_data['uin'].y = class_label_enums_dict[json_data['original_label'].strip()]
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
                print(f"Process Node Error: Node id={uin_gang_mem} dose not exist in node map.")
                continue
            else:
                graph_data['uin'].gang_mem[nodeid] = 1
    return graph_data


def count_dataset_labels(json_data):
    if json_data['original_label'].strip() in class_label_enums_dict.keys():
        gang_label = class_label_enums_dict[json_data['original_label'].strip()]
    else:
        gang_label = None
    return gang_label

if __name__ == '__main__':
    dir_path = "/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model"
    args_dict = {
        "minirbt_path": os.path.join(dir_path, "minirbt-h256"),
        "uin_gangs_enum_yaml_path": os.path.join(dir_path, "data/config/yml/uin_gangs_enum.yaml"),
        "file_path": os.path.join(dir_path, "data/uin_gangs_full_graph_dataset/valid/raw/uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive.txt"),
        "prompt_file": os.path.join(dir_path, "data/uin_gangs_full_graph_dataset/valid/raw/uin_gangs_supervise_full_graph_dataset_prompt_initialization.txt"),
        "tuning_file": os.path.join(dir_path, "data/uin_gangs_full_graph_dataset/valid/raw/uin_gangs_supervise_full_graph_dataset_prompt_tuning.txt"),
        "test_file": os.path.join(dir_path, "data/uin_gangs_full_graph_dataset/valid/raw/uin_gangs_supervise_full_graph_dataset_prompt_testing.txt")
    }
    minirbt_tokenizer = AutoTokenizer.from_pretrained(args_dict["minirbt_path"])
    undirected_edge_types = ['idcardid', 'bankcard', 'device', 'wifi', 'ipv6', 'room']
    hasher = FeatureHasher(n_features=300, input_type='string')
    with open(args_dict["uin_gangs_enum_yaml_path"], 'r', encoding='utf-8') as file:
        uin_gangs_enum = yaml.safe_load(file)
    class_label_enums_dict = uin_gangs_enum['class_label_enums']
    split_lines = {"prompt_init": {}, "prompt_tune": {}}
    prompt_init_lines = []
    prompt_tune_lines = []
    with open(args_dict["file_path"], 'r', encoding="utf-8") as file:
        for i, line in enumerate(file):
            json_data = json.loads(line)
            label = count_dataset_labels(json_data)
            if label is not None:
                if label not in split_lines["prompt_init"].keys():
                    split_lines["prompt_init"][label] = i
                    prompt_init_lines.append(line)
                if label not in split_lines["prompt_tune"].keys() and i != split_lines["prompt_init"][label]:
                    split_lines["prompt_tune"][label] = i
                    prompt_tune_lines.append(line)
    print(split_lines["prompt_init"].values())
    print(split_lines["prompt_tune"].values())
    with open(args_dict["prompt_file"], 'w', encoding="utf-8") as file:
        for line in prompt_init_lines:
            file.write(line)
    with open(args_dict["tuning_file"], 'w', encoding="utf-8") as file:
        for line in prompt_tune_lines:
            file.write(line)
    test_file = open(args_dict["test_file"], 'w', encoding="utf-8")
    except_lines = list(split_lines["prompt_init"].values()) + list(split_lines["prompt_tune"].values())
    with open(args_dict["file_path"], 'r', encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i not in except_lines:
                test_file.write(line)
    test_file.close()


