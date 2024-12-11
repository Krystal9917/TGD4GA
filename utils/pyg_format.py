import os
import json
import time
import yaml
import mmap
import torch
import random
import numpy as np
import concurrent.futures
from collections import Counter
from sklearn.feature_extraction import FeatureHasher
from transformers import AutoTokenizer, BertModel
from torch_geometric.data import HeteroData
from get_url_data import download_url_content

def preprocess(json_str_file_path, batch_size=1000, max_workers=8):
    # load data from file
    with open(json_str_file_path, 'r', encoding='utf-8') as f:
        json_data_list = f.readlines()

    data_list = []
    label_list = []
    for json_data in json_data_list:
        pyg_data = process_json_to_pyg(json_data)
        if pyg_data['uin'].x.shape[0] != 0:
            data_list.append(pyg_data)
            label_list.append(pyg_data['root_uin'].label)
    # multiprocessing
    # total_batches = (len(json_data_list) + batch_size - 1) // batch_size
    # print(f"Total batches: {total_batches}")
    #
    # for batch_index in range(total_batches):
    #     batch_start = batch_index * batch_size
    #     batch_end = min((batch_index + 1) * batch_size, len(json_data_list))
    #     batch_json_data_list = json_data_list[batch_start:batch_end]
    #     print(f"Processing batch {batch_index + 1}/{total_batches}...")
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         samples = executor.map(process_json_to_pyg, batch_json_data_list)
    #         data_list.extend([sample for sample in samples])
    #
    #         labels = executor.map(count_label, batch_json_data_list)
    #         label_list.extend([label for label in labels])
    #
    print(f"Label count: {Counter(label_list)}")
    return data_list

def count_label(json_data):
    json_data = json.loads(json_data)
    if json_data['original_label'] in class_label_enums_dict.keys():
        return class_label_enums_dict[json_data['original_label']]
    else:
        return 1


def process_json_to_pyg(json_data, control_node_number=0, control_edge_number=0):
    json_data = json.loads(json_data)
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

            hasher = FeatureHasher(n_features=300, input_type='string')
            uin_acs_categorical_feat = torch.from_numpy(hasher.transform(np.array(
                graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"]["string_list"])).toarray()).float()
            text_list = np.array(
                graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
            text_input = minirbt_tokenizer(text_list, max_length=256, padding="max_length",
                                           truncation=True, return_tensors="pt")
            uin_acs_text_feat_input_ids = text_input["input_ids"]
            uin_acs_text_feat_attention_mask = text_input["attention_mask"]
            with torch.no_grad():
                uin_acs_text_feat = minirbt_model(uin_acs_text_feat_input_ids,
                                                  uin_acs_text_feat_attention_mask).pooler_output
            graph_data['uin'].x = torch.concat([uin_acs_numberical_feat, uin_acs_categorical_feat, uin_acs_text_feat],
                                               dim=1)
            if json_data['original_label'].strip() in class_label_enums_dict.keys():
                graph_data['root_uin'].label = class_label_enums_dict[json_data['original_label'].strip()]
            else:
                graph_data['root_uin'].label = 1
        else:
            graph_data['uin'].x = torch.tensor([])
    else:
        graph_data['uin'].x = torch.tensor([])

    return graph_data

def random_select(url_file_path, output_file_path, ratio, tag, max_workers=8):
    with open(url_file_path, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]
    select_idx_list = random.sample(urls, int(len(urls) * ratio))
    select_idx_list = list(sorted(select_idx_list))
    with open(prefix_dir + f'/data/uin_gangs_full_graph_dataset/{tag}/240929_1/uin_gangs_full_graph_dataset_{tag}_240929_1_selected_{str(sample_ratio)}_index.txt',
            'w', encoding='utf-8') as file:
        for idx in select_idx_list:
            file.write(str(idx) + '\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(download_url_content, select_idx_list)
    with open(output_file_path, 'a', encoding='utf-8') as outfile:  # 注意使用追加模式 'a'
        for content in results:
            if content:
                outfile.write(content + '\n')

def sample_statistics(samples):
    sample_type_dict = {}
    for sample in samples:
        if sample['root_uin'].label not in sample_type_dict.keys():
            sample_type_dict[sample['root_uin'].label] = [samples.index(sample)]
        else:
            sample_type_dict[sample['root_uin'].label].append(samples.index(sample))
    for sample_type in sample_type_dict.keys():
        sample_type_list = sample_type_dict[sample_type]
        sample_type_items = [samples[idx] for idx in sample_type_list]
        aver_edges, aver_nodes, min_max_dict, edge_type_ratio_dict = edge_statistics(sample_type_items)
        print(f"Type: {sample_type}: count: {len(sample_type_list)}, sample ratio: {len(sample_type_list) / len(samples)}, "
              f"average edges: {aver_edges}, average nodes: {aver_nodes}, "
              f"{min_max_dict}, {edge_type_ratio_dict}")

def edge_statistics(samples):
    max_nodes = 0
    min_nodes = 1e8
    max_edges = 0
    min_edges = 1e8
    sum_edges = 0
    each_type_edges = {}
    node_count = 0
    for sample in samples:
        if sample.num_edges > max_edges:
            max_edges = sample.num_edges
        if sample.num_edges < min_edges:
            min_edges = sample.num_edges
        if sample['uin'].num_nodes > max_nodes:
            max_nodes = sample['uin'].num_nodes
        if sample['uin'].num_nodes < min_nodes:
            min_nodes = sample['uin'].num_nodes
        node_count += sample['uin'].num_nodes
        for edge_type in sample.edge_types:
            sum_edges += sample[edge_type].num_edges
            if edge_type not in each_type_edges.keys():
                each_type_edges[edge_type] = sample[edge_type].num_edges
            else:
                each_type_edges[edge_type] += sample[edge_type].num_edges
    return sum_edges / len(samples), node_count / len(samples), {'max_nodes': max_nodes, 'min_nodes': min_nodes, 'max_edges': max_edges, 'min_edges': min_edges}, \
           sorted({key: round(value / sum_edges, 5) for (key, value) in each_type_edges.items()}.items(), key=lambda x: x[1],reverse=True)


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    sample_ratio = 0.01
    prefix_dir = '/mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model'
    # prefix_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model'
    train_urls_file_path = prefix_dir + '/data/uin_gangs_full_graph_dataset/train/240929_1/uin_gangs_full_graph_dataset_train_240929_1.txt'
    train_output_selected_data_path = prefix_dir + f'/data/uin_gangs_full_graph_dataset/train/json_data/uin_gangs_full_graph_dataset_train_240929_1_selected_{str(sample_ratio)}.txt'
    random_select(train_urls_file_path, train_output_selected_data_path, sample_ratio, tag='train')
    eval_urls_file_path = prefix_dir + '/data/uin_gangs_full_graph_dataset/eval/240929_1/uin_gangs_full_graph_dataset_eval_240929_1.txt'
    eval_output_selected_data_path = prefix_dir + f'/data/uin_gangs_full_graph_dataset/eval/json_data/uin_gangs_full_graph_dataset_eval_240929_1_selected_{str(sample_ratio)}.txt'
    random_select(eval_urls_file_path, eval_output_selected_data_path, sample_ratio, tag='eval')
    # load yaml file
    uin_gangs_enum_yaml_path = prefix_dir + '/data/config/yml/uin_gangs_enum.yaml'
    minirbt_path = prefix_dir + '/minirbt-h256'
    # load language model
    minirbt_tokenizer = AutoTokenizer.from_pretrained(minirbt_path)
    minirbt_model = BertModel.from_pretrained(minirbt_path)
    for param in minirbt_model.parameters():
        param.requires_grad = False

    # load some yml files
    with open(uin_gangs_enum_yaml_path, 'r', encoding='utf-8') as file:
        uin_gangs_enum = yaml.safe_load(file)
    class_label_enums_dict = uin_gangs_enum['class_label_enums']
    undirected_edge_types = ['idcardid', 'bankcard', 'device', 'wifi', 'ipv6', 'room']

    start_time = time.time()
    train_pyg_data_list = preprocess(train_output_selected_data_path)
    print(f"Train data processed time: {time.time() - start_time} s")
    train_sample_save_path = prefix_dir + '/data/uin_gangs_full_graph_dataset/train/pyg/'
    if not os.path.exists(train_sample_save_path):
        os.makedirs(train_sample_save_path)
    torch.save(train_pyg_data_list, train_sample_save_path + f'random_selected_{str(sample_ratio)}_samples.pt')
    eval_pyg_data_list = preprocess(eval_output_selected_data_path)
    eval_sample_save_path = prefix_dir + '/data/uin_gangs_full_graph_dataset/eval/pyg/'
    if not os.path.exists(eval_sample_save_path):
        os.makedirs(eval_sample_save_path)
    torch.save(eval_pyg_data_list, eval_sample_save_path + f'random_selected_{str(sample_ratio)}_samples.pt')


