import json
import torch
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj
from transformers import AutoTokenizer, BertModel
from sklearn.feature_extraction import FeatureHasher


def read_file(in_dir, infile_name, outfile_name):
    unique_data_list = []
    unique_line_list = []
    same_count = 0
    normal_count = 599
    with open(in_dir + infile_name, 'r') as f:
        for i, line in enumerate(f):
            # if i not in [40, 54, 60, 62, 105, 108, 114, 138, 149, 199, 225, 264, 290, 291, 349, 356, 401, 411, 417, 440, 453, 471, 517, 550, 563, 583, 593, 604, 618]:
            #     data = json.loads(line)
            #     if '养号' in data['original_label']:
            #         if data not in unique_data_list:
            #             unique_data_list.append(data)
            #             unique_line_list.append(line)
            #         else:
            #             same_count += 1
            data = json.loads(line)
            if '正常' in data['original_label']:
                if data not in unique_data_list:
                    unique_data_list.append(data)
                    unique_line_list.append(line)
                    normal_count -= 1
            if normal_count == 0:
                break
    print(len(unique_data_list), same_count)
    with open(in_dir + outfile_name, 'w') as f:
        for line in unique_line_list:
            f.write(line)


def process(in_dir, infile_name):
    minirbt_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
                    'projects/mmgog_long_term_sequence_model/minirbt-h256')
    minirbt_tokenizer = AutoTokenizer.from_pretrained(minirbt_path)
    minirbt_model = BertModel.from_pretrained(minirbt_path)
    hasher = FeatureHasher(n_features=256, input_type='string')
    data_list = []
    anomaly_score = []
    scaler = MinMaxScaler()
    with open(in_dir + infile_name, 'r') as f:
        for i, line in enumerate(f):
            if i in [32, 34, 35, 36]:
                data = json.loads(line)
                pyg_data = process_json_to_pyg(data, hasher, minirbt_tokenizer, minirbt_model)
                total_edge_index = torch.concat([pyg_data[edge_type].edge_index for edge_type in pyg_data.edge_types],
                                                dim=1)
                full_adj = to_dense_adj(total_edge_index, max_num_nodes=pyg_data.num_nodes).squeeze()
                gang_mem_idx = (pyg_data['uin'].gang_mem == 1).nonzero().squeeze()
                non_gang_mem_idx = (pyg_data['uin'].gang_mem == 0).nonzero().squeeze()
                gang_non_gang_adj_1 = full_adj[gang_mem_idx, :][:, non_gang_mem_idx]
                gang_non_gang_adj_2 = full_adj[non_gang_mem_idx, :]
                if len(gang_non_gang_adj_2.shape) == 1:
                    gang_non_gang_adj_2 = gang_non_gang_adj_2[gang_mem_idx]
                else:
                    gang_non_gang_adj_2 = gang_non_gang_adj_2[:, gang_mem_idx]
                gang_non_gang_adj = gang_non_gang_adj_1.T + gang_non_gang_adj_2
                gang_score = pyg_data['uin'].score[gang_mem_idx].squeeze()
                gang_adj = full_adj[gang_mem_idx, :][:, gang_mem_idx]
                gang_density, gang_non_gang_density = gang_adj.mean().item(), gang_non_gang_adj.mean().item()
                df_data = pd.DataFrame(np.array([[i, gang_density, gang_non_gang_density]]))
                data_list.append(df_data)
            # label = data['original_label']
            # x = pyg_data['uin'].x[pyg_data['uin'].gang_mem == 1]
            # gang_score = pyg_data['uin'].score[pyg_data['uin'].gang_mem == 1]
            # score = pyg_data['uin'].score
            # anomaly_score.append(pd.DataFrame(
            #     [[gang_score.mean().item(), gang_score.std().item(), score.mean().item(), score.std().item()]]))
            # x = pyg_data['uin'].x
            # x_normalized = scaler.fit_transform(x.detach().numpy())
            # pca(x_normalized, pyg_data['uin'].gang_mem.int().numpy())

            # label = pyg_data['uin'].gang_mem.unsqueeze(1).int()
            # score = pyg_data['uin'].score
            # idx = (torch.ones_like(score) * i).type(torch.LongTensor)
            # df_data = pd.DataFrame(torch.concat([idx, label, score, x], dim=1).detach().numpy())
            # data_list.append(df_data)
    # anomaly_score = pd.concat(anomaly_score)
    data_list = pd.concat(data_list)
    print()
    # data_list.to_csv(in_dir + 'full_subgraphs_connectivity.csv', index=False)


def process_json_to_pyg(json_data, hasher, minirbt_tokenizer, minirbt_model, undirected_edge_types=None):
    if undirected_edge_types is None:
        undirected_edge_types = ['idcardid', 'bankcard', 'device', 'wifi', 'ipv6', 'room',
                                 'headimg', 'signature', 'nickname', 'android_bootid_fsid']
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
    num_feature = torch.from_numpy(
        np.array(graph_schema["node_sets"]["uin"]["data"]["uin_acs_numberical_feat"]["float_list"],
                 dtype=np.float32)).float()
    cat_feature = torch.from_numpy(hasher.transform(np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_acs_categorical_feat"]["string_list"])).toarray()).float()
    text_list = np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_acs_text_feat"]["string_list"]).squeeze().tolist()
    text_input = minirbt_tokenizer(text_list, max_length=256, padding="max_length", truncation=True,
                                   return_tensors="pt")
    txt_feature = minirbt_model(text_input["input_ids"], text_input["attention_mask"]).pooler_output
    graph_data['uin'].x = torch.concat([num_feature, cat_feature, txt_feature], dim=1)
    # obtain anomaly score
    graph_data['uin'].score = torch.from_numpy(np.array(
        graph_schema["node_sets"]["uin"]["data"]["uin_evil_score"]["float_list"], dtype=np.float32)).float()
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


def statistics_gang(in_dir, file_name, tag):
    data = pd.read_csv(in_dir + file_name, header=0, dtype={'0': np.int32})
    member_num = data['0'].to_numpy()
    idx, counts = np.unique(member_num, return_counts=True)
    print(f"{tag}' average member number: {counts.mean()}")
    mean_data = data.groupby(by=['0']).mean()
    print(f"{tag}' average member score: {mean_data['1'].mean()}")
    return mean_data.to_numpy()[:, 2:]


def statistics_subgraph(in_dir, file_name, tag, file_name2=None):
    data = pd.read_csv(in_dir + file_name, header=0, dtype={'0': np.int32})
    if file_name2 is not None:
        data2 = pd.read_csv(in_dir + file_name2, header=0, dtype={'0': np.int32})
        data = pd.concat([data, data2], axis=0)
    data_group = data.groupby(by=['0'])
    feature_std = []
    for (_, sub_data) in data_group:
        scaler = MinMaxScaler()
        features_normed = scaler.fit_transform(sub_data.iloc[:, 3:])
        feature_std.append(features_normed.std(axis=0))
    feature_std = np.array(feature_std)

    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(feature_std.shape[1]), feature_std.mean(axis=0), color='blue')
    plt.title(f'Std Mean Feature of {tag}\'s')
    plt.xlabel('Feature Column')
    plt.ylabel('Std Mean')
    plt.show()
    print()


def read_subgraph(in_dir, file_name):
    data = pd.read_csv(in_dir + file_name, header=0, dtype={'0': np.int32, '1': np.int32})
    node_label = data['1'].to_numpy()
    features = data.iloc[:, 3:].to_numpy()
    return node_label, features


def pca(x, y, tag=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], cmap='viridis', c=y, edgecolor='k', s=25)
    # if tag is not None:
    #     plt.title(f'General and Yanghao Gangs in {tag}')
    # else:
    #     plt.title(f'General and Yanghao Gangs')
    if tag is not None:
        plt.title(f'{tag}\'s Gang Members and Not')
    plt.colorbar(scatter)
    plt.show()


def t_SNE(x, y, tag=None):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    data_tsne = tsne.fit_transform(x)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title(f't-SNE Visualization of {tag} Features with Labels')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid()
    plt.show()


def simple_statistics(path):
    general_file = 'general_subgraphs.csv'
    yanghao_file = 'yanghao_subgraphs.csv'
    general_features = statistics_gang(path, general_file, 'General gangs')
    yanghao_features = statistics_gang(path, yanghao_file, 'Yanghao gangs')
    general_label = np.zeros(general_features.shape[0])
    yanghao_label = np.zeros(yanghao_features.shape[0])
    # gang_features = np.concatenate((general_gang_feature, yanghao_gang_feature), axis=0)
    # gang_labels = np.concatenate((np.zeros(general_gang_feature.shape[0]), np.ones(yanghao_gang_feature.shape[0])),
    #                              axis=0)
    # scaler = MinMaxScaler()
    # gang_features_normed = scaler.fit_transform(gang_features)
    # pca(gang_features_normed, gang_labels)
    # general_label, general_features = read_subgraph(path, general_file)
    # yanghao_label, yanghao_features = read_subgraph(path, yanghao_file)
    # scaler = MinMaxScaler()
    # general_features_normed = scaler.fit_transform(general_features)[:, 590:]
    # general_distances = pdist(general_features_normed, metric='euclidean')
    # ave_general_distance = np.mean(general_distances)
    # print("General Gang Textual Feature Average Distance:", ave_general_distance)
    # t_SNE(general_features_normed, general_label, 'General Gang Textual')
    # scaler = MinMaxScaler()
    # yanghao_features_normed = scaler.fit_transform(yanghao_features)[:, 590:]
    # yanghao_distances = pdist(yanghao_features_normed, metric='euclidean')
    # ave_yanghao_distance = np.mean(yanghao_distances)
    # print("Yanghao Gang Textual Feature Average Distance:", ave_yanghao_distance)
    # t_SNE(yanghao_features_normed, yanghao_label, 'Yanghao Gang Textual')


def new_feature_statistics(path):
    full_file = 'full_gangs.csv'
    # full_gang_feature = statistics_gang(path, full_file, 'Full gangs')
    # scaler = MinMaxScaler()
    # gang_features_normed = scaler.fit_transform(full_gang_feature)
    # gang_labels = np.zeros(full_gang_feature.shape[0])
    # pca(gang_features_normed, gang_labels)
    full_exclude_file = 'full_exclude_gangs.csv'
    full_data = pd.read_csv(path+full_file, header=0, dtype={'0': np.int32})
    full_exclude_data = pd.read_csv(path+full_exclude_file, header=0, dtype={'0': np.int32})
    labels = np.concatenate([np.ones(full_data['0'].shape[0]), np.zeros(full_exclude_data['0'].shape[0])], axis=0)
    features = np.concatenate([full_data.iloc[:, 508:].to_numpy(), full_exclude_data.iloc[:, 508:].to_numpy()], axis=0)
    # general_file = 'general_subgraphs.csv'
    # yanghao_file = 'yanghao_subgraphs.csv'
    # general_data = pd.read_csv(path + general_file, header=0, dtype={'0': np.int32})
    # yanghao_data = pd.read_csv(path + yanghao_file, header=0, dtype={'0': np.int32})
    # general_label = general_data['1'].to_numpy()
    # yanghao_label = yanghao_data['1'].to_numpy()
    # yanghao_label[yanghao_label == 0] = 2
    # yanghao_label[yanghao_label == 1] = 3
    # labels = np.concatenate([general_label, yanghao_label], axis=0)
    # features = np.concatenate([general_data.iloc[:, 3:].to_numpy(), yanghao_data.iloc[:, 3:].to_numpy()], axis=0)
    scaler = MinMaxScaler()
    features_normed = scaler.fit_transform(features)
    pca(features_normed, labels)


def gang_score_std(path):
    full_file = 'full_gangs.csv'
    data = pd.read_csv(path + full_file, header=0, dtype={'0': np.int32})
    gang_data = data.groupby(by=['0'])
    mean_data = gang_data['1'].mean()
    std_data = gang_data['1'].std()
    score_data = pd.concat([mean_data, std_data], axis=1)
    score_data.to_csv(path + 'gang_score_mean_std.csv', index_label='gang_idx')
    print()


def filter_gang(path):
    select_file = '202503241700_select.txt'
    if not os.path.exists(path + select_file):
        score_file = 'gang_score_mean_std.csv'
        score_data = pd.read_csv(path + score_file, index_col=0)
        select_gang_idx = score_data.iloc[:, 1][score_data.iloc[:, 1] > 0.15].index.to_list()
        full_file = 'uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_filter.txt'
        select_list = []
        with open(path + full_file, 'r') as f:
            for i, line in enumerate(f):
                if i in select_gang_idx:
                    select_list.append(line)
        with open(path + select_file, 'w') as f:
            f.writelines(select_list)
    process(path, select_file)


def score_histogram(path, file_name, tag):
    data = pd.read_csv(path + file_name, header=0, dtype={'0': np.int32})
    stats = data['1'].describe()
    print(tag, stats)

    plt.figure(figsize=(12, 6))

    # 直方图
    plt.subplot(1, 2, 1)
    sns.histplot(data['1'], bins=30, kde=False)
    plt.title(f'Histogram of {tag}\'s Score')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 密度图
    plt.subplot(1, 2, 2)
    sns.kdeplot(data['1'], fill=True)
    plt.title(f'Density Plot of {tag}\'s Score')
    plt.xlabel('Value')
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()


def class_score_histogram(path, file_name, tag):
    data = pd.read_csv(path + file_name, header=0, dtype={'0': np.int32, '1': np.int32})
    plt.figure(figsize=(10, 6))

    bins = np.linspace(0, 1, 10)

    subset_0 = data[data['1'] == 0]['2']
    subset_1 = data[data['1'] == 1]['2']
    hist1, bin_edges = np.histogram(subset_0, bins=bins)
    hist2, _ = np.histogram(subset_1, bins=bins)
    width = np.diff(bin_edges)

    x1 = bin_edges[:-1]
    x2 = bin_edges[:-1] + width

    plt.bar(x1, hist1 / hist1.sum(), width=width, alpha=0.5, label='0', color='blue', align='edge')
    plt.bar(x2, hist2 / hist2.sum(), width=width, alpha=0.5, label='1', color='orange', align='edge')

    plt.title(f'{tag} Anomaly Score Histogram by Gang Label')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(title='Gang Label')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    path_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/raw/'
    # filename = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive.txt'
    # read_file(path_dir, filename)

    # yanghao_filename = 'uin_gangs_supervise_full_graph_dataset_eval_250228_202502281128.txt'
    # yanghao_filename_filter = 'uin_gangs_supervise_full_graph_dataset_eval_250228_202502281128_filter.txt'
    # read_file(path_dir, yanghao_filename, yanghao_filename_filter)

    # process(path_dir, filename)

    # full_filename = 'uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_2.txt'
    # filter_filename = 'uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_exclude_none_connect_gangs.txt'
    # white_filename = 'uin_gangs_full_graph_dataset_train_202503201445_white.txt'
    # filter_white_filename = 'uin_gangs_full_graph_dataset_train_202503201445_599_white.txt'
    # read_file(path_dir, white_filename, filter_white_filename)

    # simple_statistics(path_dir)
    # process(path_dir, yanghao_filename_filter)
    new_feature_statistics(path_dir)

    # gang_score_std(path_dir)

    # filter_gang(path_dir)

    # gang_file = 'full_gangs.csv'
    # normal_file = 'full_exclude_gangs.csv'
    # score_histogram(path_dir, gang_file, tag='Gang Member')
    # score_histogram(path_dir, normal_file, tag='Normal User')

    # general_file = 'general_subgraphs.csv'
    # yanghao_file = 'yanghao_subgraphs.csv'
    # full_file = 'full_gangs.csv'
    # statistics_subgraph(path_dir, general_file, 'General Subgraph')
    # statistics_subgraph(path_dir, yanghao_file, 'Yanghao Subgraph')
    # statistics_subgraph(path_dir, full_file, 'Full Gang')
    # class_score_histogram(path_dir, general_file, 'General Subgraph')
    # class_score_histogram(path_dir, yanghao_file, 'Yanghao Subgraph')

    # full_filter_file = 'uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700_2.txt'
    # process(path_dir, full_filter_file)

    # file_name = 'gang_score_mean_std.csv'
    # density_data = pd.read_csv(path_dir + file_name, dtype={'0': np.int32})
    # filter_data_1 = density_data[(density_data['1'] < density_data['2']) | (density_data['1'] == 0)]
    # print()
