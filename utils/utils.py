import math
import glob
import logging
import subprocess
import numpy
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, auc, silhouette_score,
                             average_precision_score)
from datetime import datetime
import os
import torch
from typing import List
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj


def print_model_size(model):
    """
    打印模型参数规模
    :param model:
    :return:
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params} parameters")
    return total_params


def min_max_scaler(tensor, epsilon=1e-10):
    min_vals = torch.min(tensor, dim=0).values
    max_vals = torch.max(tensor, dim=0).values
    range_vals = max_vals - min_vals
    # print("range_vals:", range_vals)
    scaled_tensor = (tensor - min_vals) / (range_vals + epsilon)  # 添加epsilon避免除以零
    return scaled_tensor


def pad_zero_or_truncat(seq, max_len, padding_elem):
    """
    对序列截断或补足
    :param seq:
    :param max_len:
    :return:
    """
    if len(seq) > max_len:
        # 行为太长，截断
        return seq[-max_len:]
    else:
        # 行为不足, 补齐
        pad_len = max_len - len(seq)
        seq = seq + [str(padding_elem)] * pad_len
        return seq


def get_binary_cls_base_threshold_by_youden_index(y_true, y_scores):
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 计算Youden Index
    youden_index = tpr - fpr

    # 找到最大Youden Index对应的阈值
    best_threshold = thresholds[np.argmax(youden_index)]

    y_cls = (y_scores > best_threshold).astype(int)

    return y_cls


def get_multi_cls_base_threshold_by_youden_index(y_true, y_scores):
    n_classes = y_true.shape[1]
    best_thresholds = np.zeros(n_classes)

    for i in range(n_classes):
        # 计算每个类别的ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_scores[:, i])

        # 计算Youden's Index
        youden_index = tpr - fpr

        # 找到最大Youden's Index对应的阈值
        best_thresholds[i] = thresholds[np.argmax(youden_index)]

    y_cls = (y_scores > best_thresholds).astype(int)
    return y_cls


def get_indicator_of_mutil_cls_base_sigmoid(y_true: numpy.array, y_pred: numpy.array, thresholds: List[float] = None):
    # y_true:(batch_size, n_classes) y_pred:(batch_size, n_classes_prob)
    num_classes = y_true.shape[1]
    if thresholds is None:
        y_pred_cls = get_multi_cls_base_threshold_by_youden_index(y_true, y_pred)
    else:
        y_pred_cls = (y_pred > thresholds).astype(int)

    # 计算 auc, precision, recall, f1, confusion_matrix
    roc_auc_scores = {}
    pr_auc_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    confusion_mats = {}
    for i in range(num_classes):
        roc_auc_scores[i] = roc_auc_score(y_true[:, i], y_pred[:, i])
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        pr_auc_scores[i] = auc(recall, precision)
        precision_scores[i] = precision_score(y_true[:, i], y_pred_cls[:, i], zero_division=0.0)
        recall_scores[i] = recall_score(y_true[:, i], y_pred_cls[:, i], zero_division=0.0)
        f1_scores[i] = f1_score(y_true[:, i], y_pred_cls[:, i], zero_division=0.0)
        confusion_mats[i] = confusion_matrix(y_true[:, i], y_pred_cls[:, i])
    return roc_auc_scores, pr_auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats


def get_indicator_of_mutil_cls_base_softmax(y_true, y_pred, num_classes):
    # y_true:(batch_size, 1) y_pred:(batch_size, n_classes_prob)
    y_pred_cls = np.argmax(y_pred, axis=1)
    # 计算 auc, precision, recall, f1, confusion_matrix
    auc_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    confusion_mats = confusion_matrix(y_true, y_pred_cls)

    for i in range(num_classes):
        auc_scores[i] = roc_auc_score(np.eye(num_classes)[y_true.tolist()][:, i], y_pred[:, i])

    precision_per_class = precision_score(y_true, y_pred_cls, average=None, zero_division=0.0)
    recall_per_class = recall_score(y_true, y_pred_cls, average=None, zero_division=0.0)
    f1_score_per_class = f1_score(y_true, y_pred_cls, average=None, zero_division=0.0)
    for i, (precision, recall, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_score_per_class)):
        precision_scores[i] = precision
        recall_scores[i] = recall
        f1_scores[i] = f1
    return auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats


def start_log():
    # 创建一个日志记录器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # 创建一个输出到控制台的处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[<%(asctime)s> <%(filename)s:%(lineno)d> %(levelname)s]\n %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # 创建一个输出到文件的处理器
    # file_handler = TimedRotatingFileHandler(os.path.abspath(
    #     os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "log",
    #                  datetime.now().strftime("%Y%m%d%H%M%S") + ".log")), when='D', interval=1, backupCount=7)
    log_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "log",
                     datetime.now().strftime("%Y%m%d")))
    os.makedirs(log_path, exist_ok=True)
    file_handler = logging.FileHandler((os.path.join(log_path, datetime.now().strftime("%Y%m%d%H%M%S") + ".log")))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[<%(asctime)s> <%(filename)s:%(lineno)d> %(levelname)s]\n %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info('-' * 50 + 'logger start' + '-' * 50)


def export_requirements(output_path):
    """
    导出当前虚拟环境中的依赖包列表到指定路径的 requirements.txt 文件。

    参数:
    output_path (str): requirements.txt 文件的输出路径。
    """
    try:
        # 使用 subprocess 运行 pip freeze 命令并将输出写入指定文件
        with open(output_path, 'w') as f:
            subprocess.run(['pip', 'freeze'], stdout=f, check=True)
        print(f"Requirements exported to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while exporting requirements: {e}")


def draw_and_save_loss_pic(x_dict, y_dict, save_path, title="标题"):
    """
    绘制并保存折线图
    """
    xlabel = next(iter(x_dict))
    x = x_dict[xlabel]

    for ylabel, y in y_dict.items():
        plt.plot(x, y, marker='o', label=ylabel)

    plt.xlabel(xlabel)
    # 添加标题
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_and_save_pca_pic(x, y, save_path=None, title="标题"):
    """
    绘制并保存PCA图
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    # 可视化结果，使用标签来区分不同的样本
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=150)
    plt.title(title)
    plt.xlabel('uin_embedding')
    plt.ylabel('exposed_label')
    plt.colorbar(scatter)
    plt.savefig(save_path)
    plt.close()


def eval_emb_with_knn(X, k=10):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    # 计算轮廓系数
    score = silhouette_score(X, kmeans.labels_)
    return score


def visualization_fig_save(input_embedding, input_class, save_path, visual_type='PCA', is_show=False,
                           data_type='Subgraph Type'):
    if visual_type == 'PCA':
        vis = PCA(n_components=2)
    else:
        vis = TSNE(n_components=2, random_state=42)
    h_reduced = vis.fit_transform(input_embedding)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(h_reduced[:, 0], h_reduced[:, 1], c=input_class, marker='.', cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter, ticks=np.arange(0, 2))
    cbar.set_label(data_type)
    cbar.ax.set_yticklabels(np.arange(0, 2))
    plt.title(f'{visual_type} Result')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(save_path, dpi=300)
    if is_show:
        plt.show()


def neg_subgraph_based_cross_entropy(neg_sub_idx, batch, y_pred, y_true):
    neg_subgraph_loss = []
    for i in neg_sub_idx:
        neg_i_y_pred = y_pred[batch == i]
        neg_i_y_true = y_true[batch == i]
        neg_loss = -((1 - neg_i_y_true) * torch.log(1 - neg_i_y_pred + 1e-4)).mean()
        neg_subgraph_loss.append(neg_loss)
    neg_subgraph_loss = torch.stack(neg_subgraph_loss)
    return neg_subgraph_loss


def pos_subgraph_based_cross_entropy(pos_sub_idx, batch, y_pred, y_true, pos_weight=2.0, neg_weight=1.0):
    pos_subgraph_loss = []
    for i in pos_sub_idx:
        i_y_pred = y_pred[batch == i]
        n = i_y_pred.shape[0]
        i_y_true = y_true[batch == i]
        pos_i_y_idx = (i_y_true == 1).nonzero().squeeze()
        pos_i_y_pred = i_y_pred[pos_i_y_idx]
        pos_i_y_true = i_y_true[pos_i_y_idx]
        neg_i_y_idx = (i_y_true == 0).nonzero().squeeze()
        neg_i_y_pred = i_y_pred[neg_i_y_idx]
        neg_i_y_true = i_y_true[neg_i_y_idx]
        pos_part_loss = (pos_weight * pos_i_y_true * torch.log(pos_i_y_pred + 1e-4)).sum()
        neg_part_loss = (neg_weight * (1 - neg_i_y_true) * torch.log(1 - neg_i_y_pred + 1e-4)).sum()
        pos_subgraph_loss.append(-(pos_part_loss + neg_part_loss)/n)
    pos_subgraph_loss = torch.stack(pos_subgraph_loss)
    return pos_subgraph_loss


def batch_subgraph_loss_based_cross_entropy(subgraph_y, batch, y_prob, y_true, alpha, beta, wp, wn):
    pos_sub_idx = (subgraph_y == 1).nonzero().squeeze()
    if pos_sub_idx.shape == torch.Size([]):
        pos_sub_idx = torch.tensor([pos_sub_idx], device=pos_sub_idx.device)
    if pos_sub_idx.shape[0] != 0:
        pos_loss = pos_subgraph_based_cross_entropy(pos_sub_idx, batch, y_prob, y_true, pos_weight=wp, neg_weight=wn)
    else:
        pos_loss = torch.tensor([0], device=subgraph_y.device)
    neg_sub_idx = (subgraph_y == 0).nonzero().squeeze()
    if neg_sub_idx.shape == torch.Size([]):
        neg_sub_idx = torch.tensor([neg_sub_idx], device=neg_sub_idx.device)
    if neg_sub_idx.shape[0] != 0:
        neg_loss = neg_subgraph_based_cross_entropy(neg_sub_idx, batch, y_prob, y_true)
    else:
        neg_loss = torch.tensor([0], device=subgraph_y.device)
    batch_subgraph_loss = torch.concat([alpha * pos_loss, beta * neg_loss], dim=0).mean()
    return batch_subgraph_loss


def compute_homo(idx, adj):
    homo_values = []
    for i in idx:
        edge_idx = (adj[i, :] != 0).nonzero().squeeze().tolist()
        if type(edge_idx) is int:
            edge_idx = [edge_idx]
        same_class_idx = list(set(edge_idx) & set(idx))
        homo_edges = adj[i, same_class_idx].sum()
        total_edges = adj[i, :].sum()
        homo = homo_edges / total_edges
        homo_values.append(homo)
    homo_values = torch.stack(homo_values, dim=0)
    homo_weight = torch.exp(-homo_values)
    return homo_weight

def batch_loss_based_on_homo(batch, y_prob, y_true, adj):
    batch_idx = (batch.unique().max() + 1).detach().cpu().item()
    loss = []
    for i in range(batch_idx):
        node_idx = (batch == i).nonzero().squeeze().detach().cpu().tolist()
        n = len(node_idx)
        i_y_prob = y_prob[node_idx]
        i_y_true = y_true[node_idx]
        i_adj = adj[node_idx, :][:, node_idx]
        gang_idx = (i_y_true == 1).nonzero().squeeze().detach().cpu().tolist()
        gang_weight = compute_homo(gang_idx, i_adj).to(batch.device)
        non_gang_idx = (i_y_true == 0).nonzero().squeeze().detach().cpu().tolist()
        non_gang_weight = compute_homo(non_gang_idx, i_adj).to(batch.device)
        # 同一子图中不同节点的权重
        gang_loss = gang_weight * (-torch.log(i_y_prob[gang_idx] + 1e-4))
        non_gang_loss = non_gang_weight * (-torch.log(1 - i_y_prob[non_gang_idx] + 1e-4))
        sub_loss = torch.concat([gang_loss, non_gang_loss], dim=0)
        # 同一子图再乘以相同的权重
        sub_weight = torch.ones_like(i_y_prob) * torch.exp(torch.tensor(1 / n)).to(batch.device)
        loss.append(sub_loss * sub_weight)
    return torch.concat(loss, dim=0).to(batch.device).mean()

def batch_loss(batch, y_prob, y_true):
    batch_idx = (batch.unique().max() + 1).detach().cpu().item()
    loss = []
    for i in range(batch_idx):
        i_y_prob = y_prob[batch == i]
        n = i_y_prob.shape[0]
        i_y_true = y_true[batch == i]
        pos_i_y_idx = (i_y_true == 1).nonzero().squeeze()
        pos_n = pos_i_y_idx.shape[0]
        pos_i_y_prob = i_y_prob[pos_i_y_idx]
        neg_i_y_idx = (i_y_true == 0).nonzero().squeeze()
        neg_n = neg_i_y_idx.shape[0]
        neg_i_y_prob = i_y_prob[neg_i_y_idx]
        # 同一个子图里的正负节点权重取决于正负节点数量的差异
        pos_weight = torch.max(torch.tensor(1), torch.log2(torch.tensor(n / pos_n) +1e-4)).to(batch.device)
        neg_weight = torch.max(torch.tensor(1), torch.log2(torch.tensor(n / neg_n) +1e-4)).to(batch.device)
        # 同类型的节点权重相同
        pos_part_loss = -torch.log(pos_i_y_prob + 1e-4) * pos_weight
        neg_part_loss = -torch.log(1 - neg_i_y_prob + 1e-4) * neg_weight
        sub_loss = torch.concat([pos_part_loss, neg_part_loss], dim=0)
        sub_weight = torch.ones_like(i_y_prob) * torch.exp(torch.tensor(1 / n)).to(batch.device)
        loss.append(sub_loss * sub_weight)
    return torch.concat(loss, dim=0).to(batch.device).mean()


def old_batch_loss(batch, y_prob, y_true):
    batch_idx = (batch.unique().max() + 1).detach().cpu().item()
    loss = []
    pos_N = []
    for i in range(batch_idx):
        i_y_prob = y_prob[batch == i]
        n = i_y_prob.shape[0]
        i_y_true = y_true[batch == i]
        pos_i_y_idx = (i_y_true == 1).nonzero().squeeze()
        pos_n = pos_i_y_idx.shape[0]
        pos_N.append(pos_n)
        pos_i_y_prob = i_y_prob[pos_i_y_idx]
        neg_i_y_idx = (i_y_true == 0).nonzero().squeeze()
        neg_n = neg_i_y_idx.shape[0]
        neg_i_y_prob = i_y_prob[neg_i_y_idx]
        # 同一个子图里的正负节点权重取决于正负节点数量的差异
        pos_weight = torch.tensor(2 * neg_n / n).to(batch.device)
        neg_weight = torch.tensor(pos_n / n).to(batch.device)
        # 同类型的节点权重相同
        pos_part_loss = -torch.log(pos_i_y_prob + 1e-4) * pos_weight
        neg_part_loss = -torch.log(1 - neg_i_y_prob + 1e-4) * neg_weight
        sub_loss = torch.concat([pos_part_loss, neg_part_loss], dim=0).mean()
        loss.append(sub_loss * n)
    sub_weight = (torch.e / torch.tensor(pos_N)).to(batch.device)
    return (sub_weight * torch.stack(loss, dim=0)).sum().to(batch.device)


def compute_density_by_probability(y_true, y_prob, deg_vec):
    pos_idx = (y_true == 1).nonzero().squeeze()
    truth_dense = deg_vec[pos_idx].mean()
    target_dense = torch.mul(y_prob[pos_idx], deg_vec[pos_idx]).mean()
    d_loss = -torch.log(target_dense / truth_dense + 1e-1)
    return d_loss


def batch_dense_loss_based_cross_entropy(batch, y_prob):
    pos_sub_idx = (batch['uin'].gang_label == 1).nonzero().squeeze()
    if pos_sub_idx.shape == torch.Size([]):
        pos_sub_idx = torch.tensor([pos_sub_idx], device=pos_sub_idx.device)
    if pos_sub_idx.shape[0] != 0:
        dense_loss = []
        N = batch['uin'].num_nodes
        adj = to_dense_adj(torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types], dim=1),
                           max_num_nodes=N).squeeze()
        deg_vec = torch.zeros(N, device=batch['uin'].x.device)
        for i in range(N):
            deg_vec[i] = (adj[i, :] + adj[:, i]).sum()
        for pos_idx in pos_sub_idx:
            node_idx = (batch['uin'].batch == pos_idx).nonzero().squeeze()
            y_true = batch['uin'].gang_mem.long()[node_idx]
            i_d_loss = compute_density_by_probability(y_true, y_prob[node_idx], deg_vec[node_idx])
            dense_loss.append(i_d_loss)
        dense_loss = torch.stack(dense_loss).mean()
    else:
        dense_loss = torch.tensor(0, device=batch['uin'].x.device)
    return dense_loss


def connect_loss(y_prob, adj, epsilon):
    masked_adj = adj * y_prob.unsqueeze(1) * y_prob.unsqueeze(0)
    degree_adj, _ = torch.stack([masked_adj.sum(dim=1), masked_adj.sum(dim=0)], dim=0).max(dim=0)
    degree_adj = degree_adj + torch.eye(masked_adj.shape[0], device=masked_adj.device)
    L = torch.diag(degree_adj) - masked_adj
    L_sym = (L + L.T) / 2
    try:
        eig_value = torch.linalg.eigvalsh(L_sym.double())
    except Exception as e:
        print(f"Error: {e}")
        lambda_2 = torch.tensor(0.0, device=L_sym.device)
    else:
        lambda_2 = eig_value[1] if len(eig_value) > 1 else torch.tensor(0.0, device=L_sym.device)
    connectivity_penalty = torch.relu(-torch.log(epsilon + torch.relu(lambda_2)))
    return connectivity_penalty


def compute_fiedler_value(node_idx, adj, y_prob=None):
    adj = adj[node_idx, :][:, node_idx]
    if y_prob is not None:
        y_prob = y_prob[node_idx]
        y_1 = y_prob.unsqueeze(1)
        y_0 = y_prob.unsqueeze(0)
        adj_prob = adj * y_1 * y_0
    else:
        adj_prob = adj
    degree_adj, _ = torch.stack([adj_prob.sum(dim=1), adj_prob.sum(dim=0)], dim=0).max(dim=0)
    L = torch.diag(degree_adj) - adj_prob
    L_sym = (L + L.T) / 2
    try:
        eig_value = torch.linalg.eigvalsh(L_sym.double())
    except Exception as e:
        print(f"Error: {e}")
        lambda_2 = torch.tensor(0.0, device=L_sym.device)
    else:
        lambda_2 = eig_value[1] if len(eig_value) > 1 else torch.tensor(0.0, device=L_sym.device)
    return lambda_2


def batch_connect_loss(batch, y_prob, epsilon=1e-3):
    device = y_prob.device
    pos_sub_idx = (batch['uin'].gang_label == 1).nonzero().squeeze()
    if pos_sub_idx.shape == torch.Size([]):
        pos_sub_idx = torch.tensor([pos_sub_idx], device=pos_sub_idx.device)
    if pos_sub_idx.shape[0] != 0:
        cnt_loss = []
        multi_edge_index = torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types], dim=1)
        adj = to_dense_adj(multi_edge_index, max_num_nodes=batch['uin'].num_nodes).squeeze()
        for pos_idx in pos_sub_idx:
            total_node_idx = (batch['uin'].batch == pos_idx).nonzero().squeeze()
            true_node_idx = total_node_idx[batch['uin'].gang_mem[total_node_idx] == 1]
            pred_node_idx = total_node_idx[y_prob[total_node_idx] >= 0.5]
            fiedler_true = compute_fiedler_value(true_node_idx, adj)
            if pred_node_idx.shape == torch.Size([]):
                pred_node_idx = torch.tensor([pred_node_idx], device=device)
            if pred_node_idx.shape[0] != 0:
                fiedler_pred = compute_fiedler_value(pred_node_idx, adj, y_prob)
            else:
                fiedler_pred = torch.tensor(0.0, device=device)
            if fiedler_true.detach().cpu().item() > torch.tensor(0.0, device=device):
                i_cnt_loss = torch.relu(-torch.log((fiedler_pred / fiedler_true).abs() + epsilon))
            else:
                i_cnt_loss = torch.exp(-fiedler_pred)
            cnt_loss.append(i_cnt_loss)
        cnt_loss = torch.stack(cnt_loss).mean()
    else:
        cnt_loss = torch.tensor(0, device=device)
    return cnt_loss


def fetch_cnt_emb(total_idx, adj, batch_h, top_k=10):
    deg = adj.sum(dim=1) - 1
    i_cnt_emb_list = []
    for idx in total_idx:
        i_ngb_idx = (adj[idx, :] > 0).nonzero().squeeze()
        i_ngb_deg_idx = deg[i_ngb_idx].sort(descending=True).indices
        if i_ngb_deg_idx.shape == torch.Size([]):
            i_ngb_deg_idx = torch.tensor([i_ngb_deg_idx], device=i_ngb_deg_idx.device)
        if top_k >= i_ngb_deg_idx.shape[0]:
            k_i_ngb_deg_idx = i_ngb_deg_idx
        else:
            k_i_ngb_deg_idx = i_ngb_deg_idx[:top_k]
        i_h = batch_h[idx, :].unsqueeze(0)
        i_ngb_h = batch_h[k_i_ngb_deg_idx].mean(dim=0).unsqueeze(0)
        i_cnt_emb_list.append(torch.concat([i_h, i_ngb_h], dim=1))
    return torch.concat(i_cnt_emb_list, dim=0)

def cnt_inner_score(total_idx, adj, raw_score, batch_h, top_k=5):
    deg = adj.sum(dim=1) - 1
    adj = adj.fill_diagonal_(0)
    i_cnt_score_list = []
    for idx in total_idx:
        i_ngb_idx = (adj[idx, :] > 0).nonzero().squeeze()
        # isolated node
        if i_ngb_idx.shape == torch.Size([0]):
            i_cnt_score_list.append(torch.tensor(0.0, device=batch_h.device))
        else:
            # single connection
            if i_ngb_idx.shape == torch.Size([]):
                i_ngb_idx = torch.tensor([i_ngb_idx], device=i_ngb_idx.device)
            # i_ngb_deg_idx = deg[i_ngb_idx].sort(descending=True).indices
            score_deg = (1 - (raw_score[idx] - raw_score[i_ngb_idx]).abs()) * (
                        1 - (deg[idx] - deg[i_ngb_idx]).abs() / (deg[idx] + deg[i_ngb_idx] + 1))
            if score_deg.shape == torch.Size([]):
                score_deg = torch.tensor([score_deg], device=score_deg.device)
            i_ngb_deg_idx = score_deg.sort(descending=True).indices
            if i_ngb_deg_idx.shape == torch.Size([]):
                i_ngb_deg_idx = torch.tensor([i_ngb_deg_idx], device=i_ngb_deg_idx.device)
            if top_k < i_ngb_deg_idx.shape[0]:
                i_ngb_deg_idx = i_ngb_deg_idx[:top_k]
            k_i_ngb_deg_idx = i_ngb_idx[i_ngb_deg_idx]
            i_h = batch_h[idx, :]
            i_ngb_h = (batch_h[k_i_ngb_deg_idx] * score_deg[i_ngb_deg_idx].unsqueeze(1)).mean(dim=0)
            inner_score = torch.inner(i_h, i_ngb_h)
            i_cnt_score_list.append(inner_score)
    return torch.stack(i_cnt_score_list, dim=0).to(batch_h.device)


def plot_auc(y_test, y_scores, save_path):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, dpi=300)


def plot_line_figure(x_data,
    y1_data,
    y2_data,
    x_label='Layers',
    y_label='mAP',
    title='',
    y1_label='GCN',
    y2_label='GAT',
    y1_color='blue',
    y2_color='red',
    y1_linestyle='-',
    y2_linestyle='-',
    linewidth=2,
    marker1='o',
    marker2='^',
    grid=True,
    legend=True,
    save_path=None):
    plt.figure(figsize=(8, 6))

    # Plot both lines
    plt.plot(
        x_data, y1_data,
        color=y1_color,
        linestyle=y1_linestyle,
        linewidth=linewidth,
        marker=marker1,
        label=y1_label
    )
    plt.plot(
        x_data, y2_data,
        color=y2_color,
        linestyle=y2_linestyle,
        linewidth=linewidth,
        marker=marker2,
        label=y2_label
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.6)

    if legend:
        plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_multiple_roc_with_point(y_true_list, y_score_list, labels,
                                 point_fpr, point_tpr, point_label, k_shot=None,
                                 title='Multiple ROC Curves with Fraudar Point',
                                 figsize=(8, 6), diagonal=True):
    """
    参数:
    y_true_list -- 真实标签列表，每个元素是一个数组
    y_score_list -- 预测分数列表，每个元素是一个数组(正类的概率)
    labels -- 每条ROC曲线的标签列表
    point_fpr -- 要标注的点的假正率(FPR)
    point_tpr -- 要标注的点的真正率(TPR)
    point_label -- 点的标签文本
    title -- 图表标题
    figsize -- 图表大小
    diagonal -- 是否绘制对角线(随机猜测线)
    """

    plt.figure(figsize=figsize)

    # 绘制每条ROC曲线
    for y_true, y_score, label in zip(y_true_list, y_score_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    # 绘制特殊点
    plt.scatter(point_fpr, point_tpr, color='red', s=100,
                label=point_label, zorder=3)
    plt.text(point_fpr + 0.02, point_tpr - 0.02, point_label, fontsize=12)

    # 绘制对角线(随机猜测线)
    if diagonal:
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')

    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if k_shot is not None:
        plt.title(f"{k_shot} Shot {title}")
    else:
        plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.close()


def plot_pr_curve(y_true_list, y_score_list, labels, point_pre, point_rec, point_label, k_shot=None):
    plt.figure(figsize=(8, 6))

    # 为每个模型绘制曲线
    for true_labels, pred_scores, label in zip(y_true_list, y_score_list, labels):
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        ap = average_precision_score(true_labels, pred_scores)
        plt.plot(recall, precision, lw=2, label=f'{label} (AP = {ap:.2f})')

    # 绘制特殊点
    plt.scatter(point_rec, point_pre, color='red', s=100,
                label=point_label, zorder=3)
    plt.text(point_rec + 0.02, point_pre - 0.02, point_label, fontsize=12)

    # 随机分类器，可以添加水平线作为参考
    positive_ratio = sum(y_true_list[0]) / len(y_true_list[0])
    plt.axhline(y=positive_ratio, color='gray', linestyle='--', label='Random')

    # 添加图表元素
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    if k_shot is not None:
        plt.title(f"{k_shot} Shot Comparison of Precision-Recall Curves", fontsize=14)
    else:
        plt.title('Comparison of Precision-Recall Curves', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.show()
    plt.close()


def merge_csv_files(
        input_dir,  # 输入目录路径
        merge_column,  # 合并依据的列名
        how='inner',  # 合并方式: 'inner', 'outer', 'left', 'right'
        sort_column=None,  # 可选: 合并后按某列排序
        ascending=True,  # 排序顺序
        drop_duplicates=None,  # 可选: 去重列
        fillna=None  # 可选: 填充NA值
):
    csv_files = ['supervised_pred.csv', 'pretrain_no_finetune_pred.csv',
                 'pretrain_finetune_pred.csv', 'pretrain_finetune_weight_pred.csv',
                 'score_pred.csv']

    # 读取所有CSV文件
    dfs = []
    for file in csv_files:
        df = pd.read_csv(input_dir + file)
        dfs.append(df)

    if not dfs:
        raise ValueError("没有有效的CSV文件可合并")

    # 逐步合并所有DataFrame
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on=merge_column,
            how=how,
            suffixes=('', f'_{df.columns[1]}_merged')  # 自动处理重复列名
        )

    # 后处理
    if sort_column:
        merged_df.sort_values(by=sort_column, ascending=ascending, inplace=True)

    if drop_duplicates:
        merged_df.drop_duplicates(subset=drop_duplicates, inplace=True)

    if fillna is not None:
        merged_df.fillna(fillna, inplace=True)
    merged_df = merged_df.dropna().drop_duplicates(subset=['node_id'])
    return merged_df


def merge_k_shot_csv_files(shot_num=10):
    prefix_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
                   'projects/mmgog_long_term_sequence_model/data/prediction/')
    pretrain_path = prefix_path + f'pretrain2/pretrain_{shot_num}_shot_pred.csv'
    pretrain_ft_path = prefix_path + f'pretrain_ft10/ft_{shot_num}_shot_pred.csv'
    supervise_path = prefix_path + f'supervise2/supervise_{shot_num}_shot_pred.csv'
    raw_score_path = prefix_path + f'raw_score/score_10_shot_pred.csv'
    paths = [pretrain_path, pretrain_ft_path, supervise_path, raw_score_path]
    data_df = []
    for path in paths:
        path_data = pd.read_csv(path)
        if 'pretrain' in path and 'pretrain_ft' not in path:
            path_data.rename(columns={'y_true': 'pretrain_y_true', 'y_prob': 'pretrain_y_prob'}, inplace=True)
        elif 'pretrain_ft' in path:
            path_data.rename(columns={'y_true': 'pretrain_ft_y_true', 'y_prob': 'pretrain_ft_y_prob'}, inplace=True)
        elif 'supervise' in path:
            path_data.rename(columns={'y_true': 'supervise_y_true', 'y_prob': 'supervise_y_prob'}, inplace=True)
        else:
            path_data.rename(columns={'y_true': 'score_y_true', 'y_prob': 'score_y_prob'}, inplace=True)
        data_df.append(path_data)
    merged_df = data_df[0]
    for df in data_df[1:]:
        merged_df = pd.merge(
            merged_df,
            df,
            on='node_id',
            how='inner',
            suffixes=('', f'_{df.columns[1]}_merged')  # 自动处理重复列名
        )
    merged_df = merged_df.dropna().drop_duplicates(subset=['node_id'])
    return merged_df

def find_csv_files(pattern='*.csv'):
    path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
            'projects/mmgog_long_term_sequence_model/data/prediction/supervise2/')
    search_path = os.path.join(path, pattern)
    return glob.glob(search_path)

def compute_k_shot_metrics(pattern="supervise_*.csv", metric="AUC"):
    files = find_csv_files(pattern)
    metric_dict = {}
    for file in files:
        if "pretrain_ft" in pattern:
            shot_num = file.split('/')[-1].split('_')[2]
        else:
            shot_num = file.split('/')[-1].split('_')[1]
        file_data = pd.read_csv(file)
        file_data.drop_duplicates(inplace=True)
        if metric == "AUC":
            fpr, tpr, _ = roc_curve(file_data['y_true'], file_data['y_prob'])
            roc_auc = auc(fpr, tpr)
            metric_dict[int(shot_num)] = round(roc_auc, 4)
        else:
            y_pred = np.zeros_like(file_data['y_true'])
            y_pred[file_data['y_prob'] >= 0.5] = 1
            if metric == "Precision":
                pre = precision_score(file_data['y_true'], y_pred)
                metric_dict[int(shot_num)] = round(pre, 4)
            elif metric == "Recall":
                rec = recall_score(file_data['y_true'], y_pred)
                metric_dict[int(shot_num)] = round(rec, 4)
            else:
                f1 = f1_score(file_data['y_true'], y_pred)
                metric_dict[int(shot_num)] = round(f1, 4)
    order_list = sorted(metric_dict.items(), key=lambda x:x[0])
    print(order_list)
    print([item[1] for item in order_list])

def plot_k_shot_metrics(tag):
    x = np.arange(0, 10, 1)
    shot_num = [str(i) for i in np.arange(10, 110, 10)]
    if tag == 'AUC':
        pretrain_tuning = [0.7960, 0.8236, 0.8419, 0.8554, 0.8513, 0.8591, 0.8655, 0.8643, 0.8681, 0.8707]
        pretrain = [0.7865, 0.8186, 0.8262, 0.8391, 0.8388, 0.8458, 0.8491, 0.8486, 0.8551, 0.8556]
        supervise = [0.7386, 0.7501, 0.8039, 0.8596, 0.8461, 0.8647, 0.8711, 0.8763, 0.8683, 0.893]
        fraudar = [0.80] * 10
        raw_score = [0.8014] * 10

    elif tag == 'Precision':
        pretrain_tuning = [0.3655, 0.4117, 0.4386, 0.44, 0.4448, 0.4492, 0.4432, 0.4374, 0.4367, 0.4198]
        pretrain = [0.3939, 0.4466, 0.4603, 0.4699, 0.481, 0.4779, 0.481, 0.4834, 0.4806, 0.4931]
        supervise = [0.4723, 0.511, 0.5728, 0.5772, 0.5666, 0.5735, 0.5713, 0.6063, 0.6216, 0.607]
        fraudar = [0.3993] * 10
        raw_score = [0.3164] * 10

    elif tag == 'Recall':
        pretrain_tuning = [0.4486, 0.5055, 0.5424, 0.6174, 0.5935, 0.6382, 0.6879, 0.6847, 0.7109, 0.7474]
        pretrain = [0.2048, 0.3482, 0.345, 0.4433, 0.4137, 0.4707, 0.4766, 0.4702, 0.509, 0.5003]
        supervise = [0.2786, 0.3111, 0.4041, 0.497, 0.4928, 0.5582, 0.6016, 0.5861, 0.5675, 0.6204]
        fraudar = [0.8121] * 10
        raw_score = [0.7277] * 10

    else:
        pretrain_tuning = [0.4028, 0.4538, 0.485, 0.5138, 0.5085, 0.5273, 0.5391, 0.5338, 0.5411, 0.5376]
        pretrain = [0.2695, 0.3913, 0.3944, 0.4562, 0.4448, 0.4743, 0.4788, 0.4767, 0.4944, 0.4967]
        supervise = [0.3505, 0.3868, 0.4739, 0.5341, 0.5271, 0.5657, 0.586, 0.596, 0.5933, 0.6136]
        fraudar = [0.5345] * 10
        raw_score = [0.4285] * 10

    plt.figure(figsize=(8, 6))
    plt.plot(x, pretrain_tuning, label='pretrain-tuning')
    plt.plot(x, pretrain, label='pretrain')
    plt.plot(x, supervise, label='supervise')
    plt.plot(x, fraudar, '--', label='fraudar')
    plt.plot(x, raw_score, '--', label='raw-score')

    plt.xlim([0.0, 9.0])
    plt.xticks(x, shot_num)
    if tag == 'AUC':
        plt.ylim([0.70, 0.95])
    else:
        plt.ylim([0.20, 0.90])
    plt.xlabel('K shot')
    plt.ylabel(tag)
    plt.title(f'{tag}-K shot Comparison Figure')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.close()

def plot_k_shot_metrics2(tag):
    x = np.arange(0, 10, 1)
    shot_num = [str(i) for i in np.arange(10, 110, 10)]
    if tag == 'AUC':
        pt = [0.4754, 0.5852, 0.6262, 0.6927, 0.6995, 0.7215, 0.7487, 0.7598, 0.7654, 0.7643]
        pt_tuning_1 = [0.7785, 0.8194, 0.8294, 0.8404, 0.8474, 0.8543, 0.8549, 0.858, 0.8663, 0.8583]
        pt_tuning_2 = [0.6308, 0.7261, 0.7829, 0.8171, 0.816, 0.8277, 0.8423, 0.8415, 0.8465, 0.8506]
        pt_tuning_3 = [0.6819, 0.7124, 0.7259, 0.7295, 0.7354, 0.7638, 0.7607, 0.7657, 0.7757, 0.7711]
        pt_tuning_4 = [0.6661, 0.6978, 0.7055, 0.7181, 0.73, 0.7616, 0.7579, 0.7636, 0.7697, 0.7646]
        pt_tuning_5 = [0.5488, 0.5752, 0.6642, 0.7367, 0.7224, 0.7527, 0.7478, 0.7571, 0.7823, 0.7772]
        pt_tuning_6 = [0.5686, 0.7025, 0.7228, 0.798, 0.799, 0.8068, 0.8149, 0.8245, 0.8325, 0.8336]
        pt_tuning_8 = [0.7869, 0.8052, 0.8233, 0.8432, 0.8484, 0.8579, 0.8601, 0.8607, 0.8641, 0.8664]
        pt_tuning_9 = [0.7984, 0.816, 0.8339, 0.8488, 0.8562, 0.8694, 0.8695, 0.8722, 0.8747, 0.8768]
        pt_tuning_10 = [0.8021, 0.8205, 0.8353, 0.8527, 0.8592, 0.8636, 0.8701, 0.8714, 0.8724, 0.8784]
        pt_tuning_12 = [0.7983, 0.8134, 0.8311, 0.8499, 0.8563, 0.867, 0.8703, 0.8712, 0.8722, 0.8777]
        pt_tuning_13 = [0.7925, 0.8144, 0.8343, 0.8487, 0.8526, 0.8639, 0.869, 0.8692, 0.871, 0.87]
        pt_tuning_14 = [0.7817, 0.7985, 0.8164, 0.8158, 0.8348, 0.8489, 0.8541, 0.8563, 0.8657, 0.8685]
        pt_tuning_15 = [0.7764, 0.801, 0.8177, 0.8148, 0.831, 0.8463, 0.8505, 0.8529, 0.8634, 0.8668]
        pt_tuning_16 = [0.7916, 0.795, 0.8148, 0.828, 0.8358, 0.8486, 0.854, 0.8578, 0.8659, 0.8642]
        pt_tuning_17 = [0.8021, 0.8198, 0.8426, 0.8507, 0.8605, 0.8697, 0.872, 0.8732, 0.8733, 0.876]
        pt_tuning_24 = [0.7955, 0.8062, 0.8078, 0.8128, 0.8177, 0.8231, 0.8307, 0.8386, 0.8427, 0.8510]
        pt_tuning_25 = [0.8021, 0.8198, 0.8426, 0.8507, 0.8605, 0.8697, 0.872, 0.8732, 0.8733, 0.876]
        supervise = [0.7667, 0.801, 0.8288, 0.8299, 0.8324, 0.837, 0.8535, 0.8468, 0.8626, 0.8622]
        fraudar = [0.7942] * 10
        raw_score = [0.7910] * 10

    elif tag == 'Precision':
        pt = [0.3759, 0.4041, 0.4546, 0.4592, 0.4512, 0.4435, 0.4483, 0.4609, 0.4772, 0.4781]
        pt_tuning_1 = [0.3818, 0.4231, 0.4816, 0.4947, 0.5138, 0.4951, 0.5147, 0.509, 0.5127, 0.5169]
        pt_tuning_2 = [0.4608, 0.4841, 0.5271, 0.5389, 0.5356, 0.5289, 0.5359, 0.5496, 0.5466, 0.5596]
        pt_tuning_3 = [0.3926, 0.3975, 0.4144, 0.4179, 0.4349, 0.4275, 0.4201, 0.4421, 0.456, 0.4412]
        pt_tuning_4 = [0.3913, 0.4048, 0.4106, 0.4118, 0.4073, 0.4295, 0.424, 0.423, 0.4248, 0.4302]
        pt_tuning_5 = [0.4777, 0.4655, 0.521, 0.4968, 0.4996, 0.5017, 0.4995, 0.5022, 0.5136, 0.5072]
        pt_tuning_6 = [0.468, 0.5132, 0.5499, 0.5422, 0.5672, 0.5528, 0.5343, 0.5467, 0.553, 0.5579]
        pt_tuning_8 = [0.3921, 0.4473, 0.4966, 0.5072, 0.5183, 0.5191, 0.5132, 0.5151, 0.5234, 0.5198]
        pt_tuning_9 = [0.4541, 0.4785, 0.5451, 0.5293, 0.5249, 0.533, 0.5288, 0.5364, 0.5345, 0.5375]
        pt_tuning_10 = [0.4057, 0.4325, 0.4401, 0.4649, 0.4535, 0.4639, 0.4572, 0.4514, 0.44, 0.4415]
        pt_tuning_12 = [0.4452, 0.476, 0.5092, 0.5109, 0.5139, 0.5196, 0.5142, 0.5186, 0.5172, 0.525]
        pt_tuning_13 = [0.4486, 0.485, 0.5254, 0.5126, 0.5137, 0.5248, 0.5285, 0.5322, 0.5315, 0.5449]
        pt_tuning_14 = [0.4197, 0.4447, 0.4917, 0.4925, 0.496, 0.4965, 0.4989, 0.5076, 0.4959, 0.5077]
        pt_tuning_15 = [0.4109, 0.4319, 0.4792, 0.485, 0.4786, 0.4857, 0.4806, 0.49, 0.4744, 0.4931]
        pt_tuning_16 = [0.3863, 0.3939, 0.4343, 0.4346, 0.4472, 0.4571, 0.4418, 0.441, 0.4472, 0.4393]
        pt_tuning_17 = [0.4103, 0.4364, 0.4789, 0.4714, 0.4786, 0.48, 0.4685, 0.4637, 0.4661, 0.4506]
        pt_tuning_24 = [0.4474, 0.4517, 0.4619, 0.4694, 0.4756, 0.4802, 0.4897, 0.4947, 0.5046, 0.5114]
        pt_tuning_25 = [0.4505, 0.4618, 0.4812, 0.4926, 0.5012, 0.5126, 0.5238, 0.5321, 0.5412, 0.5508]
        supervise = [0.4706, 0.5394, 0.5445, 0.5301, 0.5451, 0.5312, 0.5319, 0.5392, 0.5465, 0.55]
        fraudar = [0.3993] * 10
        raw_score = [0.3058] * 10

    elif tag == 'Recall':
        pt = [0.1739, 0.2513, 0.2877, 0.3478, 0.36, 0.3985, 0.4242, 0.4384, 0.437, 0.4299]
        pt_tuning_1 = [0.2947, 0.3671, 0.4657, 0.5189, 0.5085, 0.5793, 0.5856, 0.5795, 0.6062, 0.5834]
        pt_tuning_2 = [0.2305, 0.2707, 0.3512, 0.3784, 0.3908, 0.4373, 0.442, 0.4274, 0.4757, 0.4448]
        pt_tuning_3 = [0.2645, 0.287, 0.3132, 0.308, 0.3216, 0.3646, 0.335, 0.3422, 0.3408, 0.3468]
        pt_tuning_4 = [0.2783, 0.2775, 0.2934, 0.3118, 0.3358, 0.3673, 0.3702, 0.3747, 0.3858, 0.3811]
        pt_tuning_5 = [0.2008, 0.2464, 0.281, 0.3769, 0.3709, 0.3847, 0.3986, 0.4231, 0.4386, 0.4347]
        pt_tuning_6 = [0.1688, 0.2229, 0.2504, 0.3367, 0.3037, 0.3287, 0.3405, 0.3713, 0.3717, 0.3529]
        pt_tuning_8 = [0.2217, 0.2674, 0.3101, 0.3817, 0.3846, 0.4017, 0.4339, 0.4273, 0.4395, 0.5028]
        pt_tuning_9 = [0.2527, 0.2989, 0.3021, 0.3494, 0.3717, 0.4214, 0.4347, 0.469, 0.4685, 0.4807]
        pt_tuning_10 = [0.4139, 0.4754, 0.5058, 0.5758, 0.6195, 0.6361, 0.6938, 0.7144, 0.7437, 0.7489]
        pt_tuning_12 = [0.2835, 0.3583, 0.3788, 0.4165, 0.4236, 0.4578, 0.4929, 0.5247, 0.5314, 0.5456]
        pt_tuning_13 = [0.2761, 0.3046, 0.3304, 0.3712, 0.358, 0.3924, 0.4264, 0.4426, 0.4435, 0.4447]
        pt_tuning_14 = [0.2606, 0.3117, 0.3461, 0.3862, 0.4211, 0.4443, 0.4616, 0.4932, 0.5393, 0.5216]
        pt_tuning_15 = [0.2678, 0.3448, 0.3767, 0.409, 0.4362, 0.4641, 0.4952, 0.5024, 0.5746, 0.5424]
        pt_tuning_16 = [0.3445, 0.3706, 0.4621, 0.5019, 0.5437, 0.5635, 0.6067, 0.6427, 0.6696, 0.6834]
        pt_tuning_17 = [0.4084, 0.4231, 0.4792, 0.5221, 0.5759, 0.612, 0.6641, 0.682, 0.6928, 0.7117]
        pt_tuning_24 = [0.7694, 0.775, 0.7773, 0.7771, 0.7826, 0.784, 0.7868, 0.7891, 0.7919, 0.7942]
        pt_tuning_25 = [0.7919, 0.7956, 0.7981, 0.8018, 0.8042, 0.8103, 0.8158, 0.8191, 0.8215, 0.8234]
        supervise = [0.296, 0.3361, 0.4197, 0.4706, 0.4901, 0.5179, 0.5702, 0.5822, 0.5912, 0.6162]
        fraudar = [0.8000] * 10
        raw_score = [0.7160] * 10

    else:
        pt = [0.2378, 0.3099, 0.3524, 0.3958, 0.4004, 0.4198, 0.4359, 0.4494, 0.4562, 0.4527]
        pt_tuning_1 = [0.3625, 0.4368, 0.4735, 0.5065, 0.5111, 0.5339, 0.5478, 0.542, 0.5555, 0.5482]
        pt_tuning_2 = [0.3073, 0.3473, 0.4215, 0.4446, 0.4519, 0.4788, 0.4845, 0.4809, 0.5087, 0.4956]
        pt_tuning_3 = [0.316, 0.3333, 0.3568, 0.3546, 0.3698, 0.3936, 0.3728, 0.3858, 0.3901, 0.3884]
        pt_tuning_4 = [0.3252, 0.3292, 0.3423, 0.3549, 0.3681, 0.396, 0.3953, 0.3974, 0.4044, 0.4041]
        pt_tuning_5 = [0.2828, 0.3223, 0.3651, 0.4287, 0.4258, 0.4355, 0.4434, 0.4593, 0.4731, 0.4682]
        pt_tuning_6 = [0.2481, 0.3108, 0.3441, 0.4155, 0.3955, 0.4122, 0.416, 0.4423, 0.4446, 0.4324]
        pt_tuning_8 = [0.2833, 0.3347, 0.3818, 0.4356, 0.4415, 0.4529, 0.4702, 0.4671, 0.4778, 0.5112]
        pt_tuning_9 = [0.3247, 0.368, 0.3887, 0.421, 0.4352, 0.4707, 0.4771, 0.5005, 0.4993, 0.5075]
        # 按原版子图大小确定子图权重，s_weight和d_weight与passing_weight相加
        pt_tuning_10 = [0.4098, 0.4529, 0.4706, 0.5145, 0.5236, 0.5365, 0.5512, 0.5533, 0.5529, 0.5556]
        # 按子图大小确定子图权重
        pt_tuning_12 = [0.3464, 0.4088, 0.4344, 0.4589, 0.4644, 0.4868, 0.5033, 0.5216, 0.5242, 0.5351]
        # 按子图homophily确定子图权重
        pt_tuning_13 = [0.2917, 0.3741, 0.3822, 0.4306, 0.4219, 0.449, 0.472, 0.4833, 0.4835, 0.4897]
        pt_tuning_14 = [0.3216, 0.3665, 0.4063, 0.4329, 0.4555, 0.4689, 0.4795, 0.5003, 0.5167, 0.514]
        pt_tuning_15 = [0.3243, 0.3834, 0.4218, 0.4438, 0.4564, 0.4747, 0.4878, 0.4961, 0.5197, 0.5166]
        pt_tuning_16 = [0.3642, 0.3819, 0.4478, 0.4658, 0.4908, 0.5048, 0.5113, 0.5231, 0.5363, 0.5348]
        pt_tuning_17 = [0.4093, 0.4297, 0.479, 0.4954, 0.5228, 0.538, 0.5494, 0.5521, 0.5573, 0.5518]
        pt_tuning_24 = [0.5192, 0.522, 0.5271, 0.5302, 0.5346, 0.5399, 0.5436, 0.5467, 0.5509, 0.5535]
        pt_tuning_25 = [0.5378, 0.5446, 0.5521, 0.5576, 0.5626, 0.5696, 0.5724, 0.5788, 0.5789, 0.5810]
        supervise = [0.3335, 0.3746, 0.4741, 0.4951, 0.5093, 0.5245, 0.5527, 0.5559, 0.5652, 0.5699]
        fraudar = [0.5238] * 10
        raw_score = [0.4285] * 10

    plt.figure(figsize=(8, 6))
    plt.plot(x, pt_tuning_1, label='pretrain-tuning')
    # plt.plot(x, pt_tuning_2, label='pretrain-tuning-mix')
    # plt.plot(x, pt_tuning_3, label='pretrain-tuning-yh-30')
    # plt.plot(x, pt_tuning_4, label='pretrain-tuning-yh-42')
    # plt.plot(x, pt, label='pretrain')
    # plt.plot(x, pt_tuning_10, label='pretrain-tuning-sd')
    # plt.plot(x, pt_tuning_13, label='pretrain-tuning-sd-homo')
    plt.plot(x, pt_tuning_24, label='pretrain-tuning-add-score')
    plt.plot(x, pt_tuning_25, label='pretrain-tuning-reduce-score')
    plt.plot(x, supervise, label='supervise')
    plt.plot(x, fraudar, '--', label='fraudar')
    plt.plot(x, raw_score, '--',  label='raw-score')

    plt.xlim([0.0, 9.0])
    plt.xticks(x, shot_num)
    if tag == "AUC":
        plt.ylim([0.75, 0.90])
    elif tag == "Precision":
        plt.ylim([0.30, 0.70])
    elif tag == "Recall":
        plt.ylim([0.25, 0.85])
    else:
        plt.ylim([0.25, 0.70])
    plt.xlabel('K shot')
    plt.ylabel(tag)
    plt.title(f'{tag}-K shot Comparison Figure')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # plot_line_figure(x_data=np.arange(1, 6, 1),
    #                  y1_data=np.array([0.756, 0.748, 0.732, 0.719, 0.687]),
    #                  y2_data=np.array([0.762, 0.741, 0.735, 0.723, 0.695]))

    # file_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
    #              'projects/mmgog_long_term_sequence_model/data/prediction/test/')
    # merged_csv = merge_csv_files(file_path, 'node_id')
    # all
    # fraudar_tpr = 0.8140
    # fraudar_fpr = 0.2128
    # fraudar_pre = 0.3980
    # fraduar_rec = 0.8140
    # y_true_list = [merged_csv['sp_y_true'], merged_csv['pt_y_true'], merged_csv['pf_y_true'], merged_csv['pfw_y_true'], merged_csv['score_y_true']]
    # y_score_list = [merged_csv['sp_y_prob'], merged_csv['pt_y_prob'], merged_csv['pf_y_prob'], merged_csv['pfw_y_prob'], merged_csv['score_y_prob']]
    # labels = ['supervised', 'pretrain', 'pretrain-ft', 'pretrain-ft-w', 'raw-score']
    # plot_multiple_roc_with_point(y_true_list, y_score_list, labels, fraudar_fpr, fraudar_tpr, point_label)
    # plot_pr_curve(y_true_list, y_score_list, labels, fraudar_pre, fraudar_rec, point_label)
    # few-shot
    fraudar_tpr = 0.8121
    fraudar_fpr = 0.2113
    fraudar_pre = 0.3993
    fraudar_rec = 0.8121
    point_label = 'fraudar'
    for metric in ["AUC", "Precision", "Recall", "F1-Score"]:
        print(metric)
        plot_k_shot_metrics2(metric)
        # compute_k_shot_metrics(metric=metric)
    # prefix_path = ('/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/'
    #                'projects/mmgog_long_term_sequence_model/data/prediction/')
    # raw_score_path = prefix_path + f'raw_score/score_10_shot_pred.csv'
    # raw_score_data = pd.read_csv(raw_score_path).drop_duplicates()
    # for shot_num in np.arange(10, 30, 10):
    #     merged_df = merge_k_shot_csv_files(shot_num)
        # y_true_list = [merged_df['pretrain_ft_y_true'], merged_df['supervise_y_true'], merged_df['score_y_true']]
        # y_score_list = [merged_df['pretrain_ft_y_prob'], merged_df['supervise_y_prob'], merged_df['score_y_prob']]
        # pretrain_ft_path = prefix_path + f'pretrain_ft10/ft_{shot_num}_shot_pred.csv'
        # supervise_path = prefix_path + f'supervise2/supervise_{shot_num}_shot_pred.csv'
        # supervise_path = prefix_path + f'pretrain_ft15/ft_{shot_num}_shot_pred.csv'
        # pretrain_ft_data = pd.read_csv(pretrain_ft_path).drop_duplicates()
        # supervise_data = pd.read_csv(supervise_path).drop_duplicates()
        # y_true_list = [pretrain_ft_data['y_true'], supervise_data['y_true'], raw_score_data['y_true']]
        # y_score_list = [pretrain_ft_data['y_prob'], supervise_data['y_prob'], raw_score_data['y_prob']]
        # labels = ['pretrain-tuning-sd', 'supervise', 'raw-score']
        # plot_multiple_roc_with_point(y_true_list, y_score_list, labels, fraudar_fpr, fraudar_tpr, point_label, shot_num)
        # plot_pr_curve(y_true_list, y_score_list, labels, fraudar_pre, fraudar_rec, point_label, shot_num)