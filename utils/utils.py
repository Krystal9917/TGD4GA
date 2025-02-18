import logging
import subprocess
import numpy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import (roc_curve, roc_auc_score, precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve, auc, silhouette_score)
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
    degree = masked_adj.sum(dim=1) + torch.eye(masked_adj.shape[0], device=masked_adj.device)
    L = torch.diag(degree) - masked_adj
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

def batch_connect_loss(batch, y_prob, filter_edge_types, epsilon=1e-3):
    pos_sub_idx = (batch['uin'].gang_label == 1).nonzero().squeeze()
    if pos_sub_idx.shape == torch.Size([]):
        pos_sub_idx = torch.tensor([pos_sub_idx], device=pos_sub_idx.device)
    if pos_sub_idx.shape[0] != 0:
        cnt_loss = []
        adj = to_dense_adj(torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types
                                         if edge_type in filter_edge_types], dim=1),
                           max_num_nodes=batch['uin'].num_nodes).squeeze()
        for pos_idx in pos_sub_idx:
            total_node_idx = (batch['uin'].batch == pos_idx).nonzero().squeeze()
            node_idx = total_node_idx[y_prob[total_node_idx] >= 0.5]
            pos_adj = adj[node_idx, :][:, node_idx]
            pos_y_prob = y_prob[node_idx]
            if node_idx.shape == torch.Size([]):
                node_idx = torch.tensor([node_idx], device=node_idx.device)
            if node_idx.shape[0] != 0:
                i_cnt_loss = connect_loss(pos_y_prob, pos_adj, epsilon)
                cnt_loss.append(i_cnt_loss)
        cnt_loss = torch.stack(cnt_loss).mean()
    else:
        cnt_loss = torch.tensor(0, device=batch['uin'].x.device)
    return cnt_loss

if __name__ == '__main__':
    g_y = torch.tensor([0, 0, 0, 1, 1])
    g_batch = torch.tensor([0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1,
                            2, 2, 2, 2,
                            3, 3, 3, 3, 3, 3, 3,
                            4, 4, 4, 4, 4, 4])
    node_y_pred1 = torch.tensor([0.1, 0.15, 0.1, 0.4, 0.8, 0.2,
                                0.1, 0.25, 0.0, 0.1, 0.15,
                                0.23, 0.24, 0.2, 0.01,
                                0.07, 0.34, 0.56, 0.45, 0.69, 0.63, 0.55,
                                0.24, 0.62, 0.54, 0.48, 0.46, 0.71], requires_grad=True)
    node_y_pred2 = torch.tensor([0.01, 0.02, 0.03, 0.01, 0.02, 0.05,
                                 0.01, 0.0, 0.0, 0.0, 0.02,
                                 0.03, 0.04, 0.01, 0.005,
                                 0.97, 0.92, 0.88, 0.12, 0.15, 0.78, 0.86,
                                 0.68, 0.93, 0.08, 0.79, 0.37, 0.92], requires_grad=True)
    node_y_true = torch.tensor([0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0,
                                1, 1, 1, 0, 0, 1, 1,
                                1, 1, 0, 1, 0, 1])
    loss1 = batch_subgraph_loss_based_cross_entropy(g_y, g_batch, node_y_pred1, node_y_true)
    loss2 = batch_subgraph_loss_based_cross_entropy(g_y, g_batch, node_y_pred2, node_y_true)
    print(f"Loss 1: {loss1: .4f}, Loss 2: {loss2: .4f}")

    y_prob = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    print(y_prob)
    print(y_prob[:, 1])

