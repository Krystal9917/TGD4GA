import itertools
import logging
import subprocess
from logging.handlers import TimedRotatingFileHandler

import numpy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    precision_recall_curve, auc, silhouette_score
from torch.utils.data import Dataset as tDataset
from datetime import datetime
import os
import re
import pandas as pd
import requests
import torch
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt


def print_model_size(model):
    """
    打印模型参数规模
    :param model:
    :return:
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params} parameters")
    return total_params


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


def draw_and_save(x_dict, y_dict, save_path, title="标题"):
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


def draw_and_save_pca_pic(x, y, save_path=None, title="标题"):
    """
    绘制并保存PCA图
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)
    # 可视化结果，使用标签来区分不同的点
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=150)
    plt.title('uin_gangs_embedding')
    plt.xlabel('uin_embedding')
    plt.ylabel('exposed_label')
    plt.colorbar(scatter)
    # plt.show()
    plt.savefig(save_path)


def eval_emb_with_knn(X, k=10):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X)
    # 计算轮廓系数
    score = silhouette_score(X, kmeans.labels_)
    return score


if __name__ == '__main__':
    # 生成模拟数据
    x = np.random.rand(4, 4)  # 生成一个4x16的随机矩阵
    y = np.array([1, 2, 3, 0])
    auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats = get_indicator_of_mutil_cls_base_softmax(y,
                                                                                                                     x,
                                                                                                                     4)
    print(auc_scores)
    print(precision_scores)
    print(recall_scores)
    print(f1_scores)
    print(confusion_mats)
