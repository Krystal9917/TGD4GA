import itertools
import logging
from logging.handlers import TimedRotatingFileHandler

import numpy as np
from torch.utils.data import Dataset as tDataset
import datetime
import os
import re
import pandas as pd
import requests
import torch


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
    file_handler = TimedRotatingFileHandler(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "log",
                     "my_log" + ".log")), when='D', interval=1, backupCount=7)
    # file_handler = logging.FileHandler(os.path.abspath(
    #     os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "log",
    #                  "my_log" + ".log")))
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[<%(asctime)s> <%(filename)s:%(lineno)d> %(levelname)s]\n %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info('-' * 50 + 'logger start' + '-' * 50)
