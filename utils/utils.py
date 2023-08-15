import itertools
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
