import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import datetime
import os
import re
import pandas as pd
import requests
import torch


class DataProcess:
    def __init__(self, args_dict):
        # 序列最大长度
        self.args_dict = args_dict
        # 读群序列数据
        room_seqs_df = pd.read_csv(args_dict['room_seqs_path'], sep=",", header="infer", encoding="utf-8")
        # 群序列str转list
        room_seqs_df['action_seqs'] = room_seqs_df['action_seqs'].apply(lambda x: x.split('|'))
        # 群序列补0或截断
        self.data = room_seqs_df['action_seqs'].apply(self.pad_zero_or_truncat).join(room_seqs_df['label'])
        self.label = self.data.label.values
        self.feat = self.data.iloc[:, :-1]
        # 划分训练集和测试集
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.feat, self.label, test_size=0.2,
                                                                                random_state=42, stratify=self.label)
        print(('-' * 20 + 'data process finished' + '-' * 40)[:60])
        print("train_x.head:", self.train_x.head())
        print("train_x.shape:", self.train_x.shape)
        print("train_y.shape:", self.train_y.shape)
        print("test_x.shape:", self.test_x.shape)
        print("test_y.shape:", self.test_y.shape)

    def pad_zero_or_truncat(self, seq):
        if len(seq) > self.args_dict["max_len"]:
            return pd.Series(seq[-self.args_dict["max_len"]:])
        else:
            # 行为不足, 补齐
            pad_len = self.args_dict["max_len"] - len(seq)
            seq = seq + [str(self.args_dict["padding_idx"])] * pad_len
            return pd.Series(seq)


if __name__ == '__main__':
    print(os.getcwd())
