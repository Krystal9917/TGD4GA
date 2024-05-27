import logging
import math
import os
import sys

import yaml

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
logger = logging.getLogger("my_logger")
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import PreTrainedTokenizer


class SeqTokenizer(PreTrainedTokenizer):

    def __init__(self, vocab_file_path, **kwargs):
        self.vocab = self._load_vocab(vocab_file_path)
        self.ids_to_tokens = {id_: token for token, id_ in self.vocab.items()}
        super().__init__(**kwargs)

        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"

        self.special_tokens = {
            self.pad_token: self.vocab[self.pad_token],
            self.mask_token: self.vocab[self.mask_token],
            self.unk_token: self.vocab[self.unk_token],
            self.cls_token: self.vocab[self.cls_token]
        }

    def _load_vocab(self, vocab_file):
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f.readlines()):
                token = line.strip()
                vocab[token] = index
        return vocab

    def get_vocab(self):
        return self.vocab

    def _tokenize(self, text, **kwargs):
        return text.split(',')
        # return text

    def _convert_token_to_id(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        if index in self.ids_to_tokens:
            return self.ids_to_tokens[index]
        return self.unk_token_id

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return len(self.vocab)


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


class DataProcessSeqGraph:
    def __init__(self, args_dict):
        # 序列最大长度
        self.args_dict = args_dict

        # 读取数据配置文件
        with open(args_dict["uin_seqs_enum_yaml_path"], 'r', encoding='utf-8') as file:
            self.uin_seqs_enum = yaml.safe_load(file)
        # 获取分类标签
        self.label_dict = self.uin_seqs_enum['fraud_class_enum']
        logger.info("self.label_dict:\n %s", self.label_dict)

        # 获取行为词表
        tokenizer = SeqTokenizer(args_dict["action_vocab_path"])
        self.action_vocab = tokenizer.get_vocab()
        self.action_vocab_size = tokenizer.vocab_size
        logger.info("action_vocab_size: %s", self.action_vocab_size)
        self.special_tokens = tokenizer.special_tokens
        logger.info("special_tokens: %s", self.special_tokens)

        # 读行为序列数据
        user_seqs_df = pd.read_csv(args_dict['uin_seqs_path'], sep="<@@>", header="infer", encoding="utf-8")
        logger.info("dataset original size : %s", len(user_seqs_df))
        user_seqs_df = user_seqs_df.dropna()
        logger.info("dataset size : %s", len(user_seqs_df))

        # 序列str转list
        user_seqs_df['action_seqs'] = user_seqs_df['action_seqs'].apply(lambda x: x.split('|'))

        self.target_feat_size = len(
            user_seqs_df['action_seqs'][0][0].split(':')[1].split(','))
        logger.info("self.target_feat_size: %s", self.target_feat_size)

        # 序列补0或截断
        data = user_seqs_df['action_seqs'].apply(self.pad_zero_or_truncate).join(user_seqs_df['label'])
        label = data.label.to_frame().astype(int)
        feat = data.iloc[:, :-1]

        # 计算标签权重
        self.label_cnt_dict, self.label_weights = self.get_multi_label_weights(label)
        logger.info("label_weight: %s", self.label_weights)
        logger.info("label_cnt_dict: %s", self.label_cnt_dict)

        # 划分训练集和测试集
        train_x, test_x, self.train_y, self.test_y = train_test_split(feat, label, test_size=0.2,
                                                                      random_state=42, stratify=label)
        logger.info("train_x shape: %s", train_x.shape)
        logger.info("test_x shape: %s", test_x.shape)
        logger.info("train_y shape: %s", self.train_y.shape)
        logger.info("test_y shape: %s", self.test_y.shape)

        # 切分序列特征和数值特征
        train_seq, self.train_target_feat = self.separate_seq_feat(train_x)
        test_seq, self.test_target_feat = self.separate_seq_feat(test_x)
        logger.info("self.train_target_feat shape: %s", self.train_target_feat.shape)
        logger.info("self.train_target_feat sample:\n %s", self.train_target_feat[:5])
        logger.info("self.test_target_feat shape: %s", self.test_target_feat.shape)
        logger.info("self.test_target_feat sample:\n %s", self.test_target_feat[:5])

        # 行为token转id
        self.train_seq_token_id = train_seq.applymap(lambda x: tokenizer._convert_token_to_id(x))
        self.test_seq_token_id = test_seq.applymap(lambda x: tokenizer._convert_token_to_id(x))
        logger.info("self.train_seq_token_id shape: %s", self.train_seq_token_id.shape)
        logger.info("self.train_seq_token_id sample:\n %s ", self.train_seq_token_id.head())
        logger.info("self.test_seq_token_id shape: %s ", self.test_seq_token_id.shape)
        logger.info("self.test_seq_token_id sample:\n %s ", self.test_seq_token_id.head())

    def pad_zero_or_truncate(self, seq):
        if len(seq) > self.args_dict["max_len"]:
            return pd.Series(seq[-self.args_dict["max_len"]:])
        else:
            # 行为不足, 补齐
            target_feat_pad = ','.join(['0'] * self.target_feat_size)
            token_pad = str(self.special_tokens["[PAD]"]) + ":" + target_feat_pad
            pad_len = self.args_dict["max_len"] - len(seq)
            seq = seq + [token_pad] * pad_len
            # print([token_pad] * pad_len)
            return pd.Series(seq)

    @staticmethod
    def separate_seq_feat(feat_df):
        # 初始化两个空的DataFrame，用于存储最终结果
        seq_df = pd.DataFrame()
        target_feat_df = pd.DataFrame()

        # 遍历分割
        for col in feat_df.columns:
            # 按照':'分割
            split_col = feat_df[col].str.split(':', expand=True)
            # 第0列是seq，第1列是target_feat
            seq_df = pd.concat([seq_df, split_col[0]], axis=1)
            target_feat_df = pd.concat([target_feat_df, split_col[1]], axis=1)
        # 对 target_feat_df 再次按','分割
        target_feat_df = target_feat_df.apply(lambda x: x.str.split(','))
        target_feat = np.array(target_feat_df.values.tolist(), dtype=float)
        return seq_df, target_feat

    @staticmethod
    def get_multi_label_weights(label_df):
        label_cnt_dict = label_df.iloc[:, 0].value_counts().to_dict()
        min_label_cnt = label_df.iloc[:, 0].value_counts().min()
        label_weight = label_df.iloc[:, 0].value_counts(normalize=True).apply(lambda x: 1 - x)
        # label_weight = label_df.iloc[:, 0].value_counts().apply(lambda x: min_label_cnt * 1.0 / x)
        weight_list = [0] * len(label_weight)
        for index, value in label_weight.items():
            weight_list[index] = value
        return dict(sorted(label_cnt_dict.items())), weight_list


if __name__ == '__main__':
    print(os.getcwd())
