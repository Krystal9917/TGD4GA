import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax, relu


class WideDeepNet(nn.Module):
    def __init__(self, n_feat, output_size):
        super(WideDeepNet, self).__init__()
        self.wide = nn.Linear(n_feat, output_size)
        self.deep = nn.Sequential(
            nn.Linear(n_feat, n_feat * 2),
            nn.ReLU(),
            nn.Linear(n_feat * 2, output_size * 2),
            nn.ReLU(),
            nn.Linear(output_size * 2, output_size)
        )

    def forward(self, x):
        # print("x.shape: ", x.shape)
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        out = wide_out + deep_out
        return torch.sigmoid(out)


class MultiHead(nn.Module):
    def __init__(self, n_head, model_dim, drop_rate):
        super().__init__()
        self.head_dim = model_dim // n_head
        self.n_head = n_head
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, n_head * self.head_dim)
        self.wk = nn.Linear(model_dim, n_head * self.head_dim)
        self.wv = nn.Linear(model_dim, n_head * self.head_dim)

        self.o_dense = nn.Linear(model_dim, model_dim)
        self.o_drop = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.attention = None

    def forward(self, q, k, v, mask, training):
        # residual connect
        residual = q
        dim_per_head = self.head_dim
        num_heads = self.n_head
        batch_size = q.size(0)

        # linear projection
        key = self.wk(k)  # [n, step, num_heads * head_dim]
        value = self.wv(v)  # [n, step, num_heads * head_dim]
        query = self.wq(q)  # [n, step, num_heads * head_dim]

        # split by head
        query = self.split_heads(query)  # [n, n_head, q_step, h_dim]
        key = self.split_heads(key)
        value = self.split_heads(value)  # [n, h, step, h_dim]
        context = self.scaled_dot_product_attention(query, key, value, mask)  # [n, q_step, h*dv]
        o = self.o_dense(context)  # [n, step, dim]
        o = self.o_drop(o)

        o = self.layer_norm(residual + o)  # resNet做短接
        return o

    def split_heads(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = torch.tensor(k.shape[-1]).type(torch.float)
        score = torch.matmul(q, k.permute(0, 1, 3, 2)) / (torch.sqrt(dk) + 1e-8)  # [n, n_head, step, step]
        if mask is not None:
            # 搞成无限大，softmax激活就是0
            score = score.masked_fill_(mask, -np.inf)
        self.attention = softmax(score, dim=-1)
        context = torch.matmul(self.attention, v)  # [n, num_head, step, head_dim]
        context = context.permute(0, 2, 1, 3)  # [n, step, num_head, head_dim]
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context  # [n, step, model_dim]


class PositionWiseFFN(nn.Module):
    def __init__(self, model_dim, dropout=0.0):
        super().__init__()
        dff = model_dim * 4
        self.l = nn.Linear(model_dim, dff)
        self.o = nn.Linear(dff, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        o = relu(self.l(x))
        o = self.o(o)
        o = self.dropout(o)

        o = self.layer_norm(x + o)
        return o  # [n, step, dim]


class EncoderLayer(nn.Module):

    def __init__(self, n_head, emb_dim, drop_rate):
        super().__init__()
        self.mh = MultiHead(n_head, emb_dim, drop_rate)
        self.ffn = PositionWiseFFN(emb_dim, drop_rate)

    def forward(self, xz, training, mask):
        # xz: [n, step, emb_dim]
        context = self.mh(xz, xz, xz, mask, training)  # [n, step, emb_dim]
        o = self.ffn(context)
        return o


class Encoder(nn.Module):
    def __init__(self, n_head, emb_dim, drop_rate, n_layer):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(n_head, emb_dim, drop_rate) for _ in range(n_layer)]
        )

    def forward(self, xz, training, mask):
        for encoder in self.encoder_layers:
            xz = encoder(xz, training, mask)
        return xz  # [n, step, emb_dim]


class SeqGraphUin2UinV2(nn.Module):
    # 多头多层容易过拟合
    def __init__(self, target_feat, vocab_size, max_len, n_layers=2, emb_dim=512, n_heads=2, output_size=11,
                 drop_rate=0.1,
                 padding_idx=0):
        super().__init__()
        self.max_len = max_len
        self.emb_dim = emb_dim
        self.padding_idx = torch.tensor(padding_idx)
        self.dec_v_emb = vocab_size
        self.embeddings = nn.Embedding(vocab_size, emb_dim // 2)
        self.embeddings.weight.data.normal_(0, 0.1)
        self.target_wd = WideDeepNet(target_feat, emb_dim // 2)
        self.encoder = Encoder(n_heads, emb_dim, drop_rate, n_layers)
        # self.flatten = nn.Flatten(start_dim=1)
        # self.input_wd = nn.Linear(input_feat, emb_dim)
        self.o = nn.Linear(emb_dim, output_size)

    def forward(self, seqs_x, target_x, training=None):
        # print(" x.size()", x.size())
        # print("target_x", target_x.size())
        x_embed = self.embeddings(seqs_x)
        # 行为对象的特征矩阵逐行过w&d模型
        target_x_embed = self.target_wd(target_x.view(-1, target_x.size(2))).view(target_x.size(0), target_x.size(1),
                                                                                  -1)
        # 特征fusion
        x_embed = torch.cat((x_embed, target_x_embed), 2)
        position = self._position_emb().to(self.embeddings.weight.device)
        # position emb
        x_embed = x_embed + position
        pad_mask = self._pad_mask(seqs_x)
        encoded_z = self.encoder(x_embed, training, pad_mask)  # [n, step, emb_dim]
        # fl_out = self.flatten(encoded_z)  # [n,step * emb_dim]
        # input_wd = self.input_wd(input_x)  # [n, emb_dim]
        mlp_out = self.o(encoded_z[:, -1, :])  # [n, output_size] 取序列最后一个step的embedding
        return torch.sigmoid(mlp_out)

    def _pad_bool(self, seqs):
        o = torch.eq(seqs, self.padding_idx)  # [n, step]
        return o  # 这里返回的是[true, false, ...]

    def _pad_mask(self, seqs):
        len_q = seqs.size(1)
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q, -1)  # [n, len_q, step] 这里主要用于扩维
        return mask.unsqueeze(1)  # [n, 1, len_q, step]

    def _pad_mask_embed(self, seqs, embeddings):
        padding_mask = self._pad_bool(seqs)  # [n, step]
        # mask 位置全部置 0, 这种只适合下游做sum和max操作
        mask_embeddings = embeddings.masked_fill(padding_mask.unsqueeze(-1).expand_as(embeddings), 0.0)
        return mask_embeddings

    def _position_emb(self):
        pos = np.expand_dims(np.arange(self.max_len), 1)  # [max_len, 1]
        pe = pos / np.power(1000,
                            2 * np.expand_dims(np.arange(self.emb_dim) // 2, 0) / self.emb_dim)  # [max_len, emb_dim]
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)  # [1, max_len, emb_dim]
        pe = torch.from_numpy(pe).type(torch.float32)
        return pe
