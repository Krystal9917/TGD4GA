import torch
from torch import nn
import dgl


class FeatScaler(nn.Module):
    """
    特征归一化：数值特征原始值直接进来，通过BN层进行学习
    """

    def __init__(self, feat_in_size):
        super(FeatScaler, self).__init__()
        self.bn = nn.BatchNorm1d(feat_in_size)
        self.mlp = nn.Linear(feat_in_size, feat_in_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        bn_out = self.bn(feat)
        mlp_out = self.mlp(bn_out)
        out = self.sigmoid(mlp_out)
        return out


class WideDeepNet(nn.Module):
    def __init__(self, n_feat, output_size):
        super(WideDeepNet, self).__init__()
        self.wide = nn.Linear(n_feat, output_size)
        self.deep = nn.Sequential(
            nn.Linear(n_feat, output_size),
            # nn.Linear(n_feat, n_feat * 2),
            # nn.LeakyReLU(),
            # nn.Linear(n_feat * 2, output_size * 2),
            # nn.LeakyReLU(),
            # nn.Linear(output_size * 2, output_size)
        )

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        out = wide_out + deep_out
        return torch.sigmoid(out)


class DNN(nn.Module):
    def __init__(self, uin_in_size, uin_out_size, out_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(uin_in_size, uin_out_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(uin_out_size, uin_out_size)
        self.fc3 = nn.Linear(uin_out_size, out_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class GNNLayer(nn.Module):
    def __init__(self, uin_in_size, uin_out_size, drop_rate):
        super(GNNLayer, self).__init__()
        # 为每一种边类型设置权重矩阵，构建mlp层
        self.weight = nn.ModuleDict({"uin-spread-uin": nn.Linear(uin_in_size, uin_out_size)})

    def forward(self, g, node_feat_dict):
        with g.local_scope():
            funcs = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                # 定义 uin-spread-uin 边消息传递方式
                if srctype == "uin" and etype == "uin-spread-uin" and dsttype == "uin":
                    uin2uin_wh = self.weight["uin-spread-uin"](node_feat_dict[srctype])
                    g.nodes[srctype].data["wh_%s" % etype] = uin2uin_wh
                    funcs[etype] = (dgl.function.copy_u("wh_%s" % etype, "m"), dgl.function.mean("m", "out"))
            # 同类型节点最后消息更新
            g.multi_update_all(funcs, "mean")
            return {ntype: torch.sigmoid(g.nodes[ntype].data["out"]) for ntype in g.ntypes}


class UinGangsModel(nn.Module):
    def __init__(self, uin_in_size, uin_out_size, out_size, drop_rate):
        super(UinGangsModel, self).__init__()
        self.feat_scaler = FeatScaler(uin_in_size)
        self.uin_number_feat_model = WideDeepNet(uin_in_size, uin_out_size)
        self.gnn_layer1 = GNNLayer(uin_in_size, uin_out_size, drop_rate)
        # self.gnn_layer2 = GNNLayer(uin_hidden_size, uin_out_size)
        self.mlp = nn.Linear(uin_out_size, uin_out_size)
        self.classify_mlp = nn.Linear(uin_out_size, out_size)

    def forward(self, g):
        # node_feat_dict = {ntype: self.feat_scaler(g.nodes[ntype].data["uin_number_feat"]) for ntype in g.ntypes}
        node_feat_dict = {ntype: g.nodes[ntype].data["uin_number_feat"] for ntype in g.ntypes}
        uin_out = self.uin_number_feat_model(node_feat_dict["uin"])
        # h_dict1 = self.gnn_layer1(g, node_feat_dict)
        # h_dict2 = self.gnn_layer2(g, h_dict1)
        # out_emb = self.mlp(torch.cat([uin_out, h_dict1["uin"]], dim=1))
        out_emb = uin_out
        classify_out = self.classify_mlp(out_emb)
        # classify_out = torch.sigmoid(classify_out)
        g.nodes["uin"].data["out_emb"] = out_emb
        g.nodes["uin"].data["classify_out"] = classify_out
        return g
