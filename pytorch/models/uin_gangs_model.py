import torch
from torch import nn
import dgl


class GNNLayer(nn.Module):
    def __init__(self, device, uin_in_size, uin_out_size):
        super(GNNLayer, self).__init__()
        # 为每一种边类型设置权重矩阵，构建mlp层
        self.weight = nn.ModuleDict({"uin-spread-uin": nn.Linear(uin_in_size, uin_out_size)}).to(device)

    def forward(self, g, node_feat_dict):
        with g.local_scope():
            funcs = {}
            for srctype, etype, dsttype in g.canonical_etypes:
                # 定义 uin-spread-uin 边消息传递方式
                if srctype == "uin" and etype == "uin-spread-uin" and dsttype == "uin":
                    uin2uin_wh = self.weight["uin-spread-uin"](node_feat_dict[srctype])
                    g.nodes[srctype].data["wh_%s" % etype] = uin2uin_wh
                    funcs[etype] = (dgl.function.copy_u("wh_%s" % etype, "m"), dgl.function.max("m", "out"))
            # 同类型节点最后消息更新
            g.multi_update_all(funcs, "max")
            return {ntype: nn.functional.leaky_relu(g.nodes[ntype].data["out"]) for ntype in g.ntypes}


class UinGangsModel(nn.Module):
    def __init__(self, device, uin_in_size, uin_out_size, uin_hidden_size):
        super(UinGangsModel, self).__init__()
        self.device = device
        self.gnn_layer1 = GNNLayer(device, uin_in_size, uin_hidden_size)
        self.gnn_layer2 = GNNLayer(device, uin_hidden_size, uin_out_size)
        self.mlp = nn.Linear(uin_in_size + uin_out_size, uin_out_size)

    def forward(self, g):
        g = g.to(self.device)
        node_feat_dict = {ntype: g.nodes[ntype].data["uin_number_feat"].to(self.device) for ntype in g.ntypes}
        h_dict1 = self.gnn_layer1(g, node_feat_dict)
        h_dict2 = self.gnn_layer2(g, h_dict1)
        out_emb = self.mlp(torch.cat([node_feat_dict["uin"], h_dict2["uin"]], dim=1))
        # uin根节点emb
        root_h_dict2 = h_dict2["uin"][0]
        root_emb = out_emb[0:1, :]
        print("root_emb.shape", root_emb.shape)
        return root_h_dict2, root_emb
