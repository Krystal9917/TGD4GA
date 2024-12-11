import json
import math
import time
import torch
import heapq

import networkx as nx
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.data import HeteroData, Batch



# 异常边计算
def edge_evil_score(graph, src_node, dst_node):
    edge_data = graph.get_edge_data(src_node, dst_node)
    src_node_data = graph.nodes[src_node]
    dst_node_data = graph.nodes[dst_node]
    # 边异常度
    score = src_node_data["evil_score"] * dst_node_data["evil_score"] * (
            1 - abs(src_node_data["evil_score"] - dst_node_data["evil_score"])) \
            * (math.log(1 + edge_data["edge_type_cnt"]))

    return score


# def edge_evil_score(graph, src_node, dst_node):
#     edge_type_cnt = graph['uin'].adj[src_node, dst_node]
#     src_node_data = graph['uin'].score[src_node]
#     dst_node_data = graph['uin'].score[dst_node]
#     # 边异常度
#     edge_score = src_node_data * dst_node_data * (1 - abs(src_node_data - dst_node_data)) * (
#         math.log(1 + edge_type_cnt))
#     return edge_score


# 增量更新图密度
def update_density(weight, graph, removed_node=None):
    # 更新总权重，没有移除节点
    if removed_node is None:
        # 计算初始总权重
        weight = sum(edge_evil_score(graph, u, v) for u, v in graph.edges()) + sum(
            graph.nodes[v]["evil_score"] for v in graph.nodes())

    else:
        # 更新总权重，移除与该节点相关的出边和入边
        # 处理出边
        for neighbor in list(graph.successors(removed_node)):
            if graph.has_edge(removed_node, neighbor):
                weight -= edge_evil_score(graph, removed_node, neighbor)

        # 处理入边
        for neighbor in list(graph.predecessors(removed_node)):
            if graph.has_edge(neighbor, removed_node):
                weight -= edge_evil_score(graph, neighbor, removed_node)

        # 减去移除节点的异常度
        weight -= graph.nodes[removed_node]["evil_score"]

    num_nodes = graph.number_of_nodes() - (1 if removed_node else 0)
    density = weight / num_nodes if num_nodes > 0 else 0

    return weight, density


# def update_density(weight, graph, removed_node=None):
#     # 更新总权重，没有移除节点
#     if removed_node is None:
#         # 计算初始总权重
#         weight = (sum(edge_evil_score(graph, graph['uin'].edge_index[0, col], graph['uin'].edge_index[1, col])
#                       for col in range(graph['uin'].edge_index.shape[1])) +
#                   sum(graph['uin'].score[v] for v in range(graph['uin'].num_nodes)))
#
#     else:
#         # 更新总权重，移除与该节点相关的出边和入边
#         # 处理出边
#         for neighbor in graph['uin'].idx.tolist():
#             if graph['uin'].adj[removed_node, neighbor] > 0:
#                 weight -= edge_evil_score(graph, removed_node, neighbor)
#
#         # 处理入边
#         for neighbor in graph['uin'].idx.tolist():
#             if graph['uin'].adj[neighbor, removed_node] > 0:
#                 weight -= edge_evil_score(graph, neighbor, removed_node)
#
#         # 减去移除节点的异常度
#         weight -= graph['uin'].score[removed_node]
#
#     num_nodes = len(graph['uin'].idx.tolist()) - (1 if removed_node else 0)
#     density = weight / num_nodes if num_nodes > 0 else 0
#
#     return weight, density


# def copy_graph(pyg_graph, first_copy=False):
#     new_data = HeteroData()
#     new_data['uin'].score = pyg_graph['uin'].score.clone().requires_grad_(False)
#     new_data['uin'].idx = pyg_graph['uin'].idx.clone().requires_grad_(False)
#     if first_copy:
#         edge_index = [pyg_graph[edge_type].edge_index for edge_type in pyg_graph.edge_types]
#         edge_index = torch.concat(edge_index, dim=1)
#         unique_edge_index, indices = torch.unique(edge_index, dim=1, return_inverse=True)
#         adj = to_dense_adj(edge_index, max_num_nodes=new_data['uin'].score.shape[0]).squeeze()
#     else:
#         unique_edge_index = pyg_graph['uin'].edge_index.clone().requires_grad_(False)
#         adj = pyg_graph['uin'].adj.clone().requires_grad_(False)
#     new_data['uin'].edge_index = unique_edge_index
#     new_data['uin'].adj = adj
#     return new_data


# 异常子图挖掘算法
def fraudar(graph):
    # 初始化最优子图
    current_graph = graph.copy()
    best_graph = current_graph.copy()
    current_weight, current_density = update_density(0, current_graph)  # 初始化时计算总权重和密度
    best_weight, best_density = current_weight, current_density
    # print("Initial best_density:", best_density)

    # 创建一个优先队列，以负密度模拟最大堆
    densities = []
    for node in current_graph.nodes:
        # 计算移除节点后的密度
        weight, density = update_density(current_weight, current_graph, node)
        heapq.heappush(densities, (-density, node))

    # 死循环兜底，最多只迭代1000次
    epoch = 1000
    while current_graph.number_of_nodes() > 0 and epoch > 0:
        # 训练剩余次数更新
        epoch -= 1

        # 弹最小密度的节点
        min_node = None
        while densities:
            _, min_node = heapq.heappop(densities)
            if min_node in current_graph:
                break

        if min_node not in current_graph:
            continue

        # 更新总权重和当前子图密度
        current_weight, current_density = update_density(current_weight, current_graph, min_node)

        # 移除该节点
        current_graph.remove_node(min_node)

        # 更新剩余节点的密度
        for node in current_graph.nodes:
            if node != min_node:
                # 因为每次计算节点新的密度时，旧的密度并没有弹出，所以这里需要避免重复计算
                weight, density = update_density(current_weight, current_graph, node)
                heapq.heappush(densities, (-density, node))

        if current_density > best_density:
            best_density = current_density
            best_weight = current_weight
            best_graph = current_graph.copy()

    return best_graph, best_density, best_weight


# def fraudar(graph):
#     # 初始化最优子图
#     current_graph = copy_graph(graph, first_copy=True)
#     best_graph = copy_graph(graph, first_copy=True)
#     current_weight, current_density = update_density(0, current_graph)  # 初始化时计算总权重和密度
#     best_weight, best_density = current_weight, current_density
#     # print("Initial best_density:", best_density)
#
#     # 创建一个优先队列，以负密度模拟最大堆
#     densities = []
#     for node in current_graph['uin'].idx.tolist():
#         # 计算移除节点后的密度
#         weight, density = update_density(current_weight, current_graph, node)
#         heapq.heappush(densities, (-density, node))
#
#     # 死循环兜底，最多只迭代1000次
#     epoch = 1000
#     while len(current_graph['uin'].idx) > 0 and epoch > 0:
#         # 训练剩余次数更新
#         epoch -= 1
#
#         # 弹最小密度的节点
#         min_node = None
#         while densities:
#             _, min_node = heapq.heappop(densities)
#             if min_node in current_graph['uin'].idx.tolist():
#                 break
#
#         if min_node not in current_graph['uin'].idx.tolist():
#             continue
#
#         # 更新总权重和当前子图密度
#         current_weight, current_density = update_density(current_weight, current_graph, min_node)
#
#         # 移除该节点
#         current_graph['uin'].idx = current_graph['uin'].idx[current_graph['uin'].idx != min_node]
#         current_graph['uin'].adj[min_node, :] = 0
#         current_graph['uin'].adj[:, min_node] = 0
#         current_graph['uin'].edge_index = current_graph['uin'].edge_index[:,
#                                           (current_graph['uin'].edge_index[0, :] != min_node) &
#                                           (current_graph['uin'].edge_index[1, :] != min_node)]
#
#         # 更新剩余节点的密度
#         for node in current_graph['uin'].idx.tolist():
#             if node != min_node:
#                 # 因为每次计算节点新的密度时，旧的密度并没有弹出，所以这里需要避免重复计算
#                 weight, density = update_density(current_weight, current_graph, node)
#                 heapq.heappush(densities, (-density, node))
#
#         if current_density > best_density:
#             best_density = current_density
#             best_weight = current_weight
#             best_graph = copy_graph(current_graph)
#
#     return best_graph, best_density, best_weight


# 异常子图详细信息
def get_anomaly_graph_info(graph, density, weight):
    # 计算每个节点在子图中的重要度
    node_importance_dict = {}

    for node in graph.nodes:
        # 计算移除节点后的密度
        _, remove_density = update_density(weight, graph, node)
        # 节点重要性 = (原始密度 - 移除节点后的密度) / 原始密度
        importance = (density - remove_density) * 1.0 / density if (density > 0 and density >= remove_density) else 0
        node_importance_dict[node] = importance

    anomaly_nodes_size = graph.number_of_nodes()
    anomaly_nodes_list = ",".join([str(x) for x in graph.nodes()])
    # 节点重要度以json字符串返回
    anomaly_nodes_importance = json.dumps(node_importance_dict)

    return anomaly_nodes_size, anomaly_nodes_list, anomaly_nodes_importance


def mmspamstrategycenter_acct_general_gangs_audit_222(req):
    # 创建有向图
    G = nx.DiGraph()

    AAR_uin_attr = req.get("AAR_uin_attr")
    AS_uin_gangs_interact_relationship = req.get("AS_uin_gangs_interact_relationship")

    # 添加节点
    for node in AAR_uin_attr:
        G.add_node(node["uin"], evil_score=float(node["evil_score"]))

    # 添加边
    for node in AS_uin_gangs_interact_relationship:
        G.add_edge(node["src_uin"], node["dst_uin"], edge_type_cnt=int(node["edge_type_cnt"]),
                   edge_type_list=node["edge_type_list"])

    # 异常子图输出
    anomalous_subgraph, anomalous_density, anomalous_weight = fraudar(G)
    print("anomalous_density:", anomalous_density)
    print("anomalous_weight:", anomalous_weight)

    # 异常子图信息
    anomaly_nodes_size, anomaly_nodes_list, anomaly_nodes_importance = get_anomaly_graph_info(anomalous_subgraph,
                                                                                              anomalous_density,
                                                                                              anomalous_weight)

    req.set("anomaly_nodes_size", anomaly_nodes_size)
    req.set("anomaly_nodes_list", anomaly_nodes_list)
    req.set("anomaly_nodes_importance", anomaly_nodes_importance)


def generate_pyg_data(n=101):
    graph_data = HeteroData()
    graph_data['uin'].x = torch.randn(n, 846)
    graph_data['uin'].idx = torch.arange(n)
    graph_data['uin'].score = torch.randn((n,))
    graph_data['uin'].num_nodes = n
    graph_data['uin', 'wifi', 'uin'].edge_index = torch.randint(0, n, (2, 60))
    graph_data['uin', 'friend', 'uin'].edge_index = torch.randint(0, n, (2, 80))
    graph_data['uin', 'group', 'uin'].edge_index = torch.randint(0, n, (2, 100))
    return graph_data


def generate_positive_sample(graph_data):
    G = nx.DiGraph()
    # 添加节点
    for i in range(graph_data['uin'].num_nodes):
        G.add_node(graph_data['uin'].idx[i].item(), evil_score=graph_data['uin'].score[i].item())

    edge_index = [graph_data[edge_type].edge_index for edge_type in graph_data.edge_types]
    edge_index = torch.concat(edge_index, dim=1)
    unique_edge_index, indices = torch.unique(edge_index, dim=1, return_inverse=True)
    adj = to_dense_adj(edge_index, max_num_nodes=graph_data['uin'].score.shape[0]).squeeze()

    # 添加边
    for i in range(unique_edge_index.shape[1]):
        srt = unique_edge_index[0, i].item()
        dst = unique_edge_index[1, i].item()
        G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
    best_graph, best_density, best_weight = fraudar(G)
    idx = torch.zeros(graph_data['uin'].num_nodes, dtype=torch.int)
    idx[sorted(best_graph.nodes)] = 1
    graph_data['uin'].idx = idx
    return graph_data


def extract_subgraph(batch, node_indices):
    subgraphs = []
    for i in range(batch.num_graphs):
        subgraph_node = node_indices[batch['uin'].batch == i]
        subgraph_node_indices = (subgraph_node == 1).nonzero().squeeze()

        # Get the node features for the subgraph
        x_subgraph = batch['uin'].x[subgraph_node_indices]

        # Create a new Data object for the subgraph
        subgraph_data = HeteroData()
        subgraph_data['uin'].x = x_subgraph

        # Extract the subgraph
        for edge_type in batch.edge_types:
            edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index, relabel_nodes=True)
            subgraph_data[edge_type].edge_index = edge_index
        subgraphs.append(subgraph_data)

    return Batch.from_data_list(subgraphs)


def node_level_contrastive_loss(high_malicious_h, low_malicious_h, high_mask, low_mask):
    min_num_subgraph = min(high_mask.max().item(), low_mask.max().item()) + 1
    batch_node_loss = 0
    for i in range(min_num_subgraph):
        high_h = high_malicious_h[high_mask == i]
        low_h = low_malicious_h[low_mask == i]
        high_h_norm = high_h.norm(p=2, dim=1)
        low_h_norm = low_h.norm(p=2, dim=1)
        pos_loss = torch.mm(high_h, high_h.T) / (high_h_norm ** 2).repeat(high_h.shape[0], 1)
        neg_loss = torch.mm(high_h, low_h.T) / torch.outer(high_h_norm, low_h_norm)
        node_loss = -torch.log(pos_loss.sum(dim=1) / (pos_loss.sum(dim=1) + neg_loss.sum(dim=1)) + 1e-4).mean()
        batch_node_loss += node_loss.item()
    return batch_node_loss


if __name__ == '__main__':
    pyg_data_1 = generate_pyg_data(50)
    pyg_data_2 = generate_pyg_data(88)

    pyg_data_1 = generate_positive_sample(pyg_data_1)
    pyg_data_2 = generate_positive_sample(pyg_data_2)
    batch = Batch.from_data_list([pyg_data_1, pyg_data_2])
    pos_batch = extract_subgraph(batch, batch['uin'].idx)
    fraudar_idx = (batch['uin'].idx == 1).nonzero().squeeze()
    fraudar_batch_x = batch['uin'].x[fraudar_idx]
    fraudar_batch_mask = batch['uin'].batch[fraudar_idx]
    exclude_batch_idx = [idx for idx in range(batch['uin'].num_nodes) if idx not in batch['uin'].idx]
    exclude_batch_x = batch['uin'].x[exclude_batch_idx]
    exclude_batch_mask = batch['uin'].batch[exclude_batch_idx]
    node_loss = node_level_contrastive_loss(fraudar_batch_x, exclude_batch_x, fraudar_batch_mask, exclude_batch_mask)
    print(node_loss)




