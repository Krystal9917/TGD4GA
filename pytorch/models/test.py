import os
import sys
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_networkx
from networkx.algorithms.community import modularity
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))


class RGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, num_bases=30)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations, num_bases=30)

    def forward(self, x, edge_index, edge_type):
        x_list = []
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x_list.append(x)
        x = F.log_softmax(self.conv2(x, edge_index, edge_type), dim=1)
        x_list.append(x)
        return x_list

def contrastive_loss(x, flag):
    cosine_similarity = torch.nn.CosineSimilarity()
    gang_row = flag(flag == 1).nonzero().squeeze(-1)
    innocent_row = flag(flag == 0).nonzero().squeeze(-1)
    mean_x = x.mean(dim=0)
    mean_gang_features = x[gang_row, :].mean(dim=0)
    mean_innocent_features = x[innocent_row, :].mean(dim=0)
    sim_positive = torch.exp(cosine_similarity(mean_x, mean_gang_features))
    sim_negative = torch.exp(cosine_similarity(mean_x, mean_innocent_features))
    loss = sim_positive / (sim_positive + sim_negative)
    return loss


def modularity_computation(pyg_data, flag):
    networkx_data = to_networkx(pyg_data)
    gang_community = set((flag == 1).nonzero().squeeze(-1).tolist())
    normal_community = set((flag == 0).nonzero().squeeze(-1).tolist())
    loss = modularity(networkx_data, [gang_community, normal_community])
    return loss


def degree_similarity_computation(x, adj_matrix):
    norm = 1 / torch.sum(adj_matrix)
    degree_similarity_sum = 0
    for i in range(adj_matrix.shape[0]):
        degree_i = torch.sum(adj_matrix[i, :])
        for j in range(adj_matrix.shape[1]):
            if i != j:
                degree_j = torch.sum(adj_matrix[j, :])
                if adj_matrix[i, j] == 0:
                    degree_ratio = 0
                else:
                    if degree_i + degree_j - 1 <= 0:
                        degree_ratio = adj_matrix[i, j] / (degree_i + degree_j)
                    else:
                        degree_ratio = adj_matrix[i, j] / (degree_i + degree_j - 1)
                cos_similarity = F.cosine_similarity(x[i, :].unsqueeze(0), x[j, :].unsqueeze(0))
                degree_similarity_sum += degree_ratio * cos_similarity
    return -torch.log(norm * degree_similarity_sum)


def plot_bar(x, value1, value2, y_label):
    plt.figure(figsize=(20, 16))
    bar_width = 0.4
    plt.bar(x - bar_width / 2, value1, width=bar_width, label='normal', color='green')
    plt.bar(x + bar_width / 2, value2, width=bar_width, label='abnormal', color='red')
    plt.ylabel(y_label)
    plt.xlabel('column index')
    plt.legend()
    plt.show()