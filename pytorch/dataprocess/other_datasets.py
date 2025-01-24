import os
import torch
import random
import itertools
import networkx as nx
from torch_geometric.data import HeteroData
from torch_geometric.datasets import HGBDataset
from torch_geometric.utils import dense_to_sparse, to_dense_adj, subgraph, k_hop_subgraph
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar


def load_dataset(dataset_dir, dataset_name):
    processed_data_path = os.path.join(dataset_dir, f'multi_graph_{dataset_name}.pt')
    if not os.path.exists(processed_data_path):
        dataset = HGBDataset(root=dataset_dir, name=dataset_name)[0]
        data = HeteroData()
        if dataset_name == "IMDB":
            edge_types = [('director', 'to', 'movie'), ('actor', 'to', 'movie'), ('keyword', 'to', 'movie')]
            target_node_name = "movie"
        elif dataset_name == "DBLP":
            edge_types = [('paper', 'to', 'author')]
            target_node_name = "author"
        else:
            edge_types = [('author', 'to', 'paper'), ('term', 'to', 'paper')]
            target_node_name = "paper"
        data[target_node_name] = dataset[target_node_name]
        target_node_num = dataset[target_node_name].num_nodes
        for edge_type in edge_types:
            node_name = edge_type[0]
            node_num = dataset[node_name].num_nodes
            adj_linked_by_node = torch.zeros(target_node_num, target_node_num)
            for i in range(node_num):
                edge_type_index = dataset[edge_type].edge_index[0, :]
                linked_target_nodes = dataset[edge_type].edge_index[1, :][edge_type_index == i]
                # at least 2 target_node linked by node_name
                if linked_target_nodes.shape[0] > 1:
                    linked_target_nodes_list = linked_target_nodes.detach().tolist()
                    combination_list = list(itertools.combinations(linked_target_nodes_list, 2))
                    for (target_node1, target_node2) in combination_list:
                        adj_linked_by_node[target_node1, target_node2] = 1
            # undirected edges
            adj_linked_by_node = adj_linked_by_node + adj_linked_by_node.T
            edge_index_by_node, _ = dense_to_sparse(adj_linked_by_node)
            data[(target_node_name, node_name, target_node_name)].edge_index = edge_index_by_node
        if dataset_name == "ACM":
            addition_edge_types = [('paper', 'cite', 'paper'), ('paper', 'ref', 'paper')]
            for edge_type in addition_edge_types:
                data[edge_type].edge_index = dataset[edge_type].edge_index
        # save dataset
        torch.save(data, processed_data_path)
    else:
        data = torch.load(os.path.join(dataset_dir, f'multi_graph_{dataset_name}.pt'))
    return data


def reset_edge_index_by_map(node_map, edge_index):
    reset_edge_index = []
    for col in range(edge_index.shape[1]):
        src_node, dst_node = edge_index[:, col]
        reset_edge_index.append(torch.tensor([[node_map[src_node.item()]],
                                              [node_map[dst_node.item()]]]))
    reset_edge_index = torch.concat(reset_edge_index, dim=1)
    return reset_edge_index


def induced_subgraph(sampling_list, dataset, node_name, k_hop=2, lower_bound=3, upper_bound=101):
    data_list = []
    for node_i in sampling_list:
        # extract 2-hop edge_type of neighbors
        induced_subgraph_i = HeteroData()
        nb_subset_i = []
        for edge_type in dataset.edge_types:
            nb_subset_by_type, extract_edge_index_by_type, _, _ = (
                k_hop_subgraph(node_i, num_hops=k_hop, edge_index=dataset[edge_type].edge_index))
            nb_subset_i.append(nb_subset_by_type)
            if extract_edge_index_by_type.shape[1] != 0:
                induced_subgraph_i[edge_type].edge_index = extract_edge_index_by_type
        nb_subset_i = torch.concat(nb_subset_i).unique()
        # node number bound
        if nb_subset_i.shape[0] < lower_bound:
            continue
        else:
            # beyond the upper bound, randomly select
            if nb_subset_i.shape[0] > upper_bound:
                exclude_node_i = nb_subset_i[nb_subset_i != node_i].tolist()
                random_select_subset = random.sample(exclude_node_i, upper_bound - 1)
                random_select_subset.append(node_i)
                random_select_subset = torch.tensor(list(sorted(random_select_subset)))
                node_map = {random_select_subset[i].item(): i for i in range(random_select_subset.shape[0])}
                for edge_type in dataset.edge_types:
                    max_dataset_edge_index = dataset[edge_type].edge_index.max().detach().cpu().item()
                    max_select_node_index = random_select_subset.max().detach().cpu().item()
                    if max_dataset_edge_index < max_select_node_index:
                        upgrade_random_select_subset = random_select_subset[random_select_subset < max_dataset_edge_index]
                        edge_type_index, _ = subgraph(upgrade_random_select_subset, dataset[edge_type].edge_index)
                    else:
                        edge_type_index, _ = subgraph(random_select_subset, dataset[edge_type].edge_index)
                    del induced_subgraph_i[edge_type]
                    if edge_type_index.shape[1] > 0:
                        reset_edge_type_index = reset_edge_index_by_map(node_map, edge_type_index)
                        induced_subgraph_i[edge_type].edge_index = reset_edge_type_index
                induced_subgraph_i[node_name].x = dataset[node_name].x[random_select_subset].clone()
                induced_subgraph_i[node_name].y = dataset[node_name].y[random_select_subset].clone()
            # just reset the node index
            else:
                node_map = {nb_subset_i[i].item(): i for i in range(nb_subset_i.shape[0])}
                for edge_type in induced_subgraph_i.edge_types:
                    edge_type_index = induced_subgraph_i[edge_type].edge_index
                    del induced_subgraph_i[edge_type]
                    reset_edge_type_index = reset_edge_index_by_map(node_map, edge_type_index)
                    induced_subgraph_i[edge_type].edge_index = reset_edge_type_index
                induced_subgraph_i[node_name].x = dataset[node_name].x[nb_subset_i].clone()
                induced_subgraph_i[node_name].y = dataset[node_name].y[nb_subset_i].clone()
            induced_subgraph_i[node_name].target_node = node_i
            node_i_x = dataset[node_name].x[node_i]
            induced_subgraph_i[node_name].score = torch.cosine_similarity(node_i_x, induced_subgraph_i[node_name].x)
            # upgrade the information generated by fraudar
            induced_subgraph_i = generate_positive_subgraph_by_fraudar(induced_subgraph_i, node_name)
            data_list.append(induced_subgraph_i)
    return data_list


def generate_positive_subgraph_by_fraudar(graph_data, node_name, lower_bound=3, return_nodes_num=False):
    G = nx.DiGraph()
    # add nodes to G
    for i in range(graph_data[node_name].num_nodes):
        G.add_node(i, evil_score=graph_data[node_name].score[i].item())
    edge_index = [graph_data[edge_type].edge_index for edge_type in graph_data.edge_types]
    if len(edge_index) != 0:
        edge_index = torch.concat(edge_index, dim=1)
        unique_edge_index, indices = torch.unique(edge_index, dim=1, return_inverse=True)
        adj = to_dense_adj(edge_index, max_num_nodes=graph_data[node_name].score.shape[0]).squeeze()
        # add edges to G
        for i in range(unique_edge_index.shape[1]):
            srt = unique_edge_index[0, i].item()
            dst = unique_edge_index[1, i].item()
            try:
                G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
            except Exception as e:
                print(f"{e}")
                print(adj.shape, srt, dst)
        # find the best solution
        best_graph, best_density, best_weight = fraudar(G)
        if return_nodes_num:
            return len(best_graph.nodes)
        else:
            # tag on the graph
            idx = torch.zeros(graph_data[node_name].num_nodes, dtype=torch.int)
            flag = torch.tensor([0], dtype=torch.int)
            # filter subgraph with less than 3 nodes
            if len(best_graph.nodes) >= lower_bound:
                idx[sorted(best_graph.nodes)] = 1
                flag = torch.tensor([1], dtype=torch.int)
            graph_data[node_name].idx = idx
            graph_data[node_name].flag = flag
    else:
        graph_data[node_name].idx = torch.zeros(graph_data[node_name].num_nodes, dtype=torch.int)
        graph_data[node_name].flag = torch.tensor([0], dtype=torch.int)
    return graph_data


if __name__ == '__main__':
    name = "IMDB"
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.path.pardir, os.path.pardir,
                                            "data", "hetero_datasets", name))
    data = load_dataset(data_dir, name)
    induced_subgraph([0, 1, 2, 3], data, 'movie')
