import os
import torch
import random
import itertools
from torch_geometric.datasets import HGBDataset
from torch_geometric.utils import dense_to_sparse, subgraph, k_hop_subgraph
from torch_geometric.data import HeteroData


def load_dataset(dataset_dir, dataset_name):
    processed_data_path = os.path.join(dataset_dir, f'multi_graph_{dataset_name}.pt')
    if not os.path.exists(processed_data_path):
        dataset = HGBDataset(root=dataset_dir, name=dataset_name)[0]
        data = HeteroData()
        if dataset_name == 'IMDB':
            edge_types = [('director', 'to', 'movie'), ('actor', 'to', 'movie'), ('keyword', 'to', 'movie')]
            target_node_name = 'movie'
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


def induced_subgraph(sampling_list, dataset, node_name, k_hop=2, threshold=101):
    data_list = []
    for node_i in sampling_list:
        # extract 2-hop edge_type of neighbors
        induced_subgraph_i = HeteroData()
        nb_subset_i = []
        for edge_type in dataset.edge_types:
            nb_subset_by_type, extract_edge_index_by_type, _, _ = (
                k_hop_subgraph(node_i, num_hops=k_hop, edge_index=dataset[edge_type].edge_index))
            nb_subset_i.append(nb_subset_by_type)
            induced_subgraph_i[edge_type].edge_index = extract_edge_index_by_type
        nb_subset_i = torch.concat(nb_subset_i).unique()
        # if beyond the threshold, randomly select
        if nb_subset_i.shape[0] > threshold:
            exclude_node_i = nb_subset_i[nb_subset_i != node_i].tolist()
            random_select_subset = random.sample(exclude_node_i, threshold - 1)
            random_select_subset.append(node_i)
            random_select_subset = torch.tensor(list(sorted(random_select_subset)))
            node_map = {random_select_subset[i].item(): i for i in range(random_select_subset.shape[0])}
            for edge_type in dataset.edge_types:
                edge_type_index, _ = subgraph(random_select_subset, dataset[edge_type].edge_index)
                del induced_subgraph_i[edge_type]
                reset_edge_type_index = reset_edge_index_by_map(node_map, edge_type_index)
                induced_subgraph_i[edge_type].edge_index = reset_edge_type_index
            induced_subgraph_i[node_name].x = dataset[node_name].x[random_select_subset].clone()
            induced_subgraph_i[node_name].y = dataset[node_name].y[random_select_subset].clone()
        else:
            node_map = {nb_subset_i[i]: i for i in range(nb_subset_i.shape[0])}
            for edge_type in induced_subgraph_i.edge_types:
                edge_type_index = induced_subgraph_i[edge_type].edge_index
                del induced_subgraph_i[edge_type]
                reset_edge_type_index = reset_edge_index_by_map(node_map, edge_type_index)
                induced_subgraph_i[edge_type].edge_index = reset_edge_type_index
            induced_subgraph_i[node_name].x = dataset[node_name].x[nb_subset_i].clone()
            induced_subgraph_i[node_name].y = dataset[node_name].y[nb_subset_i].clone()
        data_list.append(induced_subgraph_i)
    return data_list


if __name__ == '__main__':
    name = 'IMDB'
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.path.pardir, os.path.pardir,
                                            "data", "hetero_datasets", name))
    data = load_dataset(data_dir, name)