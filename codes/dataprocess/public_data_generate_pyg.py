import os
import dgl
import torch
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import k_hop_subgraph
from codes.dataprocess.public_data_utils import load_pt_dataset, load_mat_dataset, load_dgl_dataset, multi_layer_sampling

class WeiboDataset(InMemoryDataset):
    def __init__(self, root, name='weibo', k=1, transform=None, pre_transform=None):
        self.name = name
        self.k = k
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'{self.name}_processed')

    @property
    def processed_file_names(self):
        return [f'{self.name}_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        dataset = load_pt_dataset(self.raw_paths[0])
        node_idx = range(dataset.num_nodes)
        for idx in node_idx:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(idx, self.k, dataset.edge_index,
                                                                        relabel_nodes=True)
            if subset.shape[0] < 5 or subset.shape[0] > 100:
                continue
            else:
                extract_x = dataset.x[subset]
                extract_y = dataset.y[subset]
                data = Data(x=extract_x, edge_index=sub_edge_index, y=extract_y)
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class FacebookDataset(InMemoryDataset):
    def __init__(self, root, name='fb', k=2, transform=None, pre_transform=None):
        self.name = name
        self.k = k
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'{self.name}_processed')

    @property
    def processed_file_names(self):
        return [f'{self.name}_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        dataset = load_mat_dataset(self.raw_paths[0])
        x = dataset['X']
        x = torch.from_numpy(x).float()
        adj = dataset['A']
        y = dataset['Label'].squeeze()
        y = torch.from_numpy(y)
        row, col = np.nonzero(adj)
        edge_index = np.vstack((row, col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_idx = range(y.shape[0])
        for idx in node_idx:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(idx, self.k, edge_index, relabel_nodes=True)
            if subset.shape[0] < 5 or subset.shape[0] > 100:
                continue
            else:
                extract_x = x[subset]
                extract_y = y[subset]
                data = Data(x=extract_x, edge_index=sub_edge_index, y=extract_y)
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class AmazonDataset(InMemoryDataset):
    def __init__(self, root, name='Amazon', k=1, transform=None, pre_transform=None):
        self.name = name
        self.k = k
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'{self.name}_processed')

    @property
    def processed_file_names(self):
        return [f'{self.name}_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        dataset = load_mat_dataset(self.raw_paths[0])
        x = dataset['features'].toarray()
        adj = dataset['homo'].toarray()
        y = dataset['label'].squeeze()
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)
        row, col = np.nonzero(adj)
        edge_index = np.vstack((row, col))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        node_idx = range(y.shape[0])
        for idx in node_idx:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(idx, self.k, edge_index, relabel_nodes=True)
            if subset.shape[0] < 5 or subset.shape[0] > 100:
                continue
            else:
                extract_x = x[subset]
                extract_y = y[subset]
                data = Data(x=extract_x, edge_index=sub_edge_index, y=extract_y)
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TFinanceDataset(InMemoryDataset):
    def __init__(self, root, name='tfinance', k=1, transform=None, pre_transform=None):
        self.name = name
        self.k = k
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_dir(self):
        return os.path.join(self.root, f'{self.name}_processed')

    @property
    def processed_file_names(self):
        return [f'{self.name}_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        dataset, _ = load_dgl_dataset(self.raw_paths[0])
        graph = dataset[0]
        x = graph.ndata['feature'].float()
        y = graph.ndata['label'].argmax(dim=1)
        node_idx = range(graph.num_nodes())
        for idx in node_idx:
            subgraph, mapping = dgl.khop_out_subgraph(graph, idx, self.k)
            subset = subgraph.ndata[dgl.NID]
            src, dst = subgraph.edges()
            sub_edge_index = torch.stack([src, dst], dim=0)
            if subset.shape[0] < 5 or subset.shape[0] > 100:
                continue
            else:
                extract_x = x[subset]
                extract_y = y[subset]
                data = Data(x=extract_x, edge_index=sub_edge_index, y=extract_y)
                data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class TSocialDataset(InMemoryDataset):
    def __init__(self, root, name='tsocial', k=2, limit=10, num=50000, random_state=0, transform=None, pre_transform=None):
        self.name = name
        self.k = k
        self.limit = limit
        self.num = num
        self.random_state = random_state
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.name}']

    @property
    def processed_file_names(self):
        return [f'{self.name}_{self.k}_hop_{self.limit}_neighbors_{self.num}_samples_processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        dataset, _ = load_dgl_dataset(self.raw_paths[0])
        graph = dataset[0]
        total_node_idx = range(graph.num_nodes())
        random.seed(self.random_state)
        node_idx = random.sample(total_node_idx, self.num)
        anomaly_idx = (graph.ndata["label"] == 1).nonzero().squeeze()
        center_anomaly_count = 0
        center_normal_count = 0
        phar = tqdm(total=self.num)
        for idx in node_idx:
            if idx in anomaly_idx:
                center_anomaly_count += 1
            else:
                center_normal_count += 1
            fanouts = [self.limit for k in range(self.k)]
            blocks = multi_layer_sampling(graph, torch.tensor([idx]), fanouts)
            node_ids = torch.concat([blocks[0].srcdata['_ID'], blocks[0].dstdata['_ID'],
                                     blocks[1].srcdata['_ID'], blocks[1].dstdata['_ID']]).unique()
            extracted_graph = dgl.node_subgraph(graph, node_ids)
            if extracted_graph.num_nodes() < 5 or extracted_graph.num_nodes() > 150:
                continue
            else:
                extract_x = extracted_graph.ndata['feature'].float()
                extract_y = extracted_graph.ndata['label'].float()
                edge_index = torch.stack([extracted_graph.edges()[0], extracted_graph.edges()[1]], dim=0)
                data = Data(x=extract_x, edge_index=edge_index, y=extract_y)
                data_list.append(data)
            phar.update(1)
        phar.close()
        data, slices = self.collate(data_list)
        print(f"Anomaly Center Count: {center_anomaly_count}, Normal Center Count: {center_normal_count}")
        torch.save((data, slices), self.processed_paths[0])


def compute_ave_size():
    data = TFinanceDataset(root='../../data/public_dataset/')
    num_nodes = [graph.num_nodes for graph in data]
    ano_nodes = []
    for graph in data:
        ano_num = (graph.y == 1).nonzero().squeeze()
        if ano_num.shape != torch.Size([]):
            if ano_num.shape[0] != 0:
                ano_nodes.append(ano_num.shape[0])
    ave_size = sum(num_nodes) / len(num_nodes)
    ave_ano = sum(ano_nodes) / len(ano_nodes)
    print(f"Subgraph Ave Size: {ave_size: .2f}, Gang num:{len(ano_nodes): .2f}, Ave Gang Size: {ave_ano: .2f}")

if __name__ == '__main__':
    tsocial_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir, os.path.pardir, "data", "public_dataset", "TSocial")
    TSocialDataset(root=tsocial_root)