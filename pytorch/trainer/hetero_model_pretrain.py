import torch
import random
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.dataprocess.other_datasets import load_dataset, induced_subgraph

class ModelPreTrain:
    def __init__(self, args_dict):
        self.train_dict = args_dict
        self.conv_type = args_dict["conv_type"]
        self.dataset_dir = args_dict["dataset_dir"]
        self.dataset_name = args_dict["dataset_name"]
        self.batch_size = args_dict["batch_size"]
        self.epoch_num = args_dict["epoch_num"]
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.dataset_name == 'IMDB':
            self.target_node_name = 'movie'
        self.raw_dataset = load_dataset(self.dataset_dir, self.dataset_name)
        self.metadata = self.raw_dataset.metadata()
        self.edge_types = {self.metadata[1][i]: i for i in range(len(self.metadata[1]))}
        if self.conv_type == 'RGCN':
            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=args_dict['num_relations'])
        self.model = self.model.to(self.device)
        self.pt_subgraph_idx = (self.raw_dataset[self.target_node_name].test_mask == 1).nonzero().squeeze().tolist()
        random.shuffle(self.pt_subgraph_idx)
        pt_size = len(self.pt_subgraph_idx)
        self.batch_num = pt_size // self.batch_size if pt_size % self.batch_size == 0 else pt_size // self.batch_size + 1

    def set_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def get_edge_info(self, batch):
        edge_index = [batch[edge_type].edge_index for edge_type
                      in list(self.edge_types.keys()) if edge_type in batch.edge_types]
        edge_index = torch.concat(edge_index, dim=1)
        edge_counts = {edge_type: batch[edge_type].num_edges for edge_type in list(self.edge_types.keys()) if
                       edge_type in batch.edge_types}
        edge_type = torch.concat(
            [torch.ones(edge_counts[edge_type]) * edge_idx for edge_type, edge_idx in self.edge_types.items() if
             edge_type in batch.edge_types])
        return edge_index, edge_type.long()

    def extract_smaller_batch_subgraph(self, batch, subgraph_node_indices=None):
        if subgraph_node_indices is None:
            node_indices = batch[self.target_node_name].idx
            subgraph_node_indices = (node_indices == 1).nonzero().squeeze()
        x = torch.zeros_like(batch[self.target_node_name].x).to(self.device)
        x[subgraph_node_indices] = batch[self.target_node_name].x[subgraph_node_indices].clone()
        new_batch = batch.clone()
        new_batch[self.target_node_name].x = x
        subgraph_max_node_idx = subgraph_node_indices.max()
        for edge_type in batch.edge_types:
            try:
                current_max_node_idx = batch[edge_type].edge_index.max()
                max_node_idx = min(subgraph_max_node_idx, current_max_node_idx)
                if max_node_idx < subgraph_max_node_idx:
                    subgraph_node_indices = subgraph_node_indices[subgraph_node_indices <= max_node_idx]
                edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index)
            except Exception as e:
                print(f"Extract Subgraph Error: <{e}>")
                del new_batch[edge_type]
                continue
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    new_batch[edge_type].edge_index = edge_index
                else:
                    del new_batch[edge_type]
        return new_batch

    def pretraining(self):
        self.model.train()
        for epoch in range(1, self.epoch_num + 1):
            for i in range(self.batch_num):
                if i != self.batch_num - 1:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size:]
                batch_subgraph = induced_subgraph(batch_subgraph_idx, self.raw_dataset, self.target_node_name)
                batch_subgraph = Batch.from_data_list(batch_subgraph).to(self.device)
                batch_edge_index, batch_edge_types = self.get_edge_info(batch_subgraph)
                batch_x = torch.nn.functional.normalize(batch_subgraph[self.target_node_name].x, dim=1)
                # raw subgraph
                batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
                batch_h_g = scatter_mean(batch_h, batch_subgraph[self.target_node_name].batch, dim=0)
                # extract smaller subgraph based on fraudar
                select_node_idx = (batch_subgraph[self.target_node_name].idx).nonzero().squeeze().detach().cpu()