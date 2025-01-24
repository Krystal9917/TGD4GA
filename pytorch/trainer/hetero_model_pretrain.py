import os
import time
import torch
import random
import numpy as np
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch.utils.tensorboard import SummaryWriter
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.dataprocess.other_datasets import load_dataset, reset_edge_index_by_map, \
    induced_subgraph


class ModelPreTrain:
    def __init__(self, args_dict):
        self.train_dict = args_dict
        self.conv_type = args_dict["conv_type"]
        self.data_dir = args_dict["train_data_path"]
        self.dataset_name = args_dict["dataset_name"]
        self.batch_size = args_dict["batch_size"]
        self.epoch_num = args_dict["n_epochs"]
        self.task_type = args_dict["task_type"]
        self.print_batch = args_dict["print_batch_num"]
        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # dataset
        if self.dataset_name == "IMDB":
            self.target_node_name = "movie"
            input_dim = 3489
            num_relations = 3
        elif self.dataset_name == "DBLP":
            self.target_node_name = "author"
            input_dim = 1024
            num_relations = 1
        else:
            self.target_node_name = "paper"
            input_dim = 1902
            num_relations = 4
        self.raw_dataset = load_dataset(os.path.join(self.data_dir, self.dataset_name), self.dataset_name)
        self.metadata = self.raw_dataset.metadata()
        self.edge_types = {self.metadata[1][i]: i for i in range(len(self.metadata[1]))}
        # model
        self.model = RGCN(input_dim=input_dim,
                          hidden_dim=args_dict['hidden_dim'],
                          output_dim=args_dict["output_dim"],
                          num_relations=num_relations)
        self.model = self.model.to(self.device)
        self.lr = args_dict["lr"]
        self.best_loss = args_dict["best_loss"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # save dir
        self.log_file_path = f"{self.dataset_name}_pretraining_{self.lr}"
        loss_log_path = os.path.abspath(os.path.join(args_dict['log_dir'],
                                                     args_dict["model_states_path"].split('/')[-1],
                                                     self.log_file_path))
        if not os.path.exists(loss_log_path):
            os.makedirs(loss_log_path)
        self.writer = SummaryWriter(log_dir=loss_log_path)
        self.save_model_path = os.path.join(self.train_dict["model_states_path"],
                                            self.log_file_path)
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        self.set_seed()
        # dataloader
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

    def rgcn_fit(self, batch):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(batch)
            normalized_x = torch.nn.functional.normalize(batch[self.target_node_name].x, dim=1)
            batch_h = self.model(normalized_x, batch_edge_index, batch_edge_types)
        except Exception as e:
            print(f"{self.conv_type} Get Edge Information Error: <{e}>")
            return None
        else:
            return batch_h

    def reset_batch_node(self, batch):
        batch_item = batch.unique().detach().cpu().tolist()
        batch_map = {batch_item[i]: i for i in range(len(batch_item))}
        upgrade_batch = torch.tensor([batch_map[item] for item in batch.detach().cpu().tolist()], device=self.device)
        return upgrade_batch

    def extract_smaller_batch_subgraph(self, batch, subgraph_node_indices=None):
        if subgraph_node_indices is None:
            subgraph_node_indices = (batch[self.target_node_name].idx == 1).nonzero().squeeze()
        batch_copy = batch.clone()
        batch_copy[self.target_node_name].x = batch[self.target_node_name].x[subgraph_node_indices]
        batch_copy[self.target_node_name].y = batch[self.target_node_name].y[subgraph_node_indices]
        batch_copy[self.target_node_name].score = batch[self.target_node_name].score[subgraph_node_indices]
        batch_copy[self.target_node_name].idx = batch[self.target_node_name].idx[subgraph_node_indices]
        subgraph_node_batch_idxes = batch[self.target_node_name].batch[subgraph_node_indices]
        batch_copy[self.target_node_name].batch = self.reset_batch_node(subgraph_node_batch_idxes)
        node_map = {subgraph_node_indices[i].detach().cpu().item(): i for i in range(subgraph_node_indices.shape[0])}
        subgraph_max_node_idx = subgraph_node_indices.max()
        for edge_type in batch.edge_types:
            try:
                current_max_node_idx = batch[edge_type].edge_index.max()
                max_node_idx = min(subgraph_max_node_idx, current_max_node_idx)
                if max_node_idx < subgraph_max_node_idx:
                    subgraph_node_indices = subgraph_node_indices[subgraph_node_indices <= max_node_idx]
                edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index)
            except Exception as e:
                print(f"Extract Edge Index Error: <{e}>")
                del batch_copy[edge_type]
                continue
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    # reset the edge index
                    reset_edge_index = reset_edge_index_by_map(node_map, edge_index)
                    batch_copy[edge_type].edge_index = reset_edge_index
                else:
                    del batch_copy[edge_type]
        return batch_copy

    def preference_contrastive_loss(self, h1, h2, h3):
        t = self.train_dict["temperature"]
        try:
            if len(h1.shape) == 1:
                h1 = h1.unsqueeze(0)
            if len(h3.shape) == 1:
                h3 = h3.unsqueeze(0)
            h1_abs = h1.norm(dim=1)
            h3_abs = h3.norm(dim=1)
            pos_sim = torch.exp(torch.cosine_similarity(h1, h2) / t)
            sim_matrix = torch.einsum('ik,jk->ij', h1, h3) / torch.einsum('i,j->ij', h1_abs, h3_abs)
            sim_matrix = torch.exp(sim_matrix / t)
        except Exception as e:
            loss = torch.tensor(torch.nan).to(h1.device)
            print(f"Norm error <{e}>, h1 shape: {h1.shape}, h3 shape: {h3.shape}")
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) + 1e-4)
            loss = -torch.log(loss).mean()
        return loss

    def node_contrastive_loss(self, h1, h2):
        t = self.train_dict["temperature"]
        if len(h1.shape) == 1:
            h1 = h1.unsqueeze(0)
        if len(h2.shape) == 1:
            h2 = h2.unsqueeze(0)
        h1_abs = h1.norm(dim=1)
        h2_abs = h2.norm(dim=1)
        pos_matrix = torch.einsum('ij,jk->ik', h1, h1.T) / torch.einsum('i,j->ij', h1_abs, h1_abs)
        pos_matrix = pos_matrix - torch.eye(pos_matrix.shape[0], device=self.device)
        pos_matrix = torch.exp(pos_matrix / t)
        neg_matrix = torch.einsum('ij,jk->ik', h1, h2.T) / torch.einsum('i,j->ij', h1_abs, h2_abs)
        neg_matrix = torch.exp(neg_matrix / t)
        loss = pos_matrix.sum(dim=1) / (pos_matrix.sum(dim=1) + neg_matrix.sum(dim=1) + 1e-4)
        loss = -torch.log(loss).mean()
        return loss

    def batch_contrastive_loss(self, pos_h, neg_h, pos_batch, neg_batch):
        idx_list = pos_batch.unique().detach().cpu().tolist()
        batch_loss_list = []
        for i in idx_list:
            subgraph_pos_h = pos_h[pos_batch == i]
            subgraph_neg_h = neg_h[neg_batch == i]
            batch_loss = self.node_contrastive_loss(subgraph_pos_h, subgraph_neg_h)
            batch_loss_list.append(batch_loss)
        return torch.stack([item for item in batch_loss_list if not torch.isnan(item)]).mean()

    def cross_contrastive_loss(self, fraudar_pos_h, subgraph_pos_h, subgraph_neg_h, pos_batch, neg_batch):
        idx_list = pos_batch.unique().detach().cpu().tolist()
        batch_loss_list = []
        for i in idx_list:
            i_fraudar_pos_h = fraudar_pos_h[pos_batch == i]
            i_subgraph_pos_h = subgraph_pos_h[pos_batch == i]
            i_subgraph_neg_h = subgraph_neg_h[neg_batch == i]
            cross_loss = self.preference_contrastive_loss(i_subgraph_pos_h, i_fraudar_pos_h, i_subgraph_neg_h)
            batch_loss_list.append(cross_loss)
        return torch.stack([item for item in batch_loss_list if not torch.isnan(item)]).mean()

    def compute_anomalous_subgraph_anchor(self, raw_feature_x, batch):
        idx_list = batch.unique().detach().cpu().tolist()
        anchor_list = []
        for i in idx_list:
            anomalous_subgraph_i = raw_feature_x[batch == i]
            average_node_embedding_i = anomalous_subgraph_i.mean(dim=0)
            anchor_list.append(average_node_embedding_i)
        return torch.stack(anchor_list)

    def exclude_anomalous_nodes_from_normals(self, normal_x, anomalous_anchors, batch):
        idx_list = batch.unique().detach().cpu().tolist()
        exclude_list = []
        for i in range(len(idx_list)):
            normal_batch_i = normal_x[batch == idx_list[i]]
            anomalous_anchor_i = anomalous_anchors[i]
            average_normal_i = normal_batch_i.mean(dim=0)
            similarity_diff = (torch.cosine_similarity(normal_batch_i, anomalous_anchor_i) -
                               torch.cosine_similarity(normal_batch_i, average_normal_i))
            exclude_idx = (similarity_diff >= self.train_dict['similarity_diff']).nonzero().squeeze().detach().cpu()
            prefix_node_idx = (batch < idx_list[i]).nonzero().squeeze()
            if prefix_node_idx.shape != torch.Size([]):
                exclude_idx = exclude_idx + prefix_node_idx.shape[0]
            else:
                exclude_idx = exclude_idx + torch.tensor(1)
            if exclude_idx.shape != torch.Size([]):
                exclude_list.extend(exclude_idx.tolist())
            else:
                exclude_list.extend([exclude_idx.detach().cpu().item()])
        return exclude_list

    def pretraining(self):
        self.model.train()
        for epoch in range(1, self.epoch_num + 1):
            epoch_loss = []
            epoch_start_time = time.time()
            for i in range(self.batch_num):
                batch_start_time = time.time()
                if i != self.batch_num - 1:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size:]
                batch_subgraph = induced_subgraph(batch_subgraph_idx, self.raw_dataset, self.target_node_name)
                batch_subgraph = Batch.from_data_list(batch_subgraph).to(self.device)

                # raw subgraph
                batch_h = self.rgcn_fit(batch_subgraph)
                batch_h_g = scatter_mean(batch_h, batch_subgraph[self.target_node_name].batch, dim=0)

                pos_subgraph_idx = (
                        batch_subgraph[self.target_node_name].flag == 1).nonzero().squeeze().detach().cpu().tolist()
                neg_subgraph_idx = (
                        batch_subgraph[self.target_node_name].flag == 0).nonzero().squeeze().detach().cpu().tolist()

                # more than one subgraph
                if type(pos_subgraph_idx) is list:
                    list_flag = len(pos_subgraph_idx) != 0
                else:
                    list_flag = False
                # only one subgraph
                int_flag = type(pos_subgraph_idx) is int

                if list_flag or int_flag:
                    # subgraph generated from fraudar (the index of initial graphs)
                    fraudar_batch_subgraph = self.extract_smaller_batch_subgraph(batch_subgraph)
                    fraudar_batch_subgraph = fraudar_batch_subgraph.to(self.device)

                    fraudar_batch_h = self.rgcn_fit(fraudar_batch_subgraph)
                    fraudar_batch_h_g = scatter_mean(fraudar_batch_h,
                                                     fraudar_batch_subgraph[self.target_node_name].batch, dim=0)

                    pos_batch_h_g = batch_h_g[pos_subgraph_idx]
                    neg_batch_h_g = batch_h_g[neg_subgraph_idx]

                    inter_loss = self.preference_contrastive_loss(pos_batch_h_g, fraudar_batch_h_g, neg_batch_h_g)
                    # intra-subgraph contrastive learning (high possibility subgraphs inside)
                    if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph',
                                          'cross_subgraph', 'fine_grained_cross_subgraph']:
                        if list_flag:
                            batch_pos_neg_samples_idx = torch.concat(
                                [(batch_subgraph[self.target_node_name].batch == i).nonzero().squeeze().detach().cpu()
                                 for i in pos_subgraph_idx], dim=0).tolist()
                        else:
                            batch_pos_neg_samples_idx = (batch_subgraph[self.target_node_name].batch ==
                                                         pos_subgraph_idx).nonzero().squeeze().detach().cpu().tolist()
                        pos_samples_idx = (batch_subgraph[
                                               self.target_node_name].idx == 1).nonzero().squeeze().detach().cpu().tolist()
                        batch_pos_samples_idx = list(set(batch_pos_neg_samples_idx) & set(pos_samples_idx))
                        batch_pos_samples_idx_batch = batch_subgraph[self.target_node_name].batch[batch_pos_samples_idx]

                        batch_neg_samples_idx = list(set(batch_pos_neg_samples_idx) - set(pos_samples_idx))
                        batch_neg_samples_idx_batch = batch_subgraph[self.target_node_name].batch[batch_neg_samples_idx]

                        if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph']:
                            if self.task_type == 'fine_grained_batch_subgraph':
                                batch_anomalous_anchors = self.compute_anomalous_subgraph_anchor(
                                    batch_subgraph[self.target_node_name].x[batch_pos_samples_idx],
                                    batch_pos_samples_idx_batch)
                                batch_exclude_normal_idx = self.exclude_anomalous_nodes_from_normals(
                                    batch_subgraph[self.target_node_name].x[batch_neg_samples_idx],
                                    batch_anomalous_anchors,
                                    batch_neg_samples_idx_batch)
                                upgrade_batch_neg_samples_idx = list(
                                    set(batch_neg_samples_idx) - set(batch_exclude_normal_idx))
                                upgrade_batch_neg_samples_idx_batch = batch_subgraph[self.target_node_name].batch[
                                    upgrade_batch_neg_samples_idx]
                                batch_loss = self.batch_contrastive_loss(batch_h[batch_pos_samples_idx],
                                                                         batch_h[upgrade_batch_neg_samples_idx],
                                                                         batch_pos_samples_idx_batch,
                                                                         upgrade_batch_neg_samples_idx_batch)
                            else:
                                batch_loss = self.batch_contrastive_loss(batch_h[batch_pos_samples_idx],
                                                                         batch_h[batch_neg_samples_idx],
                                                                         batch_pos_samples_idx_batch,
                                                                         batch_neg_samples_idx_batch)
                            intra_loss = batch_loss
                        else:
                            if fraudar_batch_h is not None:
                                if self.task_type == 'fine_grained_cross_subgraph':
                                    batch_anomalous_anchors = self.compute_anomalous_subgraph_anchor(
                                        batch_subgraph[self.target_node_name].x[batch_pos_samples_idx],
                                        batch_pos_samples_idx_batch)
                                    batch_exclude_normal_idx = self.exclude_anomalous_nodes_from_normals(
                                        batch_subgraph[self.target_node_name].x[batch_neg_samples_idx],
                                        batch_anomalous_anchors,
                                        batch_neg_samples_idx_batch)
                                    upgrade_batch_neg_samples_idx = list(
                                        set(batch_neg_samples_idx) - set(batch_exclude_normal_idx))
                                    upgrade_batch_neg_samples_idx_batch = batch_subgraph[self.target_node_name].batch[
                                        upgrade_batch_neg_samples_idx]
                                    cross_loss = self.cross_contrastive_loss(fraudar_batch_h[batch_pos_samples_idx],
                                                                             batch_h[batch_pos_samples_idx],
                                                                             batch_h[upgrade_batch_neg_samples_idx],
                                                                             batch_pos_samples_idx_batch,
                                                                             upgrade_batch_neg_samples_idx_batch)
                                else:
                                    cross_loss = self.cross_contrastive_loss(fraudar_batch_h[batch_pos_samples_idx],
                                                                             batch_h[batch_pos_samples_idx],
                                                                             batch_h[batch_neg_samples_idx],
                                                                             batch_pos_samples_idx_batch,
                                                                             batch_neg_samples_idx_batch)
                                intra_loss = cross_loss
                            else:
                                intra_loss = torch.tensor(torch.nan).to(self.device)
                        loss = inter_loss + intra_loss
                    else:
                        loss = inter_loss
                else:
                    loss = torch.tensor(torch.nan).to(self.device)
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer.step()
                    loss_value = loss.detach().cpu().item()
                    epoch_loss.append(loss_value)
                    if (i + 1) % self.print_batch == 0:
                        if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph']:
                            print(
                                "Batch: {}, Loss: {:.6f}, "
                                "Batch Loss: {:.6f}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                                    i + 1,
                                    loss_value,
                                    batch_loss.detach().cpu().item(),
                                    subgraph_loss.detach().cpu().item(),
                                    time.time() - batch_start_time))
                        elif self.task_type in ['cross_subgraph', 'fine_grained_cross_subgraph']:
                            print(
                                "Batch: {}, Loss: {:.6f}, "
                                "Cross Loss: {:.6f}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                                    i + 1,
                                    loss_value,
                                    cross_loss.detach().cpu().item(),
                                    subgraph_loss.detach().cpu().item(),
                                    time.time() - batch_start_time))
                        else:
                            print(
                                "Batch: {}, Loss: {:.6f}, "
                                "Time: {:.4f} s".format(
                                    i + 1,
                                    loss_value,
                                    time.time() - batch_start_time))

            torch.cuda.empty_cache()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            save_prefix = f"{self.conv_type}_{self.task_type}_{self.lr}"
            self.writer.add_scalar(f"{save_prefix}_pretraining_loss", epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                file_name = os.path.join(self.save_model_path, f"{save_prefix}_best_loss.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Now best loss: {self.best_loss:.4f}, save model to {file_name}")
                if epoch % 4 == 0:
                    epoch_file_name = os.path.join(self.save_model_path, f"{save_prefix}_epoch_{epoch}.pth")
                    torch.save(self.model.state_dict(), epoch_file_name)
                    print(f"Now best loss: {self.best_loss:.4f}, save model to {epoch_file_name}")
        self.writer.close()
