import os
import time
import torch
import random
import numpy as np
from collections import OrderedDict
from torch_scatter import scatter_mean
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from torch.utils.tensorboard import SummaryWriter
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.dataprocess.other_datasets import load_dataset, reset_edge_index_by_map, \
    induced_subgraph
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    multilabel_confusion_matrix


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
        self.is_evaluate = args_dict["evaluate"]
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
            node_classes = 5
        else:
            self.target_node_name = "paper"
            input_dim = 1902
            num_relations = 4
            node_classes = 5
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
        self.log_file_path = f"{self.dataset_name}_pretraining_{self.lr}"
        loss_log_path = os.path.abspath(os.path.join(args_dict['log_dir'],
                                                     args_dict["model_states_path"].split('/')[-1],
                                                     self.log_file_path))
        self.save_model_path = os.path.join(self.train_dict["model_states_path"],
                                            self.log_file_path)
        if self.is_evaluate:
            # load pretrained model
            if args_dict["evaluate_epoch"] == 0:
                file_name = os.path.join(self.save_model_path,
                                         f"{self.conv_type}_{self.task_type}_{self.lr}_best_loss.pth")
            else:
                file_name = os.path.join(self.save_model_path,
                                         f"{self.conv_type}_{self.task_type}_{self.lr}_epoch_{args_dict['evaluate_epoch']}.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            rename_key_model_weight = OrderedDict()
            for key in model_weight.keys():
                key_weight = model_weight[key]
                key = key.replace('module.', '')
                rename_key_model_weight[key] = key_weight
            self.model.load_state_dict(rename_key_model_weight)
            # classifier
            self.classifier = torch.nn.Sequential(torch.nn.Linear(args_dict['output_dim'], args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], node_classes),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args_dict["evaluate_lr"])
            self.criterion = torch.nn.CrossEntropyLoss()
            # set seed
            self.set_seed()
            # dataloader
            self.ft_subgraph_idx = (
                    self.raw_dataset[self.target_node_name].train_mask == 1).nonzero().squeeze().tolist()
            random.shuffle(self.ft_subgraph_idx)
            ft_size = len(self.ft_subgraph_idx)
            ratio = 0.8
            ft_train_size = int(ft_size * ratio)
            # fine-tuning train
            self.ft_train_idx = self.ft_subgraph_idx[:ft_train_size]
            ft_train_size = len(self.ft_train_idx)
            print(f"Train Data Length: {ft_train_size}")
            self.train_batch_num = ft_train_size // self.batch_size if ft_train_size % self.batch_size == 0 \
                else ft_train_size // self.batch_size + 1
            # fine-tuning test
            self.ft_test_idx = self.ft_subgraph_idx[ft_train_size:]
            ft_test_size = len(self.ft_test_idx)
            print(f"Test Data Length: {ft_test_size}")
            self.test_batch_num = ft_test_size // self.batch_size if ft_test_size % self.batch_size == 0 \
                else ft_test_size // self.batch_size + 1
        else:
            self.best_loss = args_dict["best_loss"]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            # loss save dir
            if not os.path.exists(loss_log_path):
                os.makedirs(loss_log_path)
            self.writer = SummaryWriter(log_dir=loss_log_path)
            # model save dir
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
        ave_nodes = []
        ave_edges = []
        ave_types = []
        for epoch in range(1, self.epoch_num + 1):
            epoch_loss = []
            epoch_start_time = time.time()
            for i in range(self.batch_num):
                batch_start_time = time.time()
                if i != self.batch_num - 1:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    batch_subgraph_idx = self.pt_subgraph_idx[i * self.batch_size:]
                batch_subgraph = induced_subgraph(batch_subgraph_idx, self.raw_dataset, self.target_node_name,
                                                  lower_bound=self.train_dict["filter_node_num"], upper_bound=300)
                batch_subgraph = Batch.from_data_list(batch_subgraph).to(self.device)

                if epoch == 1:
                    ave_nodes.append(batch_subgraph[self.target_node_name].num_nodes)
                    ave_edges.append(batch_subgraph.num_edges)
                    ave_types.append(len(batch_subgraph.edge_types))

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
                    if fraudar_batch_h is not None:
                        fraudar_batch_h_g = scatter_mean(fraudar_batch_h,
                                                         fraudar_batch_subgraph[self.target_node_name].batch, dim=0)
                        pos_batch_h_g = batch_h_g[pos_subgraph_idx]
                        neg_batch_h_g = batch_h_g[neg_subgraph_idx]
                        if neg_batch_h_g.shape[0] != 0:
                            inter_loss = self.preference_contrastive_loss(pos_batch_h_g, fraudar_batch_h_g, neg_batch_h_g)
                        else:
                            inter_loss = torch.tensor(0, device=self.device)
                        # intra-subgraph contrastive learning (high possibility subgraphs inside)
                        if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph',
                                              'cross_subgraph', 'fine_grained_cross_subgraph']:
                            if list_flag:
                                batch_pos_neg_samples_idx = torch.concat(
                                    [(batch_subgraph[
                                          self.target_node_name].batch == i).nonzero().squeeze().detach().cpu()
                                     for i in pos_subgraph_idx], dim=0).tolist()
                            else:
                                batch_pos_neg_samples_idx = (batch_subgraph[self.target_node_name].batch ==
                                                             pos_subgraph_idx).nonzero().squeeze().detach().cpu().tolist()
                            pos_samples_idx = (batch_subgraph[
                                                   self.target_node_name].idx == 1).nonzero().squeeze().detach().cpu().tolist()
                            batch_pos_samples_idx = list(set(batch_pos_neg_samples_idx) & set(pos_samples_idx))
                            batch_pos_samples_idx_batch = batch_subgraph[self.target_node_name].batch[
                                batch_pos_samples_idx]

                            batch_neg_samples_idx = list(set(batch_pos_neg_samples_idx) - set(pos_samples_idx))
                            batch_neg_samples_idx_batch = batch_subgraph[self.target_node_name].batch[
                                batch_neg_samples_idx]

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
                                        upgrade_batch_neg_samples_idx_batch = \
                                            batch_subgraph[self.target_node_name].batch[
                                                upgrade_batch_neg_samples_idx]
                                        cross_loss = self.cross_contrastive_loss(fraudar_batch_h,
                                                                                 batch_h[batch_pos_samples_idx],
                                                                                 batch_h[upgrade_batch_neg_samples_idx],
                                                                                 batch_pos_samples_idx_batch,
                                                                                 upgrade_batch_neg_samples_idx_batch)
                                    else:
                                        cross_loss = self.cross_contrastive_loss(fraudar_batch_h,
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
                    # Edge extraction error, skip the batch
                    else:
                        loss = torch.tensor(torch.nan).to(self.device)
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
                                    intra_loss.detach().cpu().item(),
                                    inter_loss.detach().cpu().item(),
                                    time.time() - batch_start_time))
                        elif self.task_type in ['cross_subgraph', 'fine_grained_cross_subgraph']:
                            print(
                                "Batch: {}, Loss: {:.6f}, "
                                "Cross Loss: {:.6f}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                                    i + 1,
                                    loss_value,
                                    intra_loss.detach().cpu().item(),
                                    inter_loss.detach().cpu().item(),
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

    def jaccard_metric(self, set_a, set_b):
        intersection = torch.sum(set_a & set_b)
        union = torch.sum(set_a | set_b)
        return intersection / union if union != torch.tensor(0) else torch.tensor(0.0)

    def compute_jaccard(self, y_true, y_pred, idx, flag, batch):
        N = batch.max().detach().cpu().item() + 1
        jaccard_list = []
        for i in range(N):
            subgraph_i_idx = (batch == i).nonzero().squeeze().detach().cpu()
            if flag[i] == 1:
                target_label = y_true[idx[subgraph_i_idx].long()][0].detach().cpu().item()
                y_true_i = idx[subgraph_i_idx].detach().cpu()
                y_pred_i = (y_pred[idx[subgraph_i_idx].long()] == target_label).int().detach().cpu()
                jaccard_list.append(self.jaccard_metric(y_true_i, y_pred_i))
        return torch.tensor(jaccard_list)

    def testing(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = self.train_dict['best_test_f1']
        best_test_roc_auc = 0
        best_test_cm = np.array([[0, 0], [0, 0]])
        for epoch in range(1, self.train_dict["test_n_epochs"] + 1):
            self.classifier.train()
            epoch_loss = []
            epoch_start_time = time.time()
            for i in range(self.train_batch_num):
                self.cls_optimizer.zero_grad()
                if i != self.train_batch_num - 1:
                    batch_subgraph_idx = self.ft_train_idx[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    batch_subgraph_idx = self.ft_train_idx[i * self.batch_size:]
                batch_subgraph = induced_subgraph(batch_subgraph_idx, self.raw_dataset, self.target_node_name)
                batch_subgraph = Batch.from_data_list(batch_subgraph).to(self.device)

                # raw subgraph
                batch_h = self.rgcn_fit(batch_subgraph)
                batch_y = batch_subgraph[self.target_node_name].y.float()

                pred_y = self.classifier(batch_h)
                cls_loss = self.criterion(pred_y, batch_y)
                jaccard_index = self.compute_jaccard(batch_y.argmax(dim=1), pred_y.argmax(dim=1),
                                                     batch_subgraph[self.target_node_name].idx,
                                                     batch_subgraph[self.target_node_name].flag,
                                                     batch_subgraph[self.target_node_name].batch).to(self.device).mean()
                jaccard_penalty_loss = torch.exp(-jaccard_index)
                loss = cls_loss + jaccard_penalty_loss
                loss.backward()
                self.cls_optimizer.step()
                epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - epoch_start_time: .4f} s")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_jaccard, test_cm = self.evaluate_classifier()
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm

    def evaluate_classifier(self):
        self.classifier.eval()
        self.model.eval()
        true_y_list = []
        pred_y_list = []
        prob_y_list = []
        jaccard_list = []
        with torch.no_grad():
            for i in range(self.test_batch_num):
                if i != self.test_batch_num - 1:
                    batch_subgraph_idx = self.ft_test_idx[i * self.batch_size: (i + 1) * self.batch_size]
                else:
                    batch_subgraph_idx = self.ft_test_idx[i * self.batch_size:]
                batch_subgraph = induced_subgraph(batch_subgraph_idx, self.raw_dataset, self.target_node_name)
                batch_subgraph = Batch.from_data_list(batch_subgraph).to(self.device)

                # raw subgraph
                batch_h = self.rgcn_fit(batch_subgraph)
                batch_y = batch_subgraph[self.target_node_name].y.argmax(dim=1).long()

                prob_y = self.classifier(batch_h)
                pred_y = prob_y.argmax(dim=1)

                true_y_list.append(batch_y.detach().cpu())
                pred_y_list.append(prob_y.detach().cpu())
                prob_y_list.append(pred_y.detach().cpu())
                jaccard_list.append(self.compute_jaccard(batch_y, pred_y,
                                                         batch_subgraph[self.target_node_name].idx,
                                                         batch_subgraph[self.target_node_name].flag,
                                                         batch_subgraph[self.target_node_name].batch))
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            jaccard = torch.concat(jaccard_list, dim=0).numpy().mean().item()
        average = "micro"
        multi_class = "ovr"
        acc = accuracy_score(true_y_list, prob_y_list)
        f1 = f1_score(true_y_list, prob_y_list, average=average)
        pre = precision_score(true_y_list, prob_y_list, average=average)
        rec = recall_score(true_y_list, prob_y_list, average=average)
        roc_auc = roc_auc_score(true_y_list, pred_y_list, multi_class=multi_class)
        cm = multilabel_confusion_matrix(true_y_list, prob_y_list)
        print(f"Test ACC: {acc: .4f}, "
              f"Precision: {pre: .4f}, "
              f"Recall: {rec: .4f}, "
              f"F1: {f1: .4f}, "
              f"ROC-AUC: {roc_auc: .4f}, "
              f"Jaccard Index: {jaccard: .4f}, "
              f"Confusion Matrix: {cm.tolist()}")
        return acc, pre, rec, f1, roc_auc, jaccard, cm
