import logging
import os
import time
import random
import torch

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import torch.utils.data as Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN, MaskRGCN
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG


class UinGangsModelPreTrain:

    def __init__(self, args_dict):
        self.train_dict = args_dict
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")
        # Load Pretrained Language Model for Text Embedding
        self.minirbt_model = BertModel.from_pretrained(self.train_dict["minirbt_path"])
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        print(f"minirbt_model params size: {minirbt_model_params_size}")
        self.minirbt_model.to(self.device)
        # Initialize GNN Model for Graph
        self.conv_type = args_dict["conv_type"]
        self.task_type = args_dict["task_type"]
        # self.scaler = GradScaler()
        if self.conv_type == 'RGCN':
            self.edge_types = {('uin', 'ipv6', 'uin'): 0, ('uin', 'wifi', 'uin'): 1, ('uin', 'room', 'uin'): 2,
                               ('uin', 'friend', 'uin'): 3, ('uin', 'idcardid', 'uin'): 4, ('uin', 'device', 'uin'): 5,
                               ('uin', 'payee', 'uin'): 6, ('uin', 'payer', 'uin'): 7, ('uin', 'bankcard', 'uin'): 8,
                               ('uin', 'download_app', 'uin'): 9}
            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=args_dict['num_relations'])
        else:
            self.edge_types = {('uin', 'self_loop', 'uin'): 0, ('uin', 'ipv6', 'uin'): 1, ('uin', 'wifi', 'uin'): 2,
                               ('uin', 'room', 'uin'): 3, ('uin', 'friend', 'uin'): 4, ('uin', 'idcardid', 'uin'): 5,
                               ('uin', 'device', 'uin'): 6, ('uin', 'payee', 'uin'): 7, ('uin', 'payer', 'uin'): 8,
                               ('uin', 'bankcard', 'uin'): 9, ('uin', 'download_app', 'uin'): 10}
            self.model = MaskRGCN(
                input_dim=args_dict['input_dim'],
                hidden_dim=args_dict['hidden_dim'],
                output_dim=args_dict['output_dim'],
                num_relations=args_dict['num_relations'],
                num_bases=args_dict['num_relations'])

        lr = self.train_dict["lr"]
        control_node_num = self.train_dict["filter_node_num"]
        sampling_type = self.train_dict["sampling"]
        self.data_tag = self.train_dict["data_tag"]
        self.train_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["train_data_path"])
        if self.train_dict["sampling"] == 'random':
            self.pos_train_loader = Data.DataLoader(self.train_data,
                                                    batch_size=self.train_dict["batch_size"],
                                                    num_workers=self.train_dict["num_workers"],
                                                    collate_fn=self.train_data.pos_collate_fn_for_random)
        elif self.train_dict["sampling"] == 'fraudar':
            self.pos_train_loader = Data.DataLoader(self.train_data,
                                                    batch_size=self.train_dict["batch_size"],
                                                    num_workers=self.train_dict["num_workers"],
                                                    collate_fn=self.train_data.pos_collate_fn_for_fraudar)
        self.log_file_path = (f"{self.data_tag}{self.conv_type}_sample_{sampling_type}_filter_"
                              f"{control_node_num}_lr_{str(lr)}_{self.train_dict['lr_scheduler']}")
        log_path = os.path.abspath(os.path.join(args_dict['log_dir'],
                                                self.train_dict["model_states_path"].split('/')[-1],
                                                self.log_file_path))
        self.save_model_path = os.path.join(self.train_dict["model_states_path"], self.log_file_path)
        if not self.train_dict["is_debug"]:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            self.writer = SummaryWriter(log_dir=log_path)
            if not os.path.exists(self.save_model_path):
                os.makedirs(self.save_model_path)
        # Retrain Setting
        if self.train_dict["re_train"]:
            epoch_num = self.train_dict["start_epoch"]
            file_name = os.path.join(self.save_model_path,
                                     f"uin_gangs_{self.conv_type}_{self.task_type}_t_"
                                     f"{self.train_dict['temperature']}_model_epoch_{epoch_num}.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            if 'GPU' in file_name:
                rename_key_model_weight = OrderedDict()
                for key in model_weight.keys():
                    key_weight = model_weight[key]
                    key = key.replace('module.', '')
                    rename_key_model_weight[key] = key_weight
                self.model.load_state_dict(rename_key_model_weight)
            else:
                self.model.load_state_dict(model_weight)
            print(f"Load: {file_name}")
            self.start_epoch = epoch_num + 1
            self.end_epoch = self.start_epoch + self.train_dict["n_epochs"]
        else:
            self.start_epoch = 1
            self.end_epoch = self.train_dict["n_epochs"]
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # lr scheduler
        if self.train_dict["lr_scheduler"] == "stepLR":
            self.scheduler = StepLR(self.optimizer,
                                    step_size=self.train_dict["lr_adjust_step"],
                                    gamma=self.train_dict["lr_gamma"],
                                    verbose=True)
        elif self.train_dict["lr_scheduler"] == "reduceLR":
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               patience=self.train_dict["lr_adjust_step"],
                                               verbose=True)
        elif self.train_dict["lr_scheduler"] == "cosineLR":
            self.scheduler = CosineAnnealingLR(self.optimizer,
                                               T_max=self.train_dict["n_epochs"],
                                               eta_min=self.train_dict["lr"] * 1e-2,
                                               verbose=True)
        else:
            self.scheduler = None
        self.setup_seed()
        self.best_loss = self.train_dict["best_loss"]

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def contrastive_loss(self, h1, h2):
        r"""Compute contrastive InfoNCE loss"""
        # hyperparameter: temperature
        t = self.train_dict["temperature"]
        batch_size, _ = h1.size()
        h1_abs = h1.norm(dim=1)
        h2_abs = h2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', h1_abs, h2_abs)
        sim_matrix = torch.exp(sim_matrix / t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) + 1e-4)
        loss = -torch.log(loss).mean()
        return loss

    def preference_contrastive_loss(self, h1, h2, h3, tag):
        t = self.train_dict["temperature"]
        try:
            if len(h1.shape) == 1:
                h1 = h1.unsqueeze(0)
            if len(h3.shape) == 1:
                h3 = h3.unsqueeze(0)
            h1_abs = h1.norm(dim=1)
            h3_abs = h3.norm(dim=1)
            pos_sim = torch.exp(torch.cosine_similarity(h1, h2) / t)
            neg_sim_mat = torch.einsum('ik,jk->ij', h1, h3) / (torch.einsum('i,j->ij', h1_abs, h3_abs) + 1e-4)
            neg_sim = torch.exp(neg_sim_mat / t)
        except Exception as e:
            loss = torch.tensor(torch.nan).to(h1.device)
            print(f"Norm error <{e}>, h1 shape: {h1.shape}, h3 shape: {h3.shape}")
        else:
            _loss = pos_sim / (neg_sim.sum(dim=1) + 1e-4)
            loss = torch.relu(-torch.log(_loss)).mean()
            if loss.isnan():
                print(tag, "h1 and h2 similarity: ", torch.cosine_similarity(h1, h2),
                      "h1 and h2 exp similarity: ", pos_sim, "h1 and h3 similarity: ", neg_sim_mat,
                      "h1 and h3 exp similarity: ", neg_sim, "total vec loss: ", _loss)
                loss = torch.tensor(0, device=self.device)
        return loss

    def node_contrastive_loss(self, h1, h2):
        t = self.train_dict["temperature"]
        if len(h1.shape) == 1:
            h1 = h1.unsqueeze(0)
        if len(h2.shape) == 1:
            h2 = h2.unsqueeze(0)
        h1_abs = h1.norm(dim=1)
        h2_abs = h2.norm(dim=1)
        pos_matrix = torch.einsum('ij,jk->ik', h1, h1.T) / (torch.einsum('i,j->ij', h1_abs, h1_abs) + 1e-4)
        pos_matrix = pos_matrix - torch.eye(pos_matrix.shape[0], device=self.device)
        pos_matrix = torch.exp(pos_matrix / t)
        neg_matrix = torch.einsum('ij,jk->ik', h1, h2.T) / (torch.einsum('i,j->ij', h1_abs, h2_abs) + 1e-4)
        neg_matrix = torch.exp(neg_matrix / t)
        loss = pos_matrix.sum(dim=1) / (pos_matrix.sum(dim=1) + neg_matrix.sum(dim=1) + 1e-4)
        loss = torch.relu(-torch.log(loss)).mean()
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
            min_num = min(i_fraudar_pos_h.shape[0], i_subgraph_neg_h.shape[0])
            if i_fraudar_pos_h.shape[0] > min_num:
                pos_select_idx = [i for i in range(i_fraudar_pos_h.shape[0])]
                random.shuffle(pos_select_idx)
                i_fraudar_pos_h = i_fraudar_pos_h[pos_select_idx[:min_num]]
                i_subgraph_pos_h = i_subgraph_pos_h[pos_select_idx[:min_num]]
            if i_subgraph_neg_h.shape[0] > min_num:
                neg_select_idx = [i for i in range(i_subgraph_neg_h.shape[0])]
                random.shuffle(neg_select_idx)
                i_subgraph_neg_h = i_subgraph_neg_h[neg_select_idx[:min_num]]
            cross_loss = self.preference_contrastive_loss(i_subgraph_pos_h, i_fraudar_pos_h, i_subgraph_neg_h, tag='intra')
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

    def get_edge_info(self, batch):
        edge_index = [batch[edge_type].edge_index for edge_type in list(self.edge_types.keys()) if
                      edge_type in batch.edge_types]
        edge_index = torch.concat(edge_index, dim=1)
        edge_counts = {edge_type: batch[edge_type].num_edges for edge_type in list(self.edge_types.keys()) if
                       edge_type in batch.edge_types}
        edge_type = torch.concat(
            [torch.ones(edge_counts[edge_type]) * edge_idx for edge_type, edge_idx in self.edge_types.items() if
             edge_type in batch.edge_types])
        return edge_index, edge_type.long().to(self.device)

    def reset_batch_node(self, batch):
        batch_item = batch.unique().detach().cpu().tolist()
        batch_map = {batch_item[i]: i for i in range(len(batch_item))}
        upgrade_batch = torch.tensor([batch_map[item] for item in batch.detach().cpu().tolist()], device=self.device)
        return upgrade_batch

    def reset_edge_index_by_map(self, node_map, edge_index):
        reset_edge_index = []
        for col in range(edge_index.shape[1]):
            src_node, dst_node = edge_index[:, col]
            reset_edge_index.append(torch.tensor([[node_map[src_node.item()]],
                                                  [node_map[dst_node.item()]]]))
        reset_edge_index = torch.concat(reset_edge_index, dim=1)
        return reset_edge_index

    def extract_batch_subgraphs(self, batch, subgraph_node_indices=None):
        if subgraph_node_indices is None:
            subgraph_node_indices = (batch['uin'].idx == 1).nonzero().squeeze()
        batch_copy = batch.clone()
        del batch_copy['uin']
        batch_copy['uin'].x = batch['uin'].x[subgraph_node_indices]
        subgraph_node_batch_idxes = batch['uin'].batch[subgraph_node_indices]
        batch_copy['uin'].batch = self.reset_batch_node(subgraph_node_batch_idxes)
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
                print(f"Extract Subgraph Error: <{e}>")
                del batch_copy[edge_type]
                continue
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    # reset the edge index
                    reset_edge_index = self.reset_edge_index_by_map(node_map, edge_index)
                    batch_copy[edge_type].edge_index = reset_edge_index
                else:
                    del batch_copy[edge_type]
        return batch_copy

    def random_sampling_pretraining(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            epoch_loss = []
            epoch_start_time = time.time()
            for (i, batch) in enumerate(self.pos_train_loader):
                start_time = time.time()
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids.to(self.device)
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask.to(self.device)
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                    batch_uin_acs_text_feat = batch_uin_acs_text_feat.to(self.device)
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                batch['uin'].x = batch_x.to(self.device)
                # get edge information
                batch_edge_index, batch_edge_types = self.get_edge_info(batch)
                batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)

                pos_batch = self.extract_batch_subgraphs(batch, (batch['uin'].mask == 0).nonzero().squeeze().detach())
                pos_batch_edge_index, pos_batch_edge_types = self.get_edge_info(pos_batch)
                pos_batch_h = self.model(pos_batch['uin'].x, pos_batch_edge_index, pos_batch_edge_types)
                pos_batch_h_g = scatter_mean(pos_batch_h, pos_batch['uin'].batch, dim=0)
                # subgraph-level contrastive learning
                loss = self.contrastive_loss(batch_h_g, pos_batch_h_g)
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
                loss_value = loss.detach().cpu().item()
                epoch_loss.append(loss_value)
                print(
                    "Batch: {}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                        i + 1,
                        loss_value,
                        time.time() - start_time))
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            if not self.train_dict["is_debug"]:
                self.writer.add_scalar('pretraining_loss', epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                file_name = os.path.join(self.save_model_path, "uin_gangs_RGCN_model_best_loss.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Now best loss: {self.best_loss:.4f}, save model to {file_name}")
        if not self.train_dict["is_debug"]:
            self.writer.close()

    def hetero_fit(self, x_dict, edge_index_dict):
        try:
            out = self.model(x_dict, edge_index_dict)
        except Exception as e:
            print(f"{self.conv_type} Error: <{e}>")
            return None
        else:
            return out

    def relation_fit(self, pos_batch, batch_x):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
        except Exception as e:
            print(f"{self.conv_type} Get Edge Information Error: <{e}>")
            return None
        else:
            return batch_h

    def fraudar_sampling_pretraining(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            self.model.train()
            epoch_loss = []
            epoch_start_time = time.time()
            for (i, pos_batch) in enumerate(self.pos_train_loader):
                start_time = time.time()
                self.optimizer.zero_grad()
                pos_batch = pos_batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = pos_batch['uin'].text_feat_input_ids.to(self.device)
                batch_uin_acs_text_feat_attention_mask = pos_batch['uin'].text_feat_attention_mask.to(self.device)
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                    batch_uin_acs_text_feat = batch_uin_acs_text_feat.to(self.device)
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([pos_batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                pos_batch['uin'].x = batch_x
                if self.conv_type in ['HAN', 'HGT']:
                    try:
                        pos_x_dict = pos_batch.x_dict
                        pos_edge_index_dict = pos_batch.edge_index_dict
                    except Exception as e:
                        print(f"Error: <{e}>; data: {pos_batch}")
                        batch_h = None
                    else:
                        batch_h = self.hetero_fit(pos_x_dict, pos_edge_index_dict)
                else:
                    batch_h = self.relation_fit(pos_batch, pos_batch['uin'].x)
                if batch_h is not None:
                    batch_h_g = scatter_mean(batch_h, pos_batch['uin'].batch, dim=0)
                    pos_batch_idx = (pos_batch['uin'].flag == 1).nonzero().squeeze().detach().cpu().tolist()
                    neg_batch_idx = (pos_batch['uin'].flag == 0).nonzero().squeeze().detach().cpu().tolist()
                    # more than one subgraph
                    if type(pos_batch_idx) is list:
                        list_flag = len(pos_batch_idx) != 0
                    else:
                        list_flag = False
                    # only one subgraph
                    int_flag = type(pos_batch_idx) is int
                    # subgraph generated from fraudar (the index of initial graphs)
                    if list_flag or int_flag:
                        fraudar_batch = self.extract_batch_subgraphs(pos_batch)
                        fraudar_batch = fraudar_batch.to(self.device)
                        if self.conv_type in ['HAN', 'HGT']:
                            try:
                                x_dict = fraudar_batch.x_dict
                                edge_index_dict = fraudar_batch.edge_index_dict
                            except Exception as e:
                                print(f"Error: <{e}>; data: {pos_batch}")
                                fraudar_batch_h = None
                            else:
                                fraudar_batch_h = self.hetero_fit(x_dict, edge_index_dict)
                        else:
                            fraudar_batch_h = self.relation_fit(fraudar_batch, fraudar_batch['uin'].x)
                        if fraudar_batch_h is not None:
                            fraudar_batch_h_g = scatter_mean(fraudar_batch_h, fraudar_batch['uin'].batch, dim=0)
                            # fraudar_batch_h_g = fraudar_batch_h_g[pos_batch_idx]

                            # readout for subgraph
                            pos_batch_h_g = scatter_mean(batch_h, pos_batch['uin'].batch, dim=0)
                            pos_batch_h_g = pos_batch_h_g[pos_batch_idx]

                            if (type(neg_batch_idx) is list and len(neg_batch_idx) != 0) or type(
                                    neg_batch_idx) is int:
                                neg_batch_h_g = batch_h_g[neg_batch_idx]
                            else:
                                neg_batch_h_g = batch_h_g

                            # subgraph-level contrastive learning
                            subgraph_loss = self.preference_contrastive_loss(fraudar_batch_h_g, pos_batch_h_g,
                                                                             neg_batch_h_g, tag='inter')
                        else:
                            subgraph_loss = torch.tensor(torch.nan).to(self.device)
                        if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph',
                                              'cross_subgraph', 'fine_grained_cross_subgraph']:
                            # batch-level contrastive learning (high possibility subgraphs inside)
                            if list_flag:
                                batch_pos_neg_samples_idx = torch.concat(
                                    [(pos_batch['uin'].batch == i).nonzero().squeeze().detach().cpu() for i in
                                     pos_batch_idx], dim=0).tolist()
                            else:
                                batch_pos_neg_samples_idx = (pos_batch[
                                                                 'uin'].batch == pos_batch_idx).nonzero().squeeze().detach().cpu().tolist()
                            pos_samples_idx = (
                                    pos_batch['uin'].idx == 1).nonzero().squeeze().detach().cpu().tolist()
                            batch_pos_samples_idx = list(set(batch_pos_neg_samples_idx) & set(pos_samples_idx))
                            batch_pos_samples_idx_batch = pos_batch['uin'].batch[batch_pos_samples_idx]

                            batch_neg_samples_idx = list(set(batch_pos_neg_samples_idx) - set(pos_samples_idx))
                            batch_neg_samples_idx_batch = pos_batch['uin'].batch[batch_neg_samples_idx]
                            if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph']:
                                if self.task_type == 'fine_grained_batch_subgraph':
                                    batch_anomalous_anchors = self.compute_anomalous_subgraph_anchor(
                                        pos_batch['uin'].x[batch_pos_samples_idx],
                                        batch_pos_samples_idx_batch)
                                    batch_exclude_normal_idx = self.exclude_anomalous_nodes_from_normals(
                                        pos_batch['uin'].x[batch_neg_samples_idx],
                                        batch_anomalous_anchors,
                                        batch_neg_samples_idx_batch)
                                    upgrade_batch_neg_samples_idx = list(
                                        set(batch_neg_samples_idx) - set(batch_exclude_normal_idx))
                                    upgrade_batch_neg_samples_idx_batch = pos_batch['uin'].batch[
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
                                loss = batch_loss + subgraph_loss
                            else:
                                if fraudar_batch_h is not None:
                                    if self.task_type == 'fine_grained_cross_subgraph':
                                        batch_anomalous_anchors = self.compute_anomalous_subgraph_anchor(
                                            pos_batch['uin'].x[batch_pos_samples_idx],
                                            batch_pos_samples_idx_batch)
                                        batch_exclude_normal_idx = self.exclude_anomalous_nodes_from_normals(
                                            pos_batch['uin'].x[batch_neg_samples_idx],
                                            batch_anomalous_anchors,
                                            batch_neg_samples_idx_batch)
                                        upgrade_batch_neg_samples_idx = list(
                                            set(batch_neg_samples_idx) - set(batch_exclude_normal_idx))
                                        upgrade_batch_neg_samples_idx_batch = pos_batch['uin'].batch[
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
                                    loss = cross_loss + subgraph_loss
                                else:
                                    loss = subgraph_loss

                        else:
                            loss = subgraph_loss
                        if not torch.isnan(loss):
                            # self.scaler.scale(loss).backward()
                            # self.scaler.step(self.optimizer)
                            # self.scaler.update()
                            loss.backward()
                            self.optimizer.step()
                            if self.train_dict["lr_scheduler"] == "cosineLR":
                                self.scheduler.step()
                            loss_value = loss.detach().cpu().item()
                            epoch_loss.append(loss_value)
                            if (i + 1) % 5 == 0:
                                if self.task_type in ['batch_subgraph', 'fine_grained_batch_subgraph']:
                                    print(
                                        "Batch: {}, Loss: {:.6f}, "
                                        "Batch Loss: {:.6f}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                                            i + 1,
                                            loss_value,
                                            batch_loss.detach().cpu().item(),
                                            subgraph_loss.detach().cpu().item(),
                                            time.time() - start_time))
                                elif self.task_type in ['cross_subgraph', 'fine_grained_cross_subgraph']:
                                    print(
                                        "Batch: {}, Loss: {:.6f}, "
                                        "Cross Loss: {:.6f}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                                            i + 1,
                                            loss_value,
                                            cross_loss.detach().cpu().item(),
                                            subgraph_loss.detach().cpu().item(),
                                            time.time() - start_time))
                                else:
                                    print(
                                        "Batch: {}, Loss: {:.6f}, "
                                        "Time: {:.4f} s".format(
                                            i + 1,
                                            loss_value,
                                            time.time() - start_time))
                        torch.cuda.empty_cache()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            if not self.train_dict["is_debug"]:
                self.writer.add_scalar(f'{self.conv_type}_pretraining_loss', epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if self.train_dict["lr_scheduler"] == "stepLR":
                self.scheduler.step()
            elif self.train_dict["lr_scheduler"] == "reduceLR":
                self.scheduler.step(epoch_loss)
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                file_name = os.path.join(self.save_model_path,
                                         f"uin_gangs_{self.conv_type}_{self.task_type}_t_"
                                         f"{self.train_dict['temperature']}_model_best_loss.pth")
                torch.save(self.model.state_dict(), file_name)
                epoch_file_name = os.path.join(self.save_model_path,
                                               f"uin_gangs_{self.conv_type}_{self.task_type}_t_"
                                               f"{self.train_dict['temperature']}_model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), epoch_file_name)
                print(f"Now best loss: {self.best_loss:.4f}, save model to {epoch_file_name}")
        if not self.train_dict["is_debug"]:
            self.writer.close()
