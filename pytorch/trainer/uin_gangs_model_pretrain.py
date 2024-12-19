import logging
import os
import time
import random
import torch

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
from torch_geometric.nn import to_hetero
import torch.utils.data as Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel
from mmgog_long_term_sequence_model.utils.utils import visualization_fig_save
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.models.han_model import HAN, SHAN
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


class UinGangsModelPreTrain:

    def __init__(self, args_dict):
        self.train_dict = args_dict
        if torch.cuda.is_available() and self.train_dict["device"] == "gpu":
            print("GPU train available")
            self.device = torch.device("cuda")
        else:
            print("GPU train not available")
            self.device = torch.device("cpu")

        self.conv_type = args_dict["conv_type"]

        if self.conv_type == 'RGCN':

            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=args_dict['num_relations'])
            self.edge_types = {('uin', 'ipv6', 'uin'): 0, ('uin', 'wifi', 'uin'): 1, ('uin', 'room', 'uin'): 2,
                               ('uin', 'friend', 'uin'): 3, ('uin', 'idcardid', 'uin'): 4, ('uin', 'device', 'uin'): 5,
                               ('uin', 'payee', 'uin'): 6, ('uin', 'payer', 'uin'): 7, ('uin', 'bankcard', 'uin'): 8,
                               ('uin', 'download_app', 'uin'): 9}
        elif self.conv_type in ['HAN', 'SHAN']:
            self.metadata = (['uin'], [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                                       ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                                       ('uin', 'payee', 'uin'), ('uin', 'payer', 'uin'), ('uin', 'bankcard', 'uin'),
                                       ('uin', 'download_app', 'uin')])
            if self.conv_type == 'HAN':
                self.model = HAN(in_channels=args_dict['input_dim'],
                                 out_channels=args_dict['output_dim'],
                                 metadata=self.metadata,
                                 heads=args_dict['num_heads'])
            elif self.conv_type == 'SHAN':
                self.model = SHAN(in_channels=args_dict['input_dim'],
                                  out_channels=args_dict['output_dim'],
                                  metadata=self.metadata,
                                  heads=args_dict['num_heads'])

        self.minirbt_model = BertModel.from_pretrained(self.train_dict["minirbt_path"])
        # 冻结文本模型的参数
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        print(f"minirbt_model params size: {minirbt_model_params_size}")
        lr = self.train_dict["lr"]
        control_node_num = self.train_dict["filter_node_num"]
        sampling_type = self.train_dict["sampling"]

        if self.train_dict["is_train"]:
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
            if not self.train_dict["is_debug"]:
                self.log_file_path = f"1921_{self.conv_type}_sample_{sampling_type}_filter_{control_node_num}_lr_{str(lr)}"
                log_path = os.path.join(args_dict['log_dir'], self.train_dict["model_states_path"].split('/')[-1],
                                        self.log_file_path)
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                self.save_model_path = os.path.join(self.train_dict["model_states_path"], self.log_file_path)
                if not os.path.exists(self.save_model_path):
                    os.makedirs(self.save_model_path)
                self.writer = SummaryWriter(log_dir=log_path)
            self.alpha = self.train_dict["pretraining_alpha"]
            self.beta = self.train_dict["pretraining_beta"]
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.save_model_path = os.path.join(self.train_dict["model_states_path"],
                                                f"1921_{self.conv_type}_sample_{sampling_type}_filter_{control_node_num}_lr_{str(lr)}")
            if self.train_dict["evaluate_task"] == 'eval_subgraph_embedding':
                self.pos_train_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["test_data_path"])
                self.pos_train_loader = Data.DataLoader(self.pos_train_data,
                                                        batch_size=self.train_dict["batch_size"],
                                                        num_workers=self.train_dict["num_workers"],
                                                        collate_fn=self.pos_train_data.pos_collate_fn_for_fraudar)
            elif self.train_dict["evaluate_task"] == 'eval_labelled_subgraph_embedding':
                self.eval_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["eval_data_path"])
                self.eval_loader = Data.DataLoader(self.eval_data,
                                                   batch_size=self.train_dict["batch_size"],
                                                   num_workers=self.train_dict["num_workers"],
                                                   collate_fn=self.eval_data.collate_fn)
            elif self.train_dict["evaluate_task"] == 'eval_node_embedding':
                self.test_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["test_data_path"])
                self.test_loader = Data.DataLoader(self.test_data,
                                                   batch_size=self.train_dict["batch_size"],
                                                   num_workers=self.train_dict["num_workers"],
                                                   collate_fn=self.test_data.collate_fn)
            elif self.train_dict["evaluate_task"] == 'predict':
                self.train_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["train_data_path"])
                self.eval_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["test_data_path"])
                self.train_loader = Data.DataLoader(self.train_data,
                                                    batch_size=self.train_dict["batch_size"],
                                                    num_workers=self.train_dict["num_workers"],
                                                    collate_fn=self.train_data.collate_fn)
                self.eval_loader = Data.DataLoader(self.eval_data,
                                                   batch_size=self.train_dict["batch_size"],
                                                   num_workers=self.train_dict["num_workers"],
                                                   collate_fn=self.eval_data.collate_fn)
                self.classifier = torch.nn.Sequential(torch.nn.Linear(args_dict['output_dim'], args_dict['hidden_dim']),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                      torch.nn.Softmax(dim=1))
                self.criterion = torch.nn.CrossEntropyLoss()
                self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
            elif self.train_dict["evaluate_task"] == 'subgraph_prompt_tuning':
                self.initial_data = UinGangsDataIterablePyG(self.train_dict,
                                                            self.train_dict["prompt_initial_data_path"])
                self.tune_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["prompt_tuning_data_path"])
                self.eval_data = UinGangsDataIterablePyG(self.train_dict,
                                                         self.train_dict["prompt_evaluating_data_path"])
                self.initial_loader = Data.DataLoader(self.initial_data,
                                                      batch_size=20,
                                                      num_workers=2,
                                                      collate_fn=self.initial_data.collate_fn)
                self.tune_loader = Data.DataLoader(self.tune_data,
                                                   batch_size=20,
                                                   num_workers=2,
                                                   collate_fn=self.tune_data.collate_fn)
                self.eval_loader = Data.DataLoader(self.eval_data,
                                                   batch_size=self.train_dict["batch_size"],
                                                   num_workers=self.train_dict["num_workers"],
                                                   collate_fn=self.eval_data.collate_fn)
                self.is_prompt = self.train_dict["is_prompt_tuning"]
                if self.is_prompt:
                    cls_input = args_dict['output_dim'] * 2
                else:
                    cls_input = args_dict['output_dim']
                self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                      torch.nn.Softmax(dim=1))
                self.criterion = torch.nn.CrossEntropyLoss()
                self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def cosine_similarity(self, h1, h2):
        h1_abs = h1.norm(dim=1)
        h2_abs = h2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', h1_abs, h2_abs)
        return sim_matrix

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
            loss = torch.tensor(5.0, requires_grad=True).to(h1.device)
            print(f"Norm error <{e}>, h1 shape: {h1.shape}, h3 shape: {h3.shape}")
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) + 1e-4)
            loss = -torch.log(loss).mean()
        return loss

    def node_level_contrastive_loss(self, high_malicious_h, low_malicious_h, high_mask, low_mask):
        loop_idx = sorted(set(high_mask.tolist()))
        batch_node_loss = torch.tensor(0.0, requires_grad=True).to(high_malicious_h.device)
        t = self.train_dict["temperature"]
        for i in loop_idx:
            high_h = high_malicious_h[high_mask == i]
            low_h = low_malicious_h[low_mask == i]
            # low_h 采样
            indices = list(range(low_h.shape[0]))
            random.shuffle(indices)
            indices = indices[:high_h.shape[0]]
            low_h = low_h[indices]
            high_h_norm = high_h.norm(dim=1)
            low_h_norm = low_h.norm(dim=1)
            pos_sim = torch.einsum('ik,jk->ij', high_h, high_h) / torch.einsum('i,j->ij', high_h_norm, high_h_norm)
            pos_loss = torch.exp(pos_sim / t)
            neg_sim = torch.einsum('ik,jk->ij', high_h, low_h) / torch.einsum('i,j->ij', high_h_norm, low_h_norm)
            neg_loss = torch.exp(neg_sim / t)
            node_loss = -torch.log(pos_loss.sum(dim=1) / (pos_loss.sum(dim=1) + neg_loss.sum(dim=1) + 1e-4)).mean()
            batch_node_loss = batch_node_loss + node_loss
        return batch_node_loss / len(loop_idx)

    def get_edge_info(self, batch):
        edge_index = [batch[edge_type].edge_index for edge_type in list(self.edge_types.keys()) if edge_type in batch.edge_types]
        edge_index = torch.concat(edge_index, dim=1)
        edge_counts = {edge_type: batch[edge_type].num_edges for edge_type in list(self.edge_types.keys()) if edge_type in batch.edge_types}
        edge_type = torch.concat([torch.ones(edge_counts[edge_type]) * edge_idx for edge_type, edge_idx in self.edge_types.items() if edge_type in batch.edge_types])
        return edge_index, edge_type.long()

    def extract_batch_subgraphs(self, batch, subgraph_node_indices=None):
        if subgraph_node_indices is None:
            node_indices = batch['uin'].idx
            subgraph_node_indices = (node_indices == 1).nonzero().squeeze()
        x = torch.zeros_like(batch['uin'].x).to(batch['uin'].x.device)
        x[subgraph_node_indices] = batch['uin'].x[subgraph_node_indices].clone()
        new_batch = batch.clone()
        new_batch['uin'].x = x
        subgraph_max_node_idx = subgraph_node_indices.max()
        for edge_type in batch.edge_types:
            try:
                current_max_node_idx = batch[edge_type].edge_index.max()
                max_node_idx = min(subgraph_max_node_idx, current_max_node_idx)
                if max_node_idx < subgraph_max_node_idx:
                    subgraph_node_indices = subgraph_node_indices[subgraph_node_indices <= max_node_idx]
                edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index)
            except Exception as e:
                print(f"Extract subgraph error: <{e}>, "
                      f"edge type: {edge_type}, "
                      f"subgraph: {subgraph_node_indices}")
                del new_batch[edge_type]
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    new_batch[edge_type].edge_index = edge_index
                else:
                    del new_batch[edge_type]
        return new_batch

    def random_sampling_pretraining(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        start_epoch = 1
        end_epoch = self.train_dict["n_epochs"] + 1
        if self.train_dict["re_train"]:
            epoch_num = self.train_dict["start_epoch"]
            model_weight = torch.load(self.train_dict["model_states_path"] + f"epoch_{str(epoch_num)}.pth",
                                      map_location=self.device)
            self.model.load_state_dict(model_weight)
            start_epoch = epoch_num + 1
            end_epoch = start_epoch + self.train_dict["n_epochs"]
        best_loss = self.train_dict["best_loss"]
        for epoch in range(start_epoch, end_epoch):
            self.model.train()
            loss_sum = 0
            epoch_start_time = time.time()
            batch_num = 0
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
                loss_sum += loss_value
                batch_num += 1
                print(
                    "Batch: {}, Subgraph Loss: {:.6f}, Time: {:.4f} s".format(
                        i + 1,
                        loss_value,
                        time.time() - start_time))
            epoch_loss = loss_sum / batch_num
            if not self.train_dict["is_debug"]:
                self.writer.add_scalar('pretraining_loss', epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                file_name = os.path.join(self.save_model_path, "uin_gangs_RGCN_model_best_loss.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            if epoch % 10 == 0:
                file_name = os.path.join(self.save_model_path, f"uin_gangs_RGCN_model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Save model to {file_name}")
        if not self.train_dict["is_debug"]:
            self.writer.close()

    def han_fit(self, x_dict, edge_index_dict):
        out = self.model(x_dict, edge_index_dict)
        return out

    def shan_fit(self, x_dict, edge_index_dict, score_dict):
        out = self.model(x_dict, edge_index_dict, score_dict)
        return out

    def rgcn_fit(self, pos_batch, batch_x):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
        except Exception as e:
            print(f"Get edge information error: <{e}>, "
                  f"x max idx: {batch_x.shape[0] - 1}, "
                  f"batch information: {pos_batch}")
            return None
        else:
            return batch_h

    def pretraining(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        start_epoch = 3
        end_epoch = self.train_dict["n_epochs"] + 1
        if self.train_dict["re_train"]:
            epoch_num = self.train_dict["start_epoch"]
            if epoch_num == 0:
                file_name = os.path.join(self.save_model_path,
                                         f"uin_gangs_{self.conv_type}_model_best_loss.pth")
            else:
                file_name = os.path.join(self.save_model_path,
                                         f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
                start_epoch = epoch_num + 1
                end_epoch = start_epoch + self.train_dict["n_epochs"]
            model_weight = torch.load(file_name, map_location=self.device)
            self.model.load_state_dict(model_weight)
        # num_fraudar_nodes = 0
        # num_fraudar_edges = 0
        # num_fraudar_graphs = 0
        #
        # num_low_malicious_nodes = 0
        # num_low_malicious_edges = 0
        # num_low_malicious_graphs = 0

        best_loss = self.train_dict["best_loss"]
        for epoch in range(start_epoch + 1, end_epoch):
            self.model.train()
            loss_sum = 0
            epoch_start_time = time.time()
            batch_num = 0
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
                pos_batch['uin'].x = batch_x.to(self.device)
                if self.conv_type == 'HAN':
                    batch_h = self.han_fit(pos_batch.x_dict, pos_batch.edge_index_dict)
                elif self.conv_type == 'SHAN':
                    batch_h = self.shan_fit(pos_batch.x_dict, pos_batch.edge_index_dict,
                                            pos_batch.score_dict)
                elif self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(pos_batch, batch_x)
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
                    # count average size of positive samples generated by fraudar
                    # if epoch == 1:
                    #     try:
                    #         num_fraudar_nodes += (fraudar_batch['uin'].idx == 1).nonzero().squeeze().shape[0]
                    #         num_fraudar_edges += fraudar_batch.num_edges
                    #         num_fraudar_graphs += len(pos_batch_idx) if list_flag else pos_batch_idx
                    #     except Exception as e:
                    #         print(f"Error: <{e}>")
                    #         print((fraudar_batch['uin'].idx == 1).nonzero().squeeze().shape)
                    #         print(fraudar_batch.num_edges)
                    #         print(len(pos_batch_idx) if list_flag else pos_batch_idx)

                    if self.conv_type == 'HAN':
                        try:
                            edge_index_dict = fraudar_batch.edge_index_dict
                        except Exception as e:
                            print(f"Error: {e}>")
                            fraudar_batch_h = None
                        else:
                            fraudar_batch_h = self.han_fit(fraudar_batch.x_dict, edge_index_dict)
                    elif self.conv_type == 'SHAN':
                        try:
                            edge_index_dict = fraudar_batch.edge_index_dict
                        except Exception as e:
                            print(f"Error: {e}>")
                            fraudar_batch_h = None
                        else:
                            fraudar_batch_h = self.shan_fit(fraudar_batch.x_dict, edge_index_dict,
                                                            fraudar_batch.score_dict)
                    elif self.conv_type == 'RGCN':
                        fraudar_batch_h = self.rgcn_fit(fraudar_batch, fraudar_batch['uin'].x)

                    if fraudar_batch_h is not None:
                        fraudar_batch_h_g = scatter_mean(fraudar_batch_h, fraudar_batch['uin'].batch, dim=0)
                        fraudar_batch_h_g = fraudar_batch_h_g[pos_batch_idx]

                        # pos_batch_node_idx = (pos_batch['uin'].idx == 1).nonzero().squeeze().detach().cpu().tolist()
                        # pos_batch_h = batch_h[pos_batch_node_idx]
                        # pos_batch_h_mask = pos_batch['uin'].batch[pos_batch_node_idx]

                        # readout for subgraph
                        pos_batch_h_g = scatter_mean(batch_h, pos_batch['uin'].batch, dim=0)
                        # pos_batch_h_g = scatter_mean(pos_batch_h, pos_batch_h_mask, dim=0)
                        pos_batch_h_g = pos_batch_h_g[pos_batch_idx]

                        if (type(neg_batch_idx) is list and len(neg_batch_idx) != 0) or type(neg_batch_idx) is int:
                            neg_batch_h_g = batch_h_g[neg_batch_idx]
                        # if epoch == 1:
                        #     node_idx_list = []
                        #     for idx in neg_batch_idx:
                        #         node_idx = (pos_batch['uin'].batch == idx).nonzero().squeeze().detach().cpu().tolist()
                        #         node_idx_list.extend(node_idx)
                        #     num_low_malicious_nodes += len(node_idx_list)
                        #     normal_batch = self.extract_batch_subgraphs(pos_batch,
                        #                                                 torch.tensor(node_idx_list).to(self.device))
                        #     num_low_malicious_edges += normal_batch.num_edges
                        #     num_low_malicious_graphs += len(neg_batch_idx)
                        else:
                            neg_batch_h_g = batch_h_g
                        # if epoch == 1:
                        #     num_low_malicious_nodes += pos_batch.num_nodes
                        #     num_low_malicious_edges += pos_batch.num_edges
                        #     num_low_malicious_graphs += pos_batch.num_graphs

                        # subgraph-level contrastive learning
                        loss = self.preference_contrastive_loss(fraudar_batch_h_g, pos_batch_h_g, neg_batch_h_g)
                    else:
                        loss = torch.tensor(torch.nan).to(self.device)
                    # node-level contrastive learning
                    # fraudar_batch_idx = (pos_batch['uin'].idx == 1).nonzero().squeeze()
                    # fraudar_batch_mask = pos_batch['uin'].batch[fraudar_batch_idx]
                    # mask_fraudar_batch_x = pos_batch['uin'].x.clone().requires_grad_(False)
                    # mask掉被认为是异常的节点，即节点属性被置为全0，边的信息也需要mask
                    # mask_fraudar_batch_x[fraudar_batch_idx] = 0.0
                    # mask_fraudar_batch_x.requires_grad_(True)
                    # exclude_batch_idx = (pos_batch['uin'].idx == 0).nonzero().squeeze()
                    # 同时边也需要mask掉边
                    # mask_fraudar_batch = self.extract_batch_subgraphs(pos_batch, exclude_batch_idx)
                    # mask_fraudar_batch_edge_index, mask_fraudar_batch_edge_types = self.get_edge_info(
                    #     mask_fraudar_batch)
                    # mask_fraudar_batch_h = self.model(mask_fraudar_batch_x, mask_fraudar_batch_edge_index,
                    #                                   mask_fraudar_batch_edge_types)
                    # exclude_batch_h = mask_fraudar_batch_h[exclude_batch_idx]
                    # exclude_batch_mask = pos_batch['uin'].batch[exclude_batch_idx]
                    # fraudar_batch_h = fraudar_batch_h[fraudar_batch_idx]
                    # compute node-level loss
                    # node_loss = self.node_level_contrastive_loss(fraudar_batch_h, exclude_batch_h,
                    #                                              fraudar_batch_mask, exclude_batch_mask)
                    # loss = self.alpha * subgraph_loss + self.beta * node_loss
                    if not torch.isnan(loss):
                        loss.backward()
                        self.optimizer.step()
                        loss_value = loss.detach().cpu().item()
                        loss_sum += loss_value
                        batch_num += 1
                        print(
                            "Batch: {}, Loss: {:.6f}, Time: {:.4f} s".format(
                                i + 1,
                                loss_value,
                                time.time() - start_time))

                    torch.cuda.empty_cache()

            epoch_loss = loss_sum / batch_num
            if not self.train_dict["is_debug"]:
                self.writer.add_scalar(f'{self.conv_type}_pretraining_loss', epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            if epoch % 5 == 0:
                file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_epoch_{epoch}.pth")
                torch.save(self.model.state_dict(), file_name)
                print(f"Save model to {file_name}")
            # if epoch == 1:
            #     print(
            #         "Average fraudar graph number={:.4f}, fraudar graph size node={:.4f}, edge={:.4f}.\n"
            #         "Average highly normal graph number={:.4f}, normal graph size node={:.4f}, edge={:.4f}.".format(
            #             num_fraudar_graphs, num_fraudar_nodes / num_fraudar_graphs,
            #                                 num_fraudar_edges / num_fraudar_graphs,
            #             num_low_malicious_graphs, num_low_malicious_nodes / num_low_malicious_graphs,
            #                                 num_low_malicious_edges / num_low_malicious_graphs
            #         ))
        if not self.train_dict["is_debug"]:
            self.writer.close()

    def evaluate_node_embedding(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        if self.train_dict["eval_epoch"] != 0:
            epoch_num = self.train_dict["eval_epoch"]
            file_name = os.path.join(self.save_model_path, f"uin_gangs_RGCN_model_epoch_{epoch_num}.pth")
            fig_name = f"filter_subgraph_pca_epoch_{str(epoch_num)}.png"
        else:
            file_name = os.path.join(self.save_model_path, f"uin_gangs_RGCN_model_best_loss.pth")
            fig_name = f"filter_subgraph_pca_best_epoch.png"
        model_weight = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(model_weight)
        self.model.eval()
        normal_h = []
        anomaly_h = []
        for i, batch in enumerate(self.test_loader):
            batch = batch.to(self.device)
            batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
            batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
            with torch.no_grad():
                batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                             batch_uin_acs_text_feat_attention_mask).pooler_output
            # combine numerical, categorical and text attributes
            batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
            batch_x = torch.nn.functional.normalize(batch_x, dim=1)
            batch['uin'].x = batch_x
            # get edge information
            batch_edge_index, batch_edge_types = self.get_edge_info(batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
            root_node_idx = batch['uin'].batch[:-1].detach().cpu().tolist()
            normal_node_idx = (batch['uin'].y < 2).nonzero().squeeze().detach().cpu().tolist()
            anomaly_node_idx = (batch['uin'].y >= 2).nonzero().squeeze().detach().cpu().tolist()
            root_node_h = batch_h[root_node_idx]
            normal_node_h = root_node_h[normal_node_idx].detach().cpu()
            anomaly_node_h = root_node_h[anomaly_node_idx].detach().cpu()
            if len(normal_node_h.shape) == 1:
                normal_node_h = normal_node_h.unsqueeze(0)
            if len(anomaly_node_h.shape) == 1:
                anomaly_node_h = anomaly_node_h.unsqueeze(0)
            normal_h.append(normal_node_h)
            anomaly_h.append(anomaly_node_h)
            torch.cuda.empty_cache()
        normal_h = torch.concat(normal_h, dim=0)
        anomaly_h = torch.concat(anomaly_h, dim=0)
        all_h = torch.concat([normal_h, anomaly_h], dim=0).numpy()
        all_class = torch.concat([torch.ones(anomaly_h.shape[0]), torch.zeros(normal_h.shape[0])], dim=0).numpy()
        visualization_fig_save(input_embedding=all_h.numpy(), input_class=all_class.numpy(), data_type='Node Type',
                               save_path=os.path.join(self.train_dict["pic_path"], fig_name))

    def evaluate_labelled_subgraph_embedding(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        if self.train_dict["eval_epoch"] != 0:
            epoch_num = self.train_dict["eval_epoch"]
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
            fig_name = f"filter_retrain_{self.conv_type}_{self.train_dict['sampling']}_subgraph_pca_epoch_{str(epoch_num)}.png"
        else:
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
            fig_name = f"filter_retrain_{self.conv_type}_{self.train_dict['sampling']}_subgraph_pca_best_epoch.png"
        model_weight = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(model_weight)
        self.model.eval()
        all_h = []
        all_class = []
        positive_sim = []
        negative_sim = []
        positive_negative_sim = []
        for (i, batch) in enumerate(self.eval_loader):
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
            if self.conv_type == 'HAN':
                batch_h = self.han_fit(batch.x_dict, batch.edge_index_dict)
            elif self.conv_type == 'RGCN':
                batch_h = self.rgcn_fit(batch, batch['uin'].x)
            # readout for subgraph
            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
            normal_idx = (batch['uin'].gang_label == 0).nonzero().squeeze().detach().cpu().tolist()
            gang_idx = (batch['uin'].gang_label == 1).nonzero().squeeze().detach().cpu().tolist()
            positive_sim.extend(self.cosine_similarity(batch_h_g[gang_idx, :], batch_h_g[gang_idx, :]).mean(
                dim=1).detach().cpu().tolist())
            negative_sim.extend(self.cosine_similarity(batch_h_g[normal_idx, :], batch_h_g[normal_idx, :]).mean(
                dim=1).detach().cpu().tolist())
            positive_negative_sim.extend(self.cosine_similarity(batch_h_g[gang_idx, :], batch_h_g[normal_idx, :]).mean(
                dim=1).detach().cpu().tolist())
            all_h.append(batch_h_g.detach().cpu())
            all_class.append(batch['uin'].gang_label.detach().cpu())
            torch.cuda.empty_cache()
        pos_sim = np.mean(positive_sim)
        neg_sim = np.mean(negative_sim)
        pos_neg_sim = np.mean(positive_negative_sim)
        print(f"Positive similarity: {pos_sim:.4f}, "
              f"negative similarity: {neg_sim:.4f}, "
              f"positive negative similarity: {pos_neg_sim:.4f}")

        all_h = torch.concat(all_h, dim=0)
        all_class = torch.concat(all_class, dim=0)
        visualization_fig_save(input_embedding=all_h.numpy(), input_class=all_class.numpy(),
                               save_path=os.path.join(self.train_dict["pic_path"], fig_name))

    def evaluate_subgraph_embedding(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        if self.train_dict["eval_epoch"] != 0:
            epoch_num = self.train_dict["eval_epoch"]
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
            fig_name = f"filter_retrain_{self.conv_type}_subgraph_pca_epoch_{str(epoch_num)}.png"
        else:
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
            fig_name = f"filter_retrain_{self.conv_type}_subgraph_pca_best_epoch.png"
        model_weight = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(model_weight)
        self.model.eval()
        low_h = []
        high_h = []
        for (i, pos_batch) in enumerate(self.pos_train_loader):
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
            pos_batch['uin'].x = batch_x.to(self.device)
            # get edge information
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
            # readout for subgraph
            batch_h_g = scatter_mean(batch_h, pos_batch['uin'].batch, dim=0)
            pos_batch_idx = (pos_batch['uin'].flag == 1).nonzero().squeeze().detach().cpu().tolist()
            if (type(pos_batch_idx) is list and len(pos_batch_idx) != 0) or type(pos_batch_idx) is int:
                high_h_g = batch_h_g[pos_batch_idx]
                if len(high_h_g.shape) == 1:
                    high_h_g = high_h_g.unsqueeze(0)
                high_h.append(high_h_g.detach().cpu())
            neg_batch_idx = (pos_batch['uin'].flag == 0).nonzero().squeeze().detach().cpu().tolist()
            if (type(neg_batch_idx) is list and len(neg_batch_idx) != 0) or type(neg_batch_idx) is int:
                low_h_g = batch_h_g[neg_batch_idx]
                if len(low_h_g.shape) == 1:
                    low_h_g = low_h_g.unsqueeze(0)
                low_h.append(low_h_g.detach().cpu())
            torch.cuda.empty_cache()
        high_h = torch.concat(high_h, dim=0)
        low_h = torch.concat(low_h, dim=0)
        all_h = torch.concat([high_h, low_h], dim=0)
        all_class = torch.concat([torch.ones(high_h.shape[0]), torch.zeros(low_h.shape[0])], dim=0).numpy()
        visualization_fig_save(input_embedding=all_h.numpy(), input_class=all_class.numpy(),
                               save_path=os.path.join(self.train_dict["pic_path"], fig_name))

    def evaluate_subgraph_predict(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        if not self.train_dict["is_supervised"]:
            if self.train_dict["eval_epoch"] != 0:
                epoch_num = self.train_dict["eval_epoch"]
                file_name = os.path.join(self.save_model_path,
                                         f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
            else:
                file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            self.model.load_state_dict(model_weight)
            print(f"Load: {file_name}")
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.classifier.to(self.device)
        best_loss = self.train_dict["best_loss"]
        for epoch in range(1, self.train_dict["n_epochs"] + 1):
            st = time.time()
            self.classifier.train()
            epoch_loss = []
            for i, batch in enumerate(self.train_loader):
                self.cls_optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                batch['uin'].x = batch_x
                if self.conv_type == 'HAN':
                    batch_h = self.han_fit(batch.x_dict, batch.edge_index_dict)
                elif self.conv_type == 'SHAN':
                    batch_h = self.shan_fit(batch.x_dict, batch.edge_index_dict,
                                            batch.score_dict)
                elif self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                # get edge information
                if batch_h is not None:
                    batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                    batch_y = batch['uin'].gang_label.float()
                    pred_y = self.classifier(batch_h_g).argmax(dim=1).float().to(self.device)
                    cls_loss = self.criterion(pred_y, batch_y).requires_grad_(True)
                    cls_loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(cls_loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                file_name = self.train_dict["cls_model_states_path"] + f"{self.conv_type}_best_loss.pth"
                torch.save(self.classifier.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            self.evaluate_classifier()
            print(f"Epoch {epoch}, Cross entropy loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")

    def evaluate_classifier(self):
        self.classifier.eval()
        true_y_list = []
        pred_y_list = []
        prob_y_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                batch['uin'].x = batch_x
                if self.conv_type == 'HAN':
                    batch_h = self.han_fit(batch.x_dict, batch.edge_index_dict)
                elif self.conv_type == 'SHAN':
                    batch_h = self.shan_fit(batch.x_dict, batch.edge_index_dict,
                                            batch.score_dict)
                elif self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                if batch_h is not None:
                    batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                    batch_y = batch['uin'].gang_label
                    prob_y = self.classifier(batch_h_g)[:, 1]
                    pred_y = self.classifier(batch_h_g).argmax(dim=1)
                    true_y = batch_y.detach().cpu()
                    prob_y = prob_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    true_y_list.append(true_y)
                    pred_y_list.append(pred_y)
                    prob_y_list.append(prob_y)
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            acc = accuracy_score(true_y_list, pred_y_list)
            f1 = f1_score(true_y_list, pred_y_list)
            pre = precision_score(true_y_list, pred_y_list)
            rec = recall_score(true_y_list, pred_y_list)
            roc_auc = roc_auc_score(true_y_list, prob_y_list)
            cm = confusion_matrix(true_y_list, pred_y_list)
            print(f"Test ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}"
                  )

    def subgraph_prompt_tuning(self):
        self.setup_seed()
        self.model.to(self.device)
        self.minirbt_model.to(self.device)
        if self.train_dict["eval_epoch"] != 0:
            epoch_num = self.train_dict["eval_epoch"]
            file_name = os.path.join(self.save_model_path,
                                     f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
        else:
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
        model_weight = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(model_weight)
        print(f"Load: {file_name}")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        if self.is_prompt:
            gang_subgraphs = []
            for i, batch in enumerate(self.initial_loader):
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch['uin'].x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                gang_mems = (batch['uin'].gang_mem == 1).nonzero().squeeze().detach()
                gang_batch = self.extract_batch_subgraphs(batch, subgraph_node_indices=gang_mems)
                batch_h = self.rgcn_fit(gang_batch, torch.nn.functional.normalize(gang_batch['uin'].x, dim=1))
                try:
                    gang_h = batch_h[gang_mems]
                except Exception as e:
                    print(f"Prompt Initialized Error : <{e}>")
                    continue
                else:
                    batch_batch = batch['uin'].batch[gang_mems]
                    gang_subgraphs.append(scatter_mean(gang_h, batch_batch, dim=0))
            gang_subgraph_mean = torch.concat(gang_subgraphs, dim=0).mean(dim=0)
            subgraph_prompt = torch.nn.Parameter(gang_subgraph_mean)
        best_loss = self.train_dict["best_loss"]
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.classifier.to(self.device)
        for epoch in range(1, self.train_dict["n_epochs"] + 1):
            epoch_loss = []
            st = time.time()
            if self.is_prompt:
                subgraph_prompt.requires_grad = True
            for i, batch in enumerate(self.tune_loader):
                self.cls_optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                batch['uin'].x = batch_x
                batch_h = self.rgcn_fit(batch, batch['uin'].x)
                # get edge information
                if batch_h is not None:
                    if self.is_prompt:
                        prompted_batch_h = torch.concat([batch_h, subgraph_prompt.repeat(batch_h.shape[0], 1)], dim=1)
                        pred_y = self.classifier(prompted_batch_h).argmax(dim=1).float().to(self.device)
                    else:
                        pred_y = self.classifier(batch_h).argmax(dim=1).float().to(self.device)
                    batch_y = batch['uin'].gang_mem.float()
                    cls_loss = self.criterion(pred_y, batch_y).requires_grad_(True)
                    cls_loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(cls_loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                file_name = f"prompt_{self.conv_type}_best_loss.pth" if self.is_prompt else f"no_prompt_{self.conv_type}_best_loss.pth"
                file_name = self.train_dict["cls_model_states_path"] + file_name
                torch.save(self.classifier.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            if self.is_prompt:
                self.evaluate_prompt_classifier(prompt=subgraph_prompt)
            else:
                self.evaluate_prompt_classifier()
            print(f"Epoch {epoch}, Cross entropy loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")

    def jaccard(self, set_a, set_b):
        intersection = torch.sum(set_a & set_b)
        union = torch.sum(set_a | set_b)
        return intersection / union if union != 0 else 0.0

    def jaccard_batch(self, y_true, y_pred, batch):
        N = batch.max().detach().cpu().item() + 1
        jaccard_list = []
        for i in range(N):
            subgraph_i_idx = (batch == i).nonzero().squeeze().detach().cpu()
            y_true_i_idx = y_true[subgraph_i_idx]
            y_pred_i_idx = y_pred[subgraph_i_idx]
            jac = self.jaccard(y_true_i_idx, y_pred_i_idx)
            jaccard_list.append(jac)
        return jaccard_list

    def evaluate_prompt_classifier(self, prompt=None):
        self.classifier.eval()
        if prompt is not None:
            prompt.requires_grad = False
        true_y_list = []
        pred_y_list = []
        prob_y_list = []
        jaccard_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                batch['uin'].x = batch_x
                batch_h = self.rgcn_fit(batch, batch['uin'].x)
                if batch_h is not None:
                    if prompt is not None:
                        prompt_batch_h = torch.concat([batch_h, prompt.repeat(batch_h.shape[0], 1)], dim=1)
                        prob_y = self.classifier(prompt_batch_h)[:, 1]
                        pred_y = self.classifier(prompt_batch_h).argmax(dim=1)
                    else:
                        prob_y = self.classifier(batch_h)[:, 1]
                        pred_y = self.classifier(batch_h).argmax(dim=1)
                    batch_y = batch['uin'].gang_mem
                    true_y = batch_y.detach().cpu()
                    prob_y = prob_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    true_y_list.append(true_y)
                    pred_y_list.append(pred_y)
                    prob_y_list.append(prob_y)
                    jaccard_list.extend(self.jaccard_batch(true_y.int(), pred_y, batch['uin'].batch))
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            acc = accuracy_score(true_y_list, pred_y_list)
            f1 = f1_score(true_y_list, pred_y_list)
            pre = precision_score(true_y_list, pred_y_list)
            rec = recall_score(true_y_list, pred_y_list)
            roc_auc = roc_auc_score(true_y_list, prob_y_list)
            cm = confusion_matrix(true_y_list, pred_y_list)
            jaccard = torch.tensor(jaccard_list).mean().detach().cpu().item()
            print(f"Test ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Jaccard Coefficient: {jaccard: .4f}"
                  )
