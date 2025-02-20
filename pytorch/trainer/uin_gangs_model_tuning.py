import logging
import os
import time
import random

import pandas as pd
import torch

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
import networkx as nx
from collections import OrderedDict
import torch.utils.data as Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from transformers import BertModel
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import HeteroData
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN, AttnRGCN
from mmgog_long_term_sequence_model.pytorch.models.han_model import HAN
from mmgog_long_term_sequence_model.pytorch.models.graph_transformer import GraphTransformer, HeteroGraphTransformer
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from mmgog_long_term_sequence_model.utils.utils import batch_subgraph_loss_based_cross_entropy, \
    batch_dense_loss_based_cross_entropy, batch_connect_loss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


class UinGangsModelTuning:
    def __init__(self, args_dict):
        self.eval_dict = args_dict
        if torch.cuda.is_available() and self.eval_dict["device"] == "gpu":
            print("GPU is available")
            self.device = torch.device("cuda")
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")
        # Load Pretrained Language Model for Text Embedding
        self.minirbt_model = BertModel.from_pretrained(self.eval_dict["minirbt_path"])
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        print(f"minirbt_model params size: {minirbt_model_params_size}")
        self.minirbt_model.to(self.device)

        self.conv_type = args_dict["conv_type"]
        self.device_tag = self.eval_dict["device_tag"]
        if self.conv_type in ['RGCN', 'AttnRGCN']:
            if self.device_tag == '_GPU3' and self.conv_type == 'RGCN':
                self.model = RGCN(input_dim=args_dict['input_dim'],
                                  hidden_dim=args_dict['hidden_dim'],
                                  output_dim=args_dict['output_dim'],
                                  num_relations=args_dict['num_relations'],
                                  num_bases=args_dict['num_relations'])
            elif self.conv_type == 'AttnRGCN':
                self.attn_weight = torch.nn.Parameter(
                    torch.sigmoid(torch.Tensor([0.6, 0.6, 0.3, 0.5, 1.3, 1.4, 0.4, 0.5, 1.5, 0.8])))
                self.model = AttnRGCN(input_dim=args_dict['input_dim'],
                                      hidden_dim=args_dict['hidden_dim'],
                                      output_dim=args_dict['output_dim'],
                                      attn_weight=self.attn_weight,
                                      num_relations=args_dict['num_relations'],
                                      num_bases=args_dict['num_relations'])
            else:
                self.model = RGCN(input_dim=args_dict['input_dim'],
                                  hidden_dim=args_dict['hidden_dim'],
                                  output_dim=args_dict['output_dim'],
                                  num_relations=args_dict['num_relations'])
            self.edge_types = {('uin', 'ipv6', 'uin'): 0, ('uin', 'wifi', 'uin'): 1, ('uin', 'room', 'uin'): 2,
                               ('uin', 'friend', 'uin'): 3, ('uin', 'idcardid', 'uin'): 4, ('uin', 'device', 'uin'): 5,
                               ('uin', 'payee', 'uin'): 6, ('uin', 'payer', 'uin'): 7, ('uin', 'bankcard', 'uin'): 8,
                               ('uin', 'download_app', 'uin'): 9}
            self.filter_edge_types = [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                                      ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                                      ('uin', 'bankcard', 'uin')]
        elif self.conv_type in ['HGT', 'HAN']:
            self.metadata = (['uin'], [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                                       ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                                       ('uin', 'payee', 'uin'), ('uin', 'payer', 'uin'), ('uin', 'bankcard', 'uin'),
                                       ('uin', 'download_app', 'uin')])
            if self.conv_type == 'HGT':
                self.model = HeteroGraphTransformer(
                    in_channels=args_dict['input_dim'],
                    hidden_channels=args_dict['hidden_dim'],
                    out_channels=args_dict['output_dim'],
                    metadata=self.metadata,
                    heads=args_dict['num_heads']
                )
            else:
                self.model = HAN(in_channels=args_dict['input_dim'],
                                 out_channels=args_dict['output_dim'],
                                 metadata=self.metadata,
                                 heads=args_dict['num_heads'])
        self.pretrain_lr = self.eval_dict['pretrain_lr']
        self.control_node_num = self.eval_dict["filter_node_num"]
        self.task_type = self.eval_dict["task_type"]
        self.info_type = self.eval_dict["info_insertion_type"]
        self.save_model_path = os.path.join(self.eval_dict["model_states_path"],
                                            f'{self.eval_dict["data_tag"]}{self.conv_type}_sample_'
                                            f'{self.eval_dict["sampling"]}_filter_{self.control_node_num}_'
                                            f'lr_{self.pretrain_lr}{self.eval_dict["lr_scheduler"]}'
                                            f'{self.device_tag}')
        if not self.eval_dict["is_supervised"]:
            if self.eval_dict["eval_epoch"] != 0:
                epoch_num = self.eval_dict["eval_epoch"]
                if self.task_type == 'subgraph':
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
                elif self.task_type == 'cross_subgraph':
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_model_epoch_{epoch_num}.pth")
                else:
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_t_"
                                             f"{self.eval_dict['temperature']}_model_epoch_{epoch_num}.pth")
            else:
                if self.task_type == 'subgraph':
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_model_best_loss.pth")
                elif self.task_type == 'cross_subgraph':
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_model_best_loss.pth")
                else:
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_t_"
                                             f"{self.eval_dict['temperature']}_model_best_loss.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            if self.device_tag != '':
                rename_key_model_weight = OrderedDict()
                for key in model_weight.keys():
                    key_weight = model_weight[key]
                    key = key.replace('module.', '')
                    rename_key_model_weight[key] = key_weight
                self.model.load_state_dict(rename_key_model_weight)
            else:
                self.model.load_state_dict(model_weight)
            print(f"Load: {file_name}")
        self.model.to(self.device)
        # set classifier
        if self.info_type in ['combine_subgraph', 'combine_difference']:
            cls_input = args_dict['output_dim'] * 2
        else:
            cls_input = args_dict['output_dim']
        self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                              torch.nn.ReLU(),
                                              torch.nn.Linear(args_dict['hidden_dim'], 2),
                                              torch.nn.Softmax(dim=1))
        self.classifier.to(self.device)
        self.pred_save_path = self.eval_dict["output_save_path"]
        self.pt_info = (f'{self.conv_type}_{self.task_type}_{self.eval_dict["eval_epoch"]}_'
                        f't_{self.eval_dict["temperature"]}_lr_{self.eval_dict["cls_lr"]}{self.device_tag}')
        if self.eval_dict['ft_loss'] == "node_penalty":
            weights = self.eval_dict['cls_loss_weight'].split[' ']
            if not self.eval_dict['cls_penalty']:
                self.ft_info = f'wp_{weights[1]}_wn_{weights[0]}'
            else:
                self.ft_info = (f'wp_{weights[1]}_wn_{weights[0]}_W_node_{self.eval_dict["W_node"]}_'
                                f'W_penalty_{self.eval_dict["W_penalty"]}')
        elif self.eval_dict['ft_loss'] == 'subgraph_and_dense':
            if self.eval_dict['cls_subgraph'] and not self.eval_dict['cls_dense'] and not self.eval_dict['cls_connect']:
                self.ft_info = (f'Wp_{self.eval_dict["W_p"]}_Wn_{self.eval_dict["W_n"]}_'
                                f'wp_{self.eval_dict["w_p"]}_wn_{self.eval_dict["w_n"]}')
            elif self.eval_dict['cls_subgraph'] and self.eval_dict['cls_dense'] and not self.eval_dict['cls_connect']:
                self.ft_info = (f'W_sub_{self.eval_dict["W_sub"]}_W_den_{self.eval_dict["W_den"]}_'
                                f'Wp_{self.eval_dict["W_p"]}_Wn_{self.eval_dict["W_n"]}_'
                                f'wp_{self.eval_dict["w_p"]}_wn_{self.eval_dict["w_n"]}')
            elif self.eval_dict['cls_subgraph'] and self.eval_dict['cls_connect'] and not self.eval_dict['cls_dense']:
                self.ft_info = (f'W_sub_{self.eval_dict["W_sub"]}_W_cnt_{self.eval_dict["W_cnt"]}_'
                                f'Wp_{self.eval_dict["W_p"]}_Wn_{self.eval_dict["W_n"]}_'
                                f'wp_{self.eval_dict["w_p"]}_wn_{self.eval_dict["w_n"]}')
            else:
                self.ft_info = (f'W_sub_{self.eval_dict["W_sub"]}_W_den_{self.eval_dict["W_den"]}_'
                                f'W_cnt_{self.eval_dict["W_cnt"]}_'
                                f'Wp_{self.eval_dict["W_p"]}_Wn_{self.eval_dict["W_n"]}_'
                                f'wp_{self.eval_dict["w_p"]}_wn_{self.eval_dict["w_n"]}')
        if self.eval_dict["evaluate_task"] in ['subgraph', 'subgraph_gang_detection']:
            train_data_path = os.path.join(self.eval_dict["train_data_path"] + str(self.eval_dict["split_idx"]),
                                           self.eval_dict["train_data_file"])
            self.train_data = UinGangsDataIterablePyG(self.eval_dict, train_data_path)
            test_data_path = os.path.join(self.eval_dict["test_data_path"] + str(self.eval_dict["split_idx"]),
                                          self.eval_dict["test_data_file"])
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict, test_data_path)
            self.train_loader = Data.DataLoader(self.train_data,
                                                batch_size=self.eval_dict["batch_size"],
                                                num_workers=self.eval_dict["num_workers"],
                                                collate_fn=self.train_data.collate_fn)
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.collate_fn)
            params = []
            if self.eval_dict["is_finetune"] or self.eval_dict["is_supervised"]:
                params.append({'params': self.model.parameters(), 'lr': self.pretrain_lr})
            params.append({'params': self.classifier.parameters(), 'lr': self.eval_dict['cls_lr']})
            self.cls_optimizer = torch.optim.Adam(params)
            self.loss_weight = self.eval_dict['cls_loss_weight'].split(' ')
            self.loss_weight = [float(item) for item in self.loss_weight]
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.loss_weight, device=self.device))
        # inference based on pretrained GNN model and classifier
        else:
            self.model.eval()
            test_data_path = os.path.join(self.eval_dict["test_data_path"] + str(self.eval_dict["split_idx"]),
                                          self.eval_dict["test_data_file"])
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict, test_data_path)
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.pos_collate_fn_for_fraudar)
        # Set seed for whole environment
        self.setup_seed()

    def setup_seed(self):
        torch.manual_seed(self.eval_dict["seed"])
        torch.cuda.manual_seed(self.eval_dict["seed"])
        torch.cuda.manual_seed_all(self.eval_dict["seed"])
        np.random.seed(self.eval_dict["seed"])
        random.seed(self.eval_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def get_edge_info(self, batch):
        edge_index = [batch[edge_type].edge_index for edge_type in list(self.edge_types.keys()) if
                      edge_type in batch.edge_types]
        edge_index = torch.concat(edge_index, dim=1)
        edge_counts = {edge_type: batch[edge_type].num_edges for edge_type in list(self.edge_types.keys()) if
                       edge_type in batch.edge_types}
        edge_type = torch.concat(
            [torch.ones(edge_counts[edge_type]) * edge_idx for edge_type, edge_idx in self.edge_types.items() if
             edge_type in batch.edge_types])
        return edge_index, edge_type.long()

    def extract_single_subgraph(self, batch, subgraph_node_indices=None):
        graph_data = HeteroData()
        graph_data['uin'].x = batch['uin'].x[subgraph_node_indices].clone()
        graph_data['uin'].score = batch['uin'].score[subgraph_node_indices].clone()
        subgraph_max_node_idx = subgraph_node_indices.max()
        for edge_type in batch.edge_types:
            try:
                current_max_node_idx = batch[edge_type].edge_index.max()
                max_node_idx = min(subgraph_max_node_idx, current_max_node_idx)
                if max_node_idx < subgraph_max_node_idx:
                    subgraph_node_indices = subgraph_node_indices[subgraph_node_indices <= max_node_idx]
                edge_index, _ = subgraph(subgraph_node_indices, batch[edge_type].edge_index, relabel_nodes=True)
            except Exception as e:
                print(f"Extract Subgraph Error: <{e}>")
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    graph_data[edge_type].edge_index = edge_index
        return graph_data

    def rgcn_fit(self, pos_batch, batch_x):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
        except Exception as e:
            print(f"{self.conv_type} Get Edge Information Error: <{e}>")
            return None
        else:
            return batch_h

    def hetero_fit(self, x_dict, edge_index_dict):
        filter_edge_dict = {}
        for edge_type in list(edge_index_dict.keys()):
            if edge_type in self.metadata[1]:
                filter_edge_dict[edge_type] = edge_index_dict[edge_type]
        out = self.model(x_dict, filter_edge_dict)
        return out

    def jaccard(self, set_a, set_b):
        intersection = torch.sum(set_a & set_b)
        union = torch.sum(set_a | set_b)
        return intersection / union if union != torch.tensor(0) else torch.tensor(0.0)

    def jaccard_batch(self, y_true, y_pred, batch):
        N = batch.max().detach().cpu().item() + 1
        jaccard_list = []
        true_idx_list = []
        pred_idx_list = []
        for i in range(N):
            subgraph_i_idx = (batch == i).nonzero().squeeze()
            i_y_true = y_true[subgraph_i_idx]
            i_y_pred = y_pred[subgraph_i_idx]
            y_true_idx = (i_y_true == 1).nonzero().squeeze().detach().cpu().tolist()
            y_pred_idx = (i_y_pred == 1).nonzero().squeeze().detach().cpu().tolist()
            true_idx_list.append(str(y_true_idx))
            pred_idx_list.append(str(y_pred_idx))
            if torch.sum(i_y_true).detach().cpu().item() == 0:
                fp_items = (i_y_pred == 1).nonzero().squeeze().detach().cpu().tolist()
                if type(fp_items) == list:
                    # reverse for prediction
                    if len(fp_items) > 2:
                        jac = torch.tensor(1.0)
                    else:
                        jac = torch.tensor(0.0)
                else:
                    jac = torch.tensor(0.0)
            else:
                jac = self.jaccard(i_y_true, i_y_pred)
            jaccard_list.append(jac.detach().cpu().item())
        return jaccard_list, true_idx_list, pred_idx_list

    def subgraph_embedding_expand(self, subgraph_embedding, expand_sizes):
        n = subgraph_embedding.shape[0]
        subgraph_embedding_list = []
        for i in range(n):
            repeat_times = (expand_sizes[i + 1] - expand_sizes[i]).detach().cpu().item()
            subgraph_embedding_list.append(subgraph_embedding[i, :].repeat(repeat_times, 1))
        return torch.concat(subgraph_embedding_list, dim=0)

    def compute_penalty_loss(self, node_y_true, node_y_pred, subgraph_y_true, batch):
        N = batch.max().detach().cpu().item() + 1
        penalty_list = []
        for i in range(N):
            subgraph_i_idx = (batch == i).nonzero().squeeze().detach().cpu()
            node_y_true_i = node_y_true[subgraph_i_idx]
            node_y_pred_i = node_y_pred[subgraph_i_idx]
            if subgraph_y_true[i] == 1:
                jac_cof = self.jaccard(node_y_true_i, node_y_pred_i)
                fn_penalty = torch.exp(-jac_cof.detach().cpu())
                penalty_list.append(fn_penalty)
            else:
                fp_items = (node_y_pred_i == 1).nonzero().squeeze().detach().cpu().tolist()
                fp_num = len(fp_items) if type(fp_items) == list else 1
                fp_penalty = torch.exp(
                    torch.tensor((fp_num - self.control_node_num + 1) / node_y_pred_i.shape[0])) - torch.exp(
                    torch.tensor(-1))
                penalty_list.append(fp_penalty)
        penalty = torch.stack(penalty_list).to(self.device).mean().requires_grad_()
        return penalty

    def save_output_file(self, true_idx_list, pred_idx_list,
                         subgraph_uin_list, subgraph_y_list,
                         jaccard_list, file_name):
        true_uin_list, pred_uin_list = [], []
        for i in range(len(true_idx_list)):
            true_uin_idx = true_idx_list[i][1:-1].split(',')
            pred_uin_idx = pred_idx_list[i][1:-1].split(',')
            uin_map_list = subgraph_uin_list[i]
            if true_uin_idx != ['']:
                true_uin_list.append([uin_map_list[int(item)] for item in true_uin_idx])
            else:
                true_uin_list.append([])
            if pred_uin_idx != ['']:
                pred_uin_list.append([uin_map_list[int(item)] for item in pred_uin_idx])
            else:
                pred_uin_list.append([])
        df = pd.DataFrame(data={'label_gang_mem_list': true_uin_list,
                                'pred_gang_mem_list': pred_uin_list,
                                'jaccard': jaccard_list,
                                'subgraph_gang_label': subgraph_y_list,
                                'subgraph_node_map': subgraph_uin_list})
        df.to_csv(file_name, index=False)
        print(f"===== Save Prediction File To: {file_name} =====")

    def detect_subgraph_gang_members(self):
        if self.eval_dict["is_finetune"] or self.eval_dict["is_supervised"]:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = self.eval_dict['best_test_f1']
        best_test_roc_auc = 0
        best_test_cm = np.array([[0, 0], [0, 0]])
        best_test_pos_jac = 0
        best_test_neg_jac = 0
        best_test_time = 0
        for epoch in range(1, self.eval_dict["n_epochs"] + 1):
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
                if self.conv_type in ['RGCN', 'AttnRGCN']:
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                # get edge information
                if batch_h is not None:
                    batch_y = batch['uin'].gang_mem.long()
                    if self.info_type == 'combine_subgraph':
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        diff_h = expand_batch_h_g - batch_h
                        batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                    elif self.info_type == 'concat_subgraph':
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                    pred_y = self.classifier(batch_h)
                    if self.eval_dict['ft_loss'] == 'node_penalty':
                        if self.eval_dict['cls_node']:
                            cls_loss = self.criterion(pred_y, batch_y)
                        else:
                            cls_loss = torch.tensor(0, device=self.device)
                        if self.eval_dict['cls_penalty']:
                            sub_y = batch['uin'].gang_label.detach().cpu().int().tolist()
                            penalty_loss = self.compute_penalty_loss(batch_y,
                                                                     pred_y.argmax(dim=1),
                                                                     sub_y,
                                                                     batch['uin'].batch)
                        else:
                            penalty_loss = torch.tensor(0, device=self.device)
                        loss = self.eval_dict['W_node'] * cls_loss + self.eval_dict['W_penalty'] * penalty_loss
                    else:
                        if self.eval_dict['cls_subgraph']:
                            sub_loss = batch_subgraph_loss_based_cross_entropy(batch['uin'].gang_label.long(),
                                                                               batch['uin'].batch,
                                                                               pred_y[:, 1],
                                                                               batch_y,
                                                                               self.eval_dict['W_p'],
                                                                               self.eval_dict['W_n'],
                                                                               self.eval_dict['w_p'],
                                                                               self.eval_dict['w_n']
                                                                               )
                        else:
                            sub_loss = torch.tensor(0, device=self.device)
                        if self.eval_dict['cls_dense']:
                            dense_loss = batch_dense_loss_based_cross_entropy(batch, pred_y[:, 1])
                        else:
                            dense_loss = torch.tensor(0, device=self.device)
                        if self.eval_dict['cls_connect']:
                            connect_loss = batch_connect_loss(batch, pred_y[:, 1])
                        else:
                            connect_loss = torch.tensor(0, device=self.device)
                        loss = (self.eval_dict["W_sub"] * sub_loss + self.eval_dict["W_den"] * dense_loss +
                                self.eval_dict["W_cnt"] * connect_loss)
                    loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm,
             test_pos_jac, test_neg_jac, test_time) = self.evaluate_classifier(task="detect_gang",
                                                                               save_results=True,
                                                                               best_f1=best_test_f1)
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
                best_test_pos_jac = test_pos_jac
                best_test_neg_jac = test_neg_jac
                best_test_time = test_time
                tp_tf = f'tn_{best_test_cm[0, 0]}_tp_{best_test_cm[1, 1]}_total_{best_test_cm[0, 0] + best_test_cm[1, 1]}'
                is_finetune = '_finetune' if self.eval_dict["is_finetune"] else ''
                is_supervised = '_supervised' if self.eval_dict["is_supervised"] else ''
                file_name = f"{self.info_type}_f1_{best_test_f1:.2f}_{tp_tf}{is_finetune}{is_supervised}.pth" \
                    if self.info_type is not None else f"f1_{best_test_f1}_{tp_tf}{is_finetune}{is_supervised}.pth"
                file_dir = os.path.join(self.eval_dict["cls_model_states_path"], self.pt_info, self.ft_info)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                file_name = os.path.join(file_dir, file_name)
                torch.save(self.classifier.state_dict(), file_name)
                print(f"===== Best F1: {test_f1:.4f}, Save To: {file_name} =====")
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm, best_test_pos_jac, best_test_neg_jac, best_test_time

    def evaluate_classifier(self, task="subgraph", save_results=False, best_f1=0):
        self.classifier.eval()
        self.model.eval()
        true_y_list = []
        pred_y_list = []
        prob_y_list = []
        true_idx_list = []
        pred_idx_list = []
        if save_results:
            subgraph_y_list = []
            subgraph_uin_list = []
            jaccard_list = []
        start_time = time.time()
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
                if self.conv_type in ['RGCN', 'AttnRGCN']:
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                if batch_h is not None:
                    if task == "subgraph":
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        batch_y = batch['uin'].gang_label
                        prob_y = self.classifier(batch_h_g)[:, 1]
                        pred_y = self.classifier(batch_h_g).argmax(dim=1)
                    else:
                        batch_y = batch['uin'].gang_label.long()
                        true_gang_member = batch['uin'].gang_mem.int()
                        if self.info_type == 'combine_subgraph':
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            diff_h = expand_batch_h_g - batch_h
                            batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                        elif self.info_type == 'concat_subgraph':
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                        pred_gang_member = self.classifier(batch_h).argmax(dim=1)
                        jaccard_coeff, true_idx, pred_idx = self.jaccard_batch(true_gang_member, pred_gang_member,
                                                                               batch['uin'].batch)
                        prob_y = torch.tensor(jaccard_coeff, device=self.device)
                        pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                        pred_y[prob_y >= self.eval_dict["threshold"]] = 1
                        if save_results:
                            jaccard_list.extend(jaccard_coeff)
                            true_idx_list.extend(true_idx)
                            pred_idx_list.extend(pred_idx)
                            subgraph_y_list.extend(batch['uin'].y.detach().cpu().tolist())
                            subgraph_uin_list.extend(batch['uin'].nodeid2uin_map)
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
            pos_jaccard = np.array(jaccard_list)[true_y_list != 0].mean()
            neg_jaccard = np.array(jaccard_list)[true_y_list == 0].mean()
            eval_time = time.time() - start_time
            print(f"Test ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Pos Jaccard: {pos_jaccard: .4f}, "
                  f"Neg Jaccard: {neg_jaccard: .4f}, "
                  f"Evaluate Time: {eval_time: .4f} s"
                  )
            if f1 > best_f1 and save_results:
                file_name = os.path.join(self.pred_save_path, self.pt_info, self.ft_info)
                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                file_name = os.path.join(file_name, f"{self.info_type}_f1_{f1:.2f}.csv")
                self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                      subgraph_y_list, jaccard_list, file_name)
            return acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, eval_time

    def inference_gang_members_by_fraudar(self, evaluate_time, save_results=False):
        start_time = time.time()
        jaccard_list = []
        true_idx_list = []
        pred_idx_list = []
        subgraph_y_list = []
        subgraph_uin_list = []
        true_y_list = []
        pred_y_list = []
        prob_y_list = []
        with torch.no_grad():
            for i, pos_batch in enumerate(self.eval_loader):
                true_y = pos_batch['uin'].gang_mem.int().detach().cpu()
                pred_y = pos_batch['uin'].idx.detach().cpu()
                jaccard_coefficient, true_idx, pred_idx = self.jaccard_batch(true_y, pred_y,
                                                                             pos_batch['uin'].batch)
                true_subgraph_y = pos_batch['uin'].gang_label.int().detach().cpu()
                prob_subgraph_y = torch.tensor(jaccard_coefficient, device=self.device)
                pred_subgraph_y = torch.zeros(prob_subgraph_y.shape[0], device=self.device)
                pred_subgraph_y[prob_subgraph_y >= self.eval_dict["threshold"]] = 1
                true_y_list.append(true_subgraph_y)
                pred_y_list.append(pred_subgraph_y)
                prob_y_list.append(prob_subgraph_y)
                jaccard_list.extend(jaccard_coefficient)
                true_idx_list.extend(true_idx)
                pred_idx_list.extend(pred_idx)
                subgraph_y_list.extend(pos_batch['uin'].y.detach().cpu().tolist())
                subgraph_uin_list.extend(pos_batch['uin'].nodeid2uin_map)
            # compute metrics
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            acc = accuracy_score(true_y_list, pred_y_list)
            f1 = f1_score(true_y_list, pred_y_list)
            pre = precision_score(true_y_list, pred_y_list)
            rec = recall_score(true_y_list, pred_y_list)
            roc_auc = roc_auc_score(true_y_list, prob_y_list)
            cm = confusion_matrix(true_y_list, pred_y_list)
            pos_jaccard = np.array(jaccard_list)[true_y_list != 0].mean()
            neg_jaccard = np.array(jaccard_list)[true_y_list == 0].mean()
            run_time = time.time() - start_time
            print(f"Fraudar ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Pos Jaccard Coefficient: {pos_jaccard: .4f}, "
                  f"Neg Jaccard Coefficient: {neg_jaccard: .4f}, "
                  f"Time: {run_time: .4f} s"
                  )
        if save_results:
            # save inference results
            save_path = os.path.join(self.pred_save_path, 'fraudar')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = os.path.join(save_path, f'fraudar_{evaluate_time}.csv')
            self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                  subgraph_y_list, jaccard_list, file_name)
            return acc, pre, rec, f1, roc_auc, pos_jaccard, neg_jaccard, run_time
        else:
            return acc, pre, rec, f1, roc_auc, pos_jaccard, neg_jaccard, run_time

    def filter_by_fraudar(self, graph_data):
        G = nx.DiGraph()
        # 添加节点
        for i in range(graph_data['uin'].num_nodes):
            G.add_node(i, evil_score=graph_data['uin'].score[i].item())
        idx = torch.zeros(graph_data['uin'].num_nodes, dtype=torch.int)
        edge_index = [graph_data[edge_type].edge_index for edge_type in graph_data.edge_types]
        if len(edge_index) > 0:
            edge_index = torch.concat(edge_index, dim=1)
            unique_edge_index, indices = torch.unique(edge_index, dim=1, return_inverse=True)
            adj = to_dense_adj(edge_index, max_num_nodes=graph_data['uin'].score.shape[0]).squeeze()

            # 添加边
            for i in range(unique_edge_index.shape[1]):
                srt = unique_edge_index[0, i].item()
                dst = unique_edge_index[1, i].item()
                G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
            best_graph, best_density, best_weight = fraudar(G)
            if len(best_graph.nodes) >= self.control_node_num:
                idx[sorted(best_graph.nodes)] = 1
        return idx

    def inference_gang_members(self, fraudar_filter=True):
        start_time = time.time()
        tp_tf = f'tn_{self.eval_dict["tn"]}_tp_{self.eval_dict["tp"]}_total_{self.eval_dict["tn"] + self.eval_dict["tp"]}'
        is_finetune = '_finetune' if self.eval_dict["is_finetune"] else ''
        is_supervised = '_supervised' if self.eval_dict["is_supervised"] else ''
        file_name = f"{self.info_type}_f1_{self.eval_dict['best_f1']}_{tp_tf}{is_finetune}{is_supervised}.pth" \
            if self.info_type is not None else f"f1_{self.eval_dict['best_f1']}_{tp_tf}{is_finetune}{is_supervised}.pth"
        file_name = os.path.join(self.eval_dict["cls_model_states_path"], self.pt_info, self.ft_info, file_name)
        print(f"Load Classifier: {file_name}")
        model_weight = torch.load(file_name, map_location=self.device)
        self.classifier.load_state_dict(model_weight)
        self.classifier.eval()
        jaccard_list = []
        if self.eval_dict["is_debug"]:
            true_ave_density = []
            pred_ave_density = []
            true_ave_score = []
            pred_ave_score = []
            true_num = []
            pred_num = []
            gang_label = []
        true_idx_list = []
        pred_idx_list = []
        subgraph_y_list = []
        subgraph_uin_list = []
        true_y_list = []
        prob_y_list = []
        pred_y_list = []
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
                if self.conv_type in ['RGCN', 'AttnRGCN']:
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                if batch_h is not None:
                    if self.info_type == 'combine_subgraph':
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        diff_h = expand_batch_h_g - batch_h
                        batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                    prob_y = self.classifier(batch_h)
                    if fraudar_filter:
                        batch['uin'].score = prob_y[:, 1]
                        batch_idx, _ = batch['uin'].batch.unique().sort()
                        pred_y = []
                        for i in batch_idx:
                            node_idx_in_subgraph_i = (batch['uin'].batch == i).nonzero().squeeze()
                            graph_data = self.extract_single_subgraph(batch, node_idx_in_subgraph_i)
                            i_pred_y = self.filter_by_fraudar(graph_data)
                            pred_y.append(i_pred_y)
                        pred_y = torch.concat(pred_y, dim=0)
                    else:
                        pred_y = prob_y.argmax(dim=1)
                    batch_y = batch['uin'].gang_mem.int()
                    true_y = batch_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    jaccard_coeff, true_idx, pred_idx = self.jaccard_batch(true_y, pred_y,
                                                                           batch['uin'].batch)
                    if self.eval_dict["is_debug"]:
                        adj = to_dense_adj(
                            torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types], dim=1),
                            max_num_nodes=batch.num_nodes).squeeze()
                        score = batch['uin'].score.squeeze()
                        pred_index = (pred_y == 1).nonzero().squeeze()
                        true_index = (true_y == 1).nonzero().squeeze()
                        if pred_index.shape == torch.Size([]):
                            pred_adj = torch.tensor([])
                            pred_score = torch.tensor([])
                        else:
                            pred_adj = adj[pred_index, :][:, pred_index]
                            pred_score = score[pred_index]
                        if true_index.shape == torch.Size([]):
                            true_adj = torch.tensor([])
                            true_score = torch.tensor([])
                        else:
                            true_adj = adj[true_index, :][:, true_index]
                            true_score = score[true_index]
                        pred_ave_density.append(
                            pred_adj.sum() / pred_index.sum() if pred_index.sum().item() != 0 else torch.tensor(0))
                        true_ave_density.append(
                            true_adj.sum() / true_index.sum() if true_index.sum().item() != 0 else torch.tensor(0))
                        pred_ave_score.append(pred_score.mean() if not pred_score.mean().isnan() else torch.tensor(0))
                        true_ave_score.append(true_score.mean() if not true_score.mean().isnan() else torch.tensor(0))
                        pred_num.append(pred_y.sum())
                        true_num.append(true_y.sum())
                        gang_label.append(batch['uin'].y.int())
                    jaccard_list.extend(jaccard_coeff)
                    true_idx_list.extend(true_idx)
                    pred_idx_list.extend(pred_idx)
                    subgraph_y_list.extend(batch['uin'].y.detach().cpu().tolist())
                    subgraph_uin_list.extend(batch['uin'].nodeid2uin_map)
                    prob_y = torch.tensor(jaccard_coeff, device=self.device)
                    pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                    pred_y[prob_y >= self.eval_dict["threshold"]] = 1
                    true_y_list.append(batch['uin'].gang_label.detach().cpu().long())
                    prob_y_list.append(prob_y.detach().cpu())
                    pred_y_list.append(pred_y.detach().cpu())
            if self.eval_dict["is_debug"]:
                pred_ave_density = [round(v, 4) for v in torch.stack(pred_ave_density).tolist()]
                true_ave_density = [round(v, 4) for v in torch.stack(true_ave_density).tolist()]
                pred_ave_score = [round(v, 4) for v in torch.stack(pred_ave_score).tolist()]
                true_ave_score = [round(v, 4) for v in torch.stack(true_ave_score).tolist()]
                pred_num = torch.stack(pred_num).tolist()
                true_num = torch.stack(true_num).tolist()
                gang_label = [v[0] for v in torch.stack(gang_label).tolist()]
                df = pd.DataFrame({'gang_label': gang_label, 'jaccard': jaccard_list,
                                   'pred_num': pred_num, 'true_num': true_num,
                                   'pred_score': pred_ave_score, 'true_score': true_ave_score,
                                   'pred_density': pred_ave_density, 'true_density': true_ave_density})
                df.to_csv(os.path.join(self.pred_save_path, self.pt_info, self.ft_info, 'check.csv'), index=False)
            # compute metrics
            file_name = os.path.join(self.pred_save_path, self.pt_info, self.ft_info)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            if fraudar_filter:
                filter_tag = '_fraudar'
            else:
                filter_tag = ''
            file_name = os.path.join(file_name, f"{self.info_type}_f1_{self.eval_dict['best_f1']}_"
                                                f"{tp_tf}{is_finetune}{is_supervised}{filter_tag}.csv")
            self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                  subgraph_y_list, jaccard_list, file_name)
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            acc = accuracy_score(true_y_list, pred_y_list)
            f1 = f1_score(true_y_list, pred_y_list)
            pre = precision_score(true_y_list, pred_y_list)
            rec = recall_score(true_y_list, pred_y_list)
            roc_auc = roc_auc_score(true_y_list, prob_y_list)
            cm = confusion_matrix(true_y_list, pred_y_list)
            pos_jaccard = np.array(jaccard_list)[true_y_list != 0].mean()
            neg_jaccard = np.array(jaccard_list)[true_y_list == 0].mean()
            run_time = time.time() - start_time
            print(f"{self.conv_type} ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Pos Jaccard Coefficient: {pos_jaccard: .4f}, "
                  f"Neg Jaccard Coefficient: {neg_jaccard: .4f}, "
                  f"Time: {run_time: .4f} s"
                  )
            return acc, f1, pre, rec, roc_auc, pos_jaccard, neg_jaccard, run_time
