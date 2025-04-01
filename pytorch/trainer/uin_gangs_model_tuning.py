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
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool.sag_pool import SAGPooling
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN, MaskRGCN, AttnRGCN
from mmgog_long_term_sequence_model.pytorch.models.rgat_model import GAT
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from mmgog_long_term_sequence_model.utils.utils import batch_subgraph_loss_based_cross_entropy
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
        self.conv_type = self.eval_dict["conv_type"]
        self.device_tag = self.eval_dict["device_tag"]
        self.pooling = self.eval_dict["pooling"]
        self.prompt_type = self.eval_dict["prompt_type"]
        self.pretrain_lr = self.eval_dict['pretrain_lr']
        self.control_node_num = self.eval_dict["filter_node_num"]
        self.pretrained_task_type = self.eval_dict["pretrained_task_type"]
        self.info_type = self.eval_dict["info_insertion_type"]
        self.n_dim = self.eval_dict["uin_acs_numberical_feat_dim"]
        self.c_dim = self.n_dim + self.eval_dict["uin_acs_categorical_feat_hasher_dim"]
        self.metric = self.eval_dict["metric_learning"]
        self.num_bases = args_dict['num_bases'] if args_dict['num_bases'] != 0 else 5
        if self.pooling == 'sag_pool':
            self.sag_pooling = SAGPooling(in_channels=args_dict['output_dim'],
                                          ratio=args_dict['top_sag_pool_ratio'],
                                          GNN=GATConv).to(self.device)
        if self.prompt_type == 'single_token':
            self.prompt = torch.nn.Parameter(torch.rand(1, args_dict['input_dim'], requires_grad=True))
        # All edges (including self loop)
        self.edge_types = {('uin', 'self_loop', 'uin'): 0, ('uin', 'ipv6', 'uin'): 1, ('uin', 'wifi', 'uin'): 2,
                           ('uin', 'room', 'uin'): 3, ('uin', 'friend', 'uin'): 4, ('uin', 'idcardid', 'uin'): 5,
                           ('uin', 'device', 'uin'): 6, ('uin', 'bankcard', 'uin'): 7, ('uin', 'headimg', 'uin'): 8,
                           ('uin', 'nickname', 'uin'): 9, ('uin', 'signature', 'uin'): 10,
                           ('uin', 'android_bootid_fsid', 'uin'): 11, ('uin', 'payer', 'uin'): 12,
                           ('uin', 'payee', 'uin'): 13, ('uin', 'download_app', 'uin'): 14}
        self.num_relations = len(self.edge_types)
        print(f"Current Edges: {self.edge_types}")

        if self.conv_type == 'RGCN':
            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=self.num_relations,
                              num_bases=self.num_relations)
        elif self.conv_type == 'MaskRGCN':
            self.model = MaskRGCN(
                mlp_n_in_dim=args_dict['uin_acs_numberical_feat_dim'],
                mlp_c_in_dim=args_dict['uin_acs_categorical_feat_hasher_dim'],
                mlp_t_in_dim=args_dict['uin_acs_text_feat_dim'],
                input_dim=args_dict['input_dim'],
                hidden_dim=args_dict['hidden_dim'],
                output_dim=args_dict['output_dim'],
                adapter_type=args_dict['adapter_type'],
                num_relations=self.num_relations,
                metric_learning=self.metric,
                num_bases=self.num_bases)
        elif self.conv_type == 'AttnRGCN':
            self.model = AttnRGCN(
                mlp_n_in_dim=args_dict['uin_acs_numberical_feat_dim'],
                mlp_c_in_dim=args_dict['uin_acs_categorical_feat_hasher_dim'],
                input_dim=args_dict['input_dim'],
                hidden_dim=args_dict['hidden_dim'],
                output_dim=args_dict['output_dim'],
                num_relations=self.num_relations,
                num_bases=self.num_relations)
        elif self.conv_type == 'GAT':
            self.model = GAT(input_dim=args_dict['input_dim'],
                             hidden_dim=args_dict['hidden_dim'],
                             output_dim=args_dict['output_dim'],
                             num_heads=args_dict['num_heads'])
        elif self.conv_type == 'MLP':
            self.model = torch.nn.Sequential(torch.nn.Linear(args_dict['input_dim'], args_dict['hidden_dim']),
                                             torch.nn.ReLU(),
                                             torch.nn.Linear(args_dict['hidden_dim'], 2),
                                             torch.nn.Softmax(dim=1))
        else:
            self.model = SVC(kernel='linear')
        self.save_model_path = os.path.join(self.eval_dict["model_states_path"],
                                            f'{self.eval_dict["data_tag"]}{self.conv_type}_sample_'
                                            f'{self.eval_dict["sampling"]}_filter_{self.control_node_num}_'
                                            f'lr_{self.pretrain_lr}_edges_{self.num_relations}'
                                            f'{self.eval_dict["lr_scheduler"]}{self.device_tag}')
        if self.conv_type != "SVM":
            if not self.eval_dict["is_supervised"]:
                if self.eval_dict["eval_epoch"] != 0:
                    epoch_num = self.eval_dict["eval_epoch"]
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.pretrained_task_type}_t_"
                                             f"{self.eval_dict['temperature']}_model_epoch_{epoch_num}.pth")
                else:
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.pretrained_task_type}_t_"
                                             f"{self.eval_dict['temperature']}_model_best_loss.pth")
                model_weight = torch.load(file_name, map_location=self.device)
                if self.device_tag not in ['', '_None']:
                    rename_key_model_weight = OrderedDict()
                    for key in model_weight.keys():
                        key_weight = model_weight[key]
                        key = key.replace('module.', '')
                        rename_key_model_weight[key] = key_weight
                    self.model.load_state_dict(rename_key_model_weight)
                else:
                    self.model.load_state_dict(model_weight, strict=False)
                print(f"Load: {file_name}")
            self.model.to(self.device)
            # set classifier
            if self.info_type in ['combine_subgraph', 'combine_difference'] and self.conv_type != 'GAT':
                cls_input = args_dict['output_dim'] * 2
            elif self.info_type in ['combine_subgraph', 'combine_difference'] and self.conv_type == 'GAT':
                cls_input = args_dict['output_dim'] * args_dict['num_heads'] * 2
            else:
                cls_input = args_dict['output_dim']
            if self.eval_dict["downstream_task"] in ['subgraph_node_score_optim']:
                output_dim = 1
                self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(args_dict['hidden_dim'], output_dim))
            else:
                output_dim = 2
                self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                      torch.nn.ReLU(),
                                                      torch.nn.Linear(args_dict['hidden_dim'], output_dim),
                                                      torch.nn.Softmax(dim=1))

            self.classifier.to(self.device)
            self.pred_save_path = self.eval_dict["output_save_path"]
            if not self.eval_dict["is_supervised"]:
                self.pt_info = (f'{self.conv_type}_{self.pretrained_task_type}_{self.eval_dict["eval_epoch"]}_'
                                f't_{self.eval_dict["temperature"]}_lr_{self.eval_dict["cls_lr"]}{self.device_tag}')
            else:
                self.pt_info = f'{self.conv_type}_No_Pretrain'
            weights = self.eval_dict['cls_loss_weight'].split(' ')
            self.ft_info = f'wp_{weights[1]}_wn_{weights[0]}'
        if self.eval_dict["downstream_task"] in ['subgraph_cl', 'subgraph_gang_cl',
                                                 'subgraph_gang_cl_by_svm', 'subgraph_node_score_optim']:
            train_data_path = os.path.join(self.eval_dict["train_data_path"] + self.eval_dict["split_idx"],
                                           self.eval_dict["train_data_file"])
            self.train_data = UinGangsDataIterablePyG(self.eval_dict, train_data_path)
            test_data_path = os.path.join(self.eval_dict["test_data_path"] + self.eval_dict["split_idx"],
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
            if self.conv_type != "SVM":
                params = []
                if (self.eval_dict["is_finetune"] and self.eval_dict["adapter_type"] is None) or self.eval_dict[
                    "is_supervised"]:
                    params.append({'params': self.model.parameters(), 'lr': self.eval_dict['cls_lr']})
                elif self.eval_dict["is_finetune"] and self.eval_dict["adapter_type"] is not None:
                    params.append(
                        {'params': self.model.conv1.conv.adapter.parameters(), 'lr': self.eval_dict['cls_lr']})
                    params.append(
                        {'params': self.model.conv2.conv.adapter.parameters(), 'lr': self.eval_dict['cls_lr']})
                params.append({'params': self.classifier.parameters(), 'lr': self.eval_dict['cls_lr']})
                if self.pooling == 'sag_pool':
                    params.append({'params': self.sag_pooling.parameters(), 'lr': self.eval_dict['cls_lr']})
                if self.prompt_type == 'single_token':
                    params.append({'params': self.prompt, 'lr': self.eval_dict['cls_lr']})
                self.cls_optimizer = torch.optim.Adam(params)
                if self.eval_dict["downstream_task"] in ['subgraph_node_score_optim']:
                    self.criterion = torch.nn.MSELoss(reduction='mean')
                else:
                    self.loss_weight = self.eval_dict['cls_loss_weight'].split(' ')
                    self.loss_weight = [float(item) for item in self.loss_weight]
                    self.criterion = torch.nn.CrossEntropyLoss(
                        weight=torch.tensor(self.loss_weight, device=self.device))
        # inference based on pretrained GNN model and classifier
        else:
            self.model.eval()
            test_data_path = os.path.join(self.eval_dict["test_data_path"] + self.eval_dict["split_idx"],
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

    def relation_fit(self, pos_batch, batch_x):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
        except Exception as e:
            print(f"{self.conv_type} Get Edge Information Error: <{e}>")
            return None
        else:
            return batch_h

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
            subgraph_i_idx = (batch == i).nonzero().squeeze().detach().cpu()
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

    def detect_subgraph_by_model(self, flag):
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
        for epoch in range(1, self.eval_dict["n_epochs"] + 1):
            st = time.time()
            self.classifier.train()
            if self.pooling == 'sag_pool':
                self.sag_pooling.train()
            if self.prompt_type is not None:
                self.prompt.requires_grad_(True)
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
                if self.prompt_type == 'single_token':
                    # batch_x = batch_x + self.prompt.to(self.device)
                    batch_x = torch.mul(batch_x, self.prompt.to(self.device))
                batch['uin'].x = batch_x
                batch_h = self.relation_fit(batch, batch['uin'].x)
                if batch_h is not None:
                    if self.pooling == 'sag_pool':
                        batch_edge_index, batch_edge_types = self.get_edge_info(batch)
                        h_pool, edge_index_pool, edge_attr_pool, perm, mask, score = (
                            self.sag_pooling(batch_h, batch_edge_index, batch=batch['uin'].batch))
                        batch_h_g = scatter_mean(h_pool, perm, dim=0)
                    else:
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                    batch_y = batch['uin'].gang_label.int()
                    prob_y = self.classifier(batch_h_g)
                    loss = self.criterion(prob_y, batch_y)
                    self.cls_optimizer.step()
                    epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm = self.evaluate_classifier(flag=flag,
                                                                                                    epoch=epoch,
                                                                                                    task="subgraph_cl")
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
                is_finetune = '_finetune' if self.eval_dict["is_finetune"] else ''
                is_supervised = '_supervised' if self.eval_dict["is_supervised"] else ''
                is_sag_pool = '_sag_pool' if self.info_type == 'sag_pooling' else ''
                file_name = (f"{self.info_type}_ratio_{self.eval_dict['sag_top_ratio']}_f1_{best_test_f1:.2f}"
                             f"{is_finetune}{is_supervised}{is_sag_pool}_sample_{flag}.pth") if self.info_type is not None else \
                    f"f1_{best_test_f1:.2f}{is_finetune}{is_supervised}{is_sag_pool}_sample_{flag}.pth"
                file_dir = os.path.join(self.eval_dict["cls_model_states_path"], self.pt_info, self.ft_info)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                file_name = os.path.join(file_dir, file_name)
                params = {'cls': self.classifier.state_dict()}
                if self.info_type == 'sag_pooling':
                    params['pooling'] = self.sag_pooling.state_dict()
                if self.prompt_type is not None:
                    params['prompt'] = self.prompt
                torch.save(params, file_name)
                print(f"===== Best F1: {test_f1:.4f}, Save To: {file_name} =====")
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm

    def detect_subgraph_gang_members_by_svm(self):
        pca = PCA(n_components=2)
        x = []
        y = []
        scaler = MinMaxScaler()
        for i, batch in enumerate(self.train_loader):
            batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
            batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
            with torch.no_grad():
                batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                             batch_uin_acs_text_feat_attention_mask).pooler_output
            batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
            batch_y = batch['uin'].gang_mem.long().detach().cpu().numpy()
            x.append(batch_x.detach().cpu().numpy())
            y.append(batch_y)
        x = np.concatenate(x, axis=0)
        x = scaler.fit_transform(x)
        y = np.concatenate(y, axis=0)
        # x_pca = pca.fit_transform(x)
        self.model.fit(x, y)
        eval_start_time = time.time()
        with torch.no_grad():
            test_x = []
            test_y = []
            for i, eval_batch in enumerate(self.eval_loader):
                eval_batch_uin_acs_text_feat_input_ids = eval_batch['uin'].text_feat_input_ids
                eval_batch_uin_acs_text_feat_attention_mask = eval_batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    eval_batch_uin_acs_text_feat = self.minirbt_model(eval_batch_uin_acs_text_feat_input_ids,
                                                                      eval_batch_uin_acs_text_feat_attention_mask).pooler_output
                eval_batch_x = torch.concat([eval_batch['uin'].x, eval_batch_uin_acs_text_feat], dim=1)
                # eval_batch_x = torch.nn.functional.normalize(eval_batch_x, dim=1)
                # eval_x_pca = pca.fit_transform(eval_batch_x)
                eval_batch_y = eval_batch['uin'].gang_mem.long().detach().cpu().numpy()
                test_x.append(eval_batch_x.detach().cpu().numpy())
                test_y.append(eval_batch_y)
            test_x = np.concatenate(test_x, axis=0)
            test_y = np.concatenate(test_y, axis=0)
            test_x = scaler.transform(test_x)
            # eval_x_pca = pca.transform(test_x)
            pred_y = self.model.predict(test_x)
            prob_y = pred_y.astype(np.float32)
            test_acc = accuracy_score(test_y, pred_y)
            test_f1 = f1_score(test_y, pred_y)
            test_pre = precision_score(test_y, pred_y)
            test_rec = recall_score(test_y, pred_y)
            test_roc_auc = roc_auc_score(test_y, prob_y)
            test_cm = confusion_matrix(test_y, pred_y)
            test_time = time.time() - eval_start_time
            print(f"Test ACC: {test_acc: .4f}, "
                  f"Precision: {test_pre: .4f}, "
                  f"Recall: {test_rec: .4f}, "
                  f"F1: {test_f1: .4f}, "
                  f"ROC-AUC: {test_roc_auc: .4f}, "
                  f"Confusion Matrix: {test_cm.tolist()}, "
                  f"Evaluate Time: {test_time: .4f} s"
                  )
        return test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm, test_time

    def detect_subgraph_gang_members_by_model(self, flag, task, save_model=False):
        if (self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is None) or self.eval_dict[
            "is_supervised"]:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
            if not (self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is not None):
                for param in self.model.parameters():
                    param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_test_cm = np.array([[0, 0], [0, 0]])
        if task != "node":
            best_test_pos_jac = 0
            best_test_neg_jac = 0
        best_test_time = 0
        for epoch in range(1, self.eval_dict["n_epochs"] + 1):
            st = time.time()
            self.classifier.train()
            if self.pooling == 'sag_pool':
                self.sag_pooling.train()
            if self.prompt_type is not None:
                self.prompt.requires_grad_(True)
            if self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is not None:
                for param in self.model.conv1.conv.adapter.parameters():
                    param.requires_grad = True
                for param in self.model.conv2.conv.adapter.parameters():
                    param.requires_grad = True
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
                if self.prompt_type == 'single_token':
                    # batch_x = batch_x + self.prompt.to(self.device)
                    batch_x = torch.mul(batch_x, self.prompt.to(self.device))
                batch['uin'].x = batch_x
                if self.eval_dict['is_single_edge'] and self.conv_type == 'GAT':
                    batch_h = self.model(batch['uin'].x, batch[('uin', 'link', 'uin')].edge_index)
                else:
                    batch_h = self.relation_fit(batch, batch['uin'].x)
                # get edge information
                if batch_h is not None:
                    batch_y = batch['uin'].gang_mem.long()
                    if self.info_type in ['combine_subgraph', 'concat_subgraph']:
                        if self.pooling == 'sag_pool':
                            batch_edge_index, batch_edge_types = self.get_edge_info(batch)
                            h_pool, edge_index_pool, edge_attr_pool, perm, mask, score = (
                                self.sag_pooling(batch_h, batch_edge_index, batch=batch['uin'].batch))
                            batch_h_g = scatter_mean(h_pool, perm, dim=0)
                        else:
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        if self.info_type == 'combine_subgraph':
                            diff_h = expand_batch_h_g - batch_h
                            batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                        else:
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                    pred_y = self.classifier(batch_h)
                    if self.eval_dict['downstream_loss'] == 'node_loss':
                        loss = self.criterion(pred_y, batch_y)
                    else:
                        loss = batch_subgraph_loss_based_cross_entropy(batch['uin'].gang_label.long(),
                                                                       batch['uin'].batch,
                                                                       pred_y[:, 1],
                                                                       batch_y,
                                                                       self.eval_dict['W_p'],
                                                                       self.eval_dict['W_n'],
                                                                       self.eval_dict['w_gang'],
                                                                       self.eval_dict['w_normal']
                                                                       )
                    loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            if task == "node":
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm, test_time = self.evaluate_classifier(
                    flag=flag,
                    epoch=epoch,
                    task="node_cl")
            else:
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm,
                 test_pos_jac, test_neg_jac, test_time) = self.evaluate_classifier(flag=flag,
                                                                                   epoch=epoch,
                                                                                   task="subgraph_gang_cl",
                                                                                   save_results=True,
                                                                                   best_f1=best_test_f1)
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
                if task != "node":
                    best_test_pos_jac = test_pos_jac
                    best_test_neg_jac = test_neg_jac
                best_test_time = test_time
                # save classifier
                if save_model:
                    is_finetune = '_finetune' if self.eval_dict["is_finetune"] else ''
                    is_supervised = '_supervised' if self.eval_dict["is_supervised"] else ''
                    is_sag_pool = f'_sag_pool_{self.eval_dict["sag_top_ratio"]}' if self.info_type == 'sag_pooling' else ''
                    file_name = (f"{self.info_type}_f1_{best_test_f1:.2f}{is_finetune}{is_supervised}"
                                 f"{is_sag_pool}_sample_{flag}.pth") if self.info_type is not None else \
                        f"f1_{best_test_f1:.2f}{is_finetune}{is_supervised}{is_sag_pool}_sample_{flag}.pth"
                    file_dir = os.path.join(self.eval_dict["cls_model_states_path"], self.pt_info, self.ft_info)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    file_name = os.path.join(file_dir, file_name)
                    params = {'cls': self.classifier.state_dict()}
                    if self.info_type == 'sag_pooling':
                        params['pooling'] = self.sag_pooling.state_dict()
                    if self.prompt_type is not None:
                        params['prompt'] = self.prompt
                    torch.save(params, file_name)
                    print(f"===== Best F1: {test_f1:.4f}, Save To: {file_name} =====")
        if task == "node":
            return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm, best_test_time
        else:
            return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm, best_test_pos_jac, best_test_neg_jac, best_test_time

    def compute_target_score(self, anomaly_score, batch_idx, gang_idx):
        N = batch_idx.unique().max() + 1
        target_score = []
        for sub_idx in range(N):
            sub_score = anomaly_score[batch_idx == sub_idx]
            sub_gang_idx = gang_idx[batch_idx == sub_idx]
            gang_score = sub_score[sub_gang_idx == 1]
            if gang_score.shape == torch.Size([0]):
                sub_tgt_score = sub_score.min() * torch.ones_like(sub_score, device=self.device)
                target_score.append(sub_tgt_score)
            else:
                max_gang_score = gang_score.max()
                if max_gang_score < torch.tensor(0.5):
                    max_gang_score = torch.tensor(0.5)
                sub_tgt_score = torch.zeros_like(sub_score, device=self.device)
                sub_tgt_score[sub_gang_idx == 1] = max_gang_score
                non_gang_score = sub_score[sub_gang_idx == 0]
                min_non_gang_score = non_gang_score.min()
                sub_tgt_score[sub_gang_idx == 0] = min_non_gang_score
                target_score.append(sub_tgt_score)
        target_score = torch.concat(target_score, dim=0).to(self.device)
        return target_score

    def optim_subgraph_node_score_by_model(self, flag, task, save_model=False):
        if (self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is None) or self.eval_dict[
            "is_supervised"]:
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
            if not (self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is not None):
                for param in self.model.parameters():
                    param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_test_cm = np.array([[0, 0], [0, 0]])
        if task != "node":
            best_test_pos_jac = 0
            best_test_neg_jac = 0
        best_test_time = 0
        for epoch in range(1, self.eval_dict["n_epochs"] + 1):
            st = time.time()
            self.classifier.train()
            if self.pooling == 'sag_pool':
                self.sag_pooling.train()
            if self.prompt_type is not None:
                self.prompt.requires_grad_(True)
            if self.eval_dict["is_finetune"] and self.eval_dict['adapter_type'] is not None:
                for param in self.model.conv1.conv.adapter.parameters():
                    param.requires_grad = True
                for param in self.model.conv2.conv.adapter.parameters():
                    param.requires_grad = True
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
                if self.prompt_type == 'single_token':
                    # batch_x = batch_x + self.prompt.to(self.device)
                    batch_x = torch.mul(batch_x, self.prompt.to(self.device))
                batch['uin'].x = batch_x
                batch_h = self.relation_fit(batch, batch['uin'].x)
                # get edge information
                if batch_h is not None:
                    if self.info_type in ['combine_subgraph', 'concat_subgraph']:
                        if self.pooling == 'sag_pool':
                            batch_edge_index, batch_edge_types = self.get_edge_info(batch)
                            h_pool, edge_index_pool, edge_attr_pool, perm, mask, score = (
                                self.sag_pooling(batch_h, batch_edge_index, batch=batch['uin'].batch))
                            batch_h_g = scatter_mean(h_pool, perm, dim=0)
                        else:
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        if self.info_type == 'combine_subgraph':
                            diff_h = expand_batch_h_g - batch_h
                            batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                        else:
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                    pred_anomaly_score = self.classifier(batch_h).squeeze()
                    raw_anomaly_score = batch['uin'].score.squeeze()
                    gang_mem_idx = batch['uin'].gang_mem.long()
                    tgt_anomaly_score = self.compute_target_score(raw_anomaly_score, batch['uin'].batch, gang_mem_idx)
                    loss = torch.sqrt(self.criterion(pred_anomaly_score, tgt_anomaly_score))
                    loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            if task == "subgraph":
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm,
                 test_pos_jac, test_neg_jac, test_time) = self.evaluate_classifier(flag=flag,
                                                                                   epoch=epoch,
                                                                                   task="subgraph_score",
                                                                                   save_results=False,
                                                                                   best_f1=best_test_f1)
            else:
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm, test_time) = (
                    self.evaluate_classifier(flag=flag,
                                             epoch=epoch,
                                             task="node_score",
                                             save_results=False,
                                             best_f1=best_test_f1))
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
                if task != "node":
                    best_test_pos_jac = test_pos_jac
                    best_test_neg_jac = test_neg_jac
                best_test_time = test_time
                # save classifier
                if save_model:
                    tp_num = f'tp_{best_test_cm[1, 1]}'
                    is_finetune = '_finetune' if self.eval_dict["is_finetune"] else ''
                    is_supervised = '_supervised' if self.eval_dict["is_supervised"] else ''
                    file_name = (f"score_{self.info_type}_best_{tp_num}{is_finetune}{is_supervised}"
                                 f"_sample_{flag}.pth") if self.info_type is not None else \
                        f"score_best_{tp_num}{is_finetune}{is_supervised}_sample_{flag}.pth"
                    file_dir = os.path.join(self.eval_dict["cls_model_states_path"], self.pt_info, self.ft_info)
                    if not os.path.exists(file_dir):
                        os.makedirs(file_dir)
                    file_name = os.path.join(file_dir, file_name)
                    params = {'cls': self.classifier.state_dict()}
                    if self.info_type == 'sag_pooling':
                        params['pooling'] = self.sag_pooling.state_dict()
                    if self.prompt_type is not None:
                        params['prompt'] = self.prompt
                    torch.save(params, file_name)
                    print(f"===== Best F1: {test_f1:.4f}, Save To: {file_name} =====")
        if task == "node":
            return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm, best_test_time
        else:
            return (best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm,
                    best_test_pos_jac, best_test_neg_jac, best_test_time)

    def generate_gang_by_fraudar(self, anomaly_score, adj):
        G = nx.DiGraph()
        for i in range(anomaly_score.shape[0]):
            G.add_node(i, evil_score=anomaly_score[i].item())
        unique_edge_index = torch.nonzero(adj)
        for i in range(unique_edge_index.shape[1]):
            srt = unique_edge_index[0, i].item()
            dst = unique_edge_index[1, i].item()
            G.add_edge(srt, dst, edge_type_cnt=adj[srt, dst].item())
        best_graph, best_density, best_weight = fraudar(G)
        idx = torch.zeros_like(anomaly_score, dtype=torch.int)
        if len(best_graph.nodes) >= self.control_node_num:
            idx[sorted(best_graph.nodes)] = 1
        return idx

    def evaluate_classifier(self, flag, epoch, task="subgraph", save_results=False, best_f1=0):
        self.classifier.eval()
        self.model.eval()
        if self.pooling == 'sag_pool':
            self.sag_pooling.eval()
        if self.prompt_type is not None:
            prompt = self.prompt.detach()
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
            if epoch == self.eval_dict["n_epochs"]:
                batch_h_list = []
                label_list = []
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
                if self.prompt_type == 'single_token':
                    # batch_x = batch_x + self.prompt.to(self.device)
                    batch_x = torch.mul(batch_x, prompt.to(self.device))
                batch['uin'].x = batch_x
                if self.eval_dict['is_single_edge'] and self.conv_type == 'GAT':
                    batch_h = self.model(batch['uin'].x, batch[('uin', 'link', 'uin')].edge_index)
                else:
                    batch_h = self.relation_fit(batch, batch['uin'].x)
                if epoch == self.eval_dict["n_epochs"]:
                    batch_h_list.append(batch_h)
                    label_list.append(batch['uin'].gang_mem.int())
                if batch_h is not None:
                    if self.pooling == 'sag_pool':
                        batch_edge_index, batch_edge_types = self.get_edge_info(batch)
                        h_pool, edge_index_pool, edge_attr_pool, perm, mask, score = (
                            self.sag_pooling(batch_h, batch_edge_index, batch=batch['uin'].batch))
                        batch_h_g = scatter_mean(h_pool, perm, dim=0)
                    else:
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                    if task == "subgraph_cl":
                        batch_y = batch['uin'].gang_label.int()
                        prob_y = self.classifier(batch_h_g)[:, 1]
                        pred_y = self.classifier(batch_h_g).argmax(dim=1)
                    elif task == "node_cl":
                        batch_y = batch['uin'].gang_mem.int()
                        prob_y = self.classifier(batch_h)[:, 1]
                        pred_y = self.classifier(batch_h).argmax(dim=1)
                    elif task in ["subgraph_score", "node_score"]:
                        if self.info_type in ['combine_subgraph', 'concat_subgraph']:
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            if self.info_type == 'combine_subgraph':
                                diff_h = expand_batch_h_g - batch_h
                                batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                            else:
                                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                                batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                        pred_anomaly_score = self.classifier(batch_h)
                        sub_num = batch['uin'].batch.unique().max() + 1
                        edge_index = torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types],
                                                  dim=1)
                        batch_adj = to_dense_adj(edge_index, max_num_nodes=batch['uin'].num_nodes).squeeze()
                        pred_gang_member = []
                        for sub_i in range(sub_num):
                            start_node, end_node = batch['uin'].ptr[sub_i].item(), batch['uin'].ptr[sub_i + 1].item()
                            sub_pred_gang_member = self.generate_gang_by_fraudar(
                                pred_anomaly_score[batch['uin'].batch == sub_i],
                                batch_adj[start_node:end_node, start_node:end_node])
                            pred_gang_member.append(sub_pred_gang_member)
                        pred_gang_member = torch.concat(pred_gang_member, dim=0)
                        true_gang_member = batch['uin'].gang_mem.int()
                        jaccard_coeff, true_idx, pred_idx = self.jaccard_batch(true_gang_member,
                                                                               pred_gang_member,
                                                                               batch['uin'].batch)
                        if task == "node_score":
                            batch_y = batch['uin'].gang_mem.int()
                            pred_y = pred_gang_member
                            prob_y = pred_gang_member.type(torch.float)
                        else:
                            batch_y = batch['uin'].gang_label
                            prob_y = torch.tensor(jaccard_coeff, device=self.device)
                            pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                            pred_y[prob_y >= self.eval_dict["subgraph_tp_threshold"]] = 1
                        if save_results:
                            jaccard_list.extend(jaccard_coeff)
                            true_idx_list.extend(true_idx)
                            pred_idx_list.extend(pred_idx)
                            subgraph_y_list.extend(batch['uin'].y.detach().cpu().tolist())
                            subgraph_uin_list.extend(batch['uin'].nodeid2uin_map)
                    else:
                        batch_y = batch['uin'].gang_label.int()
                        true_gang_member = batch['uin'].gang_mem.int()
                        if self.info_type in ['combine_subgraph', 'concat_subgraph']:
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            if self.info_type == 'combine_subgraph':
                                diff_h = expand_batch_h_g - batch_h
                                batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                            else:
                                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                                batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                        pred_gang_member = self.classifier(batch_h).argmax(dim=1)
                        jaccard_coeff, true_idx, pred_idx = self.jaccard_batch(true_gang_member, pred_gang_member,
                                                                               batch['uin'].batch)
                        prob_y = torch.tensor(jaccard_coeff, device=self.device)
                        pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                        pred_y[prob_y >= self.eval_dict["subgraph_tp_threshold"]] = 1
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
            eval_time = time.time() - start_time
            if task in ["subgraph_cl", "node_cl", "node_score"]:
                print(f"Test ACC: {acc: .4f}, "
                      f"Precision: {pre: .4f}, "
                      f"Recall: {rec: .4f}, "
                      f"F1: {f1: .4f}, "
                      f"ROC-AUC: {roc_auc: .4f}, "
                      f"Confusion Matrix: {cm.tolist()}, "
                      f"Evaluate Time: {eval_time: .4f} s")
                if f1 > best_f1 and save_results:
                    file_name = os.path.join(self.pred_save_path, self.pt_info, self.ft_info)
                    if not os.path.exists(file_name):
                        os.makedirs(file_name)
                    if self.eval_dict['is_finetune']:
                        file_tag = "finetune_score_" if "score" in task else "finetune_"
                    else:
                        if self.eval_dict['is_supervised']:
                            file_tag = "supervised_score_" if "score" in task else "supervised_"
                        else:
                            file_tag = "score_" if "score" in task else ""
                    info = self.info_type + "_" if self.info_type is not None else ""
                    file_name = os.path.join(file_name, file_tag + f"{info}best_f1_sample_{flag}.csv")
                    self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                          subgraph_y_list, jaccard_list, file_name)
                return acc, pre, rec, f1, roc_auc, cm, eval_time
            else:
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

                    if self.eval_dict['is_finetune']:
                        file_tag = "finetune_score_" if "score" in task else "finetune_"
                    else:
                        if self.eval_dict['is_supervised']:
                            file_tag = "supervised_score_" if "score" in task else "supervised_"
                        else:
                            file_tag = "score_" if "score" in task else ""
                    info = self.info_type if self.info_type is not None else ""
                    file_name = os.path.join(file_name, file_tag + f"{info}_best_f1_sample_{flag}.csv")
                    self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                          subgraph_y_list, jaccard_list, file_name)
                return acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, eval_time

    def detect_subgraph_gang_members_by_fraudar(self, flag, save_results=False):
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
                true_y = pos_batch['uin'].gang_mem.int()
                pred_y = pos_batch['uin'].idx
                jaccard_coefficient, true_idx, pred_idx = self.jaccard_batch(true_y, pred_y,
                                                                             pos_batch['uin'].batch)
                true_subgraph_y = pos_batch['uin'].gang_label.int()
                prob_subgraph_y = torch.tensor(jaccard_coefficient)
                pred_subgraph_y = torch.zeros(prob_subgraph_y.shape[0])
                pred_subgraph_y[prob_subgraph_y >= self.eval_dict["subgraph_tp_threshold"]] = 1
                true_y_list.append(true_subgraph_y)
                pred_y_list.append(pred_subgraph_y)
                prob_y_list.append(prob_subgraph_y)
                jaccard_list.extend(jaccard_coefficient)
                true_idx_list.extend(true_idx)
                pred_idx_list.extend(pred_idx)
                subgraph_y_list.extend(pos_batch['uin'].y.tolist())
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
            save_path = os.path.join(self.pred_save_path, 'fraudar_full')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = os.path.join(save_path, f'fraudar_{flag}.csv')
            self.save_output_file(true_idx_list, pred_idx_list, subgraph_uin_list,
                                  subgraph_y_list, jaccard_list, file_name)
            return acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, run_time
        else:
            return acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, run_time
