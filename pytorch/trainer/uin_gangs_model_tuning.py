import logging
import os
import time
import random

import pandas as pd
import torch

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
from collections import OrderedDict
import torch.utils.data as Data
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
from transformers import BertModel
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.models.graph_transformer import GraphTransformer, HeteroGraphTransformer
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, \
    multilabel_confusion_matrix


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
        if self.conv_type == 'RGCN':
            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=args_dict['num_relations'])
            self.edge_types = {('uin', 'ipv6', 'uin'): 0, ('uin', 'wifi', 'uin'): 1, ('uin', 'room', 'uin'): 2,
                               ('uin', 'friend', 'uin'): 3, ('uin', 'idcardid', 'uin'): 4, ('uin', 'device', 'uin'): 5,
                               ('uin', 'payee', 'uin'): 6, ('uin', 'payer', 'uin'): 7, ('uin', 'bankcard', 'uin'): 8,
                               ('uin', 'download_app', 'uin'): 9}
        elif self.conv_type == 'HGT':
            self.metadata = (['uin'], [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                                       ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                                       ('uin', 'payee', 'uin'), ('uin', 'payer', 'uin'), ('uin', 'bankcard', 'uin'),
                                       ('uin', 'download_app', 'uin')])
            self.model = HeteroGraphTransformer(
                in_channels=args_dict['input_dim'],
                hidden_channels=args_dict['hidden_dim'],
                out_channels=args_dict['output_dim'],
                metadata=self.metadata,
                heads=args_dict['num_heads']
            )

        lr = self.eval_dict["lr"]
        self.control_node_num = self.eval_dict["filter_node_num"]
        sampling_type = self.eval_dict["sampling"]
        self.data_tag = self.eval_dict["data_tag"]
        self.device_tag = self.eval_dict["device_tag"]
        self.task_type = self.eval_dict["task_type"]
        self.save_model_path = os.path.join(self.eval_dict["model_states_path"],
                                            f"{self.data_tag}{self.conv_type}_sample_{sampling_type}_"
                                            f"filter_{self.control_node_num}_lr_{str(lr)}_"
                                            f"{self.eval_dict['lr_scheduler']}{self.device_tag}")
        if not self.eval_dict["is_supervised"]:
            if self.eval_dict["eval_epoch"] != 0:
                epoch_num = self.eval_dict["eval_epoch"]
                if self.task_type == 'subgraph':
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
                else:
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_model_epoch_{epoch_num}.pth")
            else:
                if self.task_type == 'subgraph':
                    file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
                else:
                    file_name = os.path.join(self.save_model_path,
                                             f"uin_gangs_{self.conv_type}_{self.task_type}_model_best_loss.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            if self.device_tag == '_GPU2':
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

        if self.eval_dict["evaluate_task"] == 'node_classification':
            self.is_prompt = False
            self.info_type = None
            self.train_data = UinGangsDataIterablePyG(self.eval_dict, self.eval_dict["node_train_data_path"])
            self.train_loader = Data.DataLoader(self.train_data,
                                                batch_size=self.eval_dict["batch_size"],
                                                num_workers=self.eval_dict["num_workers"],
                                                collate_fn=self.train_data.collate_fn)
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict, self.eval_dict["node_test_data_path"])
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.collate_fn)
            self.classifier = torch.nn.Sequential(torch.nn.Linear(args_dict['output_dim'], args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], args_dict['node_types']),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss()
            params = [{'params': self.classifier.parameters(), 'lr': self.eval_dict['cls_lr']}]
            if self.eval_dict['evaluate_task_tuning']:
                self.model.train()
                for param in self.model.parameters():
                    param.requires_grad = True
                params.append({'params': self.model.parameters(), 'lr': self.eval_dict['lr']})
            else:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
            self.cls_optimizer = torch.optim.Adam(params)
        elif self.eval_dict["evaluate_task"] in ['subgraph', 'subgraph_gang_detection']:
            self.train_data = UinGangsDataIterablePyG(self.eval_dict, self.eval_dict["train_data_path"])
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict, self.eval_dict["test_data_path"])
            self.train_loader = Data.DataLoader(self.train_data,
                                                batch_size=self.eval_dict["batch_size"],
                                                num_workers=self.eval_dict["num_workers"],
                                                collate_fn=self.train_data.collate_fn)
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.collate_fn)
            self.info_type = self.eval_dict["info_insertion_type"]
            params = []
            self.is_prompt = True if self.eval_dict["info_insertion_type"] in ['add_prompt'] else False
            if self.is_prompt:
                self.initial_data = UinGangsDataIterablePyG(self.eval_dict,
                                                            self.eval_dict["prompt_initial_data_path"],
                                                            is_shuffle=False)
                self.initial_loader = Data.DataLoader(self.initial_data,
                                                      batch_size=20,
                                                      num_workers=2,
                                                      collate_fn=self.initial_data.collate_fn)
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
                    # normalized
                    gang_batch['uin'].x = torch.nn.functional.normalize(gang_batch['uin'].x, dim=1)
                    gang_x = gang_batch['uin'].x[gang_mems]
                    batch_batch = batch['uin'].batch[gang_mems]
                    gang_subgraphs.append(scatter_mean(gang_x, batch_batch, dim=0))
                gang_subgraph_mean = torch.concat(gang_subgraphs, dim=0).mean(dim=0)
                self.prompt = torch.nn.Parameter(gang_subgraph_mean, requires_grad=True).to(self.device)
                params.append({'params': self.prompt, 'lr': self.eval_dict['prompt_lr']})
            if self.info_type in ['combine_subgraph', 'combine_difference']:
                cls_input = args_dict['output_dim'] * 2
            else:
                cls_input = args_dict['output_dim']
            self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            params.append({'params': self.classifier.parameters(), 'lr': self.eval_dict['cls_lr']})
            self.cls_optimizer = torch.optim.Adam(params)
            self.loss_weight = self.eval_dict['loss_weight'].split(' ')
            self.loss_weight = [float(item) for item in self.loss_weight]
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.loss_weight, device=self.device))
        elif self.eval_dict["evaluate_task"] in ['subgraph_prompt_tuning']:
            self.initial_data = UinGangsDataIterablePyG(self.eval_dict,
                                                        self.eval_dict["prompt_initial_data_path"],
                                                        is_shuffle=False)
            self.tune_data = UinGangsDataIterablePyG(self.eval_dict,
                                                     self.eval_dict["prompt_tuning_data_path"],
                                                     is_shuffle=False)
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict,
                                                     self.eval_dict["prompt_evaluating_data_path"])
            self.initial_loader = Data.DataLoader(self.initial_data,
                                                  batch_size=20,
                                                  num_workers=2,
                                                  collate_fn=self.initial_data.collate_fn)
            self.tune_loader = Data.DataLoader(self.tune_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.tune_data.collate_fn)
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.collate_fn)
            self.prompt_type = self.eval_dict["prompt_insertion_type"]
            self.is_prompt = True if self.prompt_type not in [None, 'concat_subgraph'] else False
            params = []
            # Initialize Prompt
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
                    # normalized
                    gang_batch['uin'].x = torch.nn.functional.normalize(gang_batch['uin'].x, dim=1)
                    if self.conv_type == 'RGCN':
                        batch_h = self.rgcn_fit(gang_batch, gang_batch['uin'].x)
                    elif self.conv_type == 'HGT':
                        batch_h = self.hetero_fit(gang_batch.x_dict, gang_batch.edge_index_dict)
                    try:
                        gang_h = batch_h[gang_mems]
                    except Exception as e:
                        print(f"Prompt Initialized Error : <{e}>")
                        continue
                    else:
                        batch_batch = batch['uin'].batch[gang_mems]
                        gang_subgraphs.append(scatter_mean(gang_h, batch_batch, dim=0))
                gang_subgraph_mean = torch.concat(gang_subgraphs, dim=0).mean(dim=0)
                self.prompt = torch.nn.Parameter(gang_subgraph_mean, requires_grad=True).to(self.device)
                params.append({'params': self.prompt, 'lr': self.eval_dict['prompt_lr']})
            if self.prompt_type in ['concat_subgraph', 'concat_prompt']:
                cls_input = args_dict['output_dim'] * 2
            elif self.prompt_type == 'concat_subgraph_prompt':
                cls_input = args_dict['output_dim'] * 3
            else:
                cls_input = args_dict['output_dim']
            self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            params.append({'params': self.classifier.parameters(), 'lr': self.eval_dict['cls_lr']})
            self.cls_optimizer = torch.optim.Adam(params)
            self.loss_weight = self.eval_dict['loss_weight'].split(' ')
            self.loss_weight = [float(item) for item in self.loss_weight]
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.loss_weight).to(self.device))
        elif self.eval_dict["evaluate_task"] in ['inference_gang_members', 'inference_gang_members_by_fraudar']:
            self.eval_data = UinGangsDataIterablePyG(self.eval_dict,
                                                     self.eval_dict["prompt_evaluating_data_path"])
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.pos_collate_fn_for_fraudar)
            self.info_type = self.eval_dict["info_insertion_type"]
            if self.info_type in ['combine_subgraph']:
                cls_input = args_dict['output_dim'] * 2
            else:
                cls_input = args_dict['output_dim']
            self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
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
                print(f"Extract Subgraph Error: <{e}>")
                del new_batch[edge_type]
            else:
                # no such type of edges
                if edge_index.shape[1] != 0:
                    new_batch[edge_type].edge_index = edge_index
                else:
                    del new_batch[edge_type]
        return new_batch

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

    def evaluate_node_classification(self):
        best_loss = self.eval_dict["best_loss"]
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_f1 = 0
        best_roc_auc = 0
        best_cm = np.array([[0, 0], [0, 0]])
        for epoch in range(self.eval_dict["n_epochs"]):
            epoch_loss = []
            st = time.time()
            self.classifier.train()
            if self.eval_dict['evaluate_task_tuning']:
                self.model.train()
            for batch in self.train_loader:
                self.cls_optimizer.zero_grad()
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                node_h = batch_h[batch['uin'].ptr[:-1]]
                y_prob = self.classifier(node_h)
                y_true = batch['uin'].y.long()
                loss = self.criterion(y_prob, y_true)
                loss.backward()
                self.cls_optimizer.step()
                epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Cross entropy loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            if current_loss < best_loss:
                best_loss = current_loss
                print(f"...Test Node Classification Task in Best Loss...")
                acc, pre, rec, f1, roc_auc, cm = self.evaluate_classifier(task="node")
                if acc > best_acc:
                    best_acc = acc
                    best_pre = pre
                    best_rec = rec
                    best_f1 = f1
                    best_roc_auc = roc_auc
                    best_cm = cm
            else:
                if epoch % 5 == 0:
                    print(f"...Test Node Classification Task in Epoch={epoch}...")
                    acc, pre, rec, f1, roc_auc, cm = self.evaluate_classifier(task="node")
                    if acc > best_acc:
                        best_acc = acc
                        best_pre = pre
                        best_rec = rec
                        best_f1 = f1
                        best_roc_auc = roc_auc
                        best_cm = cm
        print(f"Best Test Metrics: \n"
              f"ACC: {best_acc:.4f}, Precision: {best_pre:.4f}, "
              f"Recall: {best_rec:.4f}, F1-Score: {best_f1:.4f}, "
              f"ROC-AUC: {best_roc_auc:.4f}, Confusion Matrix: {best_cm.tolist()}")

    def evaluate_labelled_subgraph_predict(self):
        if not self.eval_dict["is_supervised"]:
            # Have load pretrained model in the initialization
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Don't load pretrained model
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_loss = self.eval_dict["best_loss"]
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_test_cm = np.array([[0, 0], [0, 0]])
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                # get edge information
                if batch_h is not None:
                    if self.eval_dict["is_weighted_subgraph"]:
                        exp_score = torch.exp(batch['uin'].score)
                        batch_h = batch_h * exp_score
                    batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                    batch_y = batch['uin'].gang_label.long()
                    pred_y = self.classifier(batch_h_g)
                    cls_loss = self.criterion(pred_y, batch_y)
                    cls_loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(cls_loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                file_name = self.eval_dict["cls_model_states_path"] + f"{self.conv_type}_best_loss.pth"
                torch.save(self.classifier.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            print(f"Epoch {epoch}, Cross entropy loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm = self.evaluate_classifier()
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm

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
        penalty = torch.stack(penalty_list).to(self.device).mean()
        return penalty

    def detect_subgraph_gang_members(self):
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
            epoch_loss = []
            if self.is_prompt:
                self.prompt.requires_grad = True
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
                if self.conv_type == 'RGCN':
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
                    elif self.info_type == 'combine_difference':
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                        diff_h = expand_batch_h_g - batch_h
                        batch_h = torch.concat([batch_h, diff_h], dim=1)
                    pred_y = self.classifier(batch_h)
                    cls_loss = self.criterion(pred_y, batch_y)
                    penalty_loss = self.compute_penalty_loss(batch_y,
                                                             pred_y.argmax(dim=1),
                                                             batch['uin'].gang_label.detach().cpu().int().tolist(),
                                                             batch['uin'].batch)
                    loss = cls_loss + penalty_loss
                    loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm = self.evaluate_classifier(task="detect_gang")
            if test_f1 > best_test_f1:
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cm = test_cm
                prefix = f'{self.conv_type}_{self.task_type}_{self.eval_dict["eval_epoch"]}'
                tp_tf = f'tn_{str(best_test_cm[0, 0])}_tp_{str(best_test_cm[1, 1])}'
                file_name = f"{self.info_type}_{prefix}_best_f1_{tp_tf}.pth" if self.info_type is not None \
                    else f"{prefix}_best_f1_{tp_tf}.pth"
                file_name = self.eval_dict["cls_model_states_path"] + file_name
                torch.save(self.classifier.state_dict(), file_name)
                print(f"Now best f1: {test_f1:.4f}, save model to {file_name}")
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cm

    def evaluate_classifier(self, task="subgraph"):
        self.classifier.eval()
        self.model.eval()
        if self.is_prompt:
            self.prompt.requires_grad = False
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                if batch_h is not None:
                    if self.eval_dict["is_weighted_subgraph"]:
                        exp_score = torch.exp(batch['uin'].score)
                        batch_h = batch_h * exp_score
                    if task == "subgraph":
                        batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                        batch_y = batch['uin'].gang_label
                        prob_y = self.classifier(batch_h_g)[:, 1]
                        pred_y = self.classifier(batch_h_g).argmax(dim=1)
                    elif task == "detect_gang":
                        batch_y = batch['uin'].gang_label.long()
                        true_gang_member = batch['uin'].gang_mem.int()
                        if self.info_type == 'combine_subgraph':
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            diff_h = expand_batch_h_g - batch_h
                            batch_h = torch.concat([expand_batch_h_g, diff_h], dim=1)
                        elif self.info_type == 'combine_difference':
                            batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            diff_h = expand_batch_h_g - batch_h
                            batch_h = torch.concat([batch_h, diff_h], dim=1)
                        pred_gang_member = self.classifier(batch_h).argmax(dim=1)
                        jaccard_coeff, _, _ = self.jaccard_batch(true_gang_member, pred_gang_member, batch['uin'].batch)
                        prob_y = torch.tensor(jaccard_coeff, device=self.device)
                        pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                        pred_y[prob_y >= self.eval_dict["threshold"]] = 1
                    else:
                        node_h = batch_h[batch['uin'].ptr[:-1]]
                        batch_y = batch['uin'].y
                        prob_y = self.classifier(node_h)
                        pred_y = prob_y.argmax(dim=1)
                    true_y = batch_y.detach().cpu()
                    prob_y = prob_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    true_y_list.append(true_y)
                    pred_y_list.append(pred_y)
                    prob_y_list.append(prob_y)
            true_y_list = torch.concat(true_y_list, dim=0).numpy()
            prob_y_list = torch.concat(prob_y_list, dim=0).numpy()
            pred_y_list = torch.concat(pred_y_list, dim=0).numpy()
            if task in ["subgraph", "detect_gang"]:
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
                return acc, pre, rec, f1, roc_auc, cm
            else:
                average = "micro"
                multi_class = "ovr"
                acc = accuracy_score(true_y_list, pred_y_list)
                f1 = f1_score(true_y_list, pred_y_list, average=average)
                pre = precision_score(true_y_list, pred_y_list, average=average)
                rec = recall_score(true_y_list, pred_y_list, average=average)
                roc_auc = roc_auc_score(true_y_list, prob_y_list, multi_class=multi_class)
                cm = multilabel_confusion_matrix(true_y_list, pred_y_list)
                print(f"Test ACC: {acc: .4f}, "
                      f"Precision: {pre: .4f}, "
                      f"Recall: {rec: .4f}, "
                      f"F1: {f1: .4f}, "
                      f"ROC-AUC: {roc_auc: .4f}, "
                      f"Confusion Matrix: {cm.tolist()}")
                return acc, pre, rec, f1, roc_auc, cm

    def insert_prompt(self, batch_h, batch, prompt):
        if self.prompt_type == 'concat_prompt':
            prompt_batch_h = torch.concat([batch_h, prompt.repeat(batch_h.shape[0], 1)], dim=1)
        else:
            if self.eval_dict["is_weighted_subgraph"]:
                exp_score = torch.exp(batch['uin'].score)
                new_batch_h = batch_h * exp_score
                batch_h_g = scatter_mean(new_batch_h, batch['uin'].batch, dim=0)
            else:
                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
            if self.prompt_type == 'concat_subgraph_prompt':
                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                prompt_batch_h = torch.concat(
                    [batch_h, expand_batch_h_g, prompt.repeat(batch_h.shape[0], 1)], dim=1)
        return prompt_batch_h

    def evaluate_labelled_subgraph_prompt_tuning(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        best_loss = self.eval_dict["best_loss"]
        # Prompt Tuning
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_test_cfm = np.array([])
        best_test_jac = 0
        for epoch in range(1, self.eval_dict["n_epochs"] + 1):
            epoch_loss = []
            st = time.time()
            self.classifier.train()
            if self.is_prompt:
                self.prompt.requires_grad = True
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                # get edge information
                if batch_h is not None:
                    if self.is_prompt:
                        prompt_batch_h = self.insert_prompt(batch_h, batch, self.prompt)
                        pred_y = self.classifier(prompt_batch_h)
                    else:
                        if self.prompt_type == 'concat_subgraph':
                            if self.eval_dict["is_weighted_subgraph"]:
                                exp_score = torch.exp(batch['uin'].score)
                                new_batch_h = batch_h * exp_score
                                batch_h_g = scatter_mean(new_batch_h, batch['uin'].batch, dim=0)
                            else:
                                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                        pred_y = self.classifier(batch_h)
                    batch_y = batch['uin'].gang_mem.long()
                    cls_loss = self.criterion(pred_y, batch_y)
                    cls_loss.backward()
                    self.cls_optimizer.step()
                    epoch_loss.append(cls_loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                file_name = f"{self.prompt_type}_{self.conv_type}_best_loss.pth" if self.is_prompt else f"no_prompt_{self.conv_type}_best_loss.pth"
                file_name = self.eval_dict["cls_model_states_path"] + file_name
                torch.save(self.classifier.state_dict(), file_name)
                print(f"Now best loss: {best_loss:.4f}, save model to {file_name}")
            print(f"Epoch {epoch}, Cross Entropy Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            # Prompt Evaluation
            if self.is_prompt:
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm, test_jac = self.evaluate_prompt_classifier(
                    prompt=self.prompt)
            else:
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm, test_jac = self.evaluate_prompt_classifier()
            if test_acc > best_test_acc:
                print(f"Current Best Test ACC: {test_acc: .4f}, "
                      f"Precision: {test_pre: .4f}, "
                      f"Recall: {test_rec: .4f}, "
                      f"F1: {test_f1: .4f}, "
                      f"ROC-AUC: {test_roc_auc: .4f}, "
                      f"Confusion Matrix: {test_cfm.tolist()}, "
                      f"Jaccard Coefficient: {test_jac: .4f}"
                      )
                best_test_acc = test_acc
                best_test_pre = test_pre
                best_test_rec = test_rec
                best_test_f1 = test_f1
                best_test_roc_auc = test_roc_auc
                best_test_cfm = test_cfm
                best_test_jac = test_jac
        return best_test_acc, best_test_pre, best_test_rec, best_test_f1, best_test_roc_auc, best_test_cfm, best_test_jac

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
            y_true_i_idx = y_true[subgraph_i_idx]
            y_pred_i_idx = y_pred[subgraph_i_idx]
            y_true_idx = (y_true_i_idx == 1).nonzero().squeeze().detach().cpu().tolist()
            y_pred_idx = (y_pred_i_idx == 1).nonzero().squeeze().detach().cpu().tolist()
            true_idx_list.append(str(y_true_idx))
            pred_idx_list.append(str(y_pred_idx))
            if torch.sum(y_true_i_idx).detach().cpu().item() == 0:
                fp_items = (y_pred_i_idx == 1).nonzero().squeeze().detach().cpu().tolist()
                if type(fp_items) == list:
                    if len(fp_items) > 2:
                        jac = torch.tensor(1.0)
                    else:
                        jac = torch.tensor(0.0)
                else:
                    jac = torch.tensor(0.0)
                # y_true_i_idx[:] = 1
                # y_pred_i_idx[y_pred_i_idx == 1] = -1
                # y_pred_i_idx[y_pred_i_idx == 0] = 1
                # y_pred_i_idx[y_pred_i_idx == -1] = 0
                # jac = 1 - self.jaccard(y_true_i_idx, y_pred_i_idx)
            else:
                jac = self.jaccard(y_true_i_idx, y_pred_i_idx)
            jaccard_list.append(jac.detach().cpu().item())
        return jaccard_list, true_idx_list, pred_idx_list

    def subgraph_embedding_expand(self, subgraph_embedding, expand_sizes):
        n = subgraph_embedding.shape[0]
        subgraph_embedding_list = []
        for i in range(n):
            repeat_times = (expand_sizes[i + 1] - expand_sizes[i]).detach().cpu().item()
            subgraph_embedding_list.append(subgraph_embedding[i, :].repeat(repeat_times, 1))
        return torch.concat(subgraph_embedding_list, dim=0)

    def evaluate_prompt_classifier(self, prompt=None):
        self.classifier.eval()
        if prompt is not None and self.prompt_type not in ['concat_prompted_subgraph']:
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                else:
                    batch_h = self.hetero_fit(batch.x_dict, batch.edge_index_dict)
                if batch_h is not None:
                    if prompt is not None:
                        prompt_batch_h = self.insert_prompt(batch_h, batch, prompt)
                        prob_y = self.classifier(prompt_batch_h)[:, 1]
                        pred_y = self.classifier(prompt_batch_h).argmax(dim=1)
                    else:
                        if self.prompt_type == 'concat_subgraph':
                            if self.eval_dict["is_weighted_subgraph"]:
                                exp_score = torch.exp(batch['uin'].score)
                                new_batch_h = batch_h * exp_score
                                batch_h_g = scatter_mean(new_batch_h, batch['uin'].batch, dim=0)
                            else:
                                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = torch.concat([batch_h, expand_batch_h_g], dim=1)
                        prob_y = self.classifier(batch_h)[:, 1]
                        pred_y = self.classifier(batch_h).argmax(dim=1)
                    batch_y = batch['uin'].gang_mem
                    true_y = batch_y.detach().cpu()
                    prob_y = prob_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    true_y_list.append(true_y)
                    pred_y_list.append(pred_y)
                    prob_y_list.append(prob_y)
                    jaccard_coefficient, true_idx, pred_idx = self.jaccard_batch(true_y.int(), pred_y,
                                                                                 batch['uin'].batch)
                    jaccard_list.extend(jaccard_coefficient)
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
            return acc, pre, rec, f1, roc_auc, cm, jaccard

    def inference_gang_members_by_fraudar(self, save_results=False):
        start_time = time.time()
        jaccard_list = []
        true_idx_list = []
        pred_idx_list = []
        true_uin_list = []
        pred_uin_list = []
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
            print(f"Fraudar ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Pos Jaccard Coefficient: {pos_jaccard: .4f}, "
                  f"Neg Jaccard Coefficient: {neg_jaccard: .4f}, "
                  f"Time: {time.time() - start_time: .4f} s"
                  )
        if save_results:
            # save inference results
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
            if torch.cuda.is_available():
                output_file_dir = '/mnt/cephfs'
            else:
                output_file_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common'
            file_name = '/jiujiuchen/projects/mmgog_long_term_sequence_model/data/fraudar_y.csv'
            df.to_csv(output_file_dir + file_name, index=False)
            print(f"Save to file: {output_file_dir + file_name}")

    def inference_gang_members(self, save_results=False):
        start_time = time.time()
        prefix = f'{self.conv_type}_{self.task_type}_{self.eval_dict["eval_epoch"]}'
        file_name = f"{self.info_type}_{prefix}_best_f1_tn_67_tp_44.pth" if self.info_type is not None \
            else f"{prefix}_best_f1.pth"
        print(f"Load Classifier: {self.eval_dict['cls_model_states_path'] + file_name}")
        model_weight = torch.load(self.eval_dict['cls_model_states_path'] + file_name, map_location=self.device)
        self.classifier.load_state_dict(model_weight)
        self.classifier.eval()
        jaccard_list = []
        if save_results:
            true_idx_list = []
            pred_idx_list = []
            true_uin_list = []
            pred_uin_list = []
            subgraph_y_list = []
            subgraph_uin_list = []
        else:
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
                if self.conv_type == 'RGCN':
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
                    pred_y = prob_y.argmax(dim=1)
                    batch_y = batch['uin'].gang_mem.int()
                    true_y = batch_y.detach().cpu()
                    pred_y = pred_y.detach().cpu()
                    jaccard_coeff, true_idx, pred_idx = self.jaccard_batch(true_y, pred_y,
                                                                           batch['uin'].batch)
                    jaccard_list.extend(jaccard_coeff)
                    if save_results:
                        true_idx_list.extend(true_idx)
                        pred_idx_list.extend(pred_idx)
                        subgraph_y_list.extend(batch['uin'].y.detach().cpu().tolist())
                        subgraph_uin_list.extend(batch['uin'].nodeid2uin_map)
                    else:
                        prob_y = torch.tensor(jaccard_coeff, device=self.device)
                        pred_y = torch.zeros(prob_y.shape[0], device=self.device)
                        pred_y[prob_y >= self.eval_dict["threshold"]] = 1
                        true_y_list.append(batch['uin'].gang_label.detach().cpu().long())
                        prob_y_list.append(prob_y.detach().cpu())
                        pred_y_list.append(pred_y.detach().cpu())
            # compute metrics
            if save_results:
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
                if torch.cuda.is_available():
                    output_file_dir = '/mnt/cephfs'
                else:
                    output_file_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common'
                file_name = f'/jiujiuchen/projects/mmgog_long_term_sequence_model/data/{prefix}_{self.eval_dict["filter_node_num"]}_y.csv'
                df.to_csv(output_file_dir + file_name, index=False)
                print(f"Save to file: {output_file_dir + file_name}")
            else:
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
                print(f"{self.conv_type} ACC: {acc: .4f}, "
                      f"Precision: {pre: .4f}, "
                      f"Recall: {rec: .4f}, "
                      f"F1: {f1: .4f}, "
                      f"ROC-AUC: {roc_auc: .4f}, "
                      f"Confusion Matrix: {cm.tolist()}, "
                      f"Pos Jaccard Coefficient: {pos_jaccard: .4f}, "
                      f"Neg Jaccard Coefficient: {neg_jaccard: .4f}, "
                      f"Time: {time.time() - start_time: .4f} s"
                      )
