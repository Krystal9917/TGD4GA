import logging
import os
import time
import random
import torch

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import numpy as np
from collections import OrderedDict
import torch.utils.data as Data
from torch_geometric.utils import subgraph, to_dense_adj
from torch_scatter import scatter_mean
from transformers import BertModel
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.models.graph_transformer import GraphTransformer, HeteroGraphTransformer
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, multilabel_confusion_matrix


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
        control_node_num = self.eval_dict["filter_node_num"]
        sampling_type = self.eval_dict["sampling"]
        self.data_tag = self.eval_dict["data_tag"]
        self.device_tag = self.eval_dict["device_tag"]
        self.save_model_path = os.path.join(self.eval_dict["model_states_path"],
                                            f"{self.data_tag}{self.conv_type}_sample_{sampling_type}_filter_{control_node_num}_lr_{str(lr)}{self.device_tag}")
        if not self.eval_dict["is_supervised"]:
            if self.eval_dict["eval_epoch"] != 0:
                epoch_num = self.eval_dict["eval_epoch"]
                file_name = os.path.join(self.save_model_path,
                                         f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
            else:
                file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_best_loss.pth")
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
            self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args_dict['lr'])
        elif self.eval_dict["evaluate_task"] == 'subgraph':
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
            self.classifier = torch.nn.Sequential(torch.nn.Linear(args_dict['output_dim'], args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.cls_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=5e-4)
        elif self.eval_dict["evaluate_task"] == 'subgraph_prompt_tuning':
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
                                               batch_size=20,
                                               num_workers=2,
                                               collate_fn=self.tune_data.collate_fn)
            self.eval_loader = Data.DataLoader(self.eval_data,
                                               batch_size=self.eval_dict["batch_size"],
                                               num_workers=self.eval_dict["num_workers"],
                                               collate_fn=self.eval_data.collate_fn)
            self.prompt_type = self.eval_dict["prompt_insertion_type"]
            self.is_prompt = True if self.prompt_type not in [None, 'concat_subgraph', 'concat_adj', 'multiply_adj',
                                                              'node_subtract_subgraph'] else False
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
                params.append({'params': self.prompt, 'lr': 1e-4})
            if self.prompt_type in ['concat_prompt', 'concat_subgraph',
                                    'concat_subgraph_plus_prompt', 'concat_subgraph_proj_prompt',
                                    'weighted_addition_concat_prompt', 'linear_concat_subgraph_concat_prompt']:
                cls_input = args_dict['output_dim'] * 2
                if self.prompt_type == 'linear_concat_subgraph_concat_prompt':
                    self.combine_func = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                            torch.nn.Linear(args_dict['hidden_dim'],
                                                                            args_dict['output_dim'])).to(self.device)
                    params.append({'params': self.combine_func.parameters(), 'lr': 5e-4})
            elif self.prompt_type == 'concat_adj':
                cls_input = args_dict['output_dim'] + int(args_dict['hidden_dim'] / 16)
            elif self.prompt_type == 'concat_subgraph_prompt':
                cls_input = args_dict['output_dim'] * 3
            else:
                cls_input = args_dict['output_dim']
            self.classifier = torch.nn.Sequential(torch.nn.Linear(cls_input, args_dict['hidden_dim']),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(args_dict['hidden_dim'], 2),
                                                  torch.nn.Softmax(dim=1))
            self.classifier.to(self.device)
            params.append({'params': self.classifier.parameters(), 'lr': 1e-3})
            self.cls_optimizer = torch.optim.Adam(params)
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(self.device))
        # Set seed for whole environment
        self.setup_seed()

    def setup_seed(self):
        torch.manual_seed(self.eval_dict["seed"])
        torch.cuda.manual_seed(self.eval_dict["seed"])
        torch.cuda.manual_seed_all(self.eval_dict["seed"])
        np.random.seed(self.eval_dict["seed"])
        random.seed(self.eval_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def cosine_similarity(self, h1, h2):
        h1_abs = h1.norm(dim=1)
        h2_abs = h2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', h1, h2) / torch.einsum('i,j->ij', h1_abs, h2_abs)
        return sim_matrix

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
            print(f"RGCN Get Edge Information Error: <{e}>")
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
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        best_loss = self.eval_dict["best_loss"]
        best_acc = 0
        best_pre = 0
        best_rec = 0
        best_f1 = 0
        best_roc_auc = 0
        best_cm = 0
        for epoch in range(self.eval_dict["n_epochs"]):
            epoch_loss = []
            st = time.time()
            for batch in self.train_loader:
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
                elif self.conv_type == 'HGT':
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
        best_test_cm = []
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
                elif self.conv_type == 'HGT':
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

    def evaluate_classifier(self, task="subgraph"):
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
                if self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                elif self.conv_type == 'HGT':
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
            if task == "subgraph":
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
        if self.prompt_type == 'add_prompt':
            prompt_batch_h = batch_h + prompt.repeat(batch_h.shape[0], 1)
        elif self.prompt_type == 'concat_prompt':
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
            elif self.prompt_type == 'concat_subgraph_plus_prompt':
                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                prompt_batch_h = torch.concat(
                    [batch_h, expand_batch_h_g + prompt.repeat(batch_h.shape[0], 1)], dim=1)
            elif self.prompt_type == 'concat_subgraph_proj_prompt':
                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                prompt_batch_h = torch.concat(
                    [batch_h, expand_batch_h_g * prompt.repeat(batch_h.shape[0], 1)], dim=1)
            elif self.prompt_type == 'linear_concat_subgraph_concat_prompt':
                expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                weighted_batch_h = self.combine_func(torch.concat([batch_h, expand_batch_h_g], dim=1))
                prompt_batch_h = torch.concat([weighted_batch_h, prompt.repeat(batch_h.shape[0], 1)], dim=1)
        return prompt_batch_h

    def construct_full_adj(self, batch):
        edge_index = []
        for edge_type in batch.edge_types:
            edge_index.append(batch[edge_type].edge_index)
        edge_index = torch.concat(edge_index, dim=1)
        adj = to_dense_adj(edge_index, max_num_nodes=batch['uin'].ptr[-1])
        return adj.squeeze()

    def upgrade_adj(self, adj):
        tilde_adj = adj + torch.eye(adj.shape[0]).to(self.device)
        degree_matrix_row = torch.eye(adj.shape[0]).to(self.device) * adj.sum(dim=1)
        degree_matrix_column = torch.eye(adj.shape[1]).to(self.device) * adj.sum(dim=0)
        fused_adj = torch.mul(torch.mul(degree_matrix_row ** 0.5, tilde_adj), degree_matrix_column ** 0.5)
        return fused_adj

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
        best_test_cfm = []
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
                elif self.conv_type == 'HGT':
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
                        elif self.prompt_type == 'concat_adj':
                            batch_adj = self.construct_full_adj(batch)
                            if self.eval_dict['upgrade_adj']:
                                batch_adj = self.upgrade_adj(batch_adj)
                            batch_mlp = torch.nn.Linear(batch_adj.shape[1], int(self.eval_dict['hidden_dim'] / 16), bias=False).to(self.device)
                            mlp_adj = batch_mlp(batch_adj)
                            batch_h = torch.concat([batch_h, mlp_adj], dim=1)
                        elif self.prompt_type == 'multiply_adj':
                            batch_adj = self.construct_full_adj(batch)
                            batch_adj = torch.nn.functional.normalize(batch_adj, dim=0)
                            batch_h = torch.mm(batch_adj, batch_h)
                        elif self.prompt_type == 'node_subtract_subgraph':
                            if self.eval_dict["is_weighted_subgraph"]:
                                exp_score = torch.exp(batch['uin'].score)
                                new_batch_h = batch_h * exp_score
                                batch_h_g = scatter_mean(new_batch_h, batch['uin'].batch, dim=0)
                            else:
                                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = batch_h - expand_batch_h_g
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
            if self.prompt_type == 'weighted_addition_concat_prompt':
                print(
                    f"Current Weight: {self.local_weight.detach().cpu().item()}, {self.global_weight.detach().cpu().item()}")
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
                elif self.conv_type == 'HGT':
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
                        elif self.prompt_type == 'concat_adj':
                            batch_adj = self.construct_full_adj(batch)
                            if self.eval_dict['upgrade_adj']:
                                batch_adj = self.upgrade_adj(batch_adj)
                            batch_mlp = torch.nn.Linear(batch_adj.shape[1], int(self.eval_dict['hidden_dim'] / 16), bias=False).to(self.device)
                            mlp_adj = batch_mlp(batch_adj)
                            batch_h = torch.concat([batch_h, mlp_adj], dim=1)
                        elif self.prompt_type == 'multiply_adj':
                            batch_adj = self.construct_full_adj(batch)
                            batch_adj = torch.nn.functional.normalize(batch_adj, dim=0)
                            batch_h = torch.mm(batch_adj, batch_h)
                        elif self.prompt_type == 'node_subtract_subgraph':
                            if self.eval_dict["is_weighted_subgraph"]:
                                exp_score = torch.exp(batch['uin'].score)
                                new_batch_h = batch_h * exp_score
                                batch_h_g = scatter_mean(new_batch_h, batch['uin'].batch, dim=0)
                            else:
                                batch_h_g = scatter_mean(batch_h, batch['uin'].batch, dim=0)
                            expand_batch_h_g = self.subgraph_embedding_expand(batch_h_g, batch['uin'].ptr)
                            batch_h = batch_h - expand_batch_h_g
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
            return acc, pre, rec, f1, roc_auc, cm, jaccard
