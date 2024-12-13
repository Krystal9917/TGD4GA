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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmgog_long_term_sequence_model.utils.utils import visualization_fig_save
from mmgog_long_term_sequence_model.pytorch.models.rgcn_model import RGCN
from mmgog_long_term_sequence_model.pytorch.models.han_model import HAN
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix


class UinGangsModelPreTrainDDP:

    def __init__(self, rank, args_dict, world_size):
        self.train_dict = args_dict
        self.rank = rank
        self.world_size = world_size
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)
        self.device = torch.device(f"cuda: {self.rank}")
        print(f"GPU device id: {self.rank}")
        self.conv_type = args_dict["conv_type"]
        if self.conv_type == 'RGCN':
            self.model = RGCN(input_dim=args_dict['input_dim'],
                              hidden_dim=args_dict['hidden_dim'],
                              output_dim=args_dict['output_dim'],
                              num_relations=args_dict['num_relations'])
        elif self.conv_type == 'HAN':
            self.metadata = (['uin'], [('uin', 'ipv6', 'uin'), ('uin', 'wifi', 'uin'), ('uin', 'room', 'uin'),
                                       ('uin', 'friend', 'uin'), ('uin', 'idcardid', 'uin'), ('uin', 'device', 'uin'),
                                       ('uin', 'payee', 'uin'), ('uin', 'payer', 'uin'), ('uin', 'bankcard', 'uin'),
                                       ('uin', 'download_app', 'uin')])
            self.model = HAN(in_channels=args_dict['input_dim'],
                             out_channels=args_dict['output_dim'],
                             metadata=self.metadata,
                             heads=args_dict['num_heads'])
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])

        self.minirbt_model = BertModel.from_pretrained(self.train_dict["minirbt_path"])
        # 冻结文本模型的参数
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        print(f"minirbt_model params size: {minirbt_model_params_size}")
        self.minirbt_model.to(self.device)

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
                self.log_file_path = f"1922_{self.conv_type}_sample_{sampling_type}_filter_{control_node_num}_lr_{str(lr)}"
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
                                                f"{self.conv_type}_sample_{sampling_type}_filter_{control_node_num}_lr_{str(lr)}")
            if self.train_dict["evaluate_task"] == 'eval_labelled_subgraph_embedding':
                self.eval_data = UinGangsDataIterablePyG(self.train_dict, self.train_dict["eval_data_path"])
                self.eval_loader = Data.DataLoader(self.eval_data,
                                                   batch_size=self.train_dict["batch_size"],
                                                   num_workers=self.train_dict["num_workers"],
                                                   collate_fn=self.eval_data.collate_fn)
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
        self.setup_seed()

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
            loss = torch.tensor(torch.nan).to(h1.device)
            print(f"Norm error <{e}>, h1 shape: {h1.shape}, h3 shape: {h3.shape}")
        else:
            loss = pos_sim / (sim_matrix.sum(dim=1) + 1e-4)
            loss = -torch.log(loss).mean()
        return loss

    def get_edge_info(self, batch):
        edge_index = [batch[edge_type].edge_index for edge_type in batch.edge_types]
        edge_index = torch.concat(edge_index, dim=1)
        # adj = to_dense_adj(edge_index, batch=batch['uin'].batch, batch_size=self.train_dict['batch_size'])
        edge_counts = [batch[edge_type].num_edges for edge_type in batch.edge_types]
        edge_type = torch.concat([torch.ones(edge_counts[i]) * i for i in range(len(batch.edge_types))])
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

    def rgcn_fit(self, pos_batch, batch_x):
        try:
            batch_edge_index, batch_edge_types = self.get_edge_info(pos_batch)
        except Exception as e:
            print(f"Get edge information error: <{e}>")
            return None
        else:
            batch_h = self.model(batch_x, batch_edge_index, batch_edge_types)
            return batch_h

    def pretraining_ddp(self):
        start_epoch = 1
        end_epoch = self.train_dict["n_epochs"] + 1
        if self.train_dict["re_train"]:
            epoch_num = self.train_dict["start_epoch"]
            file_name = os.path.join(self.save_model_path, f"uin_gangs_{self.conv_type}_model_epoch_{epoch_num}.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            self.model.load_state_dict(model_weight)
            start_epoch = epoch_num + 1
            end_epoch = start_epoch + self.train_dict["n_epochs"]

        best_loss = self.train_dict["best_loss"]
        for epoch in range(start_epoch, end_epoch):
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

                    if self.conv_type == 'HAN':
                        fraudar_batch_h = self.han_fit(fraudar_batch.x_dict, fraudar_batch.edge_index_dict)
                    elif self.conv_type == 'RGCN':
                        fraudar_batch_h = self.rgcn_fit(fraudar_batch, fraudar_batch['uin'].x)

                    if fraudar_batch_h is not None:
                        fraudar_batch_h_g = scatter_mean(fraudar_batch_h, fraudar_batch['uin'].batch, dim=0)
                        fraudar_batch_h_g = fraudar_batch_h_g[pos_batch_idx]

                        # readout for subgraph
                        pos_batch_h_g = scatter_mean(batch_h, pos_batch['uin'].batch, dim=0)
                        pos_batch_h_g = pos_batch_h_g[pos_batch_idx]

                        if (type(neg_batch_idx) is list and len(neg_batch_idx) != 0) or type(neg_batch_idx) is int:
                            neg_batch_h_g = batch_h_g[neg_batch_idx]
                        else:
                            neg_batch_h_g = batch_h_g

                        # subgraph-level contrastive learning
                        loss = self.preference_contrastive_loss(fraudar_batch_h_g, pos_batch_h_g, neg_batch_h_g)
                    else:
                        loss = torch.tensor(torch.nan).to(self.device)
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
        if not self.train_dict["is_debug"]:
            self.writer.close()
        dist.destroy_process_group()

    def evaluate_labelled_subgraph_embedding(self):
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

    def evaluate_subgraph_predict(self):
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
                elif self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
                # get edge information
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
                # get edge information
                batch['uin'].x = batch_x
                if self.conv_type == 'HAN':
                    batch_h = self.han_fit(batch.x_dict, batch.edge_index_dict)
                elif self.conv_type == 'RGCN':
                    batch_h = self.rgcn_fit(batch, batch['uin'].x)
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
