import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.utils.data as Data
from transformers import BertModel
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.models.ANEMONE_model import Model
from mmgog_long_term_sequence_model.pytorch.dataprocess.fraudar import fraudar
from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable_pyg import UinGangsDataIterablePyG


class BaselineRun:
    def __init__(self, args, args_dict):
        self.args = args
        if torch.cuda.is_available():
            print("GPU is available")
            self.device = torch.device("cuda")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("GPU is not available")
            self.device = torch.device("cpu")
        # Load Pretrained Language Model for Text Embedding
        self.minirbt_model = BertModel.from_pretrained(self.args.minirbt_path)
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        print(f"minirbt_model params size: {minirbt_model_params_size}")
        self.minirbt_model.to(self.device)
        self.method = self.args.baseline_name
        if self.method == 'ANEMONE':
            self.model = Model(self.args.input_dim, self.args.hidden_dim,
                               self.args.activation, self.args.negsamp_ratio_patch,
                               self.args.negsamp_ratio_context, self.args.readout).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            self.b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                                     pos_weight=torch.tensor([self.args.negsamp_ratio_patch]).to(
                                                         self.device))
            self.b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                                       pos_weight=torch.tensor([self.args.negsamp_ratio_context]).to(
                                                           self.device))
        train_data_path = os.path.join(self.args.train_data_path + self.args.split_idx,
                                       self.args.train_data_file)
        self.train_data = UinGangsDataIterablePyG(args_dict, train_data_path)
        test_data_path = os.path.join(self.args.test_data_path + self.args.split_idx,
                                      self.args.test_data_file)
        self.eval_data = UinGangsDataIterablePyG(args_dict, test_data_path)
        self.train_loader = Data.DataLoader(self.train_data,
                                            batch_size=self.args.batch_size,
                                            num_workers=self.args.num_workers,
                                            collate_fn=self.train_data.collate_fn)
        self.eval_loader = Data.DataLoader(self.eval_data,
                                           batch_size=self.args.batch_size,
                                           num_workers=self.args.num_workers,
                                           collate_fn=self.eval_data.collate_fn)
        self.set_seed(self.args.seed)

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def edge_index_to_adj(self, batch, max_n=None):
        edge_index = torch.concat([batch[edge_type].edge_index for edge_type in batch.edge_types], dim=1)
        if max_n is not None:
            adj = to_dense_adj(edge_index, batch=batch['uin'].batch, max_num_nodes=max_n,
                               batch_size=batch['uin'].batch.unique().max().item() + 1)
        else:
            adj = to_dense_adj(edge_index, max_num_nodes=batch['uin'].num_nodes).squeeze()
        return adj

    def reshape_x_into_3d(self, x, batch_size, max_n, ptr):
        n, d = x.shape
        x_3d = torch.zeros((batch_size, max_n, d))
        for i in range(batch_size):
            start_idx = ptr[i]
            end_idx = ptr[i + 1]
            diff_idx = end_idx - start_idx
            x_3d[i, :diff_idx, :] = x[start_idx:end_idx, :]
        return x_3d

    def train(self):
        cnt_wait = 0
        best = 1e5
        best_acc, best_pre, best_rec, best_f1, best_auc, best_cm, best_time = None, None, None, None, None, None, None
        for epoch in range(self.args.n_epoch):
            self.model.train()
            total_loss = []
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                cur_batch_size = batch['uin'].batch.max().item() + 1
                num_nodes_list = [batch['uin'].ptr[i + 1] - batch['uin'].ptr[i] for i in range(cur_batch_size)]
                max_num_nodes = max(num_nodes_list)
                bf = self.reshape_x_into_3d(batch_x, cur_batch_size, max_num_nodes, batch['uin'].ptr)
                ba = self.edge_index_to_adj(batch, max_num_nodes)

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, max_num_nodes)).to(self.device)
                added_adj_zero_col = torch.zeros((cur_batch_size, max_num_nodes + 1, 1)).to(self.device)
                added_adj_zero_col[:, -1, :] = 1.
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, self.args.input_dim)).to(self.device)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits_1, logits_2 = self.model(bf, ba)

                ones_cur_batch_size = torch.ones(cur_batch_size)
                patch_batch_size = torch.zeros(cur_batch_size * self.args.negsamp_ratio_patch)
                context_batch_size = torch.zeros(cur_batch_size * self.args.negsamp_ratio_context)
                lbl_patch = torch.cat((ones_cur_batch_size, patch_batch_size)).unsqueeze(1).to(self.device)
                lbl_context = torch.cat((torch.ones(cur_batch_size), context_batch_size)).unsqueeze(1).to(self.device)

                # Context-level
                loss_all_1 = self.b_xent_context(logits_1, lbl_context)
                loss_1 = torch.mean(loss_all_1)

                # Patch-level
                loss_all_2 = self.b_xent_patch(logits_2, lbl_patch)
                loss_2 = torch.mean(loss_all_2)

                loss = self.args.alpha * loss_1 + (1 - self.args.alpha) * loss_2

                loss.backward()
                self.optimizer.step()

                loss = loss.detach().cpu()
                total_loss.append(loss)

            mean_loss = torch.stack(total_loss).mean()
            print('Epoch: {} Loss: {:.8f}'.format(epoch, mean_loss), flush=True)
            if mean_loss < best:
                cnt_wait = 0
                best = mean_loss
                save_path = os.path.join(self.args.model_states_path, self.method + f'_epoch_{epoch}.pth')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(self.model.state_dict(), save_path)
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cm, test_time = self.test()
                if best_f1 < test_f1:
                    best_acc = test_acc
                    best_pre = test_pre
                    best_rec = test_rec
                    best_f1 = test_f1
                    best_auc = test_roc_auc
                    best_cm = test_cm
                    best_time = test_time
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                print('Early stopping!', flush=True)
                break

            return best_acc, best_pre, best_rec, best_f1, best_auc, best_cm, best_time

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
        idx = np.zeros_like(anomaly_score, dtype=torch.int)
        if len(best_graph.nodes) >= 3:
            idx[sorted(best_graph.nodes)] = 1
        return idx

    def detect_by_fraudar(self, batch, anomaly_score):
        sub_num = batch['uin'].batch.unique().max() + 1
        pred_gang_member = []
        adj = self.edge_index_to_adj(batch)
        for sub_i in range(sub_num):
            start_node, end_node = batch['uin'].ptr[sub_i].item(), batch['uin'].ptr[sub_i + 1].item()
            sub_pred_gang_member = self.generate_gang_by_fraudar(anomaly_score[batch['uin'].batch == sub_i],
                                                                 adj[start_node:end_node, start_node:end_node])
            pred_gang_member.append(sub_pred_gang_member)
        pred_gang_member = np.concatenate(pred_gang_member, axis=0)
        return pred_gang_member

    def test(self):
        self.model.eval()
        start_time = time.time()
        ano_score_list = []
        y_true = []
        with torch.no_grad():
            for i, batch in enumerate(self.eval_loader):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_uin_acs_text_feat_input_ids = batch['uin'].text_feat_input_ids
                batch_uin_acs_text_feat_attention_mask = batch['uin'].text_feat_attention_mask
                with torch.no_grad():
                    batch_uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                                 batch_uin_acs_text_feat_attention_mask).pooler_output
                # combine numerical, categorical and text attributes
                batch_x = torch.concat([batch['uin'].x, batch_uin_acs_text_feat], dim=1)
                batch_x = torch.nn.functional.normalize(batch_x, dim=1)
                cur_batch_size = batch['uin'].batch.max().item() + 1
                num_nodes_list = [batch['uin'].ptr[i + 1] - batch['uin'].ptr[i] for i in range(cur_batch_size)]
                max_num_nodes = max(num_nodes_list)
                bf = self.reshape_x_into_3d(batch_x, cur_batch_size, max_num_nodes, batch['uin'].ptr)
                ba = self.edge_index_to_adj(batch, max_num_nodes)

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, max_num_nodes)).to(self.device)
                added_adj_zero_col = torch.zeros((cur_batch_size, max_num_nodes + 1, 1)).to(self.device)
                added_adj_zero_col[:, -1, :] = 1.
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)

                added_feat_zero_row = torch.zeros((cur_batch_size, 1, self.args.input_dim)).to(self.device)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                select_idx = (batch['uin'].ptr[1:].detach().cpu().numpy() - 1).tolist()
                y_true.append(batch['uin'].gang_mem.int().detach().cpu().numpy()[select_idx])

                test_logits_1, test_logits_2 = self.model(bf, ba)
                test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))

                if self.args.alpha != 1.0 and self.args.alpha != 0.0:
                    if self.args.negsamp_ratio_context == 1 and self.args.negsamp_ratio_patch == 1:
                        ano_score_1 = - (test_logits_1[:cur_batch_size] - test_logits_1[
                                                                          cur_batch_size:]).detach().cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - test_logits_2[
                                                                          cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score_1 = - (
                                test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, self.args.negsamp_ratio_context), dim=1)).detach().cpu().numpy()  # context
                        ano_score_2 = - (
                                test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, self.args.negsamp_ratio_patch), dim=1)).detach().cpu().numpy()  # patch
                    ano_score = self.args.alpha * ano_score_1 + (1 - self.args.alpha) * ano_score_2
                elif self.args.alpha == 1.0:
                    if self.args.negsamp_ratio_context == 1:
                        ano_score = - (test_logits_1[:cur_batch_size] - test_logits_1[
                                                                        cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, self.args.negsamp_ratio_context), dim=1)).detach().cpu().numpy()  # context
                elif self.args.alpha == 0.0:
                    if self.args.negsamp_ratio_patch == 1:
                        ano_score = - (test_logits_2[:cur_batch_size] - test_logits_2[
                                                                        cur_batch_size:]).detach().cpu().numpy()
                    else:
                        ano_score = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, self.args.negsamp_ratio_patch), dim=1)).detach().cpu().numpy()  # patch

                ano_score_list.append(ano_score)
            ano_score_final = np.concatenate(ano_score_list, axis=0)
            y_pred = np.zeros_like(ano_score_final)
            y_pred[ano_score_final > 0] = 1
            # y_pred = self.detect_by_fraudar(batch, ano_score_final)
            y_true = np.concatenate(y_true, axis=0)
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            eval_time = time.time() - start_time
            print(f"Test ACC: {acc: .4f}, "
                  f"Precision: {pre: .4f}, "
                  f"Recall: {rec: .4f}, "
                  f"F1: {f1: .4f}, "
                  f"ROC-AUC: {roc_auc: .4f}, "
                  f"Confusion Matrix: {cm.tolist()}, "
                  f"Evaluate Time: {eval_time: .4f} s")
            return acc, pre, rec, f1, roc_auc, cm, eval_time


def set_params():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument('--baseline_name', type=str, default='ANEMONE')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--input_dim', type=int, default=762)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=250)
    parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
    parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_buffer_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--is_single_edge', type=bool, default=True)
    parser.add_argument('--filter_node_num', type=int, default=3)
    parser.add_argument('--load_text_feature', type=bool, default=True)
    parser.add_argument('--split_idx', type=str, default='', choices=['', '1', '2', '3', '4', '5'])
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--readout', type=str, default='avg')
    parser.add_argument('--negsamp_ratio_patch', type=int, default=1)
    parser.add_argument('--negsamp_ratio_context', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                     "uin_gangs_full_graph_dataset", "valid", "positive_processed", "split_")))
    parser.add_argument('--train_data_file', type=str,
                        default="uin_gangs_supervise_full_graph_dataset_train_202503241700.txt")
    parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                     "uin_gangs_full_graph_dataset", "valid", "positive_processed", "split_")))
    parser.add_argument('--test_data_file', type=str,
                        default="uin_gangs_supervise_full_graph_dataset_eval_202503241700.txt")
    parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                     "config", "yml", "uin_gangs_enum.yaml")))
    parser.add_argument('--minirbt_path', type=str, default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "minirbt-h256")))
    parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                     "saved_model", "baseline_models")))
    args = parser.parse_args()
    args_dict = vars(args)
    return args, args_dict


def compute_metrics_mean_std(acc, pre, rec, f1, roc_auc, run_time):
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    f1 = np.array(f1)
    auc = np.array(roc_auc)
    run_time = np.array(run_time)
    return (acc.mean(), acc.std(), pre.mean(), pre.std(), rec.mean(), rec.std(),
            f1.mean(), f1.std(), auc.mean(), auc.std(), run_time.mean(), run_time.std())


if __name__ == '__main__':
    params, params_dict = set_params()
    acc_list, pre_list, rec_list, f1_list, auc_list, cm_list, time_list = [], [], [], [], [], [[0, 0], [0, 0]], []
    for i in range(1, 6):
        params.split_idx = str(i)
        params_dict['split_idx'] = str(i)
        run = BaselineRun(args=params, args_dict=params_dict)
        print(f"***** Start {i} *****")
        acc, pre, rec, f1, roc_auc, cm, eval_time = run.train()
        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
        auc_list.append(roc_auc)
        cm_list += cm
        time_list.append(eval_time)
        print(f"***** End {i} *****")
    (acc_mean, acc_std, pre_mean, pre_std, rec_mean, rec_std, f1_mean, f1_std, auc_mean, auc_std,
     time_mean, time_std) = compute_metrics_mean_std(acc_list, pre_list, rec_list, f1_list, auc_list, time_list)
    print(
        f"{len(acc_list)} Times Average Best Test ACC: {acc_mean: .4f}, std: {acc_std: .4f}, "
        f"Precision: {pre_mean: .4f}, std: {pre_std: .4f}, "
        f"Recall: {rec_mean: .4f}, std: {rec_std: .4f}, "
        f"F1: {f1_mean: .4f}, std: {f1_std: .4f}, "
        f"ROC-AUC: {auc_mean: .4f}, std: {auc_std: .4f}, "
        f"Run Time: {time_std: .4f} s, std: {time_std: .4f} s")
