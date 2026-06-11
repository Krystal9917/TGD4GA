import os
import time
import json
import torch
import random
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj
from codes.models.gnn_models import GCN, GAT
from codes.models.rgcn_model import AttnRGCN
from codes.models.pca_alignment import PCARepresentationAlignment, apply_pca_alignment_to_batch
from codes.dataprocess.public_data_utils import extract_subgraph_by_fraudar, split_data_save_index, batch_loss
from codes.dataprocess.public_data_generate_pyg import WeiboDataset, FacebookDataset, AmazonDataset, TFinanceDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


class PublicDataModelTune:
    def __init__(self, args_dict):
        self.train_dict = args_dict
        self.conv_type = args_dict["conv_type"]
        self.data_dir = args_dict["data_path"]
        self.dataset_name = args_dict["dataset_name"]
        self.batch_size = args_dict["batch_size"]
        self.num_workers = args_dict["num_workers"]
        self.epoch_num = args_dict["n_epochs"]
        self.task_type = args_dict["task_type"]
        self.print_batch = args_dict["print_batch_num"]
        self.is_finetune = args_dict["is_finetune"]
        self.node_classes = args_dict["node_classes"]
        self.index_save_dir = os.path.join(self.data_dir, "index_dir")
        self.k_shot = args_dict["k_shot"]
        self.fold = args_dict["fold"]
        self.top_k = args_dict["top_k"]
        self.set_seed()
        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # dataset
        if self.dataset_name == "weibo":
            dataset = WeiboDataset(self.data_dir)
            in_channel = 400
            hidden_channel = 512
            out_channel = 400
        elif self.dataset_name == "fb":
            dataset = FacebookDataset(self.data_dir)
            in_channel = 576
            hidden_channel = 512
            out_channel = 576
        elif self.dataset_name == "Amazon":
            dataset = AmazonDataset(self.data_dir)
            in_channel = 25
            hidden_channel = 64
            out_channel = 25
        else:
            dataset = TFinanceDataset(self.data_dir)
            in_channel = 10
            hidden_channel = 32
            out_channel = 10
        path = os.path.join(self.index_save_dir, f'{self.dataset_name}.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                index_dict = json.load(f)
                tune_index = index_dict[f'{str(self.fold)}-fold'][f'{str(self.k_shot)}-shot']['tune_index']
                test_index = index_dict[f'{str(self.fold)}-fold'][f'{str(self.k_shot)}-shot']['test_index']
        else:
            index_dict = split_data_save_index(self.dataset_name, dataset, self.index_save_dir)
            tune_index = index_dict[f'{str(self.fold)}-fold'][f'{str(self.k_shot)}-shot']['tune_index']
            test_index = index_dict[f'{str(self.fold)}-fold'][f'{str(self.k_shot)}-shot']['test_index']
        tune_dataset = dataset[tune_index]
        test_dataset = dataset[test_index]
        self.tune_dataloader = DataLoader(tune_dataset, batch_size=self.batch_size,
                                        num_workers=self.num_workers, shuffle=True)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size,
                                          num_workers=self.num_workers, shuffle=False)
        # model
        if self.conv_type == "gcn":
            self.model = GCN(in_channel, hidden_channel, out_channel)
        else:
            self.model = GAT(in_channel, hidden_channel, out_channel)
        self.model = self.model.to(self.device)
        # PCA-based anomaly score estimation (replaces original ano_estimate)
        pca_components = args_dict.get("pca_components", min(in_channel // 2, 8))
        pca_hidden_dim = args_dict.get("pca_hidden_dim", 64)
        self.pca_alignment = PCARepresentationAlignment(
            pca_components=pca_components,
            hidden_dim=pca_hidden_dim,
            dropout=0.2,
        )
        self.pca_alignment = self.pca_alignment.to(self.device)
        self.lr = args_dict["lr"]
        self.log_file_path = f"{self.dataset_name}_pretraining"
        self.save_model_path = os.path.join(self.train_dict["model_states_path"],
                                            self.log_file_path)
        if self.is_finetune:
            # load pretrained model
            if args_dict["evaluate_epoch"] == 0:
                file_name = os.path.join(self.save_model_path,
                                         f"{self.conv_type}_{self.task_type}_{self.lr}_best_loss.pth")
            else:
                file_name = os.path.join(self.save_model_path,
                                         f"{self.conv_type}_{self.task_type}_{self.lr}_epoch_{args_dict['evaluate_epoch']}.pth")
            model_weight = torch.load(file_name, map_location=self.device)
            self.model.load_state_dict(model_weight["gnn"])
            # Load pca_alignment weights (new format); fall back to old "mlp" key for backward compat
            if "pca_alignment" in model_weight:
                self.pca_alignment.load_state_dict(model_weight["pca_alignment"])
            elif "mlp" in model_weight:
                # Old checkpoint format: ano_estimate was stored as "mlp" — skip,
                # pca_alignment will use random initialization
                pass
        # classifier
        self.classifier = torch.nn.Sequential(torch.nn.Linear(out_channel * 2, hidden_channel),
                                                  torch.nn.ReLU(),
                                                  torch.nn.Linear(hidden_channel, self.node_classes),
                                                  torch.nn.Softmax(dim=1))
        self.classifier.to(self.device)
        # params = [{'params': self.classifier.parameters(), 'lr': args_dict["evaluate_lr"]},
        #           {'params': self.model.parameters(), 'lr': args_dict["evaluate_lr"]},
        #           {'params': self.ano_estimate.parameters(), 'lr': args_dict["evaluate_lr"]}]
        params = [{'params': self.classifier.parameters(), 'lr': args_dict["evaluate_lr"]},
                  {'params': self.model.parameters(), 'lr': args_dict["evaluate_lr"]}]
        self.optimizer = torch.optim.Adam(params)
        self.criterion = torch.nn.CrossEntropyLoss()

    def set_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True

    def generate_node_context(self, graph_h, graph_adj, graph_score):
        n = graph_h.shape[0]
        d = graph_h.shape[1]
        graph_adj.fill_diagonal_(0)
        graph_deg = graph_adj.sum(dim=1) + graph_adj.sum(dim=0)
        ave_score = graph_score.mean()
        ave_deg = graph_deg.mean() / 2
        cnt_h_list = []
        for i in range(n):
            i_scr = graph_score[i]
            i_deg = graph_deg[i]
            if i_scr < ave_score and i_deg < ave_deg:
                cnt_h_list.append(torch.zeros(d, device=self.device))
            else:
                i_adj = graph_adj[i, :]
                i_neigh = (i_adj > 0).nonzero().squeeze()
                if i_neigh.shape == torch.Size([0]):
                    cnt_h_list.append(graph_h[i, :])
                else:
                    if i_neigh.shape == torch.Size([]):
                        i_neigh = torch.tensor([i_neigh], device=self.device)
                    i_neigh_scr = graph_score[i_neigh]
                    i_neigh_deg = graph_deg[i_neigh]
                    scr_dif = 1 - (i_scr - i_neigh_scr).abs()
                    deg_dif = 1 - (i_deg - i_neigh_deg).abs() / (i_deg + i_neigh_deg)
                    i_neigh_dif = scr_dif * deg_dif
                    i_neigh_dif_srt_idx = i_neigh_dif.sort().indices
                    if i_neigh_dif_srt_idx.shape[0] > self.top_k:
                        i_neigh_top_k = i_neigh_dif_srt_idx[:self.top_k]
                    else:
                        i_neigh_top_k = i_neigh_dif_srt_idx
                    if i_neigh_top_k.shape[0] > 1:
                        context_h = graph_h[i_neigh_top_k, :].mean(dim=0)
                    else:
                        context_h = graph_h[i_neigh_top_k, :].squeeze()
                    cnt_h_list.append(context_h)
        node_cnt_h = torch.stack(cnt_h_list, dim=0)
        return node_cnt_h.to(self.device)


    def training(self):
        best_test_acc = 0
        best_test_pre = 0
        best_test_rec = 0
        best_test_f1 = 0
        best_test_roc_auc = 0
        best_loss = 1e2
        count = 0
        for epoch in range(1, self.epoch_num + 1):
            st = time.time()
            self.model.train()
            self.classifier.train()
            self.pca_alignment.eval()
            epoch_loss = []
            for i, batch in enumerate(self.tune_dataloader):
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                batch_x = torch.nn.functional.normalize(batch.x, p=2, dim=1)
                # PCA-based anomaly score estimation (replaces original ano_estimate)
                batch_score = apply_pca_alignment_to_batch(
                    batch_x, batch.edge_index, batch.ptr, self.pca_alignment
                )
                batch_h = self.model(batch_x, batch.edge_index)
                batch_adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze()
                grh_h_cnt_list = []
                for g_num in range(batch.num_graphs):
                    g_node_idx = (batch.batch == g_num).nonzero().squeeze()
                    graph_h = batch_h[g_node_idx]
                    graph_adj = batch_adj[g_node_idx, :][:, g_node_idx]
                    graph_score = batch_score[g_node_idx]
                    graph_h_context = self.generate_node_context(graph_h, graph_adj, graph_score)
                    grh_h_cnt_list.append(graph_h_context)
                batch_h_context = torch.concat(grh_h_cnt_list, dim=0).to(self.device)
                batch_h = torch.concat([batch_h, batch_h_context], dim=1).to(self.device)
                batch_prob = self.classifier(batch_h)[:, 1]
                loss = batch_loss(batch.batch, batch_prob, batch.y) / batch.num_nodes
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.detach().cpu().item())
            current_loss = sum(epoch_loss) / len(epoch_loss)
            print(f"Epoch {epoch}, Loss: {current_loss: .4f}, Time: {time.time() - st: .4f} s")
            if current_loss < best_loss:
                best_loss = current_loss
                count = 0
            else:
                count += 1
            if epoch % 3 == 0:
                test_acc, test_f1, test_pre, test_rec, test_roc_auc = self.evaluating()
                if best_test_f1 < test_f1:
                    best_test_acc = test_acc
                    best_test_f1 = test_f1
                    best_test_pre = test_pre
                    best_test_rec = test_rec
                    best_test_roc_auc = test_roc_auc
        return best_test_acc, best_test_f1, best_test_pre, best_test_rec, best_test_roc_auc


    def evaluating(self):
        self.model.eval()
        self.pca_alignment.eval()
        self.classifier.eval()
        st = time.time()
        y_true = []
        y_pred = []
        for i, batch in enumerate(self.test_dataloader):
            batch = batch.to(self.device)
            batch_x = torch.nn.functional.normalize(batch.x, p=2, dim=1)
            # PCA-based anomaly score estimation (replaces original ano_estimate)
            batch_score = apply_pca_alignment_to_batch(
                batch_x, batch.edge_index, batch.ptr, self.pca_alignment
            )
            batch_h = self.model(batch_x, batch.edge_index)
            batch_adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze()
            grh_h_cnt_list = []
            for g_num in range(batch.num_graphs):
                g_node_idx = (batch.batch == g_num).nonzero().squeeze()
                graph_h = batch_h[g_node_idx]
                graph_adj = batch_adj[g_node_idx, :][:, g_node_idx]
                graph_score = batch_score[g_node_idx]
                graph_h_context = self.generate_node_context(graph_h, graph_adj, graph_score)
                grh_h_cnt_list.append(graph_h_context)
            batch_h_context = torch.concat(grh_h_cnt_list, dim=0).to(self.device)
            batch_h = torch.concat([batch_h, batch_h_context], dim=1).to(self.device)
            batch_prob = self.classifier(batch_h)
            batch_pred = batch_prob.argmax(dim=1)
            y_true.append(batch.y)
            y_pred.append(batch_pred)
        y_true = torch.concat(y_true).detach().cpu().numpy()
        y_pred = torch.concat(y_pred).detach().cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        print(f"Test ACC: {acc: .4f}, "
              f"Precision: {pre: .4f}, "
              f"Recall: {rec: .4f}, "
              f"F1: {f1: .4f}, "
              f"ROC-AUC: {roc_auc: .4f}, "
              f"Evaluate Time: {time.time() - st: .4f} s")
        return acc, f1, pre, rec, roc_auc

