import os
import time
import torch
import random
import numpy as np
from torch_scatter import scatter_mean
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import subgraph, to_dense_adj
from torch_geometric.data import Data, DataLoader
from codes.models.gnn_models import GCN, GAT
from codes.dataprocess.public_data_utils import extract_subgraph_by_fraudar
from codes.dataprocess.public_data_generate_pyg import WeiboDataset, FacebookDataset, AmazonDataset, TFinanceDataset, TSocialDataset


class PublicDataModelPreTrain:
    def __init__(self, args_dict):
        self.train_dict = args_dict
        self.conv_type = args_dict["conv_type"]
        self.data_dir = args_dict["train_data_path"]
        self.dataset_name = args_dict["dataset_name"]
        self.batch_size = args_dict["batch_size"]
        self.num_workers = args_dict["num_workers"]
        self.epoch_num = args_dict["n_epochs"]
        self.task_type = args_dict["task_type"]
        self.print_batch = args_dict["print_batch_num"]
        self.top_neigh_ratio = args_dict["top_neigh_ratio"]
        self.set_seed()
        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # dataset
        if self.dataset_name == "weibo":
            dataset = WeiboDataset(self.data_dir)
            perm = list(range(len(dataset)))
            random.shuffle(perm)
            perm = perm[:3135]
            pretrain_dataset = dataset[perm]
            in_channel = 400
            hidden_channel = 512
            out_channel = 400
        elif self.dataset_name == "fb":
            dataset = FacebookDataset(self.data_dir)
            perm = list(range(len(dataset)))
            random.shuffle(perm)
            perm = perm[:303]
            in_channel = 576
            hidden_channel = 512
            out_channel = 576
            pretrain_dataset = dataset[perm]
        elif self.dataset_name == "Amazon":
            dataset = AmazonDataset(self.data_dir)
            perm = list(range(len(dataset)))
            random.shuffle(perm)
            perm = perm[:1019]
            pretrain_dataset = dataset[perm]
            in_channel = 25
            hidden_channel = 64
            out_channel = 25
        elif self.dataset_name == "TFinance":
            dataset = TFinanceDataset(self.data_dir)
            perm = list(range(len(dataset)))
            random.shuffle(perm)
            perm = perm[:8064]
            pretrain_dataset = dataset[perm]
            in_channel = 10
            hidden_channel = 32
            out_channel = 10
        else:
            dataset = TSocialDataset(self.data_dir)
            perm = list(range(len(dataset)))
            random.shuffle(perm)
            pretrain_dataset = dataset[perm]
            in_channel = 10
            hidden_channel = 32
            out_channel = 10
        self.dataloader = DataLoader(pretrain_dataset, batch_size=self.batch_size,
                                     num_workers=self.num_workers, shuffle=True)
        # model
        if self.conv_type == "gcn":
            self.model = GCN(in_channel, hidden_channel, out_channel)
        else:
            self.model = GAT(in_channel, hidden_channel, out_channel)
        self.model = self.model.to(self.device)
        self.lr = args_dict["lr"]
        self.log_file_path = f"{self.dataset_name}_pretraining"
        loss_log_path = os.path.abspath(os.path.join(args_dict['log_dir'],
                                                     args_dict["model_states_path"].split('/')[-1],
                                                     self.log_file_path))
        self.save_model_path = os.path.join(self.train_dict["model_states_path"],
                                            self.log_file_path)

        self.ano_estimate = torch.nn.Sequential(torch.nn.Linear(in_channel, 1), torch.nn.Sigmoid())
        self.ano_estimate = self.ano_estimate.to(self.device)
        self.best_loss = args_dict["best_loss"]
        params = [{'params': self.ano_estimate.parameters(), 'lr': self.lr},
                  {'params': self.model.parameters(), 'lr': self.lr}]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)
        # loss save dir
        if not os.path.exists(loss_log_path):
            os.makedirs(loss_log_path)
        self.writer = SummaryWriter(log_dir=loss_log_path)
        # model save dir
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

    def set_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True


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


    def context_contrastive_loss(self, graph_h, graph_adj, graph_score):
        n = graph_h.shape[0]
        graph_deg = graph_adj.sum(dim=1) + graph_adj.sum(dim=0)
        node_idx_list = list(range(n))
        node_loss = []
        for i in node_idx_list:
            i_adj = graph_adj[i, :]
            i_adj[i] = 0
            possible_positive = (i_adj > 0).nonzero().squeeze().detach().cpu().tolist()
            if type(possible_positive) is int:
                possible_positive = [possible_positive]
            target_score = graph_score[i]
            target_degree = graph_deg[i]
            candidate_positive_score = graph_score[possible_positive]
            candidate_positive_degree = graph_deg[possible_positive]
            score_dif = -torch.log(1 - (target_score - candidate_positive_score).abs())
            deg_dif = -torch.log(1 - (target_degree - candidate_positive_degree).abs() / (target_degree + candidate_positive_degree))
            candidate_positive_dif = 0.5 * score_dif + 0.5 * deg_dif
            sort_dif = candidate_positive_dif.sort().indices
            possible_negative = (i_adj == 0).nonzero().squeeze().detach().cpu().tolist()
            if type(possible_negative) is int:
                possible_negative = [possible_negative]
            neighbor_l = int(self.top_neigh_ratio * len(possible_positive)) if (
                    int(self.top_neigh_ratio * len(possible_positive)) > 3) else 3
            if len(possible_positive) > 0:
                if neighbor_l < len(possible_positive):
                    context_idx = sort_dif[:neighbor_l]
                else:
                    context_idx = possible_positive
            else:
                continue
            neighbor_l = int(self.top_neigh_ratio * len(possible_negative)) if (
                    int(self.top_neigh_ratio * len(possible_negative)) > 3) else 3
            if len(possible_negative) > 0:
                if neighbor_l < len(possible_negative):
                    negative_idx = random.sample(possible_negative, neighbor_l)
                else:
                    negative_idx = possible_negative
            else:
                continue
            context_h = graph_h[context_idx, :]
            negative_h = graph_h[negative_idx, :]
            i_h = graph_h[i, :].repeat(context_h.shape[0], 1)
            pos_sim = torch.exp((torch.cosine_similarity(i_h, context_h) / self.train_dict["temperature"])).sum()
            i_h = graph_h[i, :].repeat(negative_h.shape[0], 1)
            neg_sim = torch.exp((torch.cosine_similarity(i_h, negative_h) / self.train_dict["temperature"])).sum()
            i_loss = -torch.log(pos_sim / (pos_sim + neg_sim))
            node_loss.append(i_loss)
        if len(node_loss) != 0:
            node_loss = torch.stack(node_loss).mean()
        else:
            node_loss = torch.tensor(0.0, device=self.device)
        return node_loss
    
    def node_context_contrastive_loss(self, batch_h, batch, adj, score):
        N = batch.shape[0] - 1
        context_loss = []
        for i in range(N):
            start_node, end_node = batch[i], batch[i + 1]
            graph_h = batch_h[start_node:end_node, :]
            graph_adj = adj[start_node:end_node, start_node:end_node]
            graph_score = score[start_node:end_node]
            node_loss = self.context_contrastive_loss(graph_h, graph_adj, graph_score)
            context_loss.append(node_loss)
        context_loss = torch.stack(context_loss).mean()
        return context_loss

    def pretraining(self):
        self.model.train()
        for epoch in range(1, self.epoch_num + 1):
            epoch_loss = []
            epoch_start_time = time.time()
            for i, batch in enumerate(self.dataloader):
                batch_start_time = time.time()
                batch = batch.to(self.device)
                batch_x = torch.nn.functional.normalize(batch.x, p=2, dim=1)
                # raw subgraph
                batch_h = self.model(batch_x, batch.edge_index)
                batch_h_g = scatter_mean(batch_h, batch.batch, dim=0)
                batch_ano_prob = self.ano_estimate(batch.x).squeeze()
                ps_sub_idx = []
                ng_sub_idx = []
                ps_node_idx = []
                ptr = batch.ptr.detach().cpu()
                for batch_id in range(batch.num_graphs):
                    graph_node_idx = (batch.batch == batch_id).nonzero().squeeze()
                    graph_edge_index, _ = subgraph(subset=graph_node_idx, edge_index=batch.edge_index,
                                          relabel_nodes=True, num_nodes=batch.num_nodes)
                    graph_x = batch.x[graph_node_idx]
                    graph_score = batch_ano_prob[graph_node_idx]
                    graph_data = Data(x=graph_x, edge_index=graph_edge_index, score=graph_score)
                    graph_data = extract_subgraph_by_fraudar(graph_data)
                    if graph_data.flag == 1:
                        ps_sub_idx.append(batch_id)
                        ps_node_idx.append((graph_data.idx == 1).nonzero().squeeze() + ptr[batch_id])
                    else:
                        ng_sub_idx.append(batch_id)
                # subgraph loss
                if len(ps_sub_idx) == 0 or len(ng_sub_idx) == 0:
                    sub_loss = torch.tensor(0.).to(self.device)
                else:
                    ps_node_idx = torch.concat(ps_node_idx, dim=0).to(self.device)
                    extract_batch_x = batch.x[ps_node_idx]
                    extract_batch = batch.batch[ps_node_idx]
                    _, reset_extract_batch = torch.unique(extract_batch, return_inverse=True)
                    extract_batch_edge_index, _ = subgraph(subset=ps_node_idx, edge_index=batch.edge_index,
                                                           relabel_nodes=True, num_nodes=batch.num_nodes)

                    extract_batch_x = torch.nn.functional.normalize(extract_batch_x, p=2, dim=1)
                    extract_batch_h = self.model(extract_batch_x, extract_batch_edge_index)
                    extract_batch_h_g = scatter_mean(extract_batch_h, reset_extract_batch, dim=0)
                    ps_sub_idx = torch.tensor(ps_sub_idx).to(self.device)
                    ng_sub_idx = torch.tensor(ng_sub_idx).to(self.device)
                    sub_loss = self.preference_contrastive_loss(batch_h_g[ps_sub_idx], extract_batch_h_g, batch_h_g[ng_sub_idx])
                # node loss
                batch_adj = to_dense_adj(batch.edge_index, max_num_nodes=batch.num_nodes).squeeze()
                node_loss = self.node_context_contrastive_loss(batch_h, batch.ptr, batch_adj, batch_ano_prob)
                loss = self.train_dict['sub_weight'] * sub_loss + self.train_dict['node_weight'] * node_loss
                if not torch.isnan(loss):
                    loss.backward()
                    self.optimizer.step()
                    loss_value = loss.detach().cpu().item()
                    epoch_loss.append(loss_value)
                    if (i + 1) % self.print_batch == 0:
                            print(
                                "Batch: {}, Subgraph Loss: {:.6f}, "
                                "Node Loss: {:.6f}, "
                                "Time: {:.4f} s".format(
                                    i + 1,
                                    sub_loss.detach().cpu().item(),
                                    node_loss.detach().cpu().item(),
                                    time.time() - batch_start_time))

            torch.cuda.empty_cache()
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            save_prefix = f"{self.conv_type}_{self.task_type}_{self.lr}"
            self.writer.add_scalar(f"{save_prefix}_pretraining_loss", epoch_loss, epoch)
            print("Epoch: {}, Loss: {:.4f}, Time: {:.4f} s".format(epoch, epoch_loss,
                                                                   time.time() - epoch_start_time))
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                model_dict = {"gnn": self.model.state_dict(),
                              "mlp": self.ano_estimate.state_dict()}
                file_name = os.path.join(self.save_model_path, f"{save_prefix}_best_loss.pth")
                torch.save(model_dict, file_name)
                print(f"Now best loss: {self.best_loss:.4f}, save model to {file_name}")
                if epoch % 10 == 0:
                    epoch_file_name = os.path.join(self.save_model_path, f"{save_prefix}_epoch_{epoch}.pth")
                    torch.save(model_dict, epoch_file_name)
                    print(f"Now best loss: {self.best_loss:.4f}, save model to {epoch_file_name}")
        print("Finished!")
        self.writer.close()
