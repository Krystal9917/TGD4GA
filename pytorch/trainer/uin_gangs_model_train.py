from datetime import datetime
import logging
import os
import time

logger = logging.getLogger("my_logger")
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import numpy as np
import random
import torch
import torch.utils.data as Data
from accelerate import Accelerator
from torch import nn
from transformers import AutoTokenizer, BertModel

from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable import UinGangsDataIterable
from mmgog_long_term_sequence_model.pytorch.models.LossFunction import InfoNCELoss, InfoNCELossV2, InfoNCELossV3, \
    WeightedFocalBalanceBCELoss, WeightedFocalLoss, CosineEmbeddingLossModule
from mmgog_long_term_sequence_model.pytorch.models.uin_gangs_model import UinGangsModel
from mmgog_long_term_sequence_model.utils.utils import print_model_size, eval_emb_with_knn, draw_and_save_pca_pic, \
    get_indicator_of_mutil_cls_base_sigmoid, get_indicator_of_mutil_cls_base_softmax, draw_and_save_loss_pic


class UinGangsModelTrain:

    def __init__(self, args_dict):
        self.train_dict = args_dict
        self.accelerator = Accelerator()
        if torch.cuda.is_available() and self.train_dict["device"] == "gpu":
            print("GPU train available")
            self.device = torch.device("cuda")
            # self.device = self.accelerator.device
        else:
            print("GPU train not available")
            self.device = torch.device("cpu")
            # self.device = self.accelerator.device

        self.model = UinGangsModel(  # device=self.device,
            uin_acs_numberical_feat_dim=self.train_dict["uin_acs_numberical_feat_dim"],
            uin_acs_text_feat_dim=self.train_dict["uin_acs_text_feat_dim"],
            uin_in_size=self.train_dict["uin_in_size"],
            uin_out_size=self.train_dict["uin_out_size"],
            drop_rate=self.train_dict["drop_rate"],
            out_size=self.train_dict["out_size"],
            # uin_hidden_size=self.train_dict["uin_hidden_size"],
        )
        self.model.to(self.device)

        # 预训练的文本embedding模型
        # self.minirbt_tokenizer = AutoTokenizer.from_pretrained(self.train_dict["minirbt_path"])
        self.minirbt_model = BertModel.from_pretrained(self.train_dict["minirbt_path"]).to(self.device)
        # 冻结文本模型的参数
        minirbt_model_params_size = 0
        for param in self.minirbt_model.parameters():
            param.requires_grad = False
            minirbt_model_params_size += param.numel()
        logger.info("minirbt_model params size: %s", minirbt_model_params_size)

        self.train_data = UinGangsDataIterable(self.train_dict, self.train_dict["train_data_path"])
        # logger.info("train_data.label_caculator:\n %s", train_data.label_caculator)
        self.test_data = UinGangsDataIterable(self.train_dict, self.train_dict["test_data_path"])
        # logger.info("test_data.label_caculator:\n %s", test_data.label_caculator)

        self.train_loader = Data.DataLoader(self.train_data,
                                            batch_size=self.train_dict["batch_size"],
                                            num_workers=self.train_dict["num_workers"],
                                            collate_fn=self.train_data.uin_gangs_collate_fn)
        self.test_loader = Data.DataLoader(self.test_data,
                                           batch_size=self.train_dict["batch_size"],
                                           num_workers=self.train_dict["num_workers"],
                                           collate_fn=self.test_data.uin_gangs_collate_fn)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_dict["lr"])
        # criterion = InfoNCELoss(temperature=self.train_dict["temperature"],
        #                         ignore_labels=[0])
        # self.criterion = InfoNCELossV2(temperature=self.train_dict["temperature"])
        self.criterion = CosineEmbeddingLossModule(
            negative_positive_ratio=self.train_dict["negative_positive_ratio"]).to(self.device)
        # self.classify_criterion = WeightedFocalBalanceBCELoss(
        #     weight=torch.tensor([1, 1, 1, 2, 3, 10], dtype=torch.float32).to(self.device),
        #     alpha=0.25,
        #     gamma=2.0,
        #     reduction='mean',
        #     max_zero_ratio=5,
        #     mask=-1)
        # self.classify_criterion = WeightedFocalLoss(
        #     weight=torch.tensor([0.5, 0.5, 0.5, 1, 1, 1], dtype=torch.float32).to(self.device),
        #     alpha=0.25,
        #     gamma=2.0,
        #     reduction='mean')
        self.classify_criterion = nn.CrossEntropyLoss()
        # criterion = InfoNCELossV3(temperature=self.train_dict["temperature"])
        # (self.model, self.optimizer, self.criterion, self.train_loader, self.test_loader) = (
        #     model, optimizer, criterion, train_loader, test_loader)
        # self.model, self.optimizer, self.criterion, self.train_loader, self.test_loader = self.accelerator.prepare(
        #     model, optimizer, criterion, train_loader, test_loader)

    def train(self):
        self.setup_seed()
        logger.info("model parameter size: %s ", print_model_size(self.model))
        print(('-' * 20 + 'training' + '-' * 40)[:60])
        x_dict = {"epoch": []}
        y_dict = {"train_similarity_loss": [], "train_classify_loss": [],
                  "train_silhouette_score": [], "test_similarity_loss": [],
                  "test_classify_loss": [], "test_silhouette_score": []}

        for epoch in range(self.train_dict["n_epochs"]):
            self.model.train()
            train_loss = []
            train_similarity_loss = []
            train_classify_loss = []
            train_root_emb_list = []
            train_label_list = []
            x_dict["epoch"].append(epoch)
            for step, batch_data in enumerate(self.train_loader):
                # print("batch_data", len(batch_data))
                start_time = time.time()
                batch_uin_acs_text_feat_input_ids = torch.cat(batch_data["batch_uin_acs_text_feat_input_ids"],
                                                              dim=0).to(self.device)
                batch_uin_acs_text_feat_attention_mask = torch.cat(batch_data["batch_uin_acs_text_feat_attention_mask"],
                                                                   dim=0).to(self.device)
                with torch.no_grad():
                    uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                           batch_uin_acs_text_feat_attention_mask).pooler_output

                batch_graph = batch_data["batch_graph"].to(self.device)
                # 补充uin类型节点的文本特征
                batch_graph.nodes['uin'].data['uin_acs_text_feat'] = uin_acs_text_feat
                batch_label = batch_data["batch_label"].to(self.device)
                # print("batch_label", batch_label)
                batch_label_one_hot = nn.functional.one_hot(batch_label, self.train_dict["cls_num"]).to(self.device)
                pre_batch_graph = self.model(batch_graph)
                # 解pre_batch_graph，取每个graph的根节点的embedding
                root_node_emb_list = [g.nodes["uin"].data["out_emb"][0] for g in dgl.unbatch(pre_batch_graph)]
                root_node_classify_out_list = [g.nodes["uin"].data["classify_out"][0] for g in
                                               dgl.unbatch(pre_batch_graph)]
                # print("root_node_emb len", len(root_node_emb_list))
                # print("root_node_emb", root_node_emb_list)
                batch_root_emb = torch.stack(root_node_emb_list).to(self.device)
                batch_root_classify_emb = torch.stack(root_node_classify_out_list).to(self.device)
                similarity_loss = self.criterion(batch_root_emb, batch_label)
                classify_loss = self.classify_criterion(batch_root_classify_emb, batch_label)
                loss = 5 * classify_loss + similarity_loss
                # loss = classify_loss
                # print("loss", loss)

                self.optimizer.zero_grad()
                # self.accelerator.backward(loss)
                loss.backward()
                self.optimizer.step()
                end_time = time.time()
                # logger.info(f"The batch training took {end_time - start_time} seconds to complete.")
                train_loss.append(loss.item())
                train_similarity_loss.append(similarity_loss.item())
                train_classify_loss.append(classify_loss.item())
                train_root_emb_list.extend(batch_root_emb.detach().cpu().numpy().tolist())
                train_label_list.extend(batch_label.detach().cpu().numpy().tolist())

                # logger.info(f"The batch training loss is:\n %s", train_loss)

                # 检验一下每个batch各个标签的分布
                # 获取唯一元素及其计数
                unique, counts = torch.unique(batch_label, return_counts=True)
                # 将结果转换为字典形式
                element_counts = dict(zip(unique.tolist(), counts.tolist()))
                # logger.info("The batch label distribution is %s", element_counts)

            # 统计训练集各标签分布
            train_label_unique, train_label_counts = torch.unique(torch.tensor(train_label_list, dtype=torch.int),
                                                                  return_counts=True)
            train_label_counts = dict(zip(train_label_unique.tolist(), train_label_counts.tolist()))

            # 计算训练集轮廓系数
            train_silhouette_score = eval_emb_with_knn(np.array(train_root_emb_list), k=self.train_dict["cls_num"])

            self.model.eval()
            test_loss = []
            test_similarity_loss = []
            test_classify_loss = []
            test_root_emb_list = []
            test_label_list = []
            y_onehot = []
            y_pred = []
            with torch.no_grad():
                for step, batch_data in enumerate(self.test_loader):
                    batch_uin_acs_text_feat_input_ids = torch.cat(batch_data["batch_uin_acs_text_feat_input_ids"],
                                                                  dim=0).to(self.device)
                    batch_uin_acs_text_feat_attention_mask = torch.cat(
                        batch_data["batch_uin_acs_text_feat_attention_mask"],
                        dim=0).to(self.device)
                    with torch.no_grad():
                        uin_acs_text_feat = self.minirbt_model(batch_uin_acs_text_feat_input_ids,
                                                               batch_uin_acs_text_feat_attention_mask).pooler_output

                    batch_graph = batch_data["batch_graph"].to(self.device)
                    # 补充uin类型节点的文本特征
                    batch_graph.nodes['uin'].data['uin_acs_text_feat'] = uin_acs_text_feat
                    batch_label = batch_data["batch_label"].to(self.device)
                    batch_label_one_hot = nn.functional.one_hot(batch_label, self.train_dict["cls_num"]).to(self.device)
                    y_onehot.extend(batch_label_one_hot.detach().cpu().numpy().tolist())
                    pre_batch_graph = self.model(batch_graph)
                    # 解pre_batch_graph，取每个graph的根节点的embedding
                    root_node_emb_list = [g.nodes["uin"].data["out_emb"][0] for g in dgl.unbatch(pre_batch_graph)]
                    root_node_classify_out_list = [g.nodes["uin"].data["classify_out"][0] for g in
                                                   dgl.unbatch(pre_batch_graph)]
                    batch_root_emb = torch.stack(root_node_emb_list).to(self.device)
                    batch_root_classify_emb = torch.stack(root_node_classify_out_list).to(self.device)
                    y_pred.extend(batch_root_classify_emb.detach().cpu().numpy().tolist())
                    similarity_loss = self.criterion(batch_root_emb, batch_label)
                    classify_loss = self.classify_criterion(batch_root_classify_emb, batch_label)
                    loss = 5 * classify_loss + similarity_loss
                    # loss = classify_loss
                    test_loss.append(loss.item())
                    test_similarity_loss.append(similarity_loss.item())
                    test_classify_loss.append(classify_loss.item())
                    test_root_emb_list.extend(batch_root_emb.detach().cpu().numpy().tolist())
                    test_label_list.extend(batch_label.detach().cpu().numpy().tolist())

            # 统计训练集各标签分布
            test_label_unique, test_label_counts = torch.unique(torch.tensor(test_label_list, dtype=torch.int),
                                                                return_counts=True)
            test_label_counts = dict(zip(test_label_unique.tolist(), test_label_counts.tolist()))

            # 计算测试集轮廓系数
            test_silhouette_score = eval_emb_with_knn(np.array(test_root_emb_list), k=self.train_dict["cls_num"])
            os.makedirs(os.path.join(self.train_dict["pic_path"], "pca"), exist_ok=True)
            pca_pic_path = os.path.join(self.train_dict["pic_path"], "pca",
                                        "pca_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png")

            draw_and_save_pca_pic(np.array(test_root_emb_list), test_label_list, save_path=pca_pic_path,
                                  title="uin_gangs")
            y_dict["train_similarity_loss"].append(np.mean(train_similarity_loss))
            y_dict["train_classify_loss"].append(np.mean(train_classify_loss))
            y_dict["train_silhouette_score"].append(train_silhouette_score)
            y_dict["test_similarity_loss"].append(np.mean(test_similarity_loss))
            y_dict["test_classify_loss"].append(np.mean(test_classify_loss))
            y_dict["test_silhouette_score"].append(test_silhouette_score)
            os.makedirs(os.path.join(self.train_dict["pic_path"], "loss"), exist_ok=True)
            loss_pic_path = os.path.join(self.train_dict["pic_path"], "loss",
                                         "loss_curve_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
            draw_and_save_loss_pic(x_dict=x_dict, y_dict=y_dict, save_path=loss_pic_path, title="loss curve")

            # if epoch % 10 == 0:
            detail_validation_info = "EPOCH %s : \n" \
                                     "train_loss=%.6f, val_loss=%.6f \n" \
                                     "train_similarity_loss=%.6f, val_similarity_loss=%.6f \n" \
                                     "train_classify_loss=%.6f, val_classify_loss=%.6f \n" \
                                     "train_silhouette_score=%.6f, val_silhouette_score=%.6f \n" % (
                                         epoch,
                                         np.mean(train_loss), np.mean(test_loss),
                                         np.mean(train_similarity_loss), np.mean(test_similarity_loss),
                                         # 0, 0,
                                         np.mean(train_classify_loss), np.mean(test_classify_loss),
                                         train_silhouette_score, test_silhouette_score)

            # 打印一下训练集和测试集的分布
            logger.info("The train label distribution is : \n %s", train_label_counts)
            logger.info("The test label distribution is : \n %s", test_label_counts)

            # thresholds = [self.train_dict["mul_cls_threshold"]] * self.train_dict["cls_num"]
            #
            # roc_auc_scores, pr_auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats = get_indicator_of_mutil_cls_base_sigmoid(
            #     np.array(y_onehot), np.array(y_pred), thresholds=thresholds)
            #
            # # 分类指标
            # for i in range(self.train_dict["cls_num"]):
            #     label_class = self.test_data.uin_gangs_enum['label_class_enums'][i]
            #     class_info = f"Class {i} = {label_class}:" \
            #                  f" roc_auc = {roc_auc_scores[i]:.4f}," \
            #                  f" pr_auc = {pr_auc_scores[i]:.4f}," \
            #                  f" precision = {precision_scores[i]:.4f}," \
            #                  f" recall = {recall_scores[i]:.4f}," \
            #                  f" f1 = {f1_scores[i]:.4f}, " \
            #                  f"confusion_matrix = \n {confusion_mats[i]} "
            #     detail_validation_info += "\n" + class_info

            auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats = get_indicator_of_mutil_cls_base_softmax(
                np.array(test_label_list), np.array(y_pred), self.train_dict["cls_num"])
            for i in range(self.train_dict["cls_num"]):
                label_class = self.test_data.uin_gangs_enum['label_class_enums'][i]
                class_info = f"Class {i} = {label_class}:" \
                             f" roc_auc = {auc_scores[i]:.4f}," \
                             f" precision = {precision_scores[i]:.4f}," \
                             f" recall = {recall_scores[i]:.4f}," \
                             f" f1 = {f1_scores[i]:.4f}, "

                detail_validation_info += "\n" + class_info

            detail_validation_info += "\n" + f"confusion_matrix = \n {confusion_mats}"

            logger.info(detail_validation_info)
            # logger.info("root_emb_list sample: \n %s", np.array(random.choices(test_root_emb_list, k=1)))
            logger.info("y_pred sample: \n %s", np.array(random.choices(y_pred, k=self.train_dict["cls_num"])))
            logger.info("test_label_list sample: \n %s",
                        np.array(random.choices(test_label_list, k=2 * self.train_dict["cls_num"])))

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True
