import os.path
import random
import logging
from datetime import datetime

import numpy as np

np.set_printoptions(suppress=True, precision=4)
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as Data
import torch
import torch.nn.functional as F

from mmgog_long_term_sequence_model.pytorch.models.LossFunction import WeightedFocalBCELoss, FocalLoss, \
    WeightedFocalLoss, WeightedFocalBalanceBCELoss
from mmgog_long_term_sequence_model.pytorch.models.basic_sequence_model import SeqBaseTransformer
from mmgog_long_term_sequence_model.utils.utils import print_model_size, \
    get_multi_cls_base_threshold_by_youden_index, get_indicator_of_mutil_cls_base_sigmoid, draw_and_save
from mmgog_long_term_sequence_model.pytorch.models.action_sequence_graph_model import SeqGraphUin2Uin
from accelerate import Accelerator

logger = logging.getLogger("my_logger")


class Train:
    def __init__(self, data, train_dict):
        self.train_dict = train_dict
        self.model = SeqBaseTransformer(vocab_size=self.train_dict["vocab_size"],
                                        max_len=self.train_dict["max_len"],
                                        n_layers=self.train_dict["n_layers"],
                                        emb_dim=self.train_dict["emb_dim"],
                                        n_heads=self.train_dict["n_heads"],
                                        output_size=self.train_dict["output_size"],
                                        drop_rate=self.train_dict["drop_rate"],
                                        padding_idx=self.train_dict["padding_idx"])
        if torch.cuda.is_available() and self.train_dict["device"] == "gpu":
            print("GPU train available")
            device = torch.device("cuda")
            self.model.cuda()
        else:
            print("GPU train not available")
            device = torch.device("cpu")
            self.model.cpu()

        train_x = torch.from_numpy(data.train_x.values.astype(int)).type(torch.LongTensor).to(
            device)
        test_x = torch.from_numpy(data.test_x.values.astype(int)).type(torch.LongTensor).to(
            device)
        train_y = torch.from_numpy(data.train_y.astype(float)).type(torch.LongTensor).to(
            device)
        test_y = torch.from_numpy(data.test_y.astype(float)).type(torch.LongTensor).to(
            device)
        train_data = Data.TensorDataset(train_x, train_y)
        test_data = Data.TensorDataset(test_x, test_y)
        self.train_loader = Data.DataLoader(dataset=train_data,
                                            batch_size=self.train_dict["batch_size"],
                                            shuffle=True)
        self.test_loader = Data.DataLoader(dataset=test_data,
                                           batch_size=self.train_dict["batch_size"],
                                           shuffle=True)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_dict["lr"])

    def train(self):
        self.setup_seed()
        print(('-' * 20 + 'model parameter size' + '-' * 40)[:60])
        print_model_size(self.model)
        print(('-' * 20 + 'training' + '-' * 40)[:60])
        for epoch in range(self.train_dict["n_epochs"]):
            self.model.train()
            train_loss = []
            for batch_idx, (x, y) in enumerate(self.train_loader):
                pred = self.model(x)
                # print("pred.shape", pred.shape)
                # print("y.shape", y.shape)
                loss = self.criterion(pred, y.float().detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            self.model.eval()
            test_loss = []
            prediction = []
            y_true = []
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(self.test_loader):
                    pred = self.model(x)
                    loss = self.criterion(pred, y.float().detach())
                    test_loss.append(loss.item())
                    prediction.extend(pred.detach().cpu().numpy().tolist())
                    y_true.extend(y.detach().cpu().numpy().tolist())
            test_auc = roc_auc_score(y_true=y_true, y_score=prediction)
            print("EPOCH %s train loss : %.5f   validation loss : %.5f   validation auc is %.5f" % (
                epoch, np.mean(train_loss), np.mean(test_loss), test_auc))
            if test_auc > self.train_dict["best_auc"]:
                torch.save(self.model.state_dict(), self.train_dict["model_states_path"])
                print(('-' * 20 + 'model saved' + '-' * 40)[:60])
                return train_loss, test_loss, test_auc
        return train_loss, test_loss, test_auc

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True


class TrainSeqGraph:
    def __init__(self, data, train_dict):
        self.train_dict = train_dict
        self.data = data
        self.accelerator = Accelerator()
        model = SeqGraphUin2Uin(input_feat=data.input_feat_size,
                                target_feat=data.target_feat_size,
                                vocab_size=data.action_vocab_size,
                                max_len=self.train_dict["max_len"],
                                n_layers=self.train_dict["n_layers"],
                                emb_dim=self.train_dict["emb_dim"],
                                n_heads=self.train_dict["n_heads"],
                                output_size=self.train_dict["output_size"],
                                drop_rate=self.train_dict["drop_rate"],
                                padding_idx=data.special_tokens["[PAD]"])

        if torch.cuda.is_available() and self.train_dict["device"] == "gpu":
            logger.info("GPU train available")
            self.device = self.accelerator.device
        else:
            logger.info("GPU train not available")
            self.device = torch.device("cpu")

        train_x = torch.from_numpy(data.train_seq_token_id.values.astype(int)).type(torch.LongTensor).to(
            self.device)
        train_input_x = torch.tensor(data.train_input_feat, dtype=torch.float32).to(self.device)
        train_target_x = torch.tensor(data.train_target_feat, dtype=torch.float32).to(self.device)

        test_x = torch.from_numpy(data.test_seq_token_id.values.astype(int)).type(torch.LongTensor).to(
            self.device)
        test_input_x = torch.tensor(data.test_input_feat, dtype=torch.float32).to(self.device)
        test_target_x = torch.tensor(data.test_target_feat, dtype=torch.float32).to(self.device)

        train_y = torch.from_numpy(data.train_y.values.astype(int)).type(torch.LongTensor).to(
            self.device).squeeze(1)
        test_y = torch.from_numpy(data.test_y.values.astype(int)).type(torch.LongTensor).to(
            self.device).squeeze(1)

        label_weights = torch.tensor(data.label_weights, dtype=torch.float32).to(self.device)
        training_data_weights = label_weights[train_y]
        generator = torch.Generator(device=self.device)
        train_sampler = WeightedRandomSampler(training_data_weights, len(training_data_weights), generator=generator)

        train_data = Data.TensorDataset(train_x, train_input_x, train_target_x, train_y)
        test_data = Data.TensorDataset(test_x, test_input_x, test_target_x, test_y)
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=self.train_dict["batch_size"],
                                       shuffle=True,
                                       )
        test_loader = Data.DataLoader(dataset=test_data,
                                      batch_size=self.train_dict["batch_size"],
                                      shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_dict["lr"])
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(data.label_weights).to(self.device))
        # self.criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        # self.criterion = WeightedFocalLoss(weight=torch.tensor(data.label_weights).to(self.device),
        #                                    alpha=0.25,
        #                                    gamma=2.0,
        #                                    reduction='mean')
        # self.criterion = WeightedFocalBCELoss(weight=torch.tensor(data.label_weights).to(self.device),
        #                                       alpha=0.25,
        #                                       gamma=2.0,
        #                                       reduction='mean')
        self.criterion = WeightedFocalBalanceBCELoss(weight=torch.tensor(data.label_weights).to(self.device),
                                                     alpha=0.25,
                                                     gamma=2.0,
                                                     reduction='mean',
                                                     max_zero_ratio=5,
                                                     mask=-1)
        print(torch.tensor(data.label_weights).shape)
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(model,
                                                                                                   optimizer,
                                                                                                   train_loader,
                                                                                                   test_loader)

    def train(self):
        self.setup_seed()
        logger.info("model parameter size: %s ", print_model_size(self.model))
        print(('-' * 20 + 'training' + '-' * 40)[:60])
        x_dict = {"epoch": []}
        y_dict = {"train_loss": [], "test_loss": []}

        for epoch in range(self.train_dict["n_epochs"]):
            self.model.train()
            train_loss = []
            for batch_idx, (x, input_x, target_x, y) in enumerate(self.train_loader):
                y = F.one_hot(y, self.data.num_classes)

                pred = self.model(x, input_x, target_x)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                train_loss.append(loss.item())

            self.model.eval()
            test_loss = []
            y_pred_original = []
            y_onehot = []
            y_pred = []
            y_true = []
            with torch.no_grad():
                for batch_idx, (x, input_x, target_x, y) in enumerate(self.test_loader):
                    y = F.one_hot(y, self.data.num_classes)
                    y_onehot.extend(y.detach().cpu().numpy().tolist())

                    pred = self.model(x, input_x, target_x)
                    loss = self.criterion(pred, y)
                    test_loss.append(loss.item())

                    # pred = torch.sigmoid(pred)

                    y_pred.extend(np.argmax(pred.detach().cpu().numpy(), axis=1))
                    # y_true.extend(y.detach().cpu().numpy().tolist())
                    y_true.extend(np.argmax(y.detach().cpu().numpy(), axis=1))
                    y_pred_original.extend(pred.detach().cpu().numpy().tolist())

                # thresholds = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
                roc_auc_scores, pr_auc_scores, precision_scores, recall_scores, f1_scores, confusion_mats = get_indicator_of_mutil_cls_base_sigmoid(
                    np.array(y_onehot), np.array(y_pred_original))

                if epoch % 10 == 0:
                    x_dict["epoch"].append(epoch)
                    y_dict["train_loss"].append(np.mean(train_loss))
                    y_dict["test_loss"].append(np.mean(test_loss))
                    # 随机打印5个预测结果
                    logger.info("y_pred_original sample:\n %s", np.array(random.choices(y_pred_original, k=5)))
                    # # 混淆矩阵
                    # logger.info("conf_matrix: \n%s", conf_matrix)
                    # # 总指标
                    # detail_validation_info = "EPOCH %s : train_loss=%.5f, validation_loss=%.5f, auc=%.5f, precision=%.5f, f1=%.5f" % (
                    #     epoch, np.mean(train_loss), np.mean(test_loss), total_auc, total_precision,
                    #     total_f1)

                    detail_validation_info = "EPOCH %s : train_loss=%.6f, validation_loss=%.6f" % (
                        epoch, np.mean(train_loss), np.mean(test_loss))

                    # 分类指标
                    for i in range(self.data.num_classes):
                        label_class = self.data.label_dict[i]
                        class_info = f"Class {i} = {label_class}:" \
                                     f" roc_auc = {roc_auc_scores[i]:.4f}," \
                                     f" pr_auc = {pr_auc_scores[i]:.4f}," \
                                     f" precision = {precision_scores[i]:.4f}," \
                                     f" recall = {recall_scores[i]:.4f}," \
                                     f" f1 = {f1_scores[i]:.4f}, " \
                                     f"confusion_matrix = \n {confusion_mats[i]} "
                        detail_validation_info += "\n" + class_info

                    logger.info(detail_validation_info)

                # 保存模型
                if np.mean(test_loss) < self.train_dict["best_loss"]:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        # 将模型从分布式状态中提取出来
                        unwrapped_model = self.accelerator.unwrap_model(self.model).to(self.train_dict["infer_device"])
                        # 设置模型为评估模式
                        unwrapped_model.eval()

                        # 将模型参数保存
                        torch.save(unwrapped_model.state_dict(), self.train_dict["model_states_path"])
                        logger.info(f"Best model states saved at {self.train_dict['model_states_path']}")

                        # 将模型保存为onnx
                        self.save_model_as_onnx(unwrapped_model)
                        logger.info(f"Best model onnx saved at {self.train_dict['model_onnx_path']}")

                    return train_loss, test_loss

        loss_pic_path = os.path.join(self.train_dict["pic_path"],
                                     "loss_curve_" + datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
        draw_and_save(x_dict=x_dict, y_dict=y_dict, save_path=loss_pic_path, title="loss curve")
        return train_loss, test_loss

    def save_model_as_onnx(self, model):
        seqs_x = torch.randint(0, 10, (1, self.train_dict["max_len"]))
        input_x = torch.randn(1, self.data.input_feat_size)
        target_x = torch.randn(1, self.train_dict["max_len"], self.data.target_feat_size)
        torch.onnx.export(model,
                          (seqs_x, input_x, target_x),
                          self.train_dict["model_onnx_path"],
                          export_params=True,
                          opset_version=self.train_dict["onnx_opset"],
                          do_constant_folding=True,
                          input_names=['seqs_x', 'input_x', 'target_x'],
                          output_names=['mutil_cls_pred'],
                          dynamic_axes={'seqs_x': {0: 'batch_size'},
                                        'input_x': {0: 'batch_size'},
                                        'target_x': {0: 'batch_size'},
                                        'mutil_cls_pred': {0: 'batch_size'}})
        return

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True
