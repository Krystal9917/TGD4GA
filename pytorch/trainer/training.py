import random
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.utils.data as Data
import torch
from mmgog_long_term_sequence_model.pytorch.models.basic_sequence_model import SeqBaseTransformer
from mmgog_long_term_sequence_model.utils.utils import print_model_size
from mmgog_long_term_sequence_model.pytorch.models.action_sequence_graph_model import SeqGraphUin2Uin, FocalLoss, \
    WeightedFocalLoss
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
        model = SeqGraphUin2Uin(target_feat=data.target_feat_size,
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
        train_target_x = torch.tensor(data.train_target_feat, dtype=torch.float32).to(self.device)

        test_x = torch.from_numpy(data.test_seq_token_id.values.astype(int)).type(torch.LongTensor).to(
            self.device)
        test_target_x = torch.tensor(data.test_target_feat, dtype=torch.float32).to(self.device)

        train_y = torch.from_numpy(data.train_y.values.astype(int)).type(torch.LongTensor).to(
            self.device).squeeze(1)
        test_y = torch.from_numpy(data.test_y.values.astype(int)).type(torch.LongTensor).to(
            self.device).squeeze(1)

        label_weights = torch.tensor(data.label_weights, dtype=torch.float32).to(self.device)
        training_data_weights = label_weights[train_y]
        generator = torch.Generator(device=self.device)
        train_sampler = WeightedRandomSampler(training_data_weights, len(training_data_weights), generator=generator)

        train_data = Data.TensorDataset(train_x, train_target_x, train_y)
        test_data = Data.TensorDataset(test_x, test_target_x, test_y)
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=self.train_dict["batch_size"],
                                       shuffle=True,
                                       )
        test_loader = Data.DataLoader(dataset=test_data,
                                      batch_size=self.train_dict["batch_size"],
                                      shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_dict["lr"])
        # self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(data.label_weights).to(self.device))
        # self.criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
        self.criterion = WeightedFocalLoss(weight=torch.tensor(data.label_weights).to(self.device),
                                           alpha=0.25,
                                           gamma=2.0,
                                           reduction='mean')
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(model,
                                                                                                   optimizer,
                                                                                                   train_loader,
                                                                                                   test_loader)

    def train(self):
        self.setup_seed()
        logger.info("model parameter size: %s ", print_model_size(self.model))
        print(('-' * 20 + 'training' + '-' * 40)[:60])

        for epoch in range(self.train_dict["n_epochs"]):
            self.model.train()
            train_loss = []
            for batch_idx, (x, target_x, y) in enumerate(self.train_loader):
                pred = self.model(x, target_x)
                # print("pred: ", pred.shape)
                # print("y: ", y.shape)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                train_loss.append(loss.item())

            self.model.eval()
            test_loss = []
            y_pred_original = []
            y_pred = []
            y_true = []
            with torch.no_grad():
                for batch_idx, (x, target_x, y) in enumerate(self.test_loader):
                    pred = self.model(x, target_x)
                    loss = self.criterion(pred, y)
                    test_loss.append(loss.item())
                    y_pred.extend(np.argmax(pred.detach().cpu().numpy(), axis=1))
                    y_true.extend(y.detach().cpu().numpy().tolist())
                    y_pred_original.extend(pred.detach().cpu().numpy().tolist())
                # with torch.no_grad():
                #     x, target_x, y = self.test_data.tensors
                #     y = y.squeeze(1)
                #     pred = self.model(x, target_x)
                #     validation_loss = self.criterion(pred, y)
                #
                #     # 转numpy
                #     y_true = y.detach().cpu().numpy()
                #     y_pred = np.argmax(pred.detach().cpu().numpy(), axis=1)

                # 计算评估指标
                total_accuracy = accuracy_score(y_true, y_pred)
                total_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
                total_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
                total_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
                conf_matrix = confusion_matrix(y_true, y_pred)

                # 计算每个类别的精确度、召回率和F1分数
                precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                f1_score_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

                if epoch % 10 == 0:
                    # 随机打印5个预测结果
                    logger.info("y_pred_original sample:\n %s", np.array(random.choices(y_pred_original, k=5)))
                    # 混淆矩阵
                    logger.info("conf_matrix: \n%s", conf_matrix)
                    # 总指标
                    detail_validation_info = "EPOCH %s : train_loss=%.5f, validation_loss=%.5f, accuracy=%.5f, precision=%.5f, recall=%.5f, f1=%.5f" % (
                        epoch, np.mean(train_loss), np.mean(test_loss), total_accuracy, total_precision, total_recall,
                        total_f1)

                    # 分类指标
                    for i, (precision, recall, f1) in enumerate(
                            zip(precision_per_class, recall_per_class, f1_score_per_class)):
                        label_class = self.data.label_dict[i]
                        class_info = f"Class {i} = {label_class}: precision = {precision:.2f}, recall = {recall:.2f}, f1 = {f1:.2f}"
                        detail_validation_info += "\n" + class_info

                    logger.info(detail_validation_info)
        return train_loss, test_loss, precision, recall, f1

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True
