import random

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch
from mmgog_long_term_sequence_model.pytorch.models.basic_sequence_model import SeqBaseTransformer
from mmgog_long_term_sequence_model.utils.utils import print_model_size


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
