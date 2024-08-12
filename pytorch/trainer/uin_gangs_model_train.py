import logging

import numpy as np
import random
import torch
import torch.utils.data as Data
from accelerate import Accelerator

from mmgog_long_term_sequence_model.pytorch.dataprocess.data_process_iterable import DataProcess, \
    DataIterable
from mmgog_long_term_sequence_model.pytorch.models.LossFunction import InfoNCELoss
from mmgog_long_term_sequence_model.pytorch.models.uin_gangs_model import UinGangsModel
from mmgog_long_term_sequence_model.utils.utils import print_model_size

logger = logging.getLogger("my_logger")


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

        model = UinGangsModel(device=self.device,
                              uin_in_size=self.train_dict["uin_in_size"],
                              uin_out_size=self.train_dict["uin_out_size"],
                              uin_hidden_size=self.train_dict["uin_hidden_size"],
                              )
        model.to(self.device)

        if torch.cuda.is_available() and self.train_dict["device"] == "gpu":
            print("GPU train available")
            self.device = torch.device("cuda")
            model.cuda()
            # self.device = self.accelerator.device
        else:
            print("GPU train not available")
            self.device = torch.device("cpu")
            model.cpu()
            # self.device = self.accelerator.device

        data_process = DataProcess(self.train_dict)
        train_data = DataIterable(self.train_dict, self.train_dict["train_data_path"])
        test_data = DataIterable(self.train_dict, self.train_dict["test_data_path"])

        train_loader = Data.DataLoader(train_data,
                                       batch_size=self.train_dict["batch_size"],
                                       num_workers=self.train_dict["num_workers"],
                                       collate_fn=data_process.uin_gangs_collate_fn)
        test_loader = Data.DataLoader(test_data,
                                      batch_size=self.train_dict["batch_size"],
                                      num_workers=self.train_dict["num_workers"],
                                      collate_fn=data_process.uin_gangs_collate_fn)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_dict["lr"])
        criterion = InfoNCELoss(device=self.device, temperature=self.train_dict["temperature"], ignore_labels=None).to(
            self.device)
        (self.model, self.optimizer, self.criterion, self.train_loader, self.test_loader) = (
            model, optimizer, criterion, train_loader, test_loader)
        # self.model, self.optimizer, self.criterion, self.train_loader, self.test_loader = self.accelerator.prepare(
        #     model, optimizer, criterion, train_loader, test_loader)

    def train(self):
        self.setup_seed()
        logger.info("model parameter size: %s ", print_model_size(self.model))
        print(('-' * 20 + 'training' + '-' * 40)[:60])

        for epoch in range(self.train_dict["n_epochs"]):
            self.model.train()
            train_loss = []
            for batch_data in self.train_loader:
                # print("batch_graph", batch_data)
                batch_graph = batch_data["batch_graph"].to(self.device)
                batch_label = batch_data["batch_label"].to(self.device)
                batch_root_h_dict2, batch_root_emb = self.model(batch_graph.to(self.device))
                loss = self.criterion(batch_root_emb, batch_label)
                self.optimizer.zero_grad()
                # self.accelerator.backward(loss)
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            self.model.eval()
            test_loss = []
            with torch.no_grad():
                for step, batch_data in enumerate(self.test_loader):
                    batch_root_h_dict2, batch_root_emb = self.model(batch_data["batch_graph"])
                    loss = self.criterion(batch_root_emb, batch_data["batch_label"])
                    test_loss.append(loss.item())

                if epoch % 10 == 0:
                    logger.info("EPOCH %s : train_loss=%.6f, validation_loss=%.6f" % (
                        epoch, np.mean(train_loss), np.mean(test_loss)))

    def setup_seed(self):
        torch.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed(self.train_dict["seed"])
        torch.cuda.manual_seed_all(self.train_dict["seed"])
        np.random.seed(self.train_dict["seed"])
        random.seed(self.train_dict["seed"])
        torch.backends.cudnn.deterministic = True
