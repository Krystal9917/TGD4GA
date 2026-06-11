import os
import sys
import logging
import argparse

import torch

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 os.path.pardir,
                 os.path.pardir,
                 os.path.pardir)))
from codes.trainer.public_data_model_pretrain import PublicDataModelPreTrain


class ArgsPublicDataGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # dataset
        parser.add_argument('--dataset_name', type=str, default="tfinance", choices=["weibo", "fb", "Amazon", "tfinance"])
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "public_dataset")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "saved_model",
                         "public_dataset_GNN_models")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--log_dir', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "losses")))
        parser.add_argument('--best_loss', type=float, default=1e2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--n_epochs', type=int, default=50)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=0.8)
        parser.add_argument('--filter_node_num', type=int, default=10)
        parser.add_argument('--print_batch_num', type=int, default=5)
        parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT'])
        parser.add_argument('--task_type', type=str, default='node_subgraph',
                            choices=['node_subgraph', 'node', 'subgraph'])
        parser.add_argument('--top_neigh_ratio', type=float, default=0.2)
        parser.add_argument('--sub_weight', type=float, default=0.5)
        parser.add_argument('--node_weight', type=float, default=0.5)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_pretraining_graph(args):
    print(f"Parameters: {args.args_dict}")
    train_model = PublicDataModelPreTrain(args.args_dict)
    print(f"======{args.args_dict['dataset_name']} Pretraining======")
    train_model.pretraining()


if __name__ == '__main__':
    args = ArgsPublicDataGangs()
    print(f"Current GPUs: {torch.cuda.device_count()}")
    run_pretraining_graph(args)
