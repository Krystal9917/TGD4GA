import os
import sys
import logging
import argparse

import torch
import torch.multiprocessing as mp

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 os.path.pardir,
                 os.path.pardir,
                 os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.trainer.hetero_model_pretrain import ModelPreTrain
from mmgog_long_term_sequence_model.pytorch.trainer.hetero_model_pretrain_ddp import ModelPreTrainDDP


class ArgsHeteroGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # dataset
        parser.add_argument('--dataset_name', type=str, default="IMDB", choices=["IMDB", "DBLP", 'ACM'])
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "hetero_datasets")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "saved_model",
                         "GNN_models")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--log_dir', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "losses")))
        parser.add_argument('--best_loss', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=1024)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--print_batch_num', type=int, default=20)
        parser.add_argument('--re_train', type=bool, default=False)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--similarity_diff', type=float, default=0.15)
        parser.add_argument('--conv_type', type=str, default='RGCN', choices=['RGCN', 'HAN', 'HGT'])
        parser.add_argument('--task_type', type=str, default='fine_grained_batch_subgraph',
                            choices=['subgraph', 'batch_subgraph', 'cross_subgraph',
                                     'fine_grained_batch_subgraph', 'fine_grained_cross_subgraph'])
        # RGCN
        parser.add_argument('--world_size', type=int, default=2)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_pretraining_graph_ddp(rank, args):
    print(f"Rank: {rank}, Parameters: {args.args_dict}")
    train_model = ModelPreTrainDDP(rank, args.args_dict)
    print(f"======{args.args_dict['dataset_name']} DDP Pretraining======")
    train_model.pretraining()


def run_pretraining_graph(args):
    print(f"Parameters: {args.args_dict}")
    train_model = ModelPreTrain(args.args_dict)
    print(f"======{args.args_dict['dataset_name']} Pretraining======")
    train_model.pretraining()


if __name__ == '__main__':
    args = ArgsHeteroGangs()
    print(f"Current GPUs: {torch.cuda.device_count()}")
    if args.args_dict["world_size"] == 1:
        run_pretraining_graph(args)
    else:
        mp.spawn(run_pretraining_graph_ddp, args=(args,),
                 nprocs=args.args_dict["world_size"], join=True)
