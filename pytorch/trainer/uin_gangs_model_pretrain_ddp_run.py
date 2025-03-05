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
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_pretrain_ddp import UinGangsModelPreTrainDDP


class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "train", "raw",
                         "uin_gangs_full_graph_dataset_train_241119_241130.txt")))
        parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "config", "yml", "uin_gangs_enum.yaml")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "pretrained_models")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--log_dir', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "losses")))
        parser.add_argument('--pic_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "pic")))
        parser.add_argument('--minirbt_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "minirbt-h256")))
        parser.add_argument('--best_loss', type=float, default=10)
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=48)
        parser.add_argument('--data_buffer_size', type=int, default=48)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--n_epochs', type=int, default=15)
        parser.add_argument('--uin_in_size', type=int, default=846)
        parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=290)
        parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
        parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=300)
        parser.add_argument('--uin_hidden_size', type=int, default=512)
        parser.add_argument('--uin_out_size', type=int, default=128)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--onnx_opset', type=int, default=15)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=64)
        parser.add_argument('--temperature', type=int, default=1)
        parser.add_argument('--input_dim', type=int, default=846)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=846)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--re_train', type=bool, default=False)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--conv_type', type=str, default='MaskRGCN', choices=['RGCN', 'MaskRGCN'])
        parser.add_argument('--task_type', type=str, default='fine_grained_cross_subgraph',
                            choices=['subgraph', 'intra_subgraph', 'batch_subgraph', 'cross_subgraph',
                                     'fine_grained_batch_subgraph', 'fine_grained_cross_subgraph'])
        # RGCN
        parser.add_argument('--num_relations', type=int, default=11)
        # HAN/HGT
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--world_size', type=int, default=2)
        parser.add_argument('--data_tag', type=str, default='order_1930_', choices=['', '1921_', '1930_', 'order_1930_'])
        parser.add_argument('--similarity_diff', type=float, default=0.15)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_pretraining_graph_ddp(rank, args):
    print(f"Rank: {rank}, Parameters: {args.args_dict}")
    sampling_type = args.args_dict["sampling"]
    train_model = UinGangsModelPreTrainDDP(rank, args.args_dict)
    if sampling_type == 'fraudar':
        print("======Fraudar DDP Pretraining======")
        train_model.pretraining_ddp()


if __name__ == '__main__':
    args = ArgsUinGangs()
    num_gpu = torch.cuda.device_count()
    print(f"Current GPUs: {num_gpu}")
    args.args_dict["world_size"] = num_gpu
    mp.spawn(run_pretraining_graph_ddp, args=(args,),
             nprocs=args.args_dict["world_size"], join=True)
