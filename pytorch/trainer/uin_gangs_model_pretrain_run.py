import os
import sys
import time
import torch
import logging
import argparse
logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_pretrain import UinGangsModelPreTrain

class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "train", "raw",
                         "uin_gangs_full_graph_dataset_train_241119_241121.txt")))
                         # "uin_gangs_full_graph_dataset", "valid", "processed",
                         # "uin_gangs_supervise_full_graph_dataset_train_241204_20241204.txt")))
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "eval", "raw",
                         "uin_gangs_full_graph_dataset_train_241119_1.txt")))
                         # "uin_gangs_full_graph_dataset", "valid", "processed",
                         # "uin_gangs_supervise_full_graph_dataset_eval_241204_20241204.txt")))
        parser.add_argument('--eval_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "raw",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241204_combined.txt")))
        parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "config", "yml", "uin_gangs_enum.yaml")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "GNN_models", "pretraining_filter_subgraph_cl")))
        parser.add_argument('--cls_model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "cls_models", "mlp_")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--log_dir', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "losses")))
        parser.add_argument('--pic_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "pic")))
        parser.add_argument('--minirbt_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "minirbt-h256")))
        parser.add_argument('--best_loss', type=float, default=5)
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=40)
        parser.add_argument('--data_buffer_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_epochs', type=int, default=100)
        parser.add_argument('--uin_in_size', type=int, default=846)
        parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=290)
        parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
        parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=300)
        parser.add_argument('--uin_hidden_size', type=int, default=512)
        parser.add_argument('--uin_out_size', type=int, default=128)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--device', type=str, default="gpu")
        parser.add_argument('--infer_device', type=str, default="cpu")
        parser.add_argument('--onnx_opset', type=int, default=15)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=1)
        parser.add_argument('--input_dim', type=int, default=846)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=846)
        parser.add_argument('--num_relations', type=int, default=10)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--filter_node_num', type=int, default=5)
        parser.add_argument('--pretraining_alpha', type=float, default=1)
        parser.add_argument('--pretraining_beta', type=float, default=1)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--is_train', type=bool, default=True)
        parser.add_argument('--re_train', type=bool, default=False)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--eval_epoch', type=int, default=0)
        parser.add_argument('--evaluate_task', type=str, default='',
                            choices=['eval_labelled_subgraph_embedding', 'eval_subgraph_embedding',
                                     'eval_node_embedding', 'predict', ''])
        parser.add_argument('--conv_type', type=str, default='RGCN', choices=['RGCN', 'HAN'])

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_pretraining_graph(args, sampling_type):
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelPreTrain(args.args_dict)
    if sampling_type == 'fraudar':
        print("Fraudar pretraining")
        train_model.pretraining()
    elif sampling_type == 'random':
        print("Random pretraining")
        train_model.random_sampling_pretraining()


def run_evaluate_node(args):
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelPreTrain(args.args_dict)
    train_model.evaluate_node_embedding()


def run_evaluate_graph(args):
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelPreTrain(args.args_dict)
    train_model.evaluate_subgraph_embedding()


def run_evaluate_labelled_graph(args):
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelPreTrain(args.args_dict)
    train_model.evaluate_labelled_subgraph_embedding()

def run_predict(args):
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelPreTrain(args.args_dict)
    train_model.evaluate_subgraph_predict()


if __name__ == '__main__':
    args = ArgsUinGangs()
    # args.args_dict["is_train"] = False
    if args.args_dict["is_train"]:
        run_pretraining_graph(args, args.args_dict["sampling"])
    else:
        if args.args_dict["evaluate_task"] == 'eval_subgraph_embedding':
            print("Subgraph evaluating")
            run_evaluate_graph(args)
        elif args.args_dict["evaluate_task"] == 'eval_labelled_subgraph_embedding':
            print("Labelled subgraph evaluating")
            run_evaluate_labelled_graph(args)
        elif args.args_dict["evaluate_task"] == 'eval_node_embedding':
            print("Node evaluating")
            run_evaluate_node(args)
        elif args.args_dict["evaluate_task"] == 'predict':
            print("Subgraph predicting")
            run_predict(args)

