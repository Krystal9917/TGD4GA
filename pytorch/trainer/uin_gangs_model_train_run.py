import argparse
import os
import sys
import logging

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.utils.utils import start_log
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_train import UinGangsModelTrain


class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ds', type=str, default="")
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "train",
                         "240812_2")))
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "eval",
                         "240812_2")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "saved_model", "model_states",
                         "optimal_uin_sequence_model" + ".bin")))
        parser.add_argument('--model_onnx_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "saved_model", "model_onnx",
                         "optimal_uin_sequence_model" + ".onnx")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--pic_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "pic")))
        parser.add_argument('--best_loss', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=4096)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_epochs', type=int, default=5)
        parser.add_argument('--uin_in_size', type=int, default=323)
        parser.add_argument('--uin_hidden_size', type=int, default=512)
        parser.add_argument('--uin_out_size', type=int, default=128)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--device', type=str, default="gpu")
        parser.add_argument('--infer_device', type=str, default="cpu")
        parser.add_argument('--onnx_opset', type=int, default=15)
        parser.add_argument('--seed', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--temperature', type=int, default=0.5)
        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_seqs_graph():
    start_log()
    args = ArgsUinGangs()
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelTrain(args.args_dict)
    train_model.train()


if __name__ == '__main__':
    run_seqs_graph()
