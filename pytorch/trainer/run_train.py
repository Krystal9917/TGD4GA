import argparse
import os
import sys
import logging

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.utils.argument import Args, ArgsSeqGraph
from mmgog_long_term_sequence_model.pytorch.trainer.training import Train, TrainSeqGraph
from mmgog_long_term_sequence_model.utils.data_process import DataProcess, DataProcessSeqGraph
from mmgog_long_term_sequence_model.utils.utils import start_log, export_requirements
from mmgog_long_term_sequence_model.pytorch.inference.optimal_uin_sequence_model_onnx_infer import SeqGraphUin2UinInfer


def run():
    args = Args()
    print(('-' * 20 + 'args_dict' + '-' * 40)[:60])
    print(args.args_dict)
    data = DataProcess(args.args_dict)
    train_model = Train(data, args.args_dict)
    train_model.train()


def run_seqs_graph():
    start_log()
    args = ArgsSeqGraph()
    logger.info("args_dict:\n %s", args.args_dict)
    data = DataProcessSeqGraph(args.args_dict)
    train_model = TrainSeqGraph(data, args.args_dict)
    train_model.train()


def run_export_requirements():
    args = ArgsSeqGraph()
    export_requirements(args.args_dict["export_requirements_path"])


def infer_seq_graph():
    args = ArgsSeqGraph()
    data = DataProcessSeqGraph(args.args_dict)
    infer_obj = SeqGraphUin2UinInfer(data, args.args_dict)
    infer_obj.infer()


if __name__ == '__main__':
    # run_export_requirements()
    # 初始化日志设置
    run_seqs_graph()
    # infer_seq_graph()
