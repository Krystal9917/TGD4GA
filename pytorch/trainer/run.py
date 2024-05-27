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
from mmgog_long_term_sequence_model.utils.utils import start_log


def run():
    args = Args()
    print(('-' * 20 + 'args_dict' + '-' * 40)[:60])
    print(args.args_dict)
    data = DataProcess(args.args_dict)
    train_model = Train(data, args.args_dict)
    train_model.train()


def run_seqs_graph():
    args = ArgsSeqGraph()
    logger.info("args_dict:\n %s", args.args_dict)
    data = DataProcessSeqGraph(args.args_dict)
    train_model = TrainSeqGraph(data, args.args_dict)
    train_model.train()


if __name__ == '__main__':
    # 初始化日志设置
    start_log()
    run_seqs_graph()
