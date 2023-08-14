import argparse
import os
import sys

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.utils.argument import Args
from mmgog_long_term_sequence_model.pytorch.trainer.training import Train
from mmgog_long_term_sequence_model.utils.data_process import DataProcess


def run():
    args = Args()
    print(('-' * 20 + 'args_dict' + '-' * 40)[:60])
    print(args.args_dict)
    data = DataProcess(args.args_dict)
    train_model = Train(data, args.args_dict)
    train_model.train()


if __name__ == '__main__':
    print(('-' * 20 + 'start' + '-' * 40)[:60])
    run()
    print(('-' * 20 + 'end' + '-' * 40)[:60])
