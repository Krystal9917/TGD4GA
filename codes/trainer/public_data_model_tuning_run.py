import os
import sys
import time
import logging
import argparse

import torch
import numpy as np

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 os.path.pardir,
                 os.path.pardir,
                 os.path.pardir)))
from codes.trainer.public_data_model_tuning import PublicDataModelTune


class ArgsPublicDataGroups:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # dataset
        parser.add_argument('--dataset_name', type=str, default="Amazon", choices=["weibo", "fb", "Amazon", "tfinance"])
        parser.add_argument('--data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "public_dataset")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "saved_model",
                         "public_dataset_GNN_models")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--n_epochs', type=int, default=30)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--print_batch_num', type=int, default=2)
        parser.add_argument('--conv_type', type=str, default='GAT', choices=['GCN', 'GAT'])
        parser.add_argument('--task_type', type=str, default='node_subgraph',
                            choices=['node_subgraph', 'node', 'subgraph'])
        # Evaluation
        parser.add_argument('--is_finetune', type=bool, default=True)
        parser.add_argument('--evaluate_times', type=int, default=5)
        parser.add_argument('--evaluate_epoch', type=int, default=0)
        parser.add_argument('--evaluate_lr', type=float, default=0.0005)
        parser.add_argument('--node_classes', type=int, default=2)
        parser.add_argument('--k_shot', type=int, default=10)
        parser.add_argument('--fold', type=int, default=1)
        parser.add_argument('--top_k', type=int, default=5)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict

def run_tuning_graph(args):
    print(f"Parameters: {args.args_dict}")
    tune_model = PublicDataModelTune(args.args_dict)
    print(f"======{args.args_dict['dataset_name']} Tuning======")
    best_acc, best_f1, best_pre, best_rec, best_roc_auc = tune_model.training()
    return best_acc, best_f1, best_pre, best_rec, best_roc_auc

def compute_metrics_ave_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list):
    acc_ = np.array(acc_list)
    pre_ = np.array(pre_list)
    rec_ = np.array(rec_list)
    f1_ = np.array(f1_list)
    roc_auc_ = np.array(roc_auc_list)
    return (acc_.std(), pre_.std(), rec_.std(), f1_.std(), roc_auc_.std(),
            acc_.mean(), pre_.mean(), rec_.mean(), f1_.mean(), roc_auc_.mean())


if __name__ == '__main__':
    args = ArgsPublicDataGroups()
    print(f"Current GPUs: {torch.cuda.device_count()}")
    dataset_name = args.args_dict["dataset_name"]
    if dataset_name == 'weibo':
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    elif dataset_name == 'fb':
        start_k_shot = 1
        end_k_shot = 21
        step_k_shot = 1
    elif dataset_name == 'Amazon':
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    else:
        start_k_shot = 10
        end_k_shot = 110
        step_k_shot = 10
    for k in range(start_k_shot, end_k_shot, step_k_shot):
        print(f"******Start {args.args_dict['dataset_name']} {str(k)}-Shot Tuning******")
        st = time.time()
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        for i in range(1, args.args_dict["evaluate_times"]+1):
            args.args_dict["fold"] = i
            args.args_dict["k_shot"] = k
            print(f"======Start {args.args_dict['dataset_name']} {str(i)}-Fold ======")
            acc, f1, pre, rec, roc_auc = run_tuning_graph(args)
            acc_list.append(acc)
            f1_list.append(f1)
            pre_list.append(pre)
            rec_list.append(rec)
            roc_auc_list.append(roc_auc)
            print(f"======End {args.args_dict['dataset_name']} {str(i)}-Fold ======")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std, acc_ave, pre_ave, rec_ave, f1_ave, roc_auc_ave = (
            compute_metrics_ave_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list))
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {acc_ave: .4f}, std: {acc_std: .4f}, "
            f"Precision: {pre_ave: .4f}, std: {pre_std: .4f}, "
            f"Recall: {rec_ave: .4f}, std: {rec_std: .4f}, "
            f"F1: {f1_ave: .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {roc_auc_ave: .4f}, std: {roc_auc_std: .4f}, "
            f"Running Time: {time.time() - st: .4f} s"
        )
        print(f"******End {args.args_dict['dataset_name']} {str(k)}-Shot Tuning******")

