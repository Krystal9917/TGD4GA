import os
import sys
import logging
import argparse

import numpy as np

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_tuning import UinGangsModelTuning


class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "processed",
                         "uin_gangs_supervise_full_graph_dataset_train_241204_20241204.txt")))
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "processed",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241204.txt")))
        parser.add_argument('--node_train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "train", "raw",
                         "uin_gangs_full_graph_dataset_train_241201_20241119.txt")))
        parser.add_argument('--node_test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "train", "raw",
                         "uin_gangs_full_graph_dataset_train_241202_20241119.txt")))
        parser.add_argument('--prompt_initial_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "raw",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive_20.txt")))
        parser.add_argument('--prompt_tuning_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "raw",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive_20_40.txt")))
        parser.add_argument('--prompt_evaluating_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "raw",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive_exclude_40.txt")))
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
        parser.add_argument('--best_loss', type=float, default=0.5)
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=40)
        parser.add_argument('--data_buffer_size', type=int, default=40)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--lr_scheduler', type=str, default='',
                            choices=['', '_stepLR', '_reduceLR', '_cosineLR'])
        parser.add_argument('--n_epochs', type=int, default=30)
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
        parser.add_argument('--temperature', type=int, default=0.5)
        parser.add_argument('--input_dim', type=int, default=846)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=846)
        parser.add_argument('--node_types', type=int, default=19)
        parser.add_argument('--num_relations', type=int, default=10)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--data_tag', type=str, default='order_1930_',
                            choices=['', '1921_', '1930_', 'order_1930_'])
        parser.add_argument('--device_tag', type=str, default='_GPU2', choices=['', '_GPU2'])
        parser.add_argument('--eval_epoch', type=int, default=9)
        parser.add_argument('--evaluate_task', type=str, default='subgraph_gang_detection',
                            choices=['node_classification', 'subgraph',
                                     'subgraph_gang_detection', 'subgraph_prompt_tuning',
                                     'inference_gang_members', 'inference_gang_members_by_fraudar'])
        parser.add_argument('--evaluate_task_tuning', type=bool, default=False)
        parser.add_argument('--is_finetune', type=bool, default=True)
        parser.add_argument('--conv_type', type=str, default='RGCN', choices=['RGCN', 'HGT', 'HAN'])
        parser.add_argument('--task_type', type=str, default='intra_subgraph',
                            choices=['subgraph', 'node_subgraph', 'batch_subgraph', 'cross_subgraph',
                                     'fine_grained_batch_subgraph', 'fine_grained_cross_subgraph',
                                     'intra_subgraph'])
        parser.add_argument('--is_supervised', type=bool, default=False)
        parser.add_argument('--is_weighted_subgraph', type=bool, default=False)
        parser.add_argument('--prompt_insertion_type', type=str, default=None,
                            choices=[None, 'add_subgraph', 'concat_subgraph', 'concat_prompt',
                                     'concat_subgraph_prompt'])
        parser.add_argument('--prompt_lr', type=float, default=1e-4)
        parser.add_argument('--info_insertion_type', type=str, default='combine_subgraph',
                            choices=[None, 'combine_subgraph', 'combine_difference'])
        parser.add_argument('--threshold', type=float, default=0.5)
        parser.add_argument('--test_times', type=int, default=5)
        parser.add_argument('--loss_weight', type=str, default='1.0 2.0',
                            choices=['1.0 2.0', '1.0 3.0', '1.0 4.0'])
        parser.add_argument('--cls_lr', type=float, default=5e-5)
        parser.add_argument('--best_test_f1', type=float, default=0.6)
        parser.add_argument('--alpha', type=float, default=0.2)
        parser.add_argument('--beta', type=float, default=0.8)
        parser.add_argument('--cls_node', type=bool, default=True)
        parser.add_argument('--cls_subgraph', type=bool, default=True)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def compute_metrics_std(acc, pre, rec, f1, roc_auc, pos_jaccard=None, neg_jaccard=None):
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    f1 = np.array(f1)
    roc_auc = np.array(roc_auc)
    if pos_jaccard is not None:
        pos_jaccard = np.array(pos_jaccard)
    if neg_jaccard is not None:
        neg_jaccard = np.array(neg_jaccard)
    if pos_jaccard is not None and neg_jaccard is not None:
        return acc.std(), pre.std(), rec.std(), f1.std(), roc_auc.std(), pos_jaccard.std(), neg_jaccard.std()
    else:
        return acc.std(), pre.std(), rec.std(), f1.std(), roc_auc.std()


if __name__ == '__main__':
    args = ArgsUinGangs()
    print(f"Tuning Parameters: {args.args_dict}")
    if args.args_dict["evaluate_task"] == 'node_classification':
        tuning_model = UinGangsModelTuning(args.args_dict)
        print("======Labelled Node Classification Evaluation======")
        tuning_model.evaluate_node_classification()
    elif args.args_dict["evaluate_task"] in ['subgraph', 'subgraph_gang_detection']:
        print(f"======Labelled {args.args_dict['evaluate_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        for i in range(args.args_dict["test_times"]):
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i]} Time Testing*****")
            if args.args_dict["evaluate_task"] == 'subgraph':
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm = tuning_model.evaluate_labelled_subgraph_predict()
            else:
                test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm = tuning_model.detect_subgraph_gang_members()
            acc_list.append(test_acc)
            pre_list.append(test_pre)
            rec_list.append(test_rec)
            f1_list.append(test_f1)
            roc_auc_list.append(test_roc_auc)
            ave_cfm += test_cfm
            print(f"*****End {times_list[i]} Time Testing*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std = compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list)
        print(f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
              f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
              f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
              f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
              f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
              f"Confusion Matrix: {ave_cfm / len(times_list)}"
              )
    elif args.args_dict["evaluate_task"] == 'subgraph_prompt_tuning':
        print("======Labelled Subgraph Prompt Tuning======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        ave_acc = 0
        ave_pre = 0
        ave_rec = 0
        ave_f1 = 0
        ave_roc_auc = 0
        ave_cfm = np.array([[0, 0], [0, 0]])
        ave_jac = 0
        for i in range(args.args_dict["test_times"]):
            print(f"*****Start {times_list[i]} Time Tuning*****")
            tuning_model = UinGangsModelTuning(args.args_dict)
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm, test_jac = tuning_model.evaluate_labelled_subgraph_prompt_tuning()
            ave_acc += test_acc
            ave_pre += test_pre
            ave_rec += test_rec
            ave_f1 += test_f1
            ave_roc_auc += test_roc_auc
            ave_cfm += test_cfm
            ave_jac += test_jac
            print(f"*****End {times_list[i]} Time Tuning*****")
        print(f"{len(times_list)} Times Average Best Test ACC: {ave_acc / len(times_list): .4f}, "
              f"Precision: {ave_pre / len(times_list): .4f}, "
              f"Recall: {ave_rec / len(times_list): .4f}, "
              f"F1: {ave_f1 / len(times_list): .4f}, "
              f"ROC-AUC: {ave_roc_auc / len(times_list): .4f}, "
              f"Confusion Matrix: {ave_cfm / len(times_list)}, "
              f"Jaccard Coefficient: {ave_jac / len(times_list): .4f}"
              )
    elif args.args_dict["evaluate_task"] == 'inference_gang_members':
        print(f"*****Start Inference*****")
        tuning_model = UinGangsModelTuning(args.args_dict)
        tuning_model.inference_gang_members()
        print(f"*****End Inference*****")
    elif args.args_dict["evaluate_task"] == 'inference_gang_members_by_fraudar':
        print("======Labelled Subgraph Prompt Tuning======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        pos_jac_list = []
        neg_jac_list = []
        for i in range(args.args_dict["test_times"]):
            print(f"*****Start {times_list[i]} Fraudar Inference*****")
            tuning_model = UinGangsModelTuning(args.args_dict)
            acc, pre, rec, f1, roc_auc, pos_jaccard, neg_jaccard = tuning_model.inference_gang_members_by_fraudar()
            acc_list.append(acc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            pos_jac_list.append(pos_jaccard)
            neg_jac_list.append(neg_jaccard)
            print(f"*****End {times_list[i]} Fraudar Inference*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std, pos_std, neg_std = (
            compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list, pos_jac_list, neg_jac_list))
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
            f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
            f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
            f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
            f"Pos Jaccard: {sum(pos_jac_list) / len(times_list): .4f}, std: {pos_std: .4f}, "
            f"Neg Jaccard: {sum(neg_jac_list) / len(times_list): .4f}, std: {neg_std: .4f}"
            )