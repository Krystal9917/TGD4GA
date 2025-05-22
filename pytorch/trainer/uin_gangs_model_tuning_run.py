import os
import sys
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score
logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_tuning import UinGangsModelTuning


class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "k_shot_processed2")))
        parser.add_argument('--train_data_file', type=str,
                            default="uin_gangs_supervise_full_graph_dataset_train_202503241700.txt")
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "k_shot_processed2")))
        parser.add_argument('--test_data_file', type=str,
                            default="uin_gangs_supervise_full_graph_dataset_eval_202503241700.txt")
        # parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
        #                  "uin_gangs_full_graph_dataset", "valid", "yanghao_processed", )))
        # parser.add_argument('--train_data_file', type=str,
        #                     default="uin_gangs_supervise_full_graph_dataset_eval_250228_train.txt")
        # parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
        #                  "uin_gangs_full_graph_dataset", "valid", "k_shot_processed")))
        # parser.add_argument('--test_data_file', type=str,
        #                     default="uin_gangs_supervise_full_graph_dataset_eval_250228_test.txt")
        parser.add_argument('--split_idx', type=str, default='', choices=['', '1', '2', '3', '4', '5'])
        parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "config", "yml", "uin_gangs_enum.yaml")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "pretrained_models")))
        parser.add_argument('--cls_model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "cls_models")))
        parser.add_argument('--output_save_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "prediction")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--log_dir', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "losses")))
        parser.add_argument('--pic_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "pic")))
        parser.add_argument('--minirbt_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "minirbt-h256")))
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--data_buffer_size', type=int, default=64)
        parser.add_argument('--pretrain_lr', type=float, default=0.001)
        parser.add_argument('--lr_scheduler', type=str, default='',
                            choices=['', '_stepLR', '_reduceLR', '_cosineLR'])
        parser.add_argument('--n_epochs', type=int, default=30)
        parser.add_argument('--patience', type=int, default=5)
        # parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=290)
        # parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
        # parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=300)
        parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=250)
        parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
        parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=256)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=1)
        # parser.add_argument('--input_dim', type=int, default=846)
        # parser.add_argument('--hidden_dim', type=int, default=1024)
        # parser.add_argument('--output_dim', type=int, default=846)
        parser.add_argument('--input_dim', type=int, default=506)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=512)
        parser.add_argument('--out_layer', type=int, default=1)
        parser.add_argument('--node_types', type=int, default=19)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--is_finetune', type=bool, default=True)
        parser.add_argument('--is_supervised', type=bool, default=False)
        parser.add_argument('--num_bases', type=int, default=15)
        parser.add_argument('--conv_type', type=str, default='MaskRGCN',
                            choices=['SVM', 'MLP', 'GAT', 'RGCN', 'MaskRGCN', 'AttnRGCN'])
        parser.add_argument('--pretrained_task_type', type=str, default='context_cl',
                            choices=['subgraph', 'node_subgraph', 'batch_subgraph', 'cross_subgraph',
                                     'fine_grained_batch_subgraph', 'fine_grained_cross_subgraph',
                                     'intra_subgraph', 'context_cl', 'new_context_cl'])
        parser.add_argument('--data_tag', type=str, default='2503_4_',
                            choices=['', '1921_', '1930_', 'order_1930_', '2503_', '2503_4_', '2504_' '2504_1_'])
        parser.add_argument('--device_tag', type=str, default='', choices=['', '_None', '_GPU2', '_GPU3', '_GPU4'])
        parser.add_argument('--eval_epoch', type=int, default=41)
        parser.add_argument('--info_insertion_type', type=str, default='concat_text',
                            choices=[None, 'combine_subgraph', 'concat_subgraph', 'concat_text'])
        parser.add_argument('--node_context', type=bool, default=True)
        parser.add_argument('--prompt_type', type=str, default=None,
                            choices=[None, 'single_token'])
        parser.add_argument('--downstream_task', type=str, default='node_cl_score',
                            choices=['node_cl', 'node_cl_score', 'subgraph_cl', 'subgraph_gang_cl', 'subgraph_gang_cl_by_fraudar',
                                     'subgraph_gang_cl_by_svm', 'node_score_optim', 'subgraph_node_score_optim',
                                     'inference_gang_members_by_fraudar', 'inference_gang_members_by_score'])
        parser.add_argument('--downstream_loss', type=str, default='new_score_loss',
                            choices=['node_loss', 'batch_loss', 'old_batch_loss', 'new_batch_loss',
                                     'score_loss', 'new_score_loss', 'subgraph_node_loss', 'score_mse_loss'])
        parser.add_argument('--ft_strategy', type=str, default='score_degree',
                            choices=[None, 'score_degree'])
        parser.add_argument('--degree_param', type=float, default=1e-3)
        # node classification weight
        parser.add_argument('--node_cls_loss_weight', type=str, default='1.0 3.0',
                            choices=['1.0 2.0', '1.0 3.0', '1.0 4.0', '1.0 7.0'])
        # inner weight for positive subgraph
        parser.add_argument('--w_gang', type=float, default=2.0)
        parser.add_argument('--w_normal', type=float, default=1.0)
        # proportion of positive and negative
        parser.add_argument('--W_p', type=float, default=0.6)
        parser.add_argument('--W_n', type=float, default=0.4)
        parser.add_argument('--cls_lr', type=float, default=1e-4)
        parser.add_argument('--pca_dim', type=int, default=64)
        parser.add_argument('--pooling', default='mean', choices=['mean', 'sag_pool'])
        parser.add_argument('--top_sag_ratio', type=float, default=0.1)
        parser.add_argument('--subgraph_tp_threshold', type=float, default=0.5)
        parser.add_argument('--test_times', type=int, default=5)
        parser.add_argument('--metric_learning', type=str, default='mlp',
                            choices=['similarity', 'mlp'])
        parser.add_argument('--adapter_type', type=str, default=None,
                            choices=[None, 'post_relation', 'post_aggregation'])
        parser.add_argument('--load_text_feature', type=bool, default=True)
        parser.add_argument('--concat_text', type=bool, default=True)
        parser.add_argument('--text_model', type=str, default=None, choices=[None, 'MLP', 'MaskRGCN'])
        parser.add_argument('--is_single_edge', type=bool, default=False)
        parser.add_argument('--k_shot', type=int, default=0)
        parser.add_argument('--top_k', type=int, default=5)
        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def compute_metrics_std(acc, pre, rec, f1, roc_auc, pos_jaccard=None, neg_jaccard=None, run_time=None):
    acc = np.array(acc)
    pre = np.array(pre)
    rec = np.array(rec)
    f1 = np.array(f1)
    roc_auc = np.array(roc_auc)
    if pos_jaccard is not None:
        pos_jaccard = np.array(pos_jaccard)
    if neg_jaccard is not None:
        neg_jaccard = np.array(neg_jaccard)
    if run_time is not None:
        run_time = np.array(run_time)
    if pos_jaccard is not None and neg_jaccard is not None and run_time is not None:
        return acc.std(), pre.std(), rec.std(), f1.std(), roc_auc.std(), pos_jaccard.std(), neg_jaccard.std(), run_time.std()
    else:
        return acc.std(), pre.std(), rec.std(), f1.std(), roc_auc.std()


def calculate_tpr_fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr

def run(k_shot):
    args = ArgsUinGangs()
    if args.args_dict['is_debug']:
        args.args_dict['train_data_file'] = 'uin_gangs_supervise_full_graph_dataset_eval_202503241700.txt'
        args.args_dict['test_data_file'] = 'uin_gangs_supervise_full_graph_dataset_train_202503241700.txt'
    print(f"Tuning Parameters: {args.args_dict}")
    if args.args_dict["downstream_task"] == 'subgraph_cl':
        print(f"======Labelled {args.args_dict['downstream_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        pos_jac_list = []
        neg_jac_list = []
        run_time_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i - 1]} Time Testing*****")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm = tuning_model.detect_subgraph_by_model(
                flag=i)
            acc_list.append(test_acc)
            pre_list.append(test_pre)
            rec_list.append(test_rec)
            f1_list.append(test_f1)
            roc_auc_list.append(test_roc_auc)
            ave_cfm += test_cfm
            print(f"*****End {times_list[i - 1]} Time Testing*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std = (
            compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list))
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
            f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
            f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
            f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
            f"Confusion Matrix: {ave_cfm / len(times_list)}")
    elif args.args_dict["downstream_task"] in ['subgraph_gang_cl', 'node_cl', 'node_cl_score']:
        args.args_dict["k_shot"] = k_shot
        print(f"======Labelled {args.args_dict['downstream_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        y_true_list = []
        y_prob_list = []
        node_id_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        if args.args_dict["downstream_task"] != 'node_cl':
            pos_jac_list = []
            neg_jac_list = []
        run_time_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i - 1]} Time Testing*****")
            if args.args_dict["downstream_task"] == 'node_cl' or args.args_dict["downstream_task"] == 'node_cl_score':
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm,
                 run_time, y_true, y_prob, node_id) = tuning_model.detect_subgraph_gang_members_by_model(
                    flag=f'{str(i)}', task="node")
            else:
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm,
                 pos_jac, neg_jac, run_time) = tuning_model.detect_subgraph_gang_members_by_model(flag=f'{str(i)}',
                                                                                                  task="subgraph")
            acc_list.append(test_acc)
            pre_list.append(test_pre)
            rec_list.append(test_rec)
            f1_list.append(test_f1)
            roc_auc_list.append(test_roc_auc)
            ave_cfm += test_cfm
            y_true_list.append(y_true)
            y_prob_list.append(y_prob)
            node_id_list.extend(node_id)
            if args.args_dict["downstream_task"] not in ['node_cl', 'node_cl_score']:
                pos_jac_list.append(pos_jac)
                neg_jac_list.append(neg_jac)
            run_time_list.append(run_time)
            print(f"*****End {times_list[i - 1]} Time Testing*****")
        if args.args_dict["downstream_task"] in ['node_cl', 'node_cl_score']:
            acc_std, pre_std, rec_std, f1_std, roc_auc_std = (
                compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list))
            print(
                f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
                f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
                f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
                f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
                f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
                f"Confusion Matrix: {ave_cfm / len(times_list)} "
            )
        else:
            acc_std, pre_std, rec_std, f1_std, roc_auc_std, pos_std, neg_std, time_std = (
                compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list, pos_jac_list, neg_jac_list,
                                    run_time_list))
            print(
                f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
                f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
                f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
                f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
                f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
                f"Pos Jaccard: {sum(pos_jac_list) / len(times_list): .4f}, std: {pos_std: .4f}, "
                f"Neg Jaccard: {sum(neg_jac_list) / len(times_list): .4f}, std: {neg_std: .4f}, "
                f"Run Time: {sum(run_time_list) / len(run_time_list): .4f} s, std: {time_std: .4f} s "
                f"Confusion Matrix: {ave_cfm / len(times_list)} "
            )
        data = pd.DataFrame(data={'node_id': node_id_list,
                                  'y_true': np.concatenate(y_true_list, axis=0),
                                  'y_prob': np.concatenate(y_prob_list, axis=0)})
        if args.args_dict['is_finetune']:
            pred_dir = os.path.join(args.args_dict['output_save_path'], "pretrain_ft27")
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            pred_path = os.path.join(pred_dir, f"ft_{k_shot}_shot_pred.csv")
        else:
            if args.args_dict['is_supervised']:
                pred_dir = os.path.join(args.args_dict['output_save_path'], "supervise4")
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                pred_path = os.path.join(pred_dir, f"supervise_{k_shot}_shot_pred.csv")
            else:
                pred_dir = os.path.join(args.args_dict['output_save_path'], "pretrain3")
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)
                pred_path = os.path.join(pred_dir, f"pretrain_{k_shot}_shot_pred.csv")
        data.to_csv(pred_path, index=False)
        print(f"Save prediction file to {pred_path}")
    elif args.args_dict["downstream_task"] in ['subgraph_node_score_optim', 'node_score_optim']:
        print(f"======Labelled {args.args_dict['downstream_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        if args.args_dict["downstream_task"] != 'node_score_optim':
            pos_jac_list = []
            neg_jac_list = []
        run_time_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i - 1]} Time Testing*****")
            if args.args_dict["downstream_task"] == 'node_score_optim':
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm, run_time) = (
                    tuning_model.optim_subgraph_node_score_by_model(flag=f'{str(i)}', task="node"))
            else:
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm,
                 pos_jac, neg_jac, run_time) = tuning_model.optim_subgraph_node_score_by_model(flag=f'{str(i)}',
                                                                                               task="subgraph")
            acc_list.append(test_acc)
            pre_list.append(test_pre)
            rec_list.append(test_rec)
            f1_list.append(test_f1)
            roc_auc_list.append(test_roc_auc)
            ave_cfm += test_cfm
            if args.args_dict["downstream_task"] != 'node_score_optim':
                pos_jac_list.append(pos_jac)
                neg_jac_list.append(neg_jac)
            run_time_list.append(run_time)
            print(f"*****End {times_list[i - 1]} Time Testing*****")
        if args.args_dict["downstream_task"] == 'node_score_optim':
            acc_std, pre_std, rec_std, f1_std, roc_auc_std = (
                compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list))
            print(
                f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
                f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
                f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
                f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
                f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
                f"Confusion Matrix: {ave_cfm / len(times_list)} "
            )
        else:
            acc_std, pre_std, rec_std, f1_std, roc_auc_std, pos_std, neg_std, time_std = (
                compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list, pos_jac_list, neg_jac_list,
                                    run_time_list))
            print(
                f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
                f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
                f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
                f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
                f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
                f"Pos Jaccard: {sum(pos_jac_list) / len(times_list): .4f}, std: {pos_std: .4f}, "
                f"Neg Jaccard: {sum(neg_jac_list) / len(times_list): .4f}, std: {neg_std: .4f}, "
                f"Run Time: {sum(run_time_list) / len(run_time_list): .4f} s, std: {time_std: .4f} s "
                f"Confusion Matrix: {ave_cfm / len(times_list)} "
            )
    elif args.args_dict["downstream_task"] == 'subgraph_gang_cl_by_svm':
        print(f"======Labelled {args.args_dict['downstream_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        run_time_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i - 1]} Time Testing*****")
            test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm, run_time = (
                tuning_model.detect_subgraph_gang_members_by_svm())
            acc_list.append(test_acc)
            pre_list.append(test_pre)
            rec_list.append(test_rec)
            f1_list.append(test_f1)
            roc_auc_list.append(test_roc_auc)
            ave_cfm += test_cfm
            run_time_list.append(run_time)
            print(f"*****End {times_list[i - 1]} Time Testing*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std = compute_metrics_std(acc_list, pre_list, rec_list, f1_list,
                                                                             roc_auc_list)
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
            f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
            f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
            f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
            f"Confusion Matrix: {ave_cfm / len(times_list)} "
        )
    elif args.args_dict["downstream_task"] == 'inference_gang_members_by_fraudar':
        print("======Inference Subgraph Gang by Fraudar======")
        args.args_dict["k_shot"] = k_shot
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        pos_jac_list = []
        neg_jac_list = []
        run_time_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        true_list = []
        pred_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            print(f"*****Start {times_list[i - 1]} Fraudar Inference*****")
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, run_time, y_true, y_pred = (
                tuning_model.detect_subgraph_gang_members_by_fraudar(i, save_results=True))
            acc_list.append(acc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            ave_cfm += cm
            pos_jac_list.append(pos_jaccard)
            neg_jac_list.append(neg_jaccard)
            run_time_list.append(run_time)
            true_list.append(y_true)
            pred_list.append(y_pred)
            print(f"*****End {times_list[i - 1]} Fraudar Inference*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std, pos_std, neg_std, time_std = (
            compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list, pos_jac_list, neg_jac_list,
                                run_time_list))
        true_list = np.concatenate(true_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        tpr, fpr = calculate_tpr_fpr(true_list, pred_list)
        pre = precision_score(true_list, pred_list)
        rec = recall_score(true_list, pred_list)
        print(f"TPR: {tpr: .4f}, FPR: {fpr: .4f}, Precision: {pre: .4f}, Recall: {rec: .4f}")
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
            f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
            f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
            f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
            f"Pos Jaccard: {sum(pos_jac_list) / len(times_list): .4f}, std: {pos_std: .4f}, "
            f"Neg Jaccard: {sum(neg_jac_list) / len(times_list): .4f}, std: {neg_std: .4f}, "
            f"Confusion Matrix: {ave_cfm / len(times_list)} "
            f"Run Time: {sum(run_time_list) / len(run_time_list): .4f} s, std: {time_std: .4f} s"
        )
    elif args.args_dict["downstream_task"] == 'inference_gang_members_by_score':
        print("======Inference Subgraph Gang by Score======")
        args.args_dict["k_shot"] = k_shot
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        pos_jac_list = []
        neg_jac_list = []
        run_time_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        true_list = []
        prob_list = []
        pred_list = []
        node_id_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            print(f"*****Start {times_list[i - 1]} Score Inference*****")
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, run_time, y_true, y_prob, y_pred, node_id = (
                tuning_model.detect_subgraph_gang_members_by_score(i, save_results=False))
            acc_list.append(acc)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            ave_cfm += cm
            pos_jac_list.append(pos_jaccard)
            neg_jac_list.append(neg_jaccard)
            run_time_list.append(run_time)
            true_list.append(y_true)
            prob_list.append(y_prob)
            pred_list.append(y_pred)
            node_id_list.extend(node_id)
            print(f"*****End {times_list[i - 1]} Score Inference*****")
        acc_std, pre_std, rec_std, f1_std, roc_auc_std, pos_std, neg_std, time_std = (
            compute_metrics_std(acc_list, pre_list, rec_list, f1_list, roc_auc_list, pos_jac_list, neg_jac_list,
                                run_time_list))
        true_list = np.concatenate(true_list, axis=0)
        prob_list = np.concatenate(prob_list, axis=0)
        pred_list = np.concatenate(pred_list, axis=0)
        data = pd.DataFrame(data={'node_id': node_id_list,
                                  'y_true': true_list,
                                  'y_prob': prob_list})
        pred_path = os.path.join(args.args_dict['output_save_path'], "raw_score")
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        data.to_csv(os.path.join(pred_path, "raw_score_few_shot_pred.csv"), index=False)
        print(f"Save prediction file to {pred_path}")
        tpr, fpr = calculate_tpr_fpr(true_list, pred_list)
        pre = precision_score(true_list, pred_list)
        rec = recall_score(true_list, pred_list)
        print(f"TPR: {tpr: .4f}, FPR: {fpr: .4f}, Precision: {pre: .4f}, Recall: {rec: .4f}")
        print(
            f"{len(acc_list)} Times Average Best Test ACC: {sum(acc_list) / len(acc_list): .4f}, std: {acc_std: .4f}, "
            f"Precision: {sum(pre_list) / len(pre_list): .4f}, std: {pre_std: .4f}, "
            f"Recall: {sum(rec_list) / len(rec_list): .4f}, std: {rec_std: .4f}, "
            f"F1: {sum(f1_list) / len(f1_list): .4f}, std: {f1_std: .4f}, "
            f"ROC-AUC: {sum(roc_auc_list) / len(roc_auc_list): .4f}, std: {roc_auc_std: .4f}, "
            f"Pos Jaccard: {sum(pos_jac_list) / len(times_list): .4f}, std: {pos_std: .4f}, "
            f"Neg Jaccard: {sum(neg_jac_list) / len(times_list): .4f}, std: {neg_std: .4f}, "
            f"Confusion Matrix: {ave_cfm / len(times_list)} "
            f"Run Time: {sum(run_time_list) / len(run_time_list): .4f} s, std: {time_std: .4f} s"
        )

if __name__ == '__main__':
    for shot_num in np.arange(10, 110, 10):
        print(f"======Shot Number {shot_num}======")
        run(shot_num)