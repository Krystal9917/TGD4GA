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
                         "uin_gangs_full_graph_dataset", "valid", "positive_processed", "split_")))
        parser.add_argument('--train_data_file', type=str,
                            default="uin_gangs_supervise_full_graph_dataset_train_202503241700.txt")
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "positive_processed", "split_")))
        parser.add_argument('--test_data_file', type=str,
                            default="uin_gangs_supervise_full_graph_dataset_eval_202503241700.txt")
        # parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
        #                  "uin_gangs_full_graph_dataset", "valid", "yanghao_processed", )))
        # parser.add_argument('--train_data_file', type=str,
        #                     default="uin_gangs_supervise_full_graph_dataset_eval_250228_train.txt")
        # parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
        #                  "uin_gangs_full_graph_dataset", "valid", "yanghao_processed", )))
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
        parser.add_argument('--input_dim', type=int, default=762)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=762)
        parser.add_argument('--out_layer', type=int, default=1)
        parser.add_argument('--node_types', type=int, default=19)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--is_finetune', type=bool, default=False)
        parser.add_argument('--is_supervised', type=bool, default=True)
        parser.add_argument('--num_bases', type=int, default=15)
        parser.add_argument('--conv_type', type=str, default='AttnRGCN',
                            choices=['SVM', 'MLP', 'GAT', 'RGCN', 'MaskRGCN', 'AttnRGCN'])
        parser.add_argument('--pretrained_task_type', type=str, default='context_cl',
                            choices=['subgraph', 'node_subgraph', 'batch_subgraph', 'cross_subgraph',
                                     'fine_grained_batch_subgraph', 'fine_grained_cross_subgraph',
                                     'intra_subgraph', 'context_cl'])
        parser.add_argument('--data_tag', type=str, default='2503_4_',
                            choices=['', '1921_', '1930_', 'order_1930_', '2503_', '2503_4_'])
        parser.add_argument('--device_tag', type=str, default='', choices=['', '_None', '_GPU2', '_GPU3', '_GPU4'])
        parser.add_argument('--eval_epoch', type=int, default=0)
        parser.add_argument('--info_insertion_type', type=str, default=None,
                            choices=[None, 'combine_subgraph', 'concat_subgraph', 'concat_text'])
        parser.add_argument('--prompt_type', type=str, default=None,
                            choices=[None, 'single_token'])
        parser.add_argument('--downstream_task', type=str, default='node_cl',
                            choices=['node_cl', 'subgraph_cl', 'subgraph_gang_cl', 'subgraph_gang_cl_by_fraudar',
                                     'subgraph_gang_cl_by_svm', 'node_score_optim', 'subgraph_node_score_optim'])
        parser.add_argument('--downstream_loss', type=str, default='node_loss',
                            choices=['node_loss', 'subgraph_node_loss', 'score_mse_loss'])
        # node classification weight
        parser.add_argument('--node_cls_loss_weight', type=str, default='1.0 2.0',
                            choices=['1.0 2.0', '1.0 3.0', '1.0 4.0'])
        # inner weight for positive subgraph
        parser.add_argument('--w_gang', type=float, default=1.5)
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
        parser.add_argument('--concat_text', type=bool, default=False)
        parser.add_argument('--is_single_edge', type=bool, default=False)
        parser.add_argument('--is_add_score', type=bool, default=False)
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


if __name__ == '__main__':
    args = ArgsUinGangs()
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
    elif args.args_dict["downstream_task"] in ['subgraph_gang_cl', 'node_cl']:
        print(f"======Labelled {args.args_dict['downstream_task']} Evaluation======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        acc_list = []
        pre_list = []
        rec_list = []
        f1_list = []
        roc_auc_list = []
        ave_cfm = np.array([[0, 0], [0, 0]])
        if args.args_dict["downstream_task"] != 'node_cl':
            pos_jac_list = []
            neg_jac_list = []
        run_time_list = []
        for i in range(1, args.args_dict["test_times"] + 1):
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            print(f"*****Start {times_list[i - 1]} Time Testing*****")
            if args.args_dict["downstream_task"] == 'node_cl':
                (test_acc, test_pre, test_rec, test_f1, test_roc_auc, test_cfm,
                 run_time) = tuning_model.detect_subgraph_gang_members_by_model(flag=f'{str(i)}', task="node")
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
            if args.args_dict["downstream_task"] != 'node_cl':
                pos_jac_list.append(pos_jac)
                neg_jac_list.append(neg_jac)
            run_time_list.append(run_time)
            print(f"*****End {times_list[i - 1]} Time Testing*****")
        if args.args_dict["downstream_task"] == 'node_cl':
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
        for i in range(1, args.args_dict["test_times"] + 1):
            print(f"*****Start {times_list[i - 1]} Fraudar Inference*****")
            args.args_dict["split_idx"] = str(i)
            tuning_model = UinGangsModelTuning(args.args_dict)
            acc, pre, rec, f1, roc_auc, cm, pos_jaccard, neg_jaccard, run_time = (
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
            print(f"*****End {times_list[i - 1]} Fraudar Inference*****")
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
            f"Confusion Matrix: {ave_cfm / len(times_list)} "
            f"Run Time: {sum(run_time_list) / len(run_time_list): .4f} s, std: {time_std: .4f} s"
        )
