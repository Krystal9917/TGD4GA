import os
import sys
import logging
import argparse
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
        parser.add_argument('--eval_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "uin_gangs_full_graph_dataset", "valid", "raw",
                         "uin_gangs_supervise_full_graph_dataset_eval_241204_20241211.txt")))
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
        parser.add_argument('--best_loss', type=float, default=1e2)
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--batch_size', type=int, default=40)
        parser.add_argument('--data_buffer_size', type=int, default=40)
        parser.add_argument('--lr', type=float, default=0.001)
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
        parser.add_argument('--temperature', type=int, default=1)
        parser.add_argument('--input_dim', type=int, default=846)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=846)
        parser.add_argument('--num_relations', type=int, default=10)
        parser.add_argument('--num_heads', type=int, default=2)
        parser.add_argument('--filter_node_num', type=int, default=5)
        parser.add_argument('--sampling', type=str, default='fraudar', choices=['fraudar', 'random'])
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--data_tag', type=str, default='1930_', choices=['', '1921_', '1930_', 'order_1930'])
        parser.add_argument('--device_tag', type=str, default='', choices=['', '_GPU2'])
        parser.add_argument('--eval_epoch', type=int, default=0)
        parser.add_argument('--evaluate_task', type=str, default='subgraph_prompt_tuning',
                            choices=['subgraph', 'subgraph_embedding', 'subgraph_prompt_tuning'])
        parser.add_argument('--conv_type', type=str, default='RGCN', choices=['RGCN', 'HGT'])
        parser.add_argument('--is_supervised', type=bool, default=False)
        parser.add_argument('--is_weighted_subgraph', type=bool, default=True)
        parser.add_argument('--prompt_insertion_type', type=str, default='concat_subgraph_prompt',
                            choices=[None, 'add_prompt', 'concat_prompt', 'concat_subgraph',
                                     'concat_subgraph_prompt', 'concat_subgraph_plus_prompt',
                                     'concat_subgraph_proj_prompt', 'concat_prompted_subgraph'])
        parser.add_argument('--prompt_tuning_times', type=int, default=5)

        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


if __name__ == '__main__':
    args = ArgsUinGangs()
    print(f"Tuning Parameters: {args.args_dict}")
    if args.args_dict["evaluate_task"] == 'subgraph':
        tuning_model = UinGangsModelTuning(args.args_dict)
        print("======Labelled Subgraph Label Evaluation======")
        tuning_model.evaluate_labelled_subgraph_predict()
    elif args.args_dict["evaluate_task"] == 'subgraph_embedding':
        tuning_model = UinGangsModelTuning(args.args_dict)
        print("======Labelled Subgraph Embedding Evaluation======")
        tuning_model.evaluate_labelled_subgraph_embedding()
    elif args.args_dict["evaluate_task"] == 'subgraph_prompt_tuning':
        print("======Labelled Subgraph Prompt Tuning======")
        times_list = ['1st', '2nd', '3rd', '4th', '5th']
        ave_acc = 0
        ave_pre = 0
        ave_rec = 0
        ave_f1 = 0
        ave_roc_auc = 0
        ave_cfm = 0
        ave_jac = 0
        for i in range(args.args_dict["prompt_tuning_times"]):
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
