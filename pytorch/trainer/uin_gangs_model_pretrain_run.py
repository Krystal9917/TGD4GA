import os
import sys
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
                         # "uin_gangs_full_graph_dataset_train_241119_241121.txt")))
                         "uin_gangs_full_graph_dataset_train_202503200423.txt")))
        parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "config", "yml", "uin_gangs_enum.yaml")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data",
                         "saved_model", "pretrained_models")))
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
        parser.add_argument('--lr_scheduler', type=str, default='',
                            choices=['', '_stepLR', '_reduceLR', '_cosineLR'])
        parser.add_argument('--lr_adjust_step', type=int, default=2)
        parser.add_argument('--lr_gamma', type=float, default=0.5)
        parser.add_argument('--n_epochs', type=int, default=50)
        parser.add_argument('--uin_acs_numberical_feat_dim', type=int, default=250)
        parser.add_argument('--uin_acs_text_feat_dim', type=int, default=256)
        parser.add_argument('--uin_acs_categorical_feat_hasher_dim', type=int, default=256)
        parser.add_argument('--drop_rate', type=float, default=0.5)
        parser.add_argument('--onnx_opset', type=int, default=15)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=1)
        parser.add_argument('--hidden_dim', type=int, default=1024)
        parser.add_argument('--output_dim', type=int, default=512)
        parser.add_argument('--filter_node_num', type=int, default=3)
        parser.add_argument('--sampling', type=str, default='fraudar')
        parser.add_argument('--drop_ratio', type=float, default=0.2)
        parser.add_argument('--re_train', type=bool, default=False)
        parser.add_argument('--is_debug', type=bool, default=False)
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--conv_type', type=str, default='MaskRGCN',
                            choices=['RGCN', 'MaskRGCN', 'AttnRGCN'])
        parser.add_argument('--task_type', type=str, default='new_context_cl',
                            choices=['subgraph', 'node_subgraph', 'context_cl', 'new_context_cl'
                                     'batch_subgraph', 'fine_grained_batch_subgraph',
                                     'cross_subgraph', 'fine_grained_cross_subgraph'])
        parser.add_argument('--load_text_feature', type=bool, default=False)
        # multi pre-training tasks weight
        parser.add_argument('--W_subgraph', type=float, default=0.2)
        parser.add_argument('--W_node', type=float, default=0.8)
        # context cl
        parser.add_argument('--random_k_ratio', type=float, default=0.2)
        parser.add_argument('--neighbor_l_ratio', type=int, default=0.1)
        # new context cl
        parser.add_argument('--s_weight', type=float, default=0.8)
        parser.add_argument('--d_weight', type=float, default=0.2)
        # cross/batch
        parser.add_argument('--n_weight', type=float, default=0.4)
        parser.add_argument('--c_weight', type=float, default=0.6)
        parser.add_argument('--similarity_diff', type=float, default=0.05)
        parser.add_argument('--data_tag', type=str, default='2504_',
                            choices=['', '1921_', '1930_', 'order_1930_', '2503_',
                                     '2503_4_', '2504_', '2504_1_'])
        parser.add_argument('--metric_learning', type=str, default='mlp',
                            choices=['similarity', 'mlp'])
        parser.add_argument('--is_single_edge', type=bool, default=False)
        parser.add_argument('--adapter_type', type=str, default=None,
                            choices=[None, 'post_relation', 'post_aggregation'])
        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_pretraining_graph(args):
    print(f"Training Parameters: {args.args_dict}")
    if args.args_dict["is_debug"]:
        args.args_dict["train_data_path"] = ("/chongqinggeminiceph1fs/geminicephfs/security-others-common/"
                                             "jiujiuchen/projects/mmgog_long_term_sequence_model/data/"
                                             "uin_gangs_full_graph_dataset/train/raw/"
                                             "uin_gangs_full_graph_dataset_train_202503201445_random_800.txt")
    train_model = UinGangsModelPreTrain(args.args_dict)
    if args.args_dict["sampling"] == 'fraudar':
        print("======Single Device Fraudar Pretraining======")
        train_model.fraudar_sampling_pretraining()


if __name__ == '__main__':
    args = ArgsUinGangs()
    run_pretraining_graph(args)
