import argparse
import multiprocessing
import os
import sys
import logging
import time

import torch

logger = logging.getLogger("my_logger")

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.utils.utils import start_log
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_train import UinGangsModelTrain
from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_train_only_classify import \
    UinGangsModelTrainOnlyClassify
from mmgog_long_term_sequence_model.pytorch.inference.uin_gangs_model_infer import UinGangsModelInfer


class ArgsUinGangs:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--ds', type=str, default="")
        parser.add_argument('--train_data_url_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "train",
                         "240904_2", "uin_gangs_train_dataset_train_240904_2.txt")))
        parser.add_argument('--test_data_url_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "eval",
                         "240904_2", "uin_gangs_train_dataset_eval_240904_2.txt")))
        parser.add_argument('--train_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "train",
                         "240904_2", "uin_gangs_train_dataset_train_240904_2_20240907.txt")))
        parser.add_argument('--test_data_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "uin_gangs_train_dataset",
                         "eval",
                         "240904_2", "uin_gangs_train_dataset_eval_240904_2_20240907.txt")))
        parser.add_argument('--uin_gangs_enum_yaml_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "config", "yml",
                         "uin_gangs_enum" + ".yaml")))
        parser.add_argument('--model_states_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "saved_model",
                         "model_states",
                         "optimal_uin_gangs_model" + ".bin")))
        parser.add_argument('--export_requirements_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, "data", "config", "requirements" + ".txt")))
        parser.add_argument('--pic_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "data", "pic")))
        parser.add_argument('--minirbt_path', type=str, default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, "minirbt-h256")))
        parser.add_argument('--best_loss', type=float, default=100)
        parser.add_argument('--cls_num', type=int, default=19)
        parser.add_argument('--out_size', type=int, default=19)
        parser.add_argument('--negative_positive_ratio', type=float, default=10)
        parser.add_argument('--mul_cls_threshold', type=int, default=0.9)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--data_buffer_size', type=int, default=100000)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--n_epochs', type=int, default=100)
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
        parser.add_argument('--seed', type=int, default=20)
        parser.add_argument('--num_workers', type=int, default=32)
        parser.add_argument('--temperature', type=int, default=0.5)
        args = parser.parse_args()
        args_dict = vars(args)
        self.args_dict = args_dict


def run_seqs_graph():
    start_log()
    args = ArgsUinGangs()
    logger.info("args_dict:\n %s", args.args_dict)
    train_model = UinGangsModelTrain(args.args_dict)
    # train_model = UinGangsModelTrainOnlyClassify(args.args_dict)
    train_model.train()


def run_infer_graph():
    args = ArgsUinGangs()
    print("args_dict", args.args_dict)
    infer_model = UinGangsModelInfer(args.args_dict)
    infer_model.init_model()
    sample = "{\"root_uin\":\"4119566402\",\"original_label\":\"其他\",\"graph_schema\":{\"nodeid2uin_map\":{\"0\":\"4119566402\",\"1\":\"2458965881\"},\"node_sets\":{\"uin\":{\"nodeids\":[\"0\",\"1\"],\"data\":{\"uin_acs_categorical_feat\":{\"string_list\":[[\"1_3\",\"2_68\",\"3_3\",\"4_4\",\"5_0\",\"6_0\",\"7_0\",\"8_0\",\"9_0\",\"10_0\",\"11_1\",\"12_0\",\"13_3\",\"14_0\",\"15_0\",\"16_0\",\"17_0\",\"18_0\",\"19_0\",\"20_1\",\"21_0\",\"22_0\",\"23_0\",\"24_0\",\"25_0\",\"26_0\",\"27_0\",\"28_0\",\"29_0\",\"30_0\",\"31_0\",\"32_0\",\"33_0\",\"34_0\",\"35_0\",\"36_0\",\"37_0\",\"38_0\",\"39_0\",\"40_-1\",\"41_0\",\"42_1\",\"43_0\",\"44_0\",\"45_0\",\"46_0\",\"47_2\",\"48_1\",\"49_18\",\"50_190\",\"51_31\",\"52_2\",\"53_2\",\"54_1\",\"55_0\",\"56_1\",\"57_18\",\"58_190\",\"59_402663724\",\"60_0\",\"61_1\",\"62_18\",\"63_190\",\"64_402666044\",\"65_0\",\"66_0\",\"67_0\",\"68_0\",\"69_0\",\"70_0\",\"71_0\",\"72_1\",\"73_18\",\"74_190\",\"75_0\",\"76_0\",\"77_15\",\"78_1\",\"79_0\",\"80_0\",\"81_0\",\"82_0\",\"83_0\",\"84_0\",\"85_0\",\"86_0\",\"87_0\",\"88_0\",\"89_0\",\"90_0\",\"91_0\",\"92_0\",\"93_0\",\"94_0\",\"95_0\",\"96_0\",\"97_0\",\"98_0\",\"99_0\",\"100_0\",\"101_0\",\"102_1\",\"103_0\",\"104_0\",\"105_0\",\"106_0\",\"107_0\",\"108_0\",\"109_0\",\"110_0\",\"111_0\",\"112_0\",\"113_1\",\"114_1\",\"115_52\",\"116_9\",\"117_84\",\"118_3\",\"119_4\",\"120_0\",\"121_0\",\"122_0\",\"123_0\",\"124_13\",\"125_0\",\"126_0\",\"127_0\",\"128_0\",\"129_0\",\"130_0\",\"131_0\",\"132_0\",\"133_0\",\"134_0\",\"135_0\",\"136_0\",\"137_0\",\"138_0\",\"139_0\"],[\"1_1\",\"2_84\",\"3_1\",\"4_0\",\"5_0\",\"6_1\",\"7_1\",\"8_0\",\"9_0\",\"10_0\",\"11_0\",\"12_0\",\"13_9\",\"14_0\",\"15_0\",\"16_0\",\"17_1\",\"18_0\",\"19_0\",\"20_0\",\"21_1\",\"22_0\",\"23_1\",\"24_0\",\"25_0\",\"26_0\",\"27_0\",\"28_0\",\"29_0\",\"30_0\",\"31_0\",\"32_0\",\"33_0\",\"34_0\",\"35_0\",\"36_0\",\"37_0\",\"38_0\",\"39_0\",\"40_0\",\"41_0\",\"42_1\",\"43_1\",\"44_0\",\"45_0\",\"46_0\",\"47_2\",\"48_1\",\"49_18\",\"50_182\",\"51_31\",\"52_2\",\"53_2\",\"54_1\",\"55_1\",\"56_0\",\"57_0\",\"58_0\",\"59_0\",\"60_0\",\"61_1\",\"62_18\",\"63_190\",\"64_402666045\",\"65_1\",\"66_18\",\"67_190\",\"68_1661537049\",\"69_0\",\"70_0\",\"71_5\",\"72_1\",\"73_18\",\"74_190\",\"75_0\",\"76_0\",\"77_15\",\"78_1\",\"79_0\",\"80_0\",\"81_0\",\"82_0\",\"83_0\",\"84_0\",\"85_0\",\"86_0\",\"87_0\",\"88_0\",\"89_0\",\"90_0\",\"91_0\",\"92_0\",\"93_0\",\"94_0\",\"95_0\",\"96_0\",\"97_0\",\"98_0\",\"99_0\",\"100_0\",\"101_0\",\"102_0\",\"103_0\",\"104_0\",\"105_0\",\"106_0\",\"107_4\",\"108_100\",\"109_2\",\"110_0\",\"111_0\",\"112_0\",\"113_0\",\"114_0\",\"115_0\",\"116_0\",\"117_0\",\"118_0\",\"119_0\",\"120_0\",\"121_0\",\"122_0\",\"123_0\",\"124_4\",\"125_0\",\"126_0\",\"127_0\",\"128_0\",\"129_0\",\"130_0\",\"131_0\",\"132_0\",\"133_0\",\"134_0\",\"135_0\",\"136_0\",\"137_0\",\"138_0\",\"139_0\"]]},\"uin_acs_text_feat\":{\"string_list\":[[\"空管精品店风声水起小贝iPhone我是顺风顺水|好，你扫码付吧，别转给我了profile_beshared\"],[\"教师示范区纪工委政府机关政府机关人员（除基层）政府贝小贝直播沟通示范区平平王贝贝的iPhonePC-202211120933我是A五菱孙晨阳15136399360轻颜_com.gorgeous.lite贝小贝vfy_hello_othermmaddcontactbiz\"]]},\"uin_acs_numberical_feat\":{\"float_list\":[[\"1164\",\"62\",\"305\",\"0\",\"0\",\"4\",\"4\",\"5\",\"0\",\"2\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"2\",\"0\",\"0\",\"0\",\"41\",\"11\",\"17\",\"0\",\"0\",\"0\",\"1\",\"1\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"6\",\"3\",\"3\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"183418\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"367\",\"78\",\"60\",\"0\",\"0\",\"1\",\"1\",\"8\",\"1\",\"0\",\"4\",\"3\",\"4\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"4\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"4\",\"0\",\"0\",\"21\",\"7\",\"10\",\"10\",\"5\",\"0\",\"0\",\"0\",\"0\",\"0\",\"5\",\"0\",\"10\",\"9\",\"1\",\"0\",\"1\",\"6\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"3\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"122\",\"122\",\"32\",\"0\",\"0\",\"49\",\"59\",\"0\",\"9\",\"5\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"55\",\"0\",\"0\",\"0\",\"0\",\"24\",\"0\",\"0\",\"0\",\"0\",\"0\",\"55\",\"0\",\"0\",\"24\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"1\",\"0\",\"1\",\"0\",\"0\",\"0\",\"0\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\"],[\"2629\",\"1815\",\"143\",\"0\",\"0\",\"0\",\"0\",\"3\",\"0\",\"14\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"97\",\"46\",\"21\",\"0\",\"0\",\"0\",\"11\",\"3\",\"11\",\"10\",\"8\",\"10\",\"1\",\"1\",\"1\",\"39\",\"26\",\"7\",\"0\",\"0\",\"0\",\"3\",\"1\",\"3\",\"3\",\"3\",\"3\",\"1\",\"1\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"7\",\"10138\",\"11\",\"1661150\",\"15\",\"1893106\",\"6\",\"1652100\",\"0\",\"0\",\"5\",\"9050\",\"103\",\"6130707\",\"110\",\"7356055\",\"60\",\"5575370\",\"0\",\"0\",\"43\",\"555337\",\"0\",\"0\",\"0\",\"4301\",\"1666\",\"458\",\"532\",\"324\",\"1\",\"1\",\"1\",\"0\",\"0\",\"12\",\"12\",\"4\",\"7\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"12\",\"11\",\"0\",\"0\",\"0\",\"0\",\"1\",\"0\",\"12\",\"4\",\"4\",\"55\",\"4\",\"2\",\"2\",\"1\",\"0\",\"0\",\"0\",\"1\",\"0\",\"0\",\"0\",\"2\",\"1\",\"0\",\"0\",\"0\",\"2\",\"1\",\"0\",\"2\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"1\",\"2\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"1\",\"2\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"18\",\"18\",\"10\",\"0\",\"0\",\"0\",\"10\",\"0\",\"0\",\"1\",\"7\",\"0\",\"7\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"2\",\"0\",\"0\",\"0\",\"0\",\"2\",\"0\",\"0\",\"0\",\"0\",\"5\",\"2\",\"0\",\"0\",\"1\",\"0\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"1\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\",\"0\"]]}}}},\"edge_sets\":{\"uin-spread-uin\":{\"features\":{},\"dst_nodetype\":\"uin\",\"src_nodetype\":\"uin\",\"edges\":[{\"src_nodeid\":\"1\",\"dst_nodeid\":\"0\"}]}},\"uin2nodeid_map\":{\"4119566402\":\"0\",\"2458965881\":\"1\"}},\"data_source\":\"train\"}"
    out = infer_model.infer(sample)
    print(out)


if __name__ == '__main__':
    run_seqs_graph()
    # start = time.time()
    # run_infer_graph()
    # end = time.time()
    # print(f"cost: {end - start}")
