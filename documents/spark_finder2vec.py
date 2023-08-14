# -*- coding: utf-8 -*-

import argparse
import os
import time
import pickle
import torch.nn.functional as F
import numpy as np
import pyspark

import torch
from pyspark.sql.session import SparkSession
from models.backbones.deep_fm import DeepFMV2
from models.simsiam import SiamNet

global BATCH_SIZE
global MODEL_PATH
global FEAT_CONFIG_FILE


def jPath(sc, filepath):
    jPathClass = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return jPathClass(filepath)


def jFileSystem(sc):
    jFileSystemClass = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_config = sc._jsc.hadoopConfiguration()
    return jFileSystemClass.get(hadoop_config)


def init_model():
    """
    Init Finder2Vec Model
    :return:
    """
    tabular_feats = pickle.load(open(FEAT_CONFIG_FILE, 'rb'))
    dnn_feature_columns = tabular_feats
    linear_feature_columns = tabular_feats
    backbone = DeepFMV2(linear_feature_columns, dnn_feature_columns)
    model = SiamNet(backbone, out_dim=128)
    model.projector.set_layers(2)
    # if model_cfg.proj_layers is not None:
    #     model.projector.set_layers(model_cfg.proj_layers)

    # assert args.eval_from is not None
    # eval_from_path = r'F:\codehub\SimSiam\ckpt\simsiam-cifar10-experiment-resnet18_cifar_variant1_0511133317.pth'
    save_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(save_dict['state_dict'])
    model = model.to("cpu")
    model = torch.nn.DataParallel(model)

    backbone_net = model.module.encoder
    backbone_net.eval()
    return backbone_net


def array2str(arr):
    return ' '.join([str(elem) for elem in arr])


def prepare_batch(batch):
    """
    Parse Sample
    :param sample:
    :return:
    """
    raw_cols = {'user_id', 'item_id', 'sample_type', 'label', 'extra_info'}
    ids = []
    feats = []
    for sample in batch:
        cur_id = sample['item_id']
        ids.append(cur_id)
        all_cols = sample.__fields__
        cur_vec = [sample[col] for col in all_cols if col not in raw_cols]
        feats.append(cur_vec)
    return ids, np.array(feats, dtype=np.float)


def predict(samples):
    """
    Batch Vector Generation
    :param samples:
    :return:
    """
    # Define model
    print("=> Loading model...")
    model = init_model()
    batch_samples = []
    for sample in samples:
        batch_samples.append(sample)
        if len(batch_samples) == BATCH_SIZE:
            t0 = time.perf_counter()
            cur_ids, image = prepare_batch(batch_samples)
            t1 = time.perf_counter()
            print("Prepare Batch:" + str(t1 - t0))
            cur_input = torch.tensor(image).float()
            feature = model(cur_input)
            batch_vec = F.normalize(feature, dim=1).detach().numpy()
            t2 = time.perf_counter()
            print("Inference:" + str(t2 - t1))
            for cur_id, cur_vec in zip(cur_ids, list(batch_vec)):
                yield cur_id, array2str(cur_vec)
            batch_samples = []
    if len(batch_samples) > 0:
        cur_ids, image = prepare_batch(batch_samples)
        cur_input = torch.tensor(image).float()
        feature = model(cur_input)
        batch_vec = F.normalize(feature, dim=1).detach().numpy()
        for cur_id, cur_vec in zip(cur_ids, list(batch_vec)):
            yield cur_id, array2str(cur_vec)


def raw_to_string_array(row, fields):
    """
    Converting row to string array for writing to tdw
    :param row:
    :param fields:
    :return:
    """
    cur_result = []
    cur_dict = row.asDict()
    for field in fields:
        if field in cur_dict:
            cur_result.append(str(cur_dict[field]))
        else:
            cur_result.append('')
    return cur_result


def max_fold(x, y):
    """
    Get value which is greater
    :param x:
    :param y:
    :return:
    """
    if x >= y:
        return x
    else:
        return y


if __name__ == '__main__':
    spark = SparkSession.builder.appName("Spark Finder2Vec Demo...").getOrCreate()
    sc = spark.sparkContext

    torch.set_default_tensor_type(torch.FloatTensor)

    model_path = os.path.join(
        r'E:\data\group_detection\siam_models\simsiam-cifar10-experiment-resnet18_cifar_variant1_0514142627.pth')
    print("\tModel Path:{}".format(model_path))
    DATA_DIR = r'E:\data\group_detection\parquet_records\p_20210513'
    OUT_DIR = r'E:\data\group_detection\pred_emb'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='file:///' + DATA_DIR,
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=model_path,
        help='Model Path.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='file:///' + OUT_DIR,
        help='Directory to put the output data.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Predict Batch Size"
    )
    parser.add_argument(
        '--date',
        type=str,
        default='20210513',
        help="Data Date"
    )
    parser.add_argument(
        '--sample',
        type=float,
        default=0.01,
        help="Sample Rate"
    )
    parser.add_argument(
        '--parts',
        type=int,
        default=50,
        help="Parts..."
    )
    parser.add_argument(
        '--out_parts',
        type=int,
        default=50,
        help="Out Parts..."
    )
    parser.add_argument(
        '--config',
        type=str,
        default=r'F:\codehub\SimSiam\configs\simsiam_tabular.yaml',
        help='Model Config File...'
    )
    args, _ = parser.parse_known_args()

    print(args)
    config_file_path = args.config

    # Data
    BATCH_SIZE = args.batch_size
    MODEL_PATH = args.model_path
    FEAT_CONFIG_FILE = args.config
    PARTS_NUM = args.parts

    if args.sample < 1.0:
        raw_records = spark.read.parquet(args.input_dir).sample(False, args.sample).repartition(PARTS_NUM)
    else:
        raw_records = spark.read.parquet(args.input_dir).repartition(PARTS_NUM)
    raw_records.persist(pyspark.StorageLevel.MEMORY_ONLY)

    data_cnt = raw_records.count()
    print("=> Raw Data Count:{}".format(data_cnt))
    cur_date = args.date
    # final_ttl = int(time.time()) + args.ttl
    rdd_result = raw_records.rdd.mapPartitions(predict) \
        .map(lambda x: " ".join([x[0], x[1]]))

    print(rdd_result.take(5))
    out_path = args.output_dir + "/" + cur_date
    print("=> Output Path:{}".format(out_path))
    rdd_result.repartition(args.out_parts).saveAsTextFile(out_path)
    sc.stop()
