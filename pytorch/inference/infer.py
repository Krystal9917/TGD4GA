import os
import sys
import time
import warnings

import pyspark
from pyspark import SparkFiles
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir)))
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
warnings.filterwarnings('ignore')

# from mmgog_long_term_sequence_model.pytorch.models.basic_sequence_model import SeqBaseTransformer
# from mmgog_long_term_sequence_model.utils.argument import Args

from pyspark.sql import SparkSession, Row, functions
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, FloatType

print("当前运行目录", os.getcwd())


# current_dir = os.getcwd()
# folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
#
# # 打印所有文件夹
# print("当前目录下的文件夹有：")
# for folder in folders:
#     print(folder)
#
# current_dir = os.path.abspath(
#     os.path.join(os.getcwd(), "trained_model"))
#
# print("current_dir", current_dir)
#
# # 获取当前目录下的所有可见文件夹
# folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f)) and not f.startswith('.')]
#
# # 打印所有文件夹
# print("当前目录下的文件夹有：")
# for folder in folders:
#     print(folder)


class Inference:
    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.model = SeqBaseTransformer(vocab_size=self.args_dict["vocab_size"],
                                        max_len=self.args_dict["max_len"],
                                        n_layers=self.args_dict["n_layers"],
                                        emb_dim=self.args_dict["emb_dim"],
                                        n_heads=self.args_dict["n_heads"],
                                        output_size=self.args_dict["output_size"],
                                        drop_rate=self.args_dict["drop_rate"],
                                        padding_idx=self.args_dict["padding_idx"])

    def init_model(self):
        self.model.load_state_dict(torch.load(self.args_dict["model_states_path"], map_location='cpu'))
        self.model = self.model.to(self.args_dict["infer_device"])
        self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def prepare_batch_data(self, batch_data):
        roomids = []
        feats = []
        for sample in batch_data:
            roomid = sample["roomid"]
            roomids.append(roomid)
            action_seq = sample["action_seqs"].split("|")
            feat = self.pad_zero_or_truncat(action_seq)
            feats.append(feat)
        return roomids, np.array(feats).astype(int)

    def predict(self, samples):
        self.init_model()
        batch_samples = []
        for sample in samples:
            batch_samples.append(sample)
            if len(batch_samples) == self.args_dict["batch_size"]:
                time_0 = time.perf_counter()
                roomids, batch_data = self.prepare_batch_data(batch_samples)
                time_1 = time.perf_counter()
                print(('-' * 20 + 'prepare batch time consume' + '-' * 40)[:60])
                print(str(time_1 - time_0) + "seconds")
                x_data = torch.tensor(batch_data, dtype=torch.int32).to(self.args_dict["infer_device"])
                infer_y = self.model(x_data)
                time_2 = time.perf_counter()
                print(('-' * 20 + 'infer time consume' + '-' * 40)[:60])
                print(str(time_2 - time_1) + "seconds")
                for roomid, label in zip(roomids, list(infer_y.tolist())):
                    yield roomid, label
                batch_samples = []

        if len(batch_samples) > 0:
            roomids, batch_data = self.prepare_batch_data(batch_samples)
            x_data = torch.tensor(batch_data, dtype=torch.int32).to(self.args_dict["infer_device"])
            infer_y = self.model(x_data)
            for roomid, label in zip(roomids, list(infer_y.tolist())):
                yield roomid, label

    def pad_zero_or_truncat(self, seq):
        if len(seq) > self.args_dict["max_len"]:
            # 行为太长，截断
            return seq[-self.args_dict["max_len"]:]
        else:
            # 行为不足, 补齐
            pad_len = self.args_dict["max_len"] - len(seq)
            seq = seq + [str(self.args_dict["padding_idx"])] * pad_len
            return seq


def get_table(spark, db, tb, pt=None, group='tl'):
    pt = ['p_%s' % str(i) for i in pt]
    return TDWSQLProvider(spark, db=db, group=group).table(tb, priParts=pt)


def drop_and_save_partition(spark, df, db, tb, pt, values_in, group='tl'):
    db_sql = TDWSQLProvider(spark, db=db, group=group)
    db = TDWUtil(dbName=db, group=group)
    assert db.tableExist(tb), '%s not exist in %s' % (tb, db)
    pt = 'p_%s' % str(pt)
    values_in = ','.join([str(val) for val in values_in])
    print(values_in)
    if db.partitionExist(tb, pt):
        print('drop partition table:%s partition:%s' % (tb, pt))
        db.dropPartition(tb, pt)
    tb_info = db.getTableInfo(tb)  # 获取表结构
    print('create partition table:%s partition:%s values_in:%s' % (tb, pt, values_in))
    db.createListPartition(tb, pt, values_in)
    db_sql.saveToTable(df.select(*tb_info.colNames), tb, pt)


# 根据sql获取spark的Dataframe格式数据
def get_df_by_sql(spark, db, tb, pt, sql):
    pt = 'p_%s' % str(pt)
    provider = TDWSQLProvider(spark, db=db, group='tl')
    provider.table(tb, priParts=[pt]).createOrReplaceTempView('data_frame')
    return spark.sql(sql)


if __name__ == '__main__':
    spark = SparkSession.builder.appName("Spark Infer Demo...").getOrCreate()
    sc = spark.sparkContext

    sc.addFile(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, r"models/basic_sequence_model.py")))
    sc.addFile(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, r"utils/argument.py")))
    sc.addFile(os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,
                     r"data/model_states/optimal_sequence_model.bin")))

    from basic_sequence_model import SeqBaseTransformer
    from argument import Args

    # 获取训练好的模型路径
    # sc.addFile(r"./optimal_sequence_model.bin")
    model_states_path = SparkFiles.get("optimal_sequence_model.bin")
    # print(file_path)
    # from basic_sequence_model import SeqBaseTransformer
    # from argument import Args

    # 程序外部传参 argv
    args = Args()
    args.args_dict["model_states_path"] = model_states_path
    sql_args_dict = {
        'ds': args.args_dict["ds"]
    }
    ds_day = args.args_dict["ds"]
    # args_dict = {
    #     'ds': sys.argv[1]
    # }
    # ds_day = sys.argv[1]
    # 测试日期写死
    # ds_day = "20230730"
    # args_dict["ds"] = "20230730"
    print('messi\'s args_dict:', args.args_dict)
    print('messi\'s sql_args_dict:', sql_args_dict)

    # 定义数据结构
    schema = StructType([
        StructField("roomid", LongType(), True),
        StructField("action_seqs", StringType(), True)
    ])
    # 创建数据
    data = [(18359862296, "2|2|2|2|2|7|2|2|8|2"),
            (19480369143, "8|7|7|2|35|35|33|33|33|33|7|8|2|2|2"),
            (20996979463,
             "2|8|8|8|8|8|8|2|8|2|35|35|33|33|33|33|33|35|35|33|33|33|33|2|2|33|33|33|33|33|33|33|33|33|33|33|33|33|33|30|35|35|33|33|33|33|8|8"),
            (47792467575, "2|2|2|2|2|2|1|1|1|1|2|2|2|2|2|2|2|2|2|2|2|2|2|2|2|2|2|2")]
    # 创建DataFrame
    room_tracker_df = spark.createDataFrame(data, schema)
    room_tracker_df.show()
    room_tracker_df.persist(pyspark.StorageLevel.MEMORY_ONLY)

    inf = Inference(args.args_dict)
    rdd_rs = room_tracker_df.rdd.mapPartitions(inf.predict).map(lambda x: Row(roomid=x[0], label=x[1]))
    rdd_rs.take(10)

    schema = StructType([
        StructField("roomid", LongType(), True),
        StructField("label", FloatType(), True)
    ])
    rs_df = spark.createDataFrame(rdd_rs, schema)
    rs_df = rs_df.withColumn("ds", functions.lit(ds_day)) \
        .select(col("ds"), col("roomid"), col("label")) \
        .repartition(100)
    rs_df.show()

    print(('-' * 20 + 'infer done' + '-' * 40)[:60])
    sc.stop()
