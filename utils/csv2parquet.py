import os
import sys

from pyspark.sql import SparkSession

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from mmgog_long_term_sequence_model.utils.argument import Args


def csv_2_parquet(args_dict):
    spark = SparkSession.builder.appName("CSV to Parquet").getOrCreate()

    df = spark.read.format("csv").option("header", "true").load(args_dict["room_seqs_path"])

    df.write.format("parquet").save(args_dict["room_seqs_parquet_path"])


if __name__ == '__main__':
    args = Args()
    print(('-' * 20 + 'args_dict' + '-' * 40)[:60])
    print(args.args_dict)
    csv_2_parquet(args.args_dict)
