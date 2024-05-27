import sys

from pytoolkit import TDWSQLProvider, TDWUtil
from pyspark.sql import SparkSession, functions, Window
from pyspark.sql.types import StructType, StructField, StringType, LongType


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


@functions.udf(returnType=StringType())
def list_2_str_udf(col=[]):
    return '|'.join(str(i) for i in col)


'''
USE wxg_gog_app_tmp;
CREATE TABLE IF NOT EXISTS t_tmp_acct_action_seqs_v1_tmp(ds STRING,
                                                        add_ds STRING,
                                                        uin BIGINT,
                                                        uin_feat_scalar_list STRING,
                                                        action_seqs STRING,
                                                        action_cnt BIGINT,
                                                        label BIGINT)
PARTITION BY LIST(ds) (PARTITION DEFAULT) STORED AS ORCFILE COMPRESS;
'''

if __name__ == '__main__':
    # 程序外部传参 argv
    args_dict = {
        'ds': sys.argv[1]
    }
    ds_day = sys.argv[1]

    # 测试日期写死
    # ds_day = 20230703
    # args_dict["ds"] = 20230703
    print('messi\'s args_dict:', args_dict)
    spark = SparkSession.builder.appName("Spark Demo...").getOrCreate()
    sc = spark.sparkContext

    sql_ = '''
SELECT *
FROM
  ( SELECT add_ds,
           uin,
           actionid,
           uin_feat_scalar_list,
           touin_feat_scalar_list,
           timestamp_,
           CASE
               WHEN label = '0' THEN 0
               WHEN label = '荐股' THEN 1
               WHEN label = '金融' THEN 2
               WHEN label = '仿冒' THEN 3
               WHEN label = '杀猪盘' THEN 4
               WHEN label = '免费送' THEN 5
               WHEN label = '兼职' THEN 6
               WHEN label = '交友' THEN 7
               WHEN label = '贷款' THEN 8
               WHEN label = '返利' THEN 9
               ELSE -1
           END AS label
   FROM data_frame
   WHERE ds = '{ds}' )
WHERE label >= 0
        '''.format(**args_dict)

    user_tracker_df = get_df_by_sql(spark, 'wxg_gog_app_tmp',
                                    't_tmp_acct_behavior_uin2uin_with_scalar_feat_sample_v1_tmp', ds_day,
                                    sql_)

    # 开窗
    window_ = Window.partitionBy('uin', 'add_ds').orderBy(functions.col('timestamp_').asc())
    user_tracker_df2 = user_tracker_df.withColumn('action_seqs', functions.collect_list(
        functions.concat_ws(':', 'actionid', 'touin_feat_scalar_list')).over
    (window_)).groupby('uin', 'add_ds').agg(functions.max('action_seqs').alias('action_seqs'),
                                            functions.count('timestamp_').alias('action_cnt'),
                                            functions.max('uin_feat_scalar_list').alias('uin_feat_scalar_list'),
                                            functions.max('label').alias('label'),
                                            )
    user_tracker_df2.show()

    user_tracker_df3 = user_tracker_df2 \
        .withColumn('action_seqs', list_2_str_udf(functions.col('action_seqs'))) \
        .withColumn('ds', functions.lit(ds_day))
    user_tracker_df3.show()

    user_tracker_df4 = user_tracker_df3.withColumn("ds", user_tracker_df3["ds"].cast(StringType())) \
        .withColumn("add_ds", user_tracker_df3["add_ds"].cast(StringType())) \
        .withColumn("uin", user_tracker_df3["uin"].cast(LongType())) \
        .withColumn("uin_feat_scalar_list", user_tracker_df3["uin_feat_scalar_list"].cast(StringType())) \
        .withColumn("action_seqs", user_tracker_df3["action_seqs"].cast(StringType())) \
        .withColumn("action_cnt", user_tracker_df3["action_cnt"].cast(LongType())) \
        .withColumn("label", user_tracker_df3["label"].cast(LongType()))
    user_tracker_df4.show()

    drop_and_save_partition(spark, user_tracker_df4, 'wxg_gog_app_tmp', 't_tmp_acct_action_seqs_v1_tmp',
                            str(ds_day), [str(ds_day)])
    print("save_result_done!!!")
    sc.stop()
