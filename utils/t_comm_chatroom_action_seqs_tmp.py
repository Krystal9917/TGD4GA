from __future__ import print_function
import os
import sys
import socket

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


# 群行为映射表
action_dict = {
    "0_pad": '0',
    '1_appmsg': '1',
    '1_picture': '2',
    '1_url_appmsg': '3',
    '1_url_appmsg_urlintext': '4',
    '1_url_record_appmsg': '5',
    '1_url_record_urlintext_chatroom': '6',
    '1_url_urlintext_chatroom': '7',
    '1_video': '8',
    '2_qrcode1': '9',
    '2_qrcode10': '10',
    '2_qrcode11': '11',
    '2_qrcode12': '12',
    '2_qrcode13': '13',
    '2_qrcode14': '14',
    '2_qrcode15': '15',
    '2_qrcode16': '16',
    '2_qrcode17': '17',
    '2_qrcode18': '18',
    '2_qrcode19': '19',
    '2_qrcode2': '20',
    '2_qrcode20': '21',
    '2_qrcode3': '22',
    '2_qrcode4': '23',
    '2_qrcode5': '24',
    '2_qrcode6': '25',
    '2_qrcode8': '26',
    '2_qrcode9': '27',
    '2_qrcode99': '28',
    '3_announcement': '29',
    '3_nickname': '30',
    '3_topic': '31',
    '3_verify': '32',
    '4_addchatroombyinvite': '33',
    '4_addchatroombyqrcode': '34',
    '4_addchatroommember': '35',
    '4_createchatroom': '36',
    '4_delchatroommember': '37',
    '4_getqrcode': '38',
    '4_quitchatroomoplog': '39',
    '4_transferchatroomowner': '40',
    '5_expose1': '41',
    '5_expose1024': '42',
    '5_expose128': '43',
    '5_expose2': '44',
    '5_expose32': '45',
    '5_expose32768': '46',
    '5_expose4': '47',
    '5_expose65536': '48',
    '5_expose8192': '49',
    '7_rechb1': '50',
    '7_rechb2': '51',
    '7_rechb21': '52',
    '7_roompay': '53',
    '7_sendhb1': '54',
    '7_sendhb2': '55',
    '7_sendhb21': '56'}


def action_2_idx_udf(action_dict_broadcasted):
    # 摆脱单机思维，要把action_dict broadcast到所有的执行节点
    def f(x):
        return action_dict_broadcasted.value.get(x, '0')

    return functions.udf(f, StringType())  # 对群行为进行编码，没有该行为则编码为'0'


@functions.udf(returnType=StringType())
def list_2_str_udf(col=[]):
    return '|'.join(str(i) for i in col)


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

    LOG_SQL = """
    SELECT
       CAST(REGEXP_EXTRACT(room_id, '^(.*)@chatroom') AS BIGINT) AS roomid,
       CONCAT(CAST(action_type AS STRING), '_', action_sub_type) AS action_str,
       update_ts
    FROM data_frame
    WHERE ds = '{ds}'
    """.format(**args_dict)

    room_tracker_df = get_df_by_sql(spark, 'wxg_gog_app_dwd', 't_comm_chatroom_room_tracker_raw_log_daily', ds_day,
                                    LOG_SQL)

    action_dict_broadcasted = spark.sparkContext.broadcast(action_dict)
    room_tracker_df1 = room_tracker_df.withColumn('action_idx', action_2_idx_udf(action_dict_broadcasted)(
        functions.col('action_str')))

    window_ = Window.partitionBy('roomid').orderBy(functions.col('update_ts').asc())
    room_tracker_df2 = room_tracker_df1.withColumn('action_seqs', functions.collect_list('action_idx').over
    (window_)).groupby('roomid').agg(functions.max('action_seqs').alias('action_seqs'),
                                     functions.countDistinct('update_ts').alias('action_cnt'))

    room_tracker_df3 = room_tracker_df2 \
        .withColumn('action_seqs', list_2_str_udf(functions.col('action_seqs'))) \
        .withColumn('ds', functions.lit(ds_day))

    room_tracker_df4 = room_tracker_df3.withColumn("ds", room_tracker_df3["ds"].cast(StringType())) \
        .withColumn("roomid", room_tracker_df3["roomid"].cast(LongType())) \
        .withColumn("action_seqs", room_tracker_df3["action_seqs"].cast(StringType())) \
        .withColumn("action_cnt", room_tracker_df3["action_cnt"].cast(LongType()))
    drop_and_save_partition(spark, room_tracker_df4, 'wxg_gog_app_tmp', 't_comm_chatroom_action_seqs_tmp',
                            str(ds_day), [str(ds_day)])
    print("save_result_done!!!")
    sc.stop()
