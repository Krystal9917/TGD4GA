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
    pt = ['p_%s' % i for i in pt]
    provider = TDWSQLProvider(spark, db=db, group='tl')
    provider.table(tb, priParts=pt).createOrReplaceTempView('data_frame')
    return spark.sql(sql)


action_dict = {
    0: "未知行为",
    1: "登录",
    2: "封号",
    3: "解封",
    4: "建群/拉人",
    5: "退群/踢人",
    6: "注册",
    7: "转移群主",
    8: "修改群名/群公告",
    9: "举报",
    10: "申诉",
    11: "绑定",
    12: "解绑",
    13: "改密",
    14: "冻结",
    15: "解冻",
    16: "修改备注",
    17: "朋友圈",
    18: "加好友",
    19: "LBS",
    20: "漂流瓶",
    21: "摇一摇",
    22: "删除好友",
    23: "搜索联系人",
    24: "授权登录",
    25: "关注公众号",
    26: "修改头像",
    27: "修改昵称",
    28: "修改签名",
    29: "二次验证",
    30: "群发助手",
    31: "抢群红包",
    32: "扫码进群",
    33: "申诉好友辅助",
    34: "扫描二维码",
    35: "公众号图文点击(阅读)",
    36: "黑名单增删",
    37: "删除朋友圈",
    38: "踢下线",
    39: "支付支出",
    40: "聊天",
    41: "voip群",
    42: "解封辅助",
    43: "注册辅助",
    44: "处罚打击",
    45: "设备",
    46: "GetA8Key",
    50: "在线状态",
    # log_22769
    51: "安装App(Android)",

    # 登录
    10001: "微信登录",
    10002: "QQ登录",
    10003: "随机验证码登录",
    10004: "邮箱登录",
    10005: "扫码登录",
    10006: "FB授权登录",
    10007: "登录扫码",

    # 解封
    20001: "解封开始",
    20002: "解封短信上行",
    20003: "解封短信下行",
    20004: "解封朋友辅助",
    20005: "解封朋友搜索",
    20006: "解封财产转移",
    20007: "解封输入手机",
    20008: "解封人脸检测",
    20009: "解封人脸检测验",
    20010: "解封验证码",

    # 举报
    30001: "举报",
    30002: "被举报",

    # 备注
    40001: "备注",
    40002: "被备注",

    # 朋友圈
    50001: "发朋友圈",
    50002: "朋友圈评论",
    50003: "朋友圈被评论",
    50004: "朋友圈广告评论",
    50005: "朋友圈修改封面",

    # 加好友
    60001: "加好友",
    60002: "被加好友",
    60003: "打招呼",
    60004: "被打招呼",

    # 删除好友
    70001: "删除好友",
    70002: "被删好友",

    # 建群拉人
    80001: "建群",
    80002: "拉人入群",
    80003: "被拉入群",

    # 退群
    90001: "退群",
    90003: "群主踢人",
    90002: "被踢出群",

    # 转移群主
    11001: "转移群主-新群主",
    11002: "转移群主-旧群主",

    # 修改群信息
    12001: "修改群名",
    12002: "修改群公告",

    # 扫描群二维码入群
    13001: "扫描二维码入群",
    13002: "群二维码分享",
    13003: "群成员生成群二维码",

    # 申诉好友辅助
    14001: "申诉好友辅助",
    14002: "被邀请好友辅助",

    # 黑名单增删
    15001: "修改黑名单",
    15002: "被修改黑名单",

    # 支付数据
    16000: "默认出账",
    16001: "WEB扫码支付",
    16002: "app间支付",
    16003: "公众号JS API支付",
    16004: "公众号原生支付",
    16005: "表情商城支付",
    16006: "商城原生支付",
    16008: "刷卡(被扫)",
    16010: "IAP支付",
    16011: "零钱充值",
    16021: "零钱提现",
    16031: "转账",
    16032: "f2f收款转账",
    16033: "固定金额f2f收款转账",
    16035: "safari支付",
    16036: "wap支付",
    16037: "红包支出",
    16038: "委托代扣",
    16041: "新wap支付",
    16042: "群收款",
    16043: "f2f红包支出",
    16044: "朋友圈红包支出",
    16045: "购买零钱通",
    16046: "小程序支出",

    # 支付重定义【 天网行为流关系探索 】【数据来源ilogs19449】
    390001: "Web扫码支付",
    390002: "App支付",
    390003: "JSAPI支付",
    390004: "商户订单native支付",
    390005: "表情商城支付",
    390006: "话费充值native支付",
    390008: "刷卡支付",
    390010: "IAP支付",
    390011: "零钱充值",
    390021: "零钱提现",
    390031: "普通转账",
    390032: "面对面付款",
    390033: "固定金额面对面支付",
    390035: "Safari收银台支付",
    390036: "wap支付",
    390037: "红包支付",
    390038: "委托代扣支付",
    390039: "摇红包支付",
    390041: "新H5支付",
    390042: "群收款",
    390043: "面对面红包支付",
    390044: "朋友圈红包支付",
    390045: "零钱通转入",
    390046: "小程序支付",
    390047: "黄金红包支付",
    390048: "二维码赞赏",
    390049: "转账到银行卡",
    390050: "商业面对面支付",
    390051: "零钱通赎回",
    390052: "零钱宝认购",
    390053: "零钱包赎回",
    390054: "人脸识别",
    390056: "转账到手机号",
    390060: "缴税",
    390061: "平台代扣",
    390062: "EVM扫码支付",
    390063: "手表支付",
    390064: "合单支付",
    390065: "个人面对面收付款",
    390066: "面对面付款码支付",
    390067: "企业互通小程序单聊支付",
    390068: "企业互通小程序群聊支付",
    390069: "微信礼物支付",
    390070: "儿童零花钱充值",
    390071: "扫儿童红包码发红包",
    390072: "刷掌支付",
    390073: "native支付的商家通支付",
    390080: "iOS的nfc交通卡",
    390082: "微信客户端互通C2B转账",
    390083: "公众号—扫码支付形态收银台",
    390084: "小程序—扫码支付形态收银台",
    390085: "商家理财",
    390086: "来华通充值",
    390087: "来华通提现",

    # 待确定文案
    18000: "解封辅助开始(eUnBan_Help_Start)",
    18001: "解封辅助支付卡(eUnBan_Help_PayCard)",
    18002: "解封辅助输入(eUnBan_Help_Input)",
    19000: "注册辅助二维码(eReg_Help_QRCode)",
    19001: "注册辅助输入(eReg_Help_Input)",
    19002: "注册辅助确认(eReg_Help_DoConfirm)",
    19003: "注册辅助检查支付(eReg_Help_CheckPay)",

    # 处罚打击
    20000: "进群实时拦截",

    # 设备
    21000: "删设备",

    # 自定义消息行为 【来源】log_16584
    5201000: "单聊-文本",
    5203000: "单聊-图片",
    5234000: "单聊-语音",
    5242000: "单聊-名片",
    5243000: "单聊-视频",
    5247000: "单聊-表情包",
    5248000: "单聊-位置",
    5249001: "单聊-APP_Text",
    5249002: "单聊-APP_Image",
    5249003: "单聊-APP_Music",
    5249004: "单聊-APP_Video",
    5249005: "单聊-APP_URL",
    5249006: "单聊-APP_File",
    5249019: "单聊-聊天记录",
    5249024: "单聊-笔记",
    5249033: "单聊-小程序",
    5249044: "单聊-仿原生小程序分享出的消息",
    5249046: "单聊-微视动态视频",
    5249047: "单聊-带链接的文本",
    5249050: "单聊-视频号名片",
    5249051: "单聊-视频号feed",
    5249057: "单聊-引用消息",
    5249059: "单聊-视频号旧话题页",
    5249063: "单聊-直播feed",
    5249068: "单聊-LITEAPP分享",
    5249069: "单聊-游戏人生游戏名片消息",
    5249071: "单聊-视频号长视频分享",
    5249072: "单聊-视频号新话题页",
    5249076: "单聊-音乐MV",
    5249082: "单聊-直播商品分享",
    5249083: "单聊-青少年模式监护人申请",
    5249084: "单聊-青少年向监护人申请临时授权",
    5249088: "单聊-付费直播",
    5249092: "单聊-听一听音频",
    5249098: "单聊-聊天多图视频合集气泡",
    5249099: "单聊-超大附件",
    5299999: "单聊-拍一拍"
}


@functions.udf(returnType=StringType())
def list_2_str_udf(col=None):
    return '|'.join(str(i) for i in col)


def action_2_idx_udf(action_dict_broadcasted):
    # 摆脱单机思维，要把action_dict broadcast到所有的执行节点
    def f(x):
        return action_dict_broadcasted.value.get(x, '未知行为')

    return functions.udf(f, StringType())


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

    sql_19594 = '''
    SELECT 
      uin_ AS uin,
      CASE
          WHEN sceneid_ = 9 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 16 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 17
               AND operateid_ IN (50002,
                                  50003,
                                  50004) THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 18 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 22 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 29 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 33 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          WHEN sceneid_ = 36 THEN SPLIT(remark_, '\\\\|') [ 0 ]
          ELSE "0"
      END AS to_uin,
      sceneid_ AS sceneid,
      CAST(IF (operateid_ = 0, sceneid_, operateid_) AS BIGINT) AS operateid,
      timestamp_,
      '' AS remark
    FROM data_frame
    WHERE sceneid_ IN (9,
                   16,
                   18,
                   22,
                   29,
                   33,
                   36)
      OR (sceneid_ = 17
          AND operateid_ IN (50002,
                             50003,
                             50004))
        '''.format(**args_dict)
    user_tracker_19594_pt = [ds_day + str(i).zfill(2) for i in range(24)]
    user_tracker_df_19594 = get_df_by_sql(spark, 'wxg_security_secretdata', 'log_19594', user_tracker_19594_pt,
                                          sql_19594)
    user_tracker_df_19594.show()

    sql_19449 = '''
    SELECT 
          uin_ AS uin,
          CAST(recv_uin_ AS STRING) AS to_uin,
          39 AS sceneid,
          CAST((390000 + pay_scene_) AS BIGINT) AS operateid,
          timestamp_,
          '' AS remark
    FROM data_frame
    WHERE amount_ > 0
     AND recv_uin_ > 100000
     AND mchid_ = 0
        '''.format(**args_dict)
    user_tracker_19449_pt = [ds_day + str(i).zfill(2) for i in range(24)]
    user_tracker_df_19449 = get_df_by_sql(spark, 'wxg_security_secretdata', 'log_19449', user_tracker_19449_pt,
                                          sql_19449)
    user_tracker_df_19449.show()

    sql_msg = '''
    SELECT from_uin AS uin,
           CAST(touin AS STRING) AS to_uin,
           52 AS sceneid,
           CASE
               WHEN msg_type = 10002
                    AND sub_msg_type = 67 THEN 5299999
               ELSE 52 * 100000 + msg_type * 1000 + sub_msg_type
           END AS operateid,
           timestamp_,
           '' AS remark
    FROM data_frame
    WHERE from_type = 1
      AND to_type = 1
      AND self_msg <> 1
      AND msg_id > 0
      AND (msg_type IN (1,
                        3,
                        34,
                        42,
                        43,
                        47,
                        48)
           OR (msg_type = 49
               AND sub_msg_type IN (1,
                                    2,
                                    3,
                                    4,
                                    5,
                                    6,
                                    19,
                                    24,
                                    33,
                                    44,
                                    46,
                                    47,
                                    50,
                                    51,
                                    57,
                                    59,
                                    63,
                                    68,
                                    69,
                                    71,
                                    72,
                                    76,
                                    82,
                                    83,
                                    84,
                                    88,
                                    92,
                                    98,
                                    99))
           OR (msg_type = 10002
               AND sub_msg_type = 67))
      AND busi_type < 100000
        '''.format(**args_dict)
    user_tracker_msg_pt = [ds_day]
    user_tracker_msg_df = get_df_by_sql(spark, 'wxg_gog_app_dwd', 't_comm_msg_raw_message_daily', user_tracker_msg_pt,
                                        sql_msg)
    user_tracker_msg_df.show()

    user_tracker_df = user_tracker_df_19594.union(user_tracker_df_19449).union(user_tracker_msg_df)
    user_tracker_df.show()

    action_dict_broadcasted = sc.broadcast(action_dict)
    user_tracker_df_1 = user_tracker_df.withColumn('action_type', action_2_idx_udf(action_dict_broadcasted)(
        functions.col('operateid'))).withColumn('action_id', user_tracker_df['operateid'].cast(LongType()))
    user_tracker_df_1.show()

    # 过滤不合法的取值
    user_tracker_df_all = user_tracker_df_1.filter(
        (user_tracker_df_1["to_uin"] > 100000) & (user_tracker_df_1["to_uin"] != user_tracker_df_1["uin"]))

    user_tracker_final = user_tracker_df_all.withColumn('ds', functions.lit(ds_day)) \
        .withColumn('uin', user_tracker_df_all['uin'].cast(LongType())) \
        .withColumn('to_uin', user_tracker_df_all['to_uin'].cast(LongType())) \
        .withColumn('action_id', user_tracker_df_all['action_id'].cast(LongType())) \
        .withColumn('action_type', user_tracker_df_all['action_type'].cast(StringType())) \
        .withColumn('timestamp_', user_tracker_df_all['timestamp_'].cast(LongType())) \
        .withColumn('remark', user_tracker_df_all['remark'].cast(LongType()))
    user_tracker_final.show()

    drop_and_save_partition(spark, user_tracker_final, 'wxg_gog_app_dwd', 't_acct_relation_uin2uin_action_daily',
                            str(ds_day), [str(ds_day)])

    sc.stop()
