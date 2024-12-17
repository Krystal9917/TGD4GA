export http_proxy="http://star-proxy.oa.com:3128"
export https_proxy="http://star-proxy.oa.com:3128"
export ftp_proxy="http://star-proxy.oa.com:3128"
export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"
pip install --upgrade pip
pip install -r /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/requirements.txt -i https://mirrors.tencent.com/pypi/simple/
pip install torch==1.13.1
pip install networkx
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/pyg_lib-0.3.1+pt113cu117-cp38-cp38-linux_x86_64.whl
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/torch_cluster-1.6.1+pt113cu117-cp38-cp38-linux_x86_64.whl
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/torch_scatter-2.1.0+pt113cu117-cp38-cp38-linux_x86_64.whl
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/torch_sparse-0.6.16+pt113cu117-cp38-cp38-linux_x86_64.whl
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/torch_geometric-2.6.1-py3-none-any.whl
pip install /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/config/libs/sentencepiece-0.1.91-cp38-cp38-manylinux1_x86_64.whl
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export MASTER_ADDR='localhost'
export MASTER_PORT='10086'
torchrun /mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/pytorch/trainer/uin_gangs_model_pretrain_ddp_run.py