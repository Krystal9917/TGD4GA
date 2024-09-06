src_dir="/mnt/wfs/mmchongqingwfssz/project_security-others-common/LLM_TRAINING_PIPELINE/DATASETS/WeSecLM-NLP/OneFileDatasets/uin_gangs_train_dataset/train/240904_2/uin_gangs_train_dataset_train_240904_2.txt"
dst_dir="/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/train/240904_2/uin_gangs_train_dataset_train_240904_2.txt"

cp -rTf "$src_dir" "$dst_dir"

src_dir="/mnt/wfs/mmchongqingwfssz/project_security-others-common/LLM_TRAINING_PIPELINE/DATASETS/WeSecLM-NLP/OneFileDatasets/uin_gangs_train_dataset/eval/240904_2/uin_gangs_train_dataset_eval_240904_2.txt"
dst_dir="/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/eval/240904_2/uin_gangs_train_dataset_eval_240904_2.txt"

cp -rTf "$src_dir" "$dst_dir"

python3 /mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/utils/get_url_data.py
