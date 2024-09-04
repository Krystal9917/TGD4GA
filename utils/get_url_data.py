import requests
import concurrent.futures
import os
import sys
import logging

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))


# from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_train_run import ArgsUinGangs


def download_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果响应状态码不是200，将引发HTTPError异常
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


def process_urls_from_file(input_file_path, output_file_path, max_workers=64):
    with open(input_file_path, 'r') as file:
        urls = [line.strip() for line in file if line.strip()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用map函数来并行处理所有URL
        results = executor.map(download_url_content, urls)

    with open(output_file_path, 'w') as outfile:
        for content in results:
            if content:
                outfile.write(content + '\n')


if __name__ == '__main__':
    # args_dict = ArgsUinGangs().args_dict
    # 指定包含URL的文件路径和输出文件路径
    train_input_file_path = '/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/train/240904_1/uin_gangs_train_dataset_train_240904_1.txt'
    train_output_file_path = '/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/train/240904_1/uin_gangs_train_dataset_train_240904_1_20240904.txt'
    process_urls_from_file(train_input_file_path, train_output_file_path)
    eval_input_file_path = '/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/eval/240904_1/uin_gangs_train_dataset_eval_240904_1.txt'
    eval_output_file_path = '/mnt/cephfs/messizeng/nlp/mmgog_long_term_sequence_model/data/uin_gangs_train_dataset/eval/240904_1/uin_gangs_train_dataset_eval_240904_1_20240904.txt'
    process_urls_from_file(eval_input_file_path, eval_output_file_path)
