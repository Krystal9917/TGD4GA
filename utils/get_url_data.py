import requests
import concurrent.futures
import os
import sys
import logging
import time

logger = logging.getLogger("my_logger")
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))


# from mmgog_long_term_sequence_model.pytorch.trainer.uin_gangs_model_train_run import ArgsUinGangs


def download_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 如果响应状态码不是200，将引发HTTPError异常
        response.encoding = 'utf-8'
        return response.text
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None


def process_urls_from_file(input_file_path, output_file_path, max_workers=32, batch_size=10000):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]

    total_batches = (len(urls) + batch_size - 1) // batch_size  # 计算总批次数
    print(f"Total batches: {total_batches}")

    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min((batch_index + 1) * batch_size, len(urls))
        batch_urls = urls[batch_start:batch_end]
        print(f"Processing batch {batch_index + 1}/{total_batches}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(download_url_content, batch_urls)

        with open(output_file_path, 'a', encoding='utf-8') as outfile:  # 注意使用追加模式 'a'
            for content in results:
                if content:
                    outfile.write(content + '\n')


if __name__ == '__main__':
    # args_dict = ArgsUinGangs().args_dict
    # 指定包含URL的文件路径和输出文件路径
    # file_names = ['24112' + str(i) for i in range(10)] + ['241130', '241201', '241202']
    prefix_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset'
    # prefix_dir = '/mnt/cephfs/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset'
    # prefix_dir = '/mnt/wfs/mmchongqingwfssz/project_security-others-common/LLM_TRAINING_PIPELINE/sz_old/DATASETS/WeSecLM-NLP/OneFileDatasets/'
    # for file_name in file_names:
    #     train_input_file_path = prefix_dir + f'/train/{file_name}_20241119/uin_gangs_full_graph_dataset_train_{file_name}_20241119.txt'
    #     train_output_file_path = prefix_dir + f'/train/raw/uin_gangs_full_graph_dataset_train_{file_name}_20241119.txt'
    #     process_urls_from_file(train_input_file_path, train_output_file_path)
        # eval_input_file_path = prefix_dir + f'/eval/{file_name}/uin_gangs_full_graph_dataset_eval_{file_name}.txt'
        # eval_output_file_path = prefix_dir + f'/eval/json_data/uin_gangs_full_graph_dataset_eval_{file_name}.txt'
        # process_urls_from_file(eval_input_file_path, eval_output_file_path)
    # date = '24'
    train_input_file_path = prefix_dir + f'/valid/250324_202503241700/uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700.txt'
    train_output_file_path = prefix_dir + f'/valid/raw/uin_gangs_supervise_full_graph_dataset_eval_250324_202503241700.txt'
    # train_output_file_path = prefix_dir + f'/train/raw/uin_gangs_full_graph_dataset_train_2503{date}_202503201445.txt'
    st = time.time()
    process_urls_from_file(train_input_file_path, train_output_file_path)
    # print(f"Process time: {time.time() - st:.4f} s")
