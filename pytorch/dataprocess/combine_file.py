import json
import random

def combine_supervise_file(in_dir, file1_name, file2_name, out_dir, outfile_name):
    # Open the output file in write mode
    with open(out_dir + outfile_name, 'w') as outfile:
        new_lines = []
        # Read from the first file and write to the output file
        with open(in_dir + file1_name, 'r') as infile1:
            infile1_len = 0
            for i, line in enumerate(infile1):
                infile1_len += 1
                new_lines.append(line)
            print(f"Malicious File Len: {infile1_len}")

        with open(in_dir + file2_name, 'r') as infile2:
            line_indices = list(range(sum(1 for _ in infile2)))
            random.shuffle(line_indices)
            print(f"Normal lines: {len(line_indices)}")
            line_indices = line_indices[:(297-19)]
            print(line_indices)

        # Read from the second file and write to the output file
        with open(in_dir + file2_name, 'r') as infile2:
            for i, line in enumerate(infile2):
                if i in line_indices:
                    new_lines.append(line)

        random.shuffle(new_lines)

        for line in new_lines:
            outfile.write(line)

    print(f"Combined {file1_name} and {file2_name} into {outfile_name}.")


def split_data(in_dir, infile_name, out_dir, train_file_name, test_file_name):
    with open(in_dir + infile_name, 'r') as f:
        lines = list(range(sum(1 for _ in f)))
        random.shuffle(lines)
    total_lines = len(lines)
    print(f"All: {total_lines}")
    ratio = 0.75
    split = int(total_lines * ratio)
    train_lines = lines[:split]
    test_lines = lines[split:]
    train_list = []
    test_list = []
    with open(in_dir + infile_name, 'r') as f:
        for i, line in enumerate(f):
            if i in train_lines:
                train_list.append(line)
            elif i in test_lines:
                test_list.append(line)

    print(f"Train: {len(train_list)}")
    print(f"Test: {len(test_list)}")

    with open(out_dir + train_file_name, 'w') as train_f:
        for line in train_list:
            train_f.write(line)

    with open(out_dir + test_file_name, 'w') as test_f:
        for line in test_list:
            test_f.write(line)


def read_file(in_dir, infile_name):
    gang_list = []
    normal_list = []
    with open(in_dir + infile_name, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if 'gangs_label' in data.keys():
                if '团伙' in data['gangs_label']:
                    gang_list.append(i)
                else:
                    normal_list.append(i)
            else:
                normal_list.append(i)
    print(len(gang_list), len(normal_list))


def filter_positive_samples(in_dir, infile_name, outfile_name):
    gang_list = []
    with open(in_dir + infile_name, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if 'gangs_label' in data.keys():
                if '异常团伙' in data['gangs_label']:
                    gang_list.append(line)
    with open(in_dir + outfile_name, 'w') as out_f:
        for line in gang_list:
            out_f.write(line)

if __name__ == '__main__':
    path_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/raw/'
    # file1 = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241211.txt'
    file2 = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241211.txt'
    file3 = 'uin_gangs_supervise_full_graph_dataset_eval_normal_subgraphs.txt'
    output_file = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241211_positive.txt'
    out_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/processed/'
    train_file = 'uin_gangs_supervise_full_graph_dataset_train_241204_20241204.txt'
    test_file = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241204.txt'
    # read_file(path_dir, file1)

    # combine_supervise_file(path_dir, file2, file3, path_dir, output_file)

    # split_data(path_dir, output_file, out_dir, train_file, test_file)

    # read_file(out_dir, train_file)

    filter_positive_samples(path_dir, file2, output_file)
