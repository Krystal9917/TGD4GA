import random
# Define the names of the input files and the output file
path_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/raw/'
file1 = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241204.txt'
file2 = 'uin_gangs_supervise_full_graph_dataset_eval_normal_subgraphs.txt'
output_file = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241204_combined.txt'
out_dir = '/chongqinggeminiceph1fs/geminicephfs/security-others-common/jiujiuchen/projects/mmgog_long_term_sequence_model/data/uin_gangs_full_graph_dataset/valid/processed/'
train_file = 'uin_gangs_supervise_full_graph_dataset_train_241204_20241204.txt'
test_file = 'uin_gangs_supervise_full_graph_dataset_eval_241204_20241204.txt'

# Open the output file in write mode
with open(path_dir+output_file, 'w') as outfile:
    new_lines = []
    # Read from the first file and write to the output file
    with open(path_dir+file1, 'r') as infile1:
        infile1_len = 0
        for i, line in enumerate(infile1):
            infile1_len += 1
            new_lines.append(line)
        print(f"Malicious File Len: {infile1_len}")

    with open(path_dir+file2, 'r') as infile2:
        line_indices = list(range(sum(1 for _ in infile2)))
        random.shuffle(line_indices)
        print(f"Normal lines: {len(line_indices)}")
        line_indices = line_indices[:infile1_len]
        print(line_indices)

    # Read from the second file and write to the output file
    with open(path_dir+file2, 'r') as infile2:
        for i, line in enumerate(infile2):
            if i in line_indices:
                new_lines.append(line)

    random.shuffle(new_lines)

    for line in new_lines:
        outfile.write(line)

print(f"Combined {file1} and {file2} into {output_file}.")

# with open(path_dir + output_file, 'r') as f:
#     lines = list(range(sum(1 for _ in f)))
#     random.shuffle(lines)
# total_lines = len(lines)
# print(f"All: {total_lines}")
# ratio = 0.75
# split = int(total_lines*ratio)
# train_lines = lines[:split]
# test_lines = lines[split:]
# train_list = []
# test_list = []
# with open(path_dir + output_file, 'r') as f:
#     for i, line in enumerate(f):
#         if i in train_lines:
#             train_list.append(line)
#         elif i in test_lines:
#             test_list.append(line)
#
# print(f"Train: {len(train_list)}")
# print(f"Test: {len(test_list)}")
#
# with open(out_dir + train_file, 'w') as train_f:
#     for line in train_list:
#         train_f.write(line)
#
# with open(out_dir + test_file, 'w') as test_f:
#     for line in test_list:
#         test_f.write(line)