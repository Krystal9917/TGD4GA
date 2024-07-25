import json
import torch
from torch.utils.data import IterableDataset


class DataProcessIterable(IterableDataset):
    """
    批数据预处理类
    """

    def __init__(self, args_dict):
        super(DataProcessIterable, self).__init__()
        self.args_dict = args_dict
        self.data_size = self.get_data_size()
        self.seqGraphDataProcess = SeqGraphDataProcess(self.args_dict)

    def get_data_size(self):
        with open(self.args_dict['file_path'], 'r', encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)
        return total_lines

    def process_line(self, line):
        json_str = json.loads(line)
        seq_graph_data = self.seqGraphDataProcess.process_data(json_str)
        return seq_graph_data

    def __iter__(self):
        # 先计算数据量分配到每个进程中
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = None
        else:  # in a worker process
            # 计算每个进程的分片
            total_lines = self.data_size
            per_worker = int(total_lines / worker_info.num_workers)
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            if worker_id == worker_info.num_workers - 1:  # 最后一个工作进程
                iter_end = total_lines

        with open(self.args_dict['file_path'], 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= iter_start and (iter_end is None or i < iter_end):
                    yield self.process_line(line.strip())
                elif i >= iter_end:
                    break

    def process_line(self, line):
        sample = self.seqGraphDataProcess.process_data(line)
        return sample


class SeqGraphDataProcess:
    """
    单条数据预处理类
    """

    def __init__(self, args_dict):
        self.args_dict = args_dict


    def process_data(self, json_str):
        # 先将json字符串解析
        data = json.loads(json_str)

        return data
