import os
from torch_geometric.datasets import HGBDataset

def load_dataset(dataset_dir, dataset_name):
    dataset = HGBDataset(root=dataset_dir, name=dataset_name)[0]
    print(dataset)
    print()

if __name__ == '__main__':
    name = 'IMDB'
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            os.path.pardir, os.path.pardir,
                                            "data", "hetero_datasets", name))
    load_dataset(data_dir, name)