import numpy as np
class DatasetUtilsForGNN:
    @staticmethod
    def shuffle_dataset(dataset, seed):
        np.random.seed(seed)
        np.random.shuffle(dataset)
        return dataset
    @staticmethod
    def split_dataset(dataset, ratio):
        n = int(ratio * len(dataset))
        dataset_1, dataset_2 = dataset[:n], dataset[n:]
        return dataset_1, dataset_2
