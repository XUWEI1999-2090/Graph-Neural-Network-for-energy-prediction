import numpy as np
import pickle
import torch
import time

class GNNFingerprintAdjacencyPropertySample(object):

    def __init__(self,
                 mol_name,
                 fingerprint,
                 adjacency,
                 y=None,
                 # 拿出GNN模型预测前一层的feature
                 embedding_feature=None
                 ):
        self.fingerprint = fingerprint
        self.adjacency = adjacency
        self.y = y
        self.mol_name = mol_name
        self.embedding_feature = embedding_feature


class RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetLoaderToTorchTensorDataset(object):
    '''
    读取DatasetMarker.RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker的数据
    这里不依赖于rdkit
    '''

    def __init__(self, dataset_dir, device):
        self.dataset_dir = dataset_dir
        self.device = device

    def load_dataset(self):
        dir_input = self.dataset_dir
        self.molecules = self.load_tensor(dir_input + 'molecules', self.device, torch.LongTensor)
        self.adjacencies = self.load_numpy(dir_input + 'adjacencies')
        self.properties = self.load_tensor(dir_input + 'properties', self.device, torch.FloatTensor)
        self.mean = self.load_numpy(dir_input + 'mean')
        self.std = self.load_numpy(dir_input + 'std')
        with open(dir_input + 'fingerprint_dict.pkl', 'rb') as f:
            fingerprint_dict = pickle.load(f)
        with open(dir_input + "./mol_name_list.pkl", "rb") as f:
            self.mol_name_list = pickle.load(f)
        self.n_fingerprint = len(fingerprint_dict)

        self.dataset = []
        for i in range(len(self.molecules)):
            self.dataset.append(GNNFingerprintAdjacencyPropertySample(
                fingerprint=self.molecules[i],
                adjacency=self.adjacencies[i],
                y=self.properties[i],
                mol_name=self.mol_name_list[i]))
        return self.dataset


    @staticmethod
    def split_dataset_by_mol_name(dataset,mol_name_for_testset):
        if isinstance(mol_name_for_testset,np.ndarray):
            mol_name_for_testset = list(mol_name_for_testset)
        train_set = []
        test_set = []
        for sample in dataset:
            if sample.mol_name in mol_name_for_testset:
                test_set.append(sample)
                mol_name_for_testset.remove(sample.mol_name)
            else:
                train_set.append(sample)
        if len(mol_name_for_testset) > 0:
            print("[Warning] Some molecules are not sampled %s" % ", ".join(mol_name_for_testset))
            time.sleep(5)

        return train_set,test_set



    @staticmethod
    def load_tensor(filename, device, dtype):
        return [dtype(d).to(device) for d in np.load(filename + '.npy',allow_pickle=True)]

    @staticmethod
    def load_numpy(filename):
        return np.load(filename + '.npy',allow_pickle=True)

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

    @staticmethod
    def split_valid(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        n1 = int(train_ratio * len(dataset))
        n2 = int(valid_ratio * len(dataset))
        dataset_1, dataset_2, dataset_3 = dataset[:n1], dataset[n1:n2], dataset[n2:]
        return dataset_1, dataset_2, dataset_3