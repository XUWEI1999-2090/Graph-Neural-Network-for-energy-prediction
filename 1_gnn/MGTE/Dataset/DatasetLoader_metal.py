import numpy as np
import pickle
import torch
import time

class GNNFingerprintAdjacencyPropertySample(object):

    def __init__(self,
                 mol_name,
                 fingerprint,
                 adjacency,
                 metal_label,
                 y=None,
                 # 拿出GNN模型预测前一层的feature
                 embedding_feature=None
                 ):
        self.fingerprint = fingerprint
        self.adjacency = adjacency
        self.y = y
        self.mol_name = mol_name
        self.embedding_feature = embedding_feature
        self.metal_label = metal_label


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
        with open(dir_input + "./metal_label.pkl", "rb") as f:
            self.metal_label = pickle.load(f)

        self.dataset = []
        for i in range(len(self.molecules)):
            self.dataset.append(GNNFingerprintAdjacencyPropertySample(
                fingerprint=self.molecules[i],
                adjacency=self.adjacencies[i],
                y=self.properties[i],
                mol_name=self.mol_name_list[i],
                metal_label = self.metal_label[i])
                )
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
    def split_dataset_by_metal(dataset,metal):
        trainset=[]
        testset=[]
        for sample in dataset:
            if sample.metal_label == metal:
                testset.append(sample)
            else:
                trainset.append(sample)
        return trainset,testset

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
    def random_dic(dicts, seed):
        np.random.seed(seed)
        dict_key_ls = list(dicts.keys())
        np.random.shuffle(dict_key_ls)
        new_dic = {}
        for key in dict_key_ls:
            new_dic[key] = dicts.get(key)
        return new_dic

    @staticmethod
    def split_dataset(dataset, ratio):
        n = int(ratio * len(dataset))
        dataset_1, dataset_2 = dataset[:n], dataset[n:]
        return dataset_1, dataset_2

    @staticmethod
    def split_dataset_dict(dataset, ratio):
        n = int(ratio * len(dataset))
        dict_name = [k for k in dataset]
        dataset_1, dataset_2 = dict_name[:n], dict_name[n:]
        return dataset_1, dataset_2
