import numpy as np
import pickle
import torch
import time

class Sample(object):

    def __init__(self,
                 mol_list,
                 mol_string,
                 y=None,
                 ):
        self.mol_list=mol_list
        self.mol_string = mol_string
        self.y = y


class Load_and_Split(object):
    '''
    读取DatasetMarker.RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker的数据
    这里不依赖于rdkit
    '''

    def __init__(self, device, dataset_dir="molecule_dataset.pkl"):
        self.dataset_dir = dataset_dir
        self.device = device

    def load_dataset(self):
        dir_input = self.dataset_dir
        with open(dir_input, 'rb') as f:
            metal, mol_list, mol_string, energy = pickle.load(f)

        self.dataset = []
        for i in range(len(mol_list)):
            self.dataset.append(Sample(
                mol_list=mol_list[i],
                mol_string=mol_string[i],
                y=energy[i],
                ))
        return self.dataset


    @staticmethod
    def split_dataset_by_mol_name(dataset,mol_name_for_testset):
        if isinstance(mol_name_for_testset,np.ndarray):
            mol_name_for_testset = list(mol_name_for_testset)
        train_mol,train_string,train_y=[],[],[]
        test_mol,test_string,test_y=[],[],[]
        for sample in dataset:
            if sample.mol_string in mol_name_for_testset:
                test_mol.append(sample.mol_list)
                test_y.append(sample.y)
                test_string.append(sample.mol_string)
                mol_name_for_testset.remove(sample.mol_string)
            else:
                train_mol.append(sample.mol_list)
                train_y.append(sample.y)
                train_string.append(sample.mol_string)
        if len(mol_name_for_testset) > 0:
            print("[Warning] Some molecules are not sampled %s" % ", ".join(mol_name_for_testset))
            time.sleep(5)

        return train_mol,train_y,train_string,test_mol,test_y,test_string

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
