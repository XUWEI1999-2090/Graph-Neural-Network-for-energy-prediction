import torch
import torch.nn as nn
import pickle
from MGTE.Dataset.DatasetLoader import \
    RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetLoaderToTorchTensorDataset as RDataset
#了解nn.embedding
#把得到fingerprints的coding加上
fingerprint = []
with open(r"C:\Users\ccmd\Desktop\实践\temp1_radius1\\fingerprint_dict.pkl", "rb") as f:
    fingerprint_dict = pickle.load(f)
    for i in fingerprint_dict.items():
        fingerprint.append(i)
        break
print(fingerprint)
print(type(fingerprint))

dataset_loader = RDataset(
    dataset_dir=dir_input,
    device=device)
# INFO： 2.载入数据集，打乱
dataset = dataset_loader.load_dataset()
dataset = RDataset.shuffle_dataset(dataset, seed=9)











#把list cat 为向量
fingerprint = torch.Tensor(fingerprint)
fingerprint = torch.cat(fingerprint)
print(type(fingerprint))
'''
embed_fingerprint = nn.Embedding(len(fingerprint), 16)
# print(embed_fingerprint)
fingerprint_vec = embed_fingerprint(fingerprint)
print(fingerprint_vec)
'''



