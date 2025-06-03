import pickle

import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol,WeaveMol
from deepchem.molnet.load_function import qm7_datasets
'''

TODO:测试：
1 查看特征化方法
    1.1 查看qm7的基于SMILES的其他方法：ECFP失败，其他可行     √
    1.2 查看toc21的特征化方法，多了一个邻接矩阵特征           √
    1.X 在deepchem.molnet.load_function中查看了数据集的load情况，大部分方法这里都包括了，目前的问题是ECFP不能使用 √
    
 
注意：下面这些代码特征化过后，还需要到deepchem相应的模型部分找提取特征，转成向量的部分
建议还是先阅读论文，了解具体的操作过后再理解特征化的过程和结果

'''

with open("molecule_dataset.pkl", "rb") as f:
    mol_list, mol_string, energy = pickle.load(f)

print(mol_list, mol_string, energy)
for i in range(len(mol_string)):
    print(mol_string[i],energy[i])
exit()
def conv_mol():
    featurizer = dc.feat.ConvMolFeaturizer()
    conv_mols = featurizer.featurize(mol_list)
    print(conv_mols)
    exit()
    mol_index_to_choose_for_example_show = 100
    example = conv_mols[mol_index_to_choose_for_example_show]
    assert isinstance(example,ConvMol)
    print(mol_string[mol_index_to_choose_for_example_show])
    print(example.get_atom_features())
    print(example.get_adjacency_list())

def weave():


    featurizer = dc.feat.WeaveFeaturizer()
    r = featurizer.featurize(mol_list)
    mol_index_to_choose_for_example_show = 100
    example = r[mol_index_to_choose_for_example_show]
    assert isinstance(example, WeaveMol)
    print(mol_string[mol_index_to_choose_for_example_show])
    print(example.get_atom_features())
    print(example.get_pair_features())

def adjacency():
    featurizer = dc.feat.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)
    r = featurizer.featurize(mol_list)
    mol_index_to_choose_for_example_show = 100
    example = r[mol_index_to_choose_for_example_show]
    print(mol_string[mol_index_to_choose_for_example_show])
    print(example)



# 下面这个运行会有问题
# dc.feat.CircularFingerprint(size=1024),


if __name__ == '__main__':
    # 测试以下4种方法
    conv_mol()
    #weave()
    #adjacency()

