#coding=UTF-8
import pickle
# INFO 1. 载入文本得到分子的string以及相应的能量，
mol_string = []
energy = []
metal_label = []
with open("E:/ethnaol reforming/C2_3.2/C2_3.2.csv", "r", encoding="utf-8") as f:
    for i in f.readlines():
        try:
            data = i.split(",")
            energy.append(float(data[2]))
            mol_string.append(data[1])
            metal_label.append(data[0])
        except:
            pass
    assert len(energy) == len(mol_string)

# 这里绘制一下能量的分布，计算均值和偏度
import matplotlib.pyplot as plt
import numpy as np
import math
# print(energy)
plt.hist(energy, bins=22)
mu = np.mean(energy)
sigma = np.var(energy)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
y_sig = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.show()
# with open("./dataset/energyData.pkl","wb") as f:
    # pickle.dump(energy,f)

from MEACRNG.MolEncoder.MoleculeEncoder import OrganicSurfaceSpeciesEncoder

from MGTE.Dataset.DatasetMaker import RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker



def calc(data):
    n = len(data)
    niu = 0.0
    niu2 = 0.0
    niu3 = 0.0
    for a in data:
        niu += a
        niu2 += a**2
        niu3 += a**3
    niu/= n   #这是求E(X)
    niu2 /= n #这是E(X^2)
    niu3 /= n #这是E(X^3)
    sigma = math.sqrt(niu2 - niu*niu) #这是D（X）的开方，标准差
    return [niu,sigma,niu3] #返回[E（X）,标准差，E（X^3）]

def calc_stat(data):
    [niu,sigma,niu3] = calc(data)
    n = len(data)
    niu4 = 0.0
    for a in data:
        a -= niu
        niu4 += a ** 4
    niu4 /= n
    skew = (niu3 - 3*niu*sigma**2 - niu**3)/(sigma**3)
    kurt =  niu4/(sigma**2)
    return [niu,sigma,skew,kurt] #返回了均值，标准差，偏度，峰度

#print(calc_stat(energy))


# TODO :根据embedding的feature （回归的前几层）进行相似度检测！！！

# INFO：2. 将string编码为mol
encoder = OrganicSurfaceSpeciesEncoder()
# 数据只需要rdkit的mol的list，以及对应的能量list
mol_list = [encoder.encode(i) for i in mol_string]
with open("E:/ethnaol reforming/C2_3.3/molecule_dataset.pkl", "wb") as f:
    pickle.dump([metal_label, mol_list, mol_string, energy], f)

# INFO 3. 将数据集进行转化，转成fingerprint和邻接矩阵
# TODO: dataset maker也需要fit和transform，fit已有分子，用于transform没有出现过的分子！！！(只要fingerprint等在fit时的字典中即可！)

dataset_marker = RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetMaker(
    mol_list[:1],
    energy[:1],
    metal_label[:1],
    mol_name_list=mol_string[:1],
    fingerprint_radius=1  # 发现radius等于1比等于2好？可能是因为训练调参不好，过拟合严重
)
print(mol_string[:1])
exit()
dataset_marker.save_data_as_npy_and_pkl_files("E:/ethnaol reforming/C2_3.3/temp1_radius1/")