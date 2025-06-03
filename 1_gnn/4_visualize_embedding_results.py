'''

可视化fingerprint以及molecule embedding的结果

'''
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import  PCA
import numpy as np

# TODO：使用能量作为PCA结果的颜色层

def visualize_molecule_embedding():
    '''
    对分子的特征向量进行降维展示
    :return:
    '''

    with open("E:/ethnaol reforming/C2_2.1/molName2vec_dict_3.2.pkl", "rb") as f:
        total_mol2vec_dict = pickle.load(f)
    mol2vec_dict = dict()
    for i in total_mol2vec_dict.keys():
        #if 'K' in i:
            mol2vec_dict[i] = total_mol2vec_dict[i]


    molecules = sorted(list(mol2vec_dict.keys()))
    matrix = []
    for m in molecules:
        matrix.append(mol2vec_dict[m])
    matrix = np.array(matrix)
    print(matrix.shape)

    pca = PCA(n_components=2)
    result = pca.fit_transform(matrix)
    print(result.shape)
    # 为了便于展示，加上一个颜色值，颜色采用包含的CHO的数目

    y = []
    for m in molecules:
        _y = 0
        for c in m:
            if c in ["C","H","O"]:
                _y += 1
        y.append(_y)

    energy_dict = dict()
    with open("E:/ethnaol reforming/C2_2.1/C2_4.0.csv", "r") as f:
        for i in f.readlines():
             data = i.split(",")
             energy_dict[data[1]] = float(data[2])
    energy = []
    for m in molecules:
        energy.append(energy_dict[m])
    print(len(energy))

    cls = []
    cls_dict = {'K':0,'Y':1,'V':2,'W':3}
    for m in molecules:
        for i in ['K','Y','V','W']:
            if i in m:
                cls.append(cls_dict[i])
                break
    print(len(cls))
    #plt.scatter(result[:,0],result[:,1],c=energy,cmap="rainbow")
    #plt.scatter(result[:, 0], result[:, 1], c=y, cmap="rainbow")
    #plt.scatter(result[:, 0], result[:, 1], c=cls, cmap="rainbow")
    plt.colorbar()
    #for i in range(result.shape[0]):
     #   plt.text(result[i,0],result[i,1],s=molecules[i],fontdict={"fontsize":6})
    #plt.savefig("molecule_embedding.png",dpi=600)

    plt.show()

def visualize_fingerprint_embedding():
    '''
    展示fingerprint的embedding
    :return:
    '''
    save_dir = "E:/ethnaol reforming/C2_2.1/temp1_radius1/"
    with open(save_dir + "/fingerprint_dict.pkl", "rb") as f:
        fingerprint_dict = pickle.load(f)

    print(fingerprint_dict)
    fingerprint_embed = np.load("./fingerprint_embed_matrix.npy")
    #fingerprint_embed = np.load(save_dir + "./molecules.npy",allow_pickle=True)
    print(fingerprint_embed.shape)
    # 之前的fingerprint dict是从边到index，TODO 需要转到index到fingerprint
    fingerprint_dict_reverse = dict()
    # FIXME： 目前的fingerprint是一堆难以辨识的数字，需要添加readable的string
    pca = PCA(n_components=2)
    result = pca.fit_transform(fingerprint_embed)
    print(result.shape)

    plt.scatter(result[:, 0], result[:, 1])
    plt.show()


if __name__ == '__main__':
    #visualize_fingerprint_embedding()
    visualize_molecule_embedding()