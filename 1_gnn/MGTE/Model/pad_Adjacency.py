#coding=utf-8
import torch
import numpy as np
def pad(matrices):
    '''输入是多个邻接矩阵，输出一个大矩阵，
    比如输入是3x3和4x4的两个邻接矩阵
    输入是7x7的大矩阵，其中新增的部分用0填充，
    这样做的目的是为了批量计算，加快矩阵乘法的速度，
    用大矩阵计算了过后还要根据结果重新分成3和4的两个向量
    '''
    sizes = [m.shape[0] for m in matrices]
    M = sum(sizes)
    pad_matrices = np.zeros((M, M))
    i = 0
    for j, m in enumerate(matrices):
        j = sizes[j]
        pad_matrices[i:i + j, i:i + j] = m
        i += j
    return torch.FloatTensor(pad_matrices)


a = np.random.randint(0, 10, (4,4))
b = np.random.randint(0, 10, (3,3))
matrices = [a,b]
'''
sizes = [m.shape[0] for m in matrices]
M = sum(sizes)
pad_matrices = np.zeros((M,M))
n = 0
for i, j in enumerate(matrices):
    i = sizes[i]
    pad_matrices[n:n+i, n:n+i] = j
    n += i
print(pad_matrices)
'''
print(pad(matrices))
