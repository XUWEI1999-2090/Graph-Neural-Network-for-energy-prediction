import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 重要：embedding层：每个fingerprint有一个25长度的特征，可以看成
#  每个fingerprint就是一个词，这个词用25的向量代表fingerprint的特征
'''
       这里的算法是，将每个不同的fingerprint用一个dim大小的特征向量去代替
       相当于词的embedding，
       然后一个分子有n个fingerprint，这里采用的直接将fingerprint的特征加在一起
       然后这样用加载一起的特征作为分子的特征，然后回归到能量！
       和获取句子的embedding非常相似！
       TODO： 验证下面说法的正确性
       注意，所有fingerprint会cat到一起，然后embedding，
       那么cat到一起怎么区分呢？靠的是sum操作？
       
               # TODO：所以可以使用word2vec进行分子相似度的工作！
        # TODO:这里的concat操作是在哪一维度？
        # 最终把26种类的fingerprint embedding了，每个用25维度的向量代替

       '''


class GraphNeuralNetwork(nn.Module):
    def __init__(self,
                 device,  # 在哪个设备（cpu或cuda）上运行
                 update_method,  # update和output method选择sum或者mean，对二维的特征采用降维求和或者降维平均得到一维特征
                 output_method,
                 n_fingerprint,  # fingerprint的总数目，也就是fingerprint dict的长度
                 dim,  # fingerprint的embedding维度，同时也是隐藏层的维度
                 hidden_layer,  # 隐藏层的数目
                 output_layer,  # output部分隐藏层的数目
                 with_batch_norm=True,  # 是否使用batchNorm
                 ratio_of_variance_as_loss=0.0  # 研究发现对于物种能量预测应用到动力学，误差的方差较误差更为重要，因此这里以一定的比例最小化误差的方法，而不是仅仅优化MSE
                 # 如果这个值等于0，就是仅仅优化MSE，如果等于1，就是仅仅优化variance
                 ):
        super(GraphNeuralNetwork, self).__init__()
        self.ratio_of_variance_as_loss = ratio_of_variance_as_loss
        print("ratio_of_variance_as_loss is %s" % ratio_of_variance_as_loss)
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        '''
        embedding层把输入的fingerprint的index转成向量，
        比如nn.Embedding(3, 5)，是这样的矩阵：
        [
        [1,2,3,4,5],
        [5,4,3,2,1],
        [6,7,8,9,0]
        ]
        那么输入[0,1]，则embedding层输出[[1,2,3,4,5],[5,4,3,2,1]]
        输入[2]，则embedding层输出[[6,7,8,9,0]]
        如果输出[-1]或者输入[3]及以上超出范围的index，则报错

        embedding可以理解为“查表”，只是表就是一个矩阵，输入index，给出向量
        embedding中的这些向量都可以被优化
        '''
        self.drop_out = nn.Dropout(0.5)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(output_layer)])
        '''
        上面两个可以看成是全连接层的数组，之后会用到，同时激活函数也在后面设置
        '''
        self.W_property = nn.Linear(dim, 1)  # 回归问题，最终输出为标量
        self.device = device
        self.update_method = update_method
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.output_method = output_method
        self.std, self.mean = None, None
        self.dim = dim
        self.bn = with_batch_norm

    def pad(self, matrices, pad_value):
        """输入是多个邻接矩阵，输出一个大矩阵，
        比如输入是3x3和4x4的两个邻接矩阵
        输入是7x7的大矩阵，其中新增的部分用0填充，
        这样做的目的是为了批量计算，加快矩阵乘法的速度，
        用大矩阵计算了过后还要根据结果重新分成3和4的两个向量
        """
        sizes = [m.shape[0] for m in matrices]
        M = sum(sizes)
        pad_matrices = pad_value + np.zeros((M, M))
        i = 0
        for j, m in enumerate(matrices):
            j = sizes[j]
            pad_matrices[i:i + j, i:i + j] = m
            i += j
        return torch.FloatTensor(pad_matrices).to(self.device)

    def sum_axis(self, xs, axis):
        y = [torch.sum(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = [torch.mean(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def update(self, xs, A, M, i):
        """
        Update the node vectors in a graph
        considering their neighboring node vectors (i.e., sum or mean),
        which are non-linear transformed by neural network."""
        self.u_xs = xs  # embedding过后的fingerprint，比如shape为7x16
        self.u_A = A  # 大的邻接矩阵，比如shape为7x7
        self.u_M = M  # 用于求平均值的系数，比如[4,4,4,4,3,3,3]
        self.u_hs = torch.selu(self.W_fingerprint[i](self.u_xs))  # 对fingerprint进行非线性变换
        if self.update_method == 'sum':
            return self.u_xs + torch.matmul(self.u_A, self.u_hs)  # 然后和邻接矩阵点乘，取sum或者平均
        if self.update_method == 'mean':
            return self.u_xs + torch.matmul(self.u_A, self.u_hs) / (self.u_M - 1)

    # 模型中最主要的前向传播，如果不能理解，可以先运行之后的一个小测试，从实际数据上理解
    def forward(self, fingerprints, adjacencies, embedding=False):
        # 输入多个样本，这里得到每个样本fingerprint的数目，如[4,3]
        self.f_axis = [len(f) for f in fingerprints]
        # 将每个样本的数值len(f)重复len(f)次,如[4,4,4,4,3,3,3]，用于后续求平均值
        self.f_M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        self.f_M = torch.unsqueeze(torch.FloatTensor(self.f_M), 1)

        self.f_fingerprints = torch.cat(fingerprints)
        '''
        将fingerprint cat过后转成向量，比如输入[1,2,3,4]和[1,2,3]的两组fingerprint，cat成
        [1,2,3,4,1,2,3]，然后每个index转成dim大小的向量，得到比如7 x 16的矩阵
        '''
        self.f_fingerprint_vectors = self.embed_fingerprint(self.f_fingerprints)
        # 将多个邻接矩阵拼成大矩阵，便于计算，比如3x3和2x2的矩阵拼成5x5的，之后点乘过后得到5长度的向量还需要拆回去得到3+2的两个向量
        self.f_adjacencies = self.pad(adjacencies, 0)

        # 接下来几层会持续将fingerprint的embedding结果进行非线性变换，然后和邻接矩阵作点乘，最终得到dim维度的向量
        for i in range(self.hidden_layer):
            if self.bn:
                self.f_fingerprint_vectors = nn.BatchNorm1d(self.dim)(self.update(self.f_fingerprint_vectors,
                                                                                  self.f_adjacencies, self.f_M, i))
            else:
                self.f_fingerprint_vectors = self.update(self.f_fingerprint_vectors, self.f_adjacencies, self.f_M, i)
        '''
        特征向量会按照样本求和或者求平均
        比如之前是两个fingerprint长度为4和3的样本，cat到了7，然后输出7x dim 大小的矩阵
        在这里就会拆分成4 x dim 和 3 x dim 的矩阵，对应到每个样本
        然后在第一个维度进行降维求和或者求平均，最终得到两个dim长度的向量
        '''
        if self.output_method == 'sum':
            self.f_molecular_vectors = self.sum_axis(self.f_fingerprint_vectors, self.f_axis)
        if self.output_method == 'mean':
            self.f_molecular_vectors = self.mean_axis(self.f_fingerprint_vectors, self.f_axis)
        # 最终得到的dim长度的向量经过一系列非线性变换输出预测的能量值
        for j in range(self.output_layer):
            self.f_molecular_vectors = self.drop_out(self.f_molecular_vectors)
            if self.bn:
                self.f_molecular_vectors = nn.BatchNorm1d(self.dim)(
                    torch.selu(self.W_output[j](self.f_molecular_vectors)))
            else:
                self.f_molecular_vectors = torch.selu(self.W_output[j](self.f_molecular_vectors))
        if embedding:
            return self.f_molecular_vectors.data.numpy()
        #self.f_molecular_vectors = self.drop_out(self.f_molecular_vectors)
        molecular_properties = self.W_property(self.f_molecular_vectors)

        return molecular_properties

    def fit(self,
            dataset,
            epoch,
            lr,
            weight_decay,
            lr_decay,
            lr_decay_interval,
            batch_size,
            std_of_y,
            mean_of_y):
        loss_curve = []
        optimizer = optim.Adam(self.parameters(),
                               lr=lr, weight_decay=weight_decay)
        self.std = std_of_y
        self.mean = mean_of_y
        for _ in range(epoch):
            if epoch % lr_decay_interval == 0:
                optimizer.param_groups[0]['lr'] *= lr_decay
            np.random.shuffle(dataset)
            N = len(dataset)
            loss_total = 0
            for i in range(0, N-1, batch_size):
                data_batch = dataset[i:i + batch_size]
                loss = self(data_batch, std=self.std, mean=self.mean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_total += loss.to('cpu').data.numpy()
            print("Total loss on step %s is: %s" % (_, loss_total))
            loss_curve.append(loss_total)
        return loss_curve

    def predict(self, dataset):
        return self(dataset, std=self.std, mean=self.mean, train=False)

    def predict_embedding_feature(self, dataset):
        return self(dataset, std=self.std, mean=self.mean, train=False, output_embedding=True)

    def __call__(self,
                 data_batch,
                 mean,
                 std,
                 train=True,
                 # 如果为True，拿出回归前的最后一层embedding的结果向量，而不是拿出回归结果
                 output_embedding=False):

        fingerprints = [i.fingerprint for i in data_batch]
        adjacencies = [i.adjacency for i in data_batch]

        if data_batch[0].y is not None:
            correct_properties = torch.cat([i.y for i in data_batch])
        else:
            correct_properties = None

        if output_embedding:
            return self.forward(fingerprints, adjacencies, embedding=True)

        predicted_properties = self.forward(fingerprints, adjacencies)

        if train:
            mse_loss = F.mse_loss(correct_properties, predicted_properties)
            error = predicted_properties - correct_properties

            variance_loss = torch.var(error)
            return mse_loss * (1 - self.ratio_of_variance_as_loss) + variance_loss * self.ratio_of_variance_as_loss
        else:
            """Transform the normalized property (i.e., mean 0 and std 1)
            to the unit-based property (e.g., eV and kcal/mol)."""
            predicted_properties = predicted_properties.to('cpu').data.numpy()
            predicted_properties = std * np.concatenate(predicted_properties) + mean
            return predicted_properties

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

# class GNNTrainer(object):
#     def __init__(self, model, lr, weight_decay):
#         self.model = model
#         self.optimizer = optim.Adam(self.model.parameters(),
#                                     lr=lr, weight_decay=weight_decay)
#
#     def train(self, dataset, batch_size, std, mean):
#         np.random.shuffle(dataset)
#         N = len(dataset)
#         loss_total = 0
#         for i in range(0, N, batch_size):
#             data_batch = list(zip(*dataset[i:i + batch_size]))
#             loss = self.model(data_batch, std=std, mean=mean)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#             loss_total += loss.to('cpu').data.numpy()
#         return loss_total
