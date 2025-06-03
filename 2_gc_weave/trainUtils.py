# -*- coding=utf-8 -*-
import copy
import pickle

import deepchem as dc

import matplotlib.pyplot as plt

plt.switch_backend('agg')  # 避免多线程错误
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from deepchemUtils import featurize_dataset

import sys
sys.path.append("..")
from MGTE.Dataset.Load_and_Split_for_GC_WEAVE import Load_and_Split as RDataset

class TrainProcess():
    '''
     每次训练需要训练到收敛，所以需要观察训练集和验证集上的loss，确定训练次数
    '''

    def __init__(self, model_name, featurizer, model, train_ratio=0.8, random_seed=None,dataset_filename="./Dataset_corrected/molecule_dataset.pkl"):
        self.id = model_name
        '''
        # 读取数据
        with open(dataset_filename, "rb") as f:
            metal, mol_list, mol_string, energy = pickle.load(f)
        self.id = model_name
        self.train_ratio = train_ratio

        # 进行特征化，转成数据集
        self.dataset = featurize_dataset(mol_list, energy, mol_string, featurizer)
        
        self.splitter = dc.splits.RandomSplitter()
        # 分训练集和测试集
        self.train_dataset, _, self.test_dataset = self.splitter.train_valid_test_split(
            self.dataset, frac_train=train_ratio, frac_valid=0., frac_test=1 - train_ratio, seed=random_seed)
        '''
        # 加载原始数据（未特征化）,按照指定分子名称得到测试集
        with open("E:/ethnaol reforming/C2_3.3/mol_id_string_on_testset_9.27.pkl", "rb") as f:
            mol_string_for_test = pickle.load(f)
        self.splitter = dc.splits.RandomSplitter()
        dataset_dir = "E:/ethnaol reforming/C2_3.3/molecule_dataset.pkl"
        device = 'cpu'
        dataset_loader = RDataset(
            dataset_dir=dataset_dir,
            device=device)
        dataset = dataset_loader.load_dataset()
        total = RDataset.split_dataset_by_mol_name(dataset, mol_string_for_test)
        # 再进行特征化
        self.train_dataset = featurize_dataset(total[0],total[1],total[2], featurizer)
        self.test_dataset = featurize_dataset(total[3],total[4],total[5], featurizer)
        print(len(self.train_dataset),len(self.test_dataset))

        assert isinstance(self.train_dataset, dc.data.DiskDataset)
        assert isinstance(self.test_dataset, dc.data.DiskDataset)
        self.model = model
        self.seed = random_seed

    def train_and_tuning_and_test(self, n_fold=8, iter_=10, nb_epoch_of_one_iter=5, plot_error_curve=False,
                                  skip_tuning_stage=False, output_train_test_true_pred=False, **kwargs):
        '''

        :param n_fold:  n折交叉验证
        :param fold_num_to_train:  训练多少折，
        :param kwargs:
        :return:
        '''
        if not skip_tuning_stage:
            # 按k折交叉验证来分数据集
            kf_dataset = self.splitter.k_fold_split(self.train_dataset, k=n_fold, seed=self.seed)
            print("K Fold results")
            for i in kf_dataset:
                t, v = i
                print(len(t), len(v))

            # 训练集和验证集上每一折的每一步的RMSE，key是折的index，value是RMSE的dict
            total_RMSE_on_train_of_every_step_every_fold = dict()
            total_RMSE_on_valid_of_every_step_every_fold = dict()

            for now_fold_index in range(n_fold):
                print("Train fold %s" % now_fold_index)
                trainset, validset = kf_dataset[now_fold_index]

                train_RMSE_on_every_step = []
                valid_RMSE_on_every_step = []
                model = copy.deepcopy(self.model)  # 需要使用一个全新的模型来训练
                for t in range(iter_):
                    print("Iter %s" % t)
                    model.fit(dataset=trainset,
                              nb_epoch=nb_epoch_of_one_iter,
                              **kwargs)
                    pred_train,std_train = model.predict_uncertainty(trainset)
                    true_train = trainset.y.reshape(-1)
                    pred_valid,std_valid = model.predict_uncertainty(validset)
                    true_valid = validset.y.reshape(-1)
                    if t == iter_ - 1: # 最后一步展示一下
                       self.p_plot(pred_train, true_train, pred_valid, true_valid)
                    train_RMSE = np.sqrt(mean_squared_error(true_train, pred_train))
                    valid_RMSE = np.sqrt(mean_squared_error(true_valid, pred_valid))
                    train_RMSE_on_every_step.append(train_RMSE)
                    valid_RMSE_on_every_step.append(valid_RMSE)

                    print(train_RMSE, valid_RMSE)
                total_RMSE_on_train_of_every_step_every_fold[now_fold_index] = train_RMSE_on_every_step
                total_RMSE_on_valid_of_every_step_every_fold[now_fold_index] = valid_RMSE_on_every_step
            # 最后展示error的训练曲线，在每一折上取平均值，用于找合适的epoch

            total_train_RMSE_on_every_step_averaged_by_fold = np.array(total_RMSE_on_train_of_every_step_every_fold[0])
            total_valid_RMSE_on_every_step_averaged_by_fold = np.array(total_RMSE_on_valid_of_every_step_every_fold[0])
            for i in range(1, n_fold):
                total_train_RMSE_on_every_step_averaged_by_fold += np.array(
                    total_RMSE_on_train_of_every_step_every_fold[i])
                total_valid_RMSE_on_every_step_averaged_by_fold += np.array(
                    total_RMSE_on_valid_of_every_step_every_fold[i])

            total_train_RMSE_on_every_step_averaged_by_fold /= n_fold
            total_valid_RMSE_on_every_step_averaged_by_fold /= n_fold

            print("Total mean RMSE on every step")
            print(total_train_RMSE_on_every_step_averaged_by_fold)
            print(total_valid_RMSE_on_every_step_averaged_by_fold)

            plt.plot(total_train_RMSE_on_every_step_averaged_by_fold)
            plt.plot(total_valid_RMSE_on_every_step_averaged_by_fold)
            if plot_error_curve:
                plt.show()
                plt.close()
            else:
                plt.savefig("E:/BNN/gc_error_curve.png", dpi=600)
                plt.close()

            print("Final RMSE of train and valid is:")
            print(total_train_RMSE_on_every_step_averaged_by_fold[-1])
            print(total_valid_RMSE_on_every_step_averaged_by_fold[-1])
        else:
            total_train_RMSE_on_every_step_averaged_by_fold = []
            total_valid_RMSE_on_every_step_averaged_by_fold = []
        # 把用于测试的分子的string记录下来，便于之后的模型进行测试
        mol_string = list([d[3] for d in self.test_dataset.itershards()][0])
        print(mol_string)
        with open("E:/ethnaol reforming/C2_3.3/mol_id_string_on_testset.pkl", "wb") as f:
            pickle.dump(mol_string, f)
        # 调参完成过后进行测试
        # 这里的思路是：K折用来调参，参数确定后训练模型
        # 参数确定后，不用K折了，直接用下面代码
        print("_____Test_____")
        model = copy.deepcopy(self.model)  # 需要使用一个全新的模型来训练，按照原有的参数！

        trainset,validset = self.splitter.k_fold_split(self.train_dataset, k=n_fold, seed=self.seed)[0]

        train_RMSE_on_every_iter = []
        valid_RMSE_on_every_iter = []
        train_std_on_every_iter = []
        valid_std_on_every_iter = []
        for t in range(iter_):
            print("Iter %s" % t)
            model.fit(dataset=trainset,
                      nb_epoch=nb_epoch_of_one_iter,
                      **kwargs)
            p_train = model.predict(trainset).reshape(-1)
            t_train = trainset.y.reshape(-1)
            p_valid = model.predict(validset).reshape(-1)
            t_valid = validset.y.reshape(-1)
            train_RMSE = np.sqrt(mean_squared_error(t_train, p_train))
            valid_RMSE = np.sqrt(mean_squared_error(t_valid, p_valid))
            train_RMSE_on_every_iter.append(train_RMSE)
            valid_RMSE_on_every_iter.append(valid_RMSE)
        plt.plot(train_RMSE_on_every_iter)
        plt.plot(valid_RMSE_on_every_iter)
        plt.show()
        plt.savefig('./gc_RMSE_curve.png')
        plt.close()
        pred_train = model.predict(self.train_dataset).reshape(-1)
        true_train = self.train_dataset.y.reshape(-1)
        pred_test = model.predict(self.test_dataset).reshape(-1)
        true_test = self.test_dataset.y.reshape(-1)
        self.p_plot(pred_train, true_train, pred_test, true_test)

        final_test_stage_train_RMSE = np.sqrt(mean_squared_error(true_train, pred_train))
        final_test_stage_test_RMSE = np.sqrt(mean_squared_error(true_test, pred_test))

        final_test_stage_train_MAE = mean_absolute_error(true_train, pred_train)
        final_test_stage_test_MAE = mean_absolute_error(true_test, pred_test)


        # 保存预测结果
        with open("train_true_pred_and_test_true_pred_of_%s.pkl" % self.id, "wb") as f:
            pickle.dump(
                dict(
                    true_train=true_train,
                    pred_train=pred_train,
                    true_test=true_test,
                    pred_test=pred_test), f)

        print("RMSE and MAE on final test stage: ")
        print(final_test_stage_train_RMSE, final_test_stage_test_RMSE)
        print(final_test_stage_train_MAE, final_test_stage_test_MAE)

        # MAE
        if output_train_test_true_pred == False:
            return total_train_RMSE_on_every_step_averaged_by_fold, \
                   total_valid_RMSE_on_every_step_averaged_by_fold, \
                   final_test_stage_train_RMSE, \
                   final_test_stage_test_RMSE, \
                   final_test_stage_train_MAE, \
                   final_test_stage_test_MAE
        else:
            return total_train_RMSE_on_every_step_averaged_by_fold, \
                   total_valid_RMSE_on_every_step_averaged_by_fold, \
                   final_test_stage_train_RMSE, \
                   final_test_stage_test_RMSE, \
                   final_test_stage_train_MAE, \
                   final_test_stage_test_MAE, \
                   true_train, \
                   pred_train, \
                   true_test, \
                   pred_test

    def train_and_test_repeated(self, iter_, nb_epoch_of_one_iter, repeat_count=50, **kwargs):
        '''
                进行重复数据集采样测试

        :param iter_:  这个参数需要是之前调参得到的最佳参数
        :param nb_epoch_of_one_iter:  这个参数需要是之前调参的得到的最佳参数
        :param repeat_count:
        :return:
        '''
        total_train_RMSE = []
        total_test_RMSE = []

        total_train_MAE = []
        total_test_MAE = []

        for _ in range(repeat_count):
            splitter = dc.splits.RandomSplitter()
            # 分训练集和测试集
            train_dataset, _, test_dataset = splitter.train_valid_test_split(
                self.dataset, frac_train=self.train_ratio, frac_valid=0., frac_test=1 - self.train_ratio)
            model = copy.deepcopy(self.model)  # 需要使用一个全新的模型来训练

            for t in range(iter_):
                print("Iter %s" % t)
                model.fit(dataset=train_dataset,
                          nb_epoch=nb_epoch_of_one_iter,
                          **kwargs)
            pred_train = model.predict(train_dataset)
            true_train = train_dataset.y
            pred_test = model.predict(test_dataset)
            true_test = test_dataset.y
            final_test_stage_train_RMSE = np.sqrt(mean_squared_error(true_train, pred_train))
            final_test_stage_test_RMSE = np.sqrt(mean_squared_error(true_test, pred_test))
            total_train_RMSE.append(final_test_stage_train_RMSE)
            total_test_RMSE.append(final_test_stage_test_RMSE)

            total_train_MAE.append(mean_absolute_error(true_train, pred_train))
            total_test_MAE.append(mean_absolute_error(true_test, pred_test))
        return np.mean(total_train_RMSE), np.mean(total_test_RMSE), np.mean(total_train_MAE), np.mean(total_test_MAE)

    def p_plot(self, pred_train, true_train, pred_test=None, true_test=None):
        #print("RMSE on train is %s" % np.sqrt(mean_squared_error(true_train, pred_train)))
        #print("RMSE on test is %s" % np.sqrt(mean_squared_error(true_test, pred_test)))

        plt.scatter(true_train, pred_train)
        plt.scatter(true_test, pred_test)
        plt.savefig("./gc_p_plot_show.png")
        plt.show()
        plt.close()

    def test_and_show_figs(self, iter_, nb_epoch_of_one_iter, **kwargs):
        '''
        splitter = dc.splits.RandomSplitter()
        # 分训练集和测试集
        train_dataset, _, test_dataset = splitter.train_valid_test_split(
            self.dataset, frac_train=self.train_ratio, frac_valid=0., frac_test=1 - self.train_ratio)
        '''
        model = copy.deepcopy(self.model)  # 需要使用一个全新的模型来训练

        for t in range(iter_):
            print("Iter %s" % t)
            model.fit(dataset=self.train_dataset,
                      nb_epoch=nb_epoch_of_one_iter,
                      **kwargs)
        pred_train = model.predict(self.train_dataset)
        true_train = self.train_dataset.y
        pred_test = model.predict(self.test_dataset)
        true_test = self.test_dataset.y

        final_train_RMSE = np.sqrt(mean_squared_error(true_train, pred_train))
        final_test_RMSE = np.sqrt(mean_squared_error(true_test, pred_test))

        final_train_MAE = mean_absolute_error(true_train, pred_train)
        final_test_MAE = mean_absolute_error(true_test, pred_test)

        print("RMSE on train and test, MAE on train and test is")
        print(final_train_RMSE, final_test_RMSE)
        print(final_train_MAE, final_test_MAE)

        self.p_plot(pred_train, true_train, pred_test, true_test)
