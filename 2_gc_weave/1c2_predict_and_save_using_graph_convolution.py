import pickle

import deepchem as dc
import numpy as np
import tensorflow as tf

from trainUtils import TrainProcess
import time

'''
图片绘制说明：

energy distribution是在GNN molecules里面

训练曲线，和error parity plot是在这里
error distribution是在z_plot_error_distribution中

'''
ALL_RANDOM_SEED = 123 # 0.25


import copy

np.random.seed(ALL_RANDOM_SEED)
tf.random.set_random_seed(ALL_RANDOM_SEED)

parameter = dict(
    graph_conv_layers=[128,64],
    batch_size=32,
    learning_rate=0.001,
    drop_out=0.4
)

# 特征化
featurizer = dc.feat.ConvMolFeaturizer()
all_configs = []
all_loss_curve = []
all_train_RMSE = []
all_valid_RMSE = []
all_test_RMSE = []
now_index = 0

a = copy.deepcopy(parameter)

# 模型
model = dc.models.GraphConvModel(
    n_tasks=1,
    graph_conv_layers=a["graph_conv_layers"],
    batch_size=a["batch_size"],
    learning_rate=a["learning_rate"],
    dropout=a["drop_out"],
    number_atom_features=featurizer.feature_length(), mode="regression")

# print(featurizer.feature_length())
trainer = TrainProcess(featurizer=featurizer,
                       model=model, random_seed=ALL_RANDOM_SEED,
                       model_name="graphConvol",dataset_filename="molecule_dataset.pkl")
# 调参，这个函数会返回K折交叉验证的训练集和验证集上的误差，以及最后test阶段的误差
total_train_RMSE_on_every_step_averaged_by_fold, \
total_valid_RMSE_on_every_step_averaged_by_fold, \
final_test_stage_train_RMSE, \
final_test_stage_test_RMSE, \
final_test_stage_train_MAE, \
final_test_stage_test_MAE, \
true_train, \
pred_train, \
true_test, \
pred_test = trainer.train_and_tuning_and_test(
    output_train_test_true_pred=True,
    skip_tuning_stage=True,
    n_fold=3,
    iter_=30,
    nb_epoch_of_one_iter=30,
    plot_error_curve=False,
)
with open("E:/ethnaol reforming/C2_3.3/final_graph_convolution_data_new_randomSeed_%s.pkl"%ALL_RANDOM_SEED, "wb") as f:
    pickle.dump([total_train_RMSE_on_every_step_averaged_by_fold, \
                 total_valid_RMSE_on_every_step_averaged_by_fold, \
                 final_test_stage_train_RMSE, \
                 final_test_stage_test_RMSE, \
                 final_test_stage_train_MAE, \
                 final_test_stage_test_MAE, \
                 true_train, \
                 pred_train, \
                 true_test, \
                 pred_test], f)
