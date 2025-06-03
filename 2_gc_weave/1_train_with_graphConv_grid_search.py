import deepchem as dc
import numpy as np
import tensorflow as tf
import pickle
from trainUtils import TrainProcess
import time
'''
图片绘制说明：

energy distribution是在GNN molecules里面

训练曲线，和error parity plot是在这里
error distribution是在z_plot_error_distribution中

'''
ALL_RANDOM_SEED = 123
import copy

np.random.seed(ALL_RANDOM_SEED)
tf.random.set_random_seed(ALL_RANDOM_SEED)

parameter = dict(
    graph_conv_layers=[128,64],
    batch_size=32,
    learning_rate=0.001,
    drop_out=0.25
)

# 特征化
featurizer = dc.feat.ConvMolFeaturizer()
all_configs = []
all_loss_curve = []
all_train_RMSE = []
all_valid_RMSE = []
all_test_RMSE = []
now_index = 0
for graph_conv_layers in [[128,64], [32, 64, 32],[256]]:
    for batch_size in [16, 32]:
        for learning_rate in [1e-2, 1e-3]:
            for drop_out in [0.2,0.3,0.4]:
                start_time = time.time()
                now_index += 1

                a = copy.deepcopy(parameter)
                a["batch_size"] = batch_size
                a["graph_conv_layers"] = graph_conv_layers
                a["learning_rate"] = learning_rate
                a["drop_out"] = drop_out
                # 模型
                model = dc.models.GraphConvModel(
                    n_tasks=1,
                    graph_conv_layers=a["graph_conv_layers"],
                    batch_size=a["batch_size"],
                    learning_rate=a["learning_rate"],
                    dropout=a["drop_out"],
                    number_atom_features=featurizer.feature_length(),
                    mode="regression",
                    uncertainty=True)

                #print(featurizer.feature_length())
                trainer = TrainProcess(featurizer=featurizer,
                                       model=model, random_seed=ALL_RANDOM_SEED,
                                       model_name="graphConvol",
                                       dataset_filename="molecule_dataset.pkl")

                # TODO: 在GPU服务器上跑！

                # ------ tuning stage ------
                # 调参，这个函数会返回K折交叉验证的训练集和验证集上的误差，以及最后test阶段的误差
                total_train_RMSE_on_every_step_averaged_by_fold, \
                total_valid_RMSE_on_every_step_averaged_by_fold, \
                final_test_stage_train_RMSE, \
                final_test_stage_test_RMSE, \
                final_test_stage_train_MAE, \
                final_test_stage_test_MAE = trainer.train_and_tuning_and_test(
                    n_fold=5, iter_=30, nb_epoch_of_one_iter=30, plot_error_curve=False,
                    skip_tuning_stage=False)
                all_configs.append(a)
                all_loss_curve.append(total_train_RMSE_on_every_step_averaged_by_fold)
                all_train_RMSE.append(total_train_RMSE_on_every_step_averaged_by_fold[-1])
                all_valid_RMSE.append(total_valid_RMSE_on_every_step_averaged_by_fold[-1])
                all_test_RMSE.append(final_test_stage_test_RMSE)
                print("--------Now index %s--------use time:%s" % (now_index,time.time()-start_time))
print(all_configs, all_loss_curve, all_train_RMSE, all_valid_RMSE,all_test_RMSE)

with open("graphConv_grid_search_new_dataset.pkl", "wb") as f:
    pickle.dump([all_configs, all_loss_curve, all_train_RMSE, all_valid_RMSE,all_test_RMSE], f)
