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
    n_atom_feat=75,
    n_pair_feat=14,
    n_hidden=64,
    n_graph_feat=128,
    batch_size=16,
    learning_rate=0.0001,
    drop_out=0.2
)
# 特征化
featurizer = dc.feat.WeaveFeaturizer()
all_configs = []
all_loss_curve = []
all_train_RMSE = []
all_valid_RMSE = []
all_test_RMSE = []

now_index = 0

# 模型

for n_hidden in [64]:#[32, 64]
    for n_graph_feat in [128]:#[64, 128]
        for batch_size in [16, 32]:
            for learning_rate in [1e-3]:
                for drop_out in [0.2, 0.3, 0.4]:
                    start_time = time.time()

                    now_index += 1
                    a = copy.deepcopy(parameter)

                    a["n_hidden"] = n_hidden
                    a["n_graph_feat"] = n_graph_feat
                    a["batch_size"] = batch_size
                    a["learning_rate"] = learning_rate
                    a["drop_out"] = drop_out
                    model = dc.models.WeaveModel(
                        n_tasks=1,
                        n_atom_feat=a["n_atom_feat"],
                        n_pair_feat=a["n_pair_feat"],
                        n_hidden=a["n_hidden"],
                        n_graph_feat=a["n_graph_feat"],
                        batch_size=a["batch_size"],
                        learning_rate=a["learning_rate"],
                        drop_out=a["drop_out"],
                        mode="regression")
                    trainer = TrainProcess(featurizer=featurizer,
                                           model=model, random_seed=ALL_RANDOM_SEED,
                                           model_name="weave")
                    # 调参，这个函数会返回K折交叉验证的训练集和验证集上的误差，以及最后test阶段的误差
                    total_train_RMSE_on_every_step_averaged_by_fold, \
                    total_valid_RMSE_on_every_step_averaged_by_fold, \
                    final_test_stage_train_RMSE, \
                    final_test_stage_test_RMSE, \
                    final_test_stage_train_MAE, \
                    final_test_stage_test_MAE = trainer.train_and_tuning_and_test(
                        skip_tuning_stage=False,
                        n_fold=5,
                        iter_=30,
                        nb_epoch_of_one_iter=30,
                        plot_error_curve=False, )
                    all_configs.append(a)
                    all_loss_curve.append(total_train_RMSE_on_every_step_averaged_by_fold)
                    all_train_RMSE.append(total_train_RMSE_on_every_step_averaged_by_fold[-1])
                    all_valid_RMSE.append(total_valid_RMSE_on_every_step_averaged_by_fold[-1])
                    all_test_RMSE.append(final_test_stage_test_RMSE)
                    print("--------Now index %s--------use time:%s" % (now_index, time.time() - start_time))
                    # 因为训练很慢，所以每次都保存pkl
                    with open("E:/ethnaol reforming/C2_3.3/weave_grid_search_new_dataset.pkl", "wb") as f:
                        pickle.dump([all_configs, all_loss_curve, all_train_RMSE, all_valid_RMSE, all_test_RMSE], f)