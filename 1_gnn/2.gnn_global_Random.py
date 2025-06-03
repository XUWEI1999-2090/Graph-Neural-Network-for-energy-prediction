import copy
import pickle

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from MGTE.Dataset.DatasetLoader_metal import \
    RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetLoaderToTorchTensorDataset as RDataset
from MGTE.Model.GNNModel import GraphNeuralNetwork
from MGTE.demos.c2_molecules.utils import plotout_results

def hyper_parameter_search(parameter_config):
    # 数据集目录，固定
    dir_input = "E:/ethnaol reforming/C2_3.3/temp1_radius1/"
    # 参数表
    update_method = parameter_config["update_method"]
    output_method = parameter_config["output_method"]
    dim = parameter_config["dim"]
    hidden_layer = parameter_config["hidden_layer"]
    output_layer = parameter_config["output_layer"]
    batch_size = parameter_config["batch_size"]
    weight_decay = parameter_config["weight_decay"]
    lr = parameter_config["lr"]
    lr_decay = parameter_config["lr_decay"]
    decay_interval = parameter_config["decay_interval"]
    iteration = parameter_config["iteration"]

    '''
    # 按照指定分子名称得到测试集（用于对照）
    with open("./MGTE_deepchem/mol_id_string_on_testset", "rb") as f:
        mol_string_for_test = pickle.load(f)
    
    trainset, testset = RDataset.split_dataset_by_mol_name(dataset, mol_string_for_test)
    '''
    # INFO： 2.载入数据集，打乱
    torch.manual_seed(4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Use cpu")

    dataset_loader = RDataset(
        dataset_dir=dir_input,
        device=device)

    dataset = dataset_loader.load_dataset()
    dataset = RDataset.shuffle_dataset(dataset, seed=9)
    # 得到训练集和测试集

    mol_string_for_test = ['CH3CH2O-H(-K)','CH3CH(-H)OH','H-CH2CH2OH']
    test2be, testseting = RDataset.split_dataset_by_mol_name(dataset, mol_string_for_test)
    #print(len(testseting),len(mol_string_for_test))
    trainset,testset = RDataset.split_dataset(test2be, ratio=0.7)
    testset = testset + testseting
    print(len(trainset), len(testset))
    '''
    #trainset,testset = RDataset.split_dataset_by_mol_name(dataset, mol_string_for_test)
    trainset, testset = RDataset.split_dataset(dataset, ratio=0.8)
    print(len(trainset),len(testset))
    '''
    trainset_orig = copy.deepcopy(trainset)
    data_num_per_fold = None
    # 储存测试集
    testset_mols = [i.mol_name for i in testset]
    with open('E:/ethnaol reforming/C2_3.3/mol_id_string_on_testset_%s.pkl'%TRAIN_LABEL,'wb') as f:
        pickle.dump(testset_mols,f)


    model = GraphNeuralNetwork(
        device=device,
        update_method=update_method,
        output_method=output_method,
        n_fingerprint=dataset_loader.n_fingerprint,
        dim=dim,
        hidden_layer=hidden_layer,
        output_layer=output_layer,
        with_batch_norm=False).to(device)

    # INFO  拟合全部训练集
    loss_curve = model.fit(
        dataset=trainset_orig,
        epoch=iteration,
        lr=lr,
        lr_decay=lr_decay,
        lr_decay_interval=decay_interval,
        weight_decay=weight_decay,
        batch_size=batch_size,
        std_of_y=dataset_loader.std,
        mean_of_y=dataset_loader.mean, )

    # INFO 7 预测训练集和测试集
    pred_train = model.predict(trainset)
    true_train = np.array([i.y for i in trainset])

    pred_test = model.predict(testset)
    true_test = np.array([i.y for i in testset])

    # INFO 8 展示结果，注意预测结果需要标准化还原！
    train_RMSE, train_MAE, test_RMSE, test_MAE = plotout_results(
        pred_test=model.predict(testset),
        true_test=np.array([i.y for i in testset]),
        label_of_test=[i.mol_name for i in testset],
        pred_train=model.predict(trainset_orig),
        true_train=np.array([i.y for i in trainset_orig]),
        label_of_train=None,
        std=dataset_loader.std,
        mean=dataset_loader.mean,
        plot_fig=False,  # 是否展示图片，默认展示
        plot_hist=False,
    )



    # 将标准化还原的预测结果写入文件， 注意，之后绘图可以在MGTE_deepchem中绘制error distribution那里绘制
    '''
    with open("./Multi_task/train_true_pred_and_test_true_pred_of_GNN.pkl", "wb") as f:
        pickle.dump(
            dict(
                true_train=true_train * dataset_loader.std + dataset_loader.mean,
                pred_train=pred_train,
                true_test=true_test * dataset_loader.std + dataset_loader.mean,
                pred_test=pred_test), f)
    '''


    with open("E:/ethnaol reforming/C2_3.3/final_gnn_data_%s.pkl"%TRAIN_LABEL, "wb") as f:
        pickle.dump([true_train * dataset_loader.std + dataset_loader.mean,
                     pred_train,
                     true_test * dataset_loader.std + dataset_loader.mean,
                     pred_test], f)

    # INFO 8 为了添加额外的信息（通过concat），将embedding层的特征向量拿出来！
    # NOTICE：上面使用70%训练，这里预测100%所有dataset的embedding

    # INFO 9最终把每个分子对应的feature写入，类似mol_string 2 vector
    result = model.predict_embedding_feature(dataset)
    print(len(trainset))
    # 这里是每个分子对应的embedding结果，可以进行高维可视化等操作！
    print(result.shape)
    #print(result)
    # 将embedding得到的feature写入数据集中！
    for i in range(len(dataset)):
        dataset[i].embedding_feature = result[i, :]
    with open("E:/ethnaol reforming/C2_3.3/embedded_dataset_%s.pkl"%TRAIN_LABEL, "wb") as f:
        pickle.dump(dataset, f)
    # 最终得到的是GNNFingerprintAdjacencyPropertySample的list
    #print(dataset)
    
    # 然后写入mol name to vec的字典到pkl文件中
    molName2vec_dict = dict()
    energy_dict = dict()
    for i in dataset:
        molName2vec_dict[i.mol_name] = i.embedding_feature
        energy_dict[i.mol_name] = i.y
    with open("E:/ethnaol reforming/C2_3.3/molName2vec_dict_%s.pkl"%TRAIN_LABEL, "wb") as f:
        pickle.dump(molName2vec_dict, f)

    return train_RMSE, train_MAE, test_RMSE, test_MAE, loss_curve

default_parameter = dict(
    update_method="sum",
    output_method="mean",
    dim=64,  # 16  # 32
    hidden_layer=4,  # 4, 8
    output_layer=2,  # 4
    batch_size=64,  # 64 # 32  # 16
    weight_decay=1e-6,  # 1  # 1e-6
    lr=0.01,  # 1e-3
    lr_decay=0.99,  # 0.999
    decay_interval=10,  # 100
    iteration=1000,  # 4000  # iteration太高会过拟合，注意观察曲线
)
TRAIN_LABEL= 9.27
a = copy.deepcopy(default_parameter)
r = hyper_parameter_search(a)
all_errors = r[:4]
all_loss_curve = r[4]
_ = [(i+1) for i in range(len(all_loss_curve))]
plt.plot(_[10:], all_loss_curve[10:])
plt.show()