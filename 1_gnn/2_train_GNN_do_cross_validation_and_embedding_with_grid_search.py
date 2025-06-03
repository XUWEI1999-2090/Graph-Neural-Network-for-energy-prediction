# os.system("nvidia-smi")
# time.sleep(0.5)
import copy
import pickle

import numpy as np
import torch

from MGTE.Dataset.DatasetLoader_metal import \
    RDKitMolToFloatPropertyOfFingerprintAndAdjacencyDatasetLoaderToTorchTensorDataset as RDataset
from MGTE.Model.GNNModel import GraphNeuralNetwork
from MGTE.demos.c2_molecules.utils import plotout_results

'''


算法流程
1. 拿出70%的物种-能量对，作为唯一的训练集，之后不论训练embedding还是GBR，都只能用这70%
2. 使用GNN监督学习训练embedding，尽可能不要过拟合，误差尽可能小！NOTICE：按理来说，调参需要使得验证集小，而不是使得测试集小，
然后最终预测测试集，因为这里样本很少所以没有设置验证集，同时，这里不会进行回归任务，所以这里不设置验证集
3. 然后把训练集和测试集所用的分子记录下，同时写入mol name 2 vec字典
4. 用GBR方法加上额外的与金属相关的能量，使用训练embedding的70%分子作为训练集（如果使用全部数据集随机划分，会出现数据泄露）
预测剩下的所有物种以及其他金属上所有物种的能量


关于过拟合问题：
    现在的问题是过拟合严重，尤其是当使用radius等于2的时候！会因为feature
    数目多出sample太多而过拟合！
    所以需要减少有效feature个数

    使用radius等于2，相当于增加相邻node信息，会大大增加feature
    能够使得拟合更快，但是过拟合更为严重！

    为了减少feature或者平均化feature，增加drop out！

    radius2能够包括更多有用信息，但是会因为模型的投机取巧而
    过拟合，因此不断优化dropout，使得radius2的情况下能够拟合好！

加Dropout：

    因为出现了训练集上的完美拟合，所以需要加dropout避免完美拟合，引入噪音
    实测发现，没有dropout，能够只有0.0几的error
    有了dropout，收敛时的loss明显变大，变成0.3
    这样有利于测试集
    
    dropout越大，拟合越差，过拟合也越难发生，但前提是dropout不能太大避免根本不能拟合！

    在某些情况下，调整dropout能够使得训练集error和测试集error十分相近
    这种情况下，怎么过训练也不会过拟合

    靠着图模型的信息能够达到100%拟合，但是在测试集上效果差，表明存在变分因素，可能很难
    实现完全的拟合，也就是模型的极限了，剩下的就是数据集的原因

加BatchNorm
    使用了BatchNorm1d，效果较好，不需要dropout调参，于是放弃了dropout


'''


def hyper_parameter_search(parameter_config):
    # 数据集目录，固定
    dir_input = "E:/ethnaol reforming/C2_3.2/temp1_radius1/"
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

    all_train_RMSE = []
    all_train_MAE = []
    all_test_RMSE = []
    all_test_MAE = []


    '''
    # 按照指定分子名称得到测试集（用于对照）
    with open("mol_id_string_on_testset", "rb") as f:
        mol_string_for_test = pickle.load(f)
    '''
    torch.manual_seed(4)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Use cpu")

    dataset_loader = RDataset(
        dataset_dir=dir_input,
        device=device)
    # INFO： 2.载入数据集，打乱
    dataset = dataset_loader.load_dataset()
    dataset = RDataset.shuffle_dataset(dataset, seed=9)
    # 得到训练集和测试集
    #trainset, testset = RDataset.split_dataset_by_mol_name(dataset, mol_string_for_test)
    train_set, test_set = RDataset.split_dataset(dataset, ratio=0.7)
    '''
    # Rh上关键物种划为测试集，Rh上其他物种0.8也划为测试集
    metal_else, metal_K = RDataset.split_dataset_by_metal(dataset, metal='K')
    mol_string_for_test = ['CH3CH2O-H(-K)','C(-K)H3CH(-H)OH','H-CH2CH2OH(-K)','C(-K)H3CH2O','C(-K)H3CHOH','C(-K)H2CH2OH',
                           'C(-K)H3CH(-H)O','H-CH2CHOH(-K)','C(-K)H(-H)CH2OH','C(-K)H3CHO','C(-K)H2CHOH','C(-K)HCH2OH',
                           'C(-K)H3C(-H)O','CH2CHO-H(-K)','H-CCH2OH(-K)','C(-K)H3CO','C(-K)H2CHO','C(-K)CH2OH',
                           'H-CH2CO(-K)','C(-K)H2C(-H)O','C(-K)CH(-H)OH','C(-K)H2CO','CCHO-H(-K)','C(-K)HCO',
                           'C(-K)CHO','H-CCO(-K)','C(-K)C(-H)O','C(-K)CO','C-CO(-K)','CH-CO(-K)',]
    for i in metal_K:
        if i in mol_string_for_test:
            print(i)
    test2be, testseting = RDataset.split_dataset_by_mol_name(metal_K, mol_string_for_test)
    trainset, testset = RDataset.split_dataset(test2be, ratio=0.7)
    train_set = trainset + metal_else
    test_set = testset + testseting
    '''
    trainset_orig = copy.deepcopy(train_set)

    # INFO：4 把测试集和训练集的物种保存一下，因为后续回归任务的测试物种也是这里没有embedding的物种
    # NOTICE：这里也可以发现，使用70%训练embedding，预测剩余30%的embedding，这也是迁移性的验证操作、
    trainset_mols = [i.mol_name for i in train_set]
    testset_mols = [i.mol_name for i in test_set]
    print(trainset_mols)
    with open("E:/ethnaol reforming/C2_3.2/test_mol_names.pkl", "wb") as f:
        pickle.dump(testset_mols,f)

    validation_K = 5  # 5折交叉验证
    data_num_per_fold = None
    if validation_K != 1:
        data_num_per_fold = len(train_set) // validation_K   #还是len(dataset)?
        print("%s data/fold" % data_num_per_fold)
        print("Total data num %s" % len(dataset))
        print("Total species", [i.mol_name for i in dataset])

    for now_fold in range(validation_K):
        # 把训练集进行分割，注意测试集现在不使用
        temp_data = np.repeat(train_set, 2)
        m = 0
        validset = temp_data[m:m + data_num_per_fold]
        trainset = temp_data[m + data_num_per_fold:m + len(train_set)]
        m += data_num_per_fold
        print("-----now_fold: %i-----" % (now_fold+1))
        print(len(trainset), len(validset))

        # INFO：4 把测试集和训练集的物种保存一下，因为后续回归任务的测试物种也是这里没有embedding的物种
        # NOTICE：这里也可以发现，使用70%训练embedding，预测剩余30%的embedding，这也是迁移性的验证操作、
        trainset_mols = [i.mol_name for i in trainset]
        validset_mols = [i.mol_name for i in validset]
        print(trainset_mols)
        # with open("./dataset/train_test_mol_names_iter%s.pkl" % now_fold, "wb") as f:
        #     pickle.dump((trainset_mols, testset_mols), f)
        # with open("./dataset/dataset_std_mean_iter%s.pkl" % now_fold, "wb") as f:
        #     pickle.dump((dataset_loader.std, dataset_loader.mean), f)

        # INFO 5: 建立模型
        model = GraphNeuralNetwork(
            device=device,
            update_method=update_method,
            output_method=output_method,
            n_fingerprint=dataset_loader.n_fingerprint,
            dim=dim,
            hidden_layer=hidden_layer,
            output_layer=output_layer,
            with_batch_norm=True ).to(device)

        # INFO 6 拟合训练集
        loss_curve = model.fit(
            dataset=trainset,
            epoch=iteration,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_interval=decay_interval,
            weight_decay=weight_decay,
            batch_size=batch_size,
            std_of_y=dataset_loader.std,
            mean_of_y=dataset_loader.mean, )

        # INFO 7 预测训练集和测试集
        pred_valid = model.predict(validset)
        true_valid = np.array([i.y for i in validset])

        pred_train = model.predict(trainset)
        true_train = np.array([i.y for i in trainset])

        # INFO 8 展示结果，注意预测结果需要标准化还原！
        train_RMSE, train_MAE, valid_RMSE, valid_MAE = plotout_results(
            pred_test=pred_valid,
            true_test=true_valid,
            label_of_test=[i.mol_name for i in validset],
            pred_train=pred_train,
            true_train=true_train,
            label_of_train=None,
            std=dataset_loader.std,
            mean=dataset_loader.mean,
            plot_fig=False  # 是否展示图片，默认展示
        )
        all_train_RMSE.append(train_RMSE)
        all_train_MAE.append(train_MAE)
        all_test_RMSE.append(valid_RMSE)
        all_test_MAE.append(valid_MAE)
        '''
        # 将标准化还原的预测结果写入文件， 注意，之后绘图可以在MGTE_deepchem中绘制error distribution那里绘制
        with open("train_true_pred_and_test_true_pred_of_GNN_grid_search.pkl", "wb") as f:
             pickle.dump(
                dict(
                     true_train=true_train * dataset_loader.std + dataset_loader.mean,
                     pred_train=pred_train,
                     true_test=true_test * dataset_loader.std + dataset_loader.mean,
                     pred_valid=pred_valid), f)

        # INFO 8 为了添加额外的信息（通过concat），将embedding层的特征向量拿出来！
        # NOTICE：上面使用70%训练，这里预测100%所有dataset的embedding
       
        # INFO 9最终把每个分子对应的feature写入，类似mol_string 2 vector
        result = model.predict_embedding_feature(dataset)
        print(len(trainset))
        # 这里是每个分子对应的embedding结果，可以进行高维可视化等操作！
        print(result.shape)
        # 将embedding得到的feature写入数据集中！
        for i in range(len(dataset)):
            dataset[i].embedding_feature = result[i, :]
        with open("./dataset/embedded_dataset_iter%s.pkl" % now_fold, "wb") as f:
            pickle.dump(dataset, f)
        # 最终得到的是GNNFingerprintAdjacencyPropertySample的list
        print(dataset)

        # 然后写入mol name to vec的字典到pkl文件中
        molName2vec_dict = dict()
        for i in dataset:
            molName2vec_dict[i.mol_name] = i.embedding_feature
        with open("./dataset/molName2vec_dict_iter%s.pkl" % now_fold, "wb") as f:
            pickle.dump(molName2vec_dict, f)
        '''
        print("All Train RMSE, MAE and test RMSE, MAE")
        print(all_train_RMSE)
        print(all_train_MAE)
        print(all_test_RMSE)
        print(all_test_MAE)

        print("Mean Train RMSE, MAE and test RMSE, MAE")
        print(np.mean(all_train_RMSE))
        print(np.mean(all_train_MAE))
        print(np.mean(all_test_RMSE))
        print(np.mean(all_test_MAE))

        '''
        # 最后预测一下测试集上的结果，使用全部训练集训练
        
        model = GraphNeuralNetwork(
            device=device,
            update_method=update_method,
            output_method=output_method,
            n_fingerprint=dataset_loader.n_fingerprint,
            dim=dim,
            hidden_layer=hidden_layer,
            output_layer=output_layer,
            with_batch_norm=True).to(device)

        # INFO  拟合全部训练集
        _ = model.fit(
            dataset=trainset_orig,
            epoch=iteration,
            lr=lr,
            lr_decay=lr_decay,
            lr_decay_interval=decay_interval,
            weight_decay=weight_decay,
            batch_size=batch_size,
            std_of_y=dataset_loader.std,
            mean_of_y=dataset_loader.mean, )

        # INFO 8 展示结果，注意预测结果需要标准化还原！
        _, _, test_RMSE, test_MAE = plotout_results(
            pred_test=model.predict(testset),
            true_test=np.array([i.y for i in testset]),
            label_of_test=[i.mol_name for i in validset],
            pred_train=model.predict(trainset_orig),
            true_train=np.array([i.y for i in trainset_orig]),
            label_of_train=None,
            std=dataset_loader.std,
            mean=dataset_loader.mean,
            plot_fig=False  # 是否展示图片，默认展示
        )

        
        # 最后存储一下训练过后的embedding权重
        fingerprint_embed = model.embed_fingerprint.weight.data.numpy()
        print(fingerprint_embed.shape)
        np.save("./fingerprint_embed_matrix_new.npy", fingerprint_embed)
        '''
        #return np.mean(all_train_RMSE), np.mean(all_test_RMSE), np.mean(all_train_MAE), np.mean(
        #    all_test_MAE), loss_curve,test_RMSE,test_MAE
    return np.mean(all_train_RMSE), np.mean(all_train_MAE), np.mean(all_test_RMSE), np.mean(
            all_test_MAE), loss_curve


# grid search 的调参表格
default_parameter = dict(
    update_method="sum",
    output_method="mean",
    dim=16,  # 16  # 32
    hidden_layer=4,  # 4, 8
    output_layer=2,  # 4
    batch_size=32,  # 64 # 32  # 16
    weight_decay=1e-6,  # 1  # 1e-6
    lr=1e-2,  # 1e-3
    lr_decay=0.99,  # 0.999
    decay_interval=10,  # 100
    iteration=1000,  # 4000  # iteration太高会过拟合，注意观察曲线
)
configs = []
all_errors = []
all_loss_curve = []
all_test_RMSE_and_MAE = []
now = 0
TRAIN_LABEL = "97" # 训练批号
for dim in [16, 32, 64]:
    for hidden_layer in [4,8]:
       for output_layer in [2,4]:
            for batch_size in [16,32,64]:
                for lr in [1e-2,1e-3]:
                    a = copy.deepcopy(default_parameter)
                    a["dim"] = dim
                    a["hidden_layer"] = hidden_layer
                    a["output_layer"] = output_layer
                    a["batch_size"] = batch_size
                    a["lr"] = lr
                    configs.append(a)
                    r = hyper_parameter_search(a)
                    all_errors.append(r[:4])
                    all_loss_curve.append(r[4])
                    #all_test_RMSE_and_MAE.append(r[5:])
                    now += 1
                    print("-------------Finished one now: %s--------------" % now)
print(configs)

print(all_loss_curve)
print(all_errors)
#print(all_test_RMSE_and_MAE)

with open("E:/ethnaol reforming/C2_3.2/grid_search_cv_new_%s.pkl"%TRAIN_LABEL, "wb") as f:
    pickle.dump([configs, all_errors], f)  # all_loss_curve,all_test_RMSE_and_MAE

# 最后保存一下结果到csv文件

import pandas

# 首先用字典构造dataframe的数据内容
result_data = dict()
error_col_name = [

    "all_train_RMSE",
    "all_train_MAE",
    "all_valid_RMSE",
    "all_valid_MAE",



]

for i in range(len(configs)):
    target_config = configs[i]
    for key in target_config:
        if key not in result_data.keys():
            result_data[key] = []
        result_data[key].append(target_config[key])
    for j in range(len(all_errors[i])):
        if error_col_name[j] not in result_data.keys():
            result_data[error_col_name[j]] = []
        result_data[error_col_name[j]].append(all_errors[i][j])
    '''
    for k in ["test_RMSE","test_MAE"]:
        if k not in result_data.keys():
            result_data[k] = []
    result_data["test_RMSE"].append(all_test_RMSE_and_MAE[i][0])
    result_data["test_MAE"].append(all_test_RMSE_and_MAE[i][1])
    '''
data = pandas.DataFrame.from_dict(result_data)
data.to_csv("E:/ethnaol reforming/C2_3.2/grid_search_gnn_new_dataset_%s.csv"%TRAIN_LABEL)
