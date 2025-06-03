import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


from utils import p_plot
def deal_with_one_pkl_file(pkl_filename):
    pass

    with open(pkl_filename, "rb") as f:
        data = pickle.load(f)
    #print(data)
    true_train=data[-4]
    pred_train=data[-3]
    true_test=data[-2]
    pred_test=data[-1]


    '''
    true_train = data["true_train"].reshape(-1)
    pred_train = data["pred_train"].reshape(-1)
    true_test = data["true_test"].reshape(-1)
    pred_test = data["pred_test"].reshape(-1)
    '''
    p_plot(pred_train, true_train, pred_test, true_test)



    error_train = true_train - pred_train
    error_test = true_test - pred_test

    print(error_train.shape)
    import matplotlib.pyplot as plt

    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.hist(error_train,bins=10)
    plt.hist(error_test,bins=10)

    plt.show()

    final_test_stage_train_RMSE = np.sqrt(mean_squared_error(true_train, pred_train))
    final_test_stage_test_RMSE = np.sqrt(mean_squared_error(true_test, pred_test))

    final_test_stage_train_MAE = mean_absolute_error(true_train, pred_train)
    final_test_stage_test_MAE = mean_absolute_error(true_test, pred_test)

    print(final_test_stage_train_RMSE,final_test_stage_test_RMSE)
    print(final_test_stage_train_MAE,final_test_stage_test_MAE)

    # 文字标注outlier

    # 将error和true形成dict方便对照
    error_energy_dict = dict()
    for i in range(len(error_train)):
        error_energy_dict[true_train[i]] = error_train[i]
    for j in range(len(error_test)):
        error_energy_dict[true_test[j]] = error_test[j]

    # 取大于平均数的error

    true=[]
    for i in error_test:
        if abs(i) >= final_test_stage_test_MAE:
            true.append(list(error_energy_dict.keys())[list(error_energy_dict.values()).index(i)])
            print(round(i,3))
    print(true)



deal_with_one_pkl_file("E:/ethnaol reforming/C2_3.3/final_gnn_data_9.27.pkl")
#deal_with_one_pkl_file("E:/ethnaol reforming/C2_3.3/final_weave_data_new_randomSeed_123.pkl")
#deal_with_one_pkl_file("E:/ethnaol reforming/C2_3.3/final_graph_convolution_data_new_randomSeed_123.pkl")




