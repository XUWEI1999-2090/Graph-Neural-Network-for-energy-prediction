import pickle

#pkl_filename = "./weave_grid_search_new_dataset.pkl"
pkl_filename = "./graphConv_grid_search_new_dataset.pkl"

with open(pkl_filename, "rb") as f:
    all_configs, all_loss_curve, all_train_RMSE, all_valid_RMSE,all_test_RMSE = pickle.load(f)

example_config = all_configs[0]
data = dict()

data["train_RMSE"] = []
data["valid_RMSE"] = []
data["test_RMSE"] = []
for key in example_config.keys():
    data[key] = []

for i in range(len(all_configs)):
    target_config = all_configs[i]
    for key in target_config:
        data[key].append(target_config[key])
    data["train_RMSE"].append(all_train_RMSE[i])
    data["valid_RMSE"].append(all_valid_RMSE[i])
    data["test_RMSE"].append(all_test_RMSE[i])


import pandas

data_frame = pandas.DataFrame.from_dict(data)
data_frame.to_csv(pkl_filename+".csv")