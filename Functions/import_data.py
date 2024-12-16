
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Functions.insert_row import Insert_row

def import_data(data_param,model_param):
    """
    import historic data for training and testing
    --------------------------------
    param: (str) path: file path of csv file
    param: (arraylike) file_names: list of data file names
    param: (dict) model_param: model parameters
    param: (int) months_train: length of training in months, default is half of dataset
    return (df) data_train: data for training, (df) data_test: data for testing
    """
    data_name = data_param["file_path"] + data_param["file_names"][0]
    data1_name = data_param["file_path"] + data_param["file_names"][1]
    data = pd.read_csv(data_name)
    data1 = pd.read_csv(data1_name,sep=";",decimal = ',')
    data1 = data1.loc[data1['PriceArea'] == "DK2"]
    data1 =  data1.reset_index()[["HourDK", "ImbalancePriceEUR"]] # ImbalancePriceEUR
    data1["ImbalancePriceEUR"] = data1["ImbalancePriceEUR"].astype(float)

    # data1 missing 2h from time time switch
    data1 = Insert_row(7177,data1,['2019-10-27 02:00', data["DW"].iloc[7177]])
    data1.at[7178,"ImbalancePriceEUR"] = data["DW"].loc[7178].copy()
    data1 = Insert_row(15913,data1,['2020-10-25 02:00', data["DW"].iloc[15913]])
    data1.at[15914,"ImbalancePriceEUR"] = data["DW"].loc[15914].copy()

    data.insert(0,"lambda_DA_FC",np.maximum(data["forward_FC"].to_numpy(),0)) # Set negative prices to 0
    data.insert(0,"lambda_DA_RE",np.maximum(data["forward_RE"].to_numpy(),0)) # Set negative prices to 0
    data.insert(0,"lambda_IM",np.maximum(data1["ImbalancePriceEUR"].to_numpy(),0))
    data.insert(0,"E_RE",np.maximum(data["production_RE"].to_numpy()*model_param["max_wind_capacity"],0))
    data.insert(0,"E_FC",np.maximum(data["production_FC"].to_numpy()*model_param["max_wind_capacity"],0))
    data.insert(0,"lambda_UP",np.maximum(data["UP"].to_numpy(),0))
    data.insert(0,"lambda_DW",np.maximum(data["DW"].to_numpy(),0))

    # Add addtitional features
    E_RE_day_before = np.zeros(len(data))
    lambda_RE_day_before = np.zeros(len(data))
    lambda_IM_day_before = np.zeros(len(data))
    E_RE_day_before_mean = np.zeros(len(data))    # mean of day before
    lambda_RE_day_before_mean = np.zeros(len(data))
    lambda_IM_day_before_mean = np.zeros(len(data))
    E_RE_day_before_10 = np.zeros(len(data))    # 10% quantile of day before
    lambda_RE_day_before_10 = np.zeros(len(data))
    lambda_IM_day_before_10 = np.zeros(len(data))
    E_RE_day_before_90 = np.zeros(len(data))     # 90% quantile of day before
    lambda_RE_day_before_90 = np.zeros(len(data))
    lambda_IM_day_before_90 = np.zeros(len(data))


    for t in range(24,len(data)):
        E_RE_day_before[t] = data["E_RE"].iloc[t-24]
        lambda_RE_day_before[t] = data["lambda_DA_RE"].iloc[t-24]
        lambda_IM_day_before[t] = data["lambda_IM"].iloc[t-24]
        if t % 24 == 23:
            E_RE_day_before_mean[t-23:t+1] = np.mean(E_RE_day_before[t-23:t+1])
            lambda_RE_day_before_mean[t-23:t+1] = np.mean(lambda_RE_day_before[t-23:t+1])
            lambda_IM_day_before_mean[t-23:t+1] = np.mean(lambda_IM_day_before[t-23:t+1])
            E_RE_day_before_10[t-23:t+1] = np.quantile(E_RE_day_before[t-23:t+1],0.1)
            lambda_RE_day_before_10[t-23:t+1] = np.quantile(lambda_RE_day_before[t-23:t+1],0.1)
            lambda_IM_day_before_10[t-23:t+1] = np.quantile(lambda_IM_day_before[t-23:t+1],0.1)
            E_RE_day_before_90[t-23:t+1] = np.quantile(E_RE_day_before[t-23:t+1],0.9)
            lambda_RE_day_before_90[t-23:t+1] = np.quantile(lambda_RE_day_before[t-23:t+1],0.9)
            lambda_IM_day_before_90[t-23:t+1] = np.quantile(lambda_IM_day_before[t-23:t+1],0.9)

    data.insert(0,"E_RE_day_before",E_RE_day_before)
    data.insert(0,"lambda_RE_day_before",lambda_RE_day_before)
    data.insert(0,"lambda_IM_day_before",lambda_IM_day_before)
    data.insert(0,"E_RE_day_before_mean",E_RE_day_before_mean)
    data.insert(0,"lambda_RE_day_before_mean",lambda_RE_day_before_mean)
    data.insert(0,"lambda_IM_day_before_mean",lambda_IM_day_before_mean)
    data.insert(0,"E_RE_day_before_10",E_RE_day_before_10)
    data.insert(0,"lambda_RE_day_before_10",lambda_RE_day_before_10)
    data.insert(0,"lambda_IM_day_before_10",lambda_IM_day_before_10)
    data.insert(0,"E_RE_day_before_90",E_RE_day_before_90)
    data.insert(0,"lambda_RE_day_before_90",lambda_RE_day_before_90)
    data.insert(0,"lambda_IM_day_before_90",lambda_IM_day_before_90)

    # Remove first day
    data = data.iloc[24:]

    data_train, data_test = train_test_split(data, test_size=0.5, shuffle=False)
    train_mod = len(data_train) % 24
    if train_mod != 0:
        data_train = pd.concat([data_train, data_test.iloc[:train_mod]])
        data_test = data_test.iloc[train_mod:]
    data_train, data_valid = train_test_split(data_train, test_size=0.3, shuffle=False)
    train_mod = int(len(data_train) % 24)
    if train_mod != 0:
        data_train = pd.concat([data_train, data_valid.iloc[:train_mod]])
        data_valid = data_valid.iloc[train_mod:]
    print("Training length: ", len(data_train)/24, "Days")
    print("Validation length: ", len(data_valid)/24, "Days")
    print("Testing length: ", len(data_test)/24, "Days")

    return data_train, data_valid, data_test