import numpy as np
import pandas as pd
import os

def feature_x(data_hist):
    """
    create feature vector x from historic data
    --------------------------------
    param: (df) data_hist: historic data
    param: (bool) rf: reduced form of feature vector including only deterministic wind production forecast
    return (df) x: feature vector
    """
    # Read in the top features from the specified files
    os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
    top_features_ridgecv = pd.read_csv('Results/Feature selection/top_features_RidgeCV.txt', header=None).squeeze().tolist()
    #top_features_shap = pd.read_csv('Results/Feature selection/top_features_SHAP.txt', header=None).squeeze().tolist()

    x = data_hist[["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1",
                       "production_FC","lambda_DA_FC",
                       "E_RE_day_before","lambda_RE_day_before","lambda_IM_day_before",
                       "E_RE_day_before_mean","lambda_RE_day_before_mean","lambda_IM_day_before_mean",
                       "E_RE_day_before_10","lambda_RE_day_before_10","lambda_IM_day_before_10",
                       "E_RE_day_before_90","lambda_RE_day_before_90","lambda_IM_day_before_90"]]

    x = data_hist[top_features_ridgecv]
    print("Features: ", x.columns)
    #for name in x.columns:
    #    x.insert(0,"{} ^2".format(name),x[name]**2)
    x.insert(0,"offset",1.)
    x.insert(0,"lambda_DA_RE",np.maximum(data_hist["lambda_DA_RE"].to_numpy(),0))

    return x