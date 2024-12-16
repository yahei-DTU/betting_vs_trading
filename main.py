import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from Functions.feature_x import feature_x
from Functions.import_data import import_data
from Functions.test_policies import test_policies, plot_bidding
from Functions import electrolyzer_efficiency 
import importlib
import Models
import os
import Models.hindsight
import time

def single_run(data_param,model_param):
    """
    run model for single set of input parameters
    --------------------------------
    param: (dict) data_param: data parameters
    param: (dict) model_param: model parameters
    """
    # Run model
    data_train, data_valid, data_test = import_data(data_param,model_param)
    module_name = "Models." + model_param["model_name"]
    module = importlib.import_module(module_name)
    name_type = model_param["model_name"] + str(model_param["model_type"])
    model = getattr(module,name_type)
    if model_param["model_name"] == "hindsight":
        # Train results
        value, df_results = model(data_train,model_param)
        print("ObjVal_hindsight_train:", value)
        if data_param["save_results"]:
            np.savetxt("{}hindsight_value_train".format(data_param["save_path"]), np.array([value]))
            df_results.to_csv("{}p_hindsight_train.csv".format(data_param["save_path"]),index=False)
        # Validation results
        value, df_results = model(data_valid,model_param)
        print("ObjVal_hindsight_valid:", value)
        if data_param["save_results"]:
            np.savetxt("{}hindsight_value_valid".format(data_param["save_path"]), np.array([value]))
            df_results.to_csv("{}p_hindsight_valid.csv".format(data_param["save_path"]),index=False)
        # Test results
        t0 = time.time()
        value, df_results = model(data_test,model_param)
        t1 = time.time()
        print("Time: ", t1-t0)
        print("ObjVal_hindsight:", value)
        if data_param["save_results"]:
            np.savetxt("{}hindsight_value_test".format(data_param["save_path"]), np.array([value]))
            df_results.to_csv("{}p_hindsight_test.csv".format(data_param["save_path"]),index=False)
    elif model_param["model_name"] == "HAPD":
        price_quantile = np.array([])
        for i in range(len(model_param["price_quantile"])):
            quantile = np.quantile(data_train["lambda_DA_RE"], model_param["price_quantile"][i])
            price_quantile = np.append(price_quantile,quantile)
        price_quantile = np.sort(price_quantile)
        x_train = feature_x(data_train)
        t0 = time.time()
        q_DA, q_H, value_train = model(data_train,model_param,x_train,price_quantile)
        t1 = time.time()
        if data_param["save_results"]:
            np.savetxt("{0}{1}_train_value".format(data_param["save_path"],model_param["model_name"]), np.array([value_train]))
        print("Training time: ", t1-t0)
        print("ObjVal on training data: ",value_train)
        # Validation results
        x_valid = feature_x(data_valid)
        value_valid, df_results = test_policies(q_DA,q_H,data_valid,x_valid,model_param,price_quantile)
        print("ObjVal on validation data: ",value_valid)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_valid_value".format(data_param["save_path"],model_param["model_name"]), np.array([value_valid]))

        # Test results
        x_test = feature_x(data_test)
        plot_bidding(q_DA,q_H,data_test,x_test,model_param,price_quantile)
        value_test, df_results = test_policies(q_DA,q_H,data_test,x_test,model_param,price_quantile)
        print("ObjVal on testing data: ",value_test)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_test_value".format(data_param["save_path"],model_param["model_name"]), np.array([value_test]))

            df_q_DA = pd.DataFrame(data=zip(q_DA), columns = np.array(["q_DA"]))
            df_q_H = pd.DataFrame(data=zip(q_H), columns = np.array(["q_H"]))
            df_q_DA.to_csv("{0}q_DA_{1}.csv".format(data_param["save_path"],model_param["model_name"]),index=False)
            df_q_H.to_csv("{0}q_H_{1}.csv".format(data_param["save_path"],model_param["model_name"]),index=False)

            df_results.to_csv("{0}p_{1}.csv".format(data_param["save_path"],model_param["model_name"]),index=False)
    else:
        print("No model chosen")

def main(): 
    os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
    #os.chdir("/zhome/25/9/211757/0_try_and_error/Day_ahead_formulation")
    
    config_dict = {
      'eff_type': 2 # Choose model for hydorgen production curve1 (1:HYP-MIL, 2: HYP-L, 3: HYP-SOC, 4: HYP_MISOC)
      }
    
    data_param = {"file_path": "Data/",
                  "file_names": np.array(["2020_data.csv","RegulatingBalancePowerdata.csv"]),
                  "save_path": "Results/",
                  "save_results": True
                  }
    
    model_param = {"model_name": "hindsight",    # "HAPD" or "hindsight"
                   "model_type": 1,    # 1: selling+buying, 2: only selling, 3: conditional buying
                   "risk_model": 2,    # 1: expected value of absolute imbalance, 2: CVaR, 3: max imbalance
                   "max_wind_capacity": 10,    # MW
                   "max_electrolyzer_capacity": 10,    # MW
                   "min_production": 880,     # 880 kg_H2/day (= 5MWh)
                   "max_production": 2640,     #  kg_H2/day 
                   "price_quantile": np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),    # Price quantiles in % for price domains
                   "lambda_H": 6,     # €/kg_H2
                   "eta_storage": 0.88,    # efficiency for compression and additional energy losses when storing hydrogen
                   "p_min": 0.15,    # Minimum load of electrolyzer consumption
                   "n_segments": 2,    # Number of segments for hydrogen production modelling
                   "K_su": 50,    # cold-startup cost for electrolyzer in €/MWh
                   "p_sb": 0.01,    # standby power consumption of electrolyzer
                   "alpha": 0.95,    # confidence level for CVaR
    }
    p_IM_abs_mean = np.loadtxt("Results/mean(p_IM_abs)")
    p_IM_abs_max = np.loadtxt("Results/max(p_IM_abs)")
    CVaR_95 = np.loadtxt("Results/CVaR_95")

    model_param.update({'p_IM_abs_mean': p_IM_abs_mean*0.5,
                        'p_IM_abs_max': p_IM_abs_max*0.5,
                        'CVaR_95': CVaR_95*0.5})
    P_eta_max = electrolyzer_efficiency.p_eta_max_fun(model_param) # around 28% of maximum power

    P_segments = [[model_param['p_min'],1], #1
                  [model_param['p_min'],P_eta_max,1], #2
                  [model_param['p_min'],P_eta_max,0.64, 1], #3
                  [model_param['p_min'],P_eta_max,0.52,	0.76, 1], #4
                  [model_param['p_min'],P_eta_max,0.46,	0.64, 	0.82, 1] #5
                  ]

    p_val = np.array(P_segments[model_param["n_segments"]-1])*model_param["max_electrolyzer_capacity"]

    model_param.update({'p_val': p_val})
    electrolyzer_efficiency.initialize_electrolyzer(model_param,config_dict) # Initialize electrolyzer (approximation coeff., etc.)
    #electrolyzer_efficiency.plot_el_curves(model_param, config_dict) # Plot the nonlinear and approximated curve
    
    single_run(data_param,model_param)

if __name__ == "__main__":
    main()