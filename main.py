import os
import numpy as np
import pandas as pd
from Functions.feature_x import feature_x
from Functions.import_data import import_data
from Functions.test_policies import test_policies, plot_bidding
from Functions import electrolyzer_efficiency 
from Models.model import Model
import time


def main():
    cwd = os.path.dirname(__file__)
    os.chdir(cwd)
   
    # Set parameters
    config_dict = {
      'eff_type': 2 # Choose model for hydorgen production curve1 (1:HYP-MIL, 2: HYP-L, 3: HYP-SOC, 4: HYP_MISOC)
      }
    
    data_param = {"file_path": "Data/",
                  "file_names": np.array(["2020_data.csv","RegulatingBalancePowerdata.csv"]),
                  "save_path": "Results/",
                  "save_results": True
                  }
    
    model_param = {"model_type": 2,    # 1: "policies", 2: "hindsight"
                   "model_conditions": 1,    # 1: selling+buying, 2: only selling, 3: conditional buying
                   "model_risk": 0,    # 0: no risk constraints, 1: mean imbalance, 2: CVaR, 3: max imbalance
                   "max_wind_capacity": 10,    # MW
                   "max_electrolyzer_capacity": 10,    # MW
                   "min_production": 880,     # 880 kg_H2/day (ca 5MWh)
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
    
    # get risk constraint limits
    p_IM_abs_mean = np.loadtxt("Results/mean(p_IM_abs)")
    p_IM_abs_max = np.loadtxt("Results/max(p_IM_abs)")
    CVaR_95 = np.loadtxt("Results/CVaR_95")

    model_param.update({'p_IM_abs_mean': p_IM_abs_mean*0.5,
                        'p_IM_abs_max': p_IM_abs_max*0.5,
                        'CVaR_95': CVaR_95*0.5})
    
    # Initialize electrolyzer (approximation coeff., etc.)
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
    
    # Run model
    data_train, data_valid, data_test = import_data(data_param,model_param)
    if model_param["model_type"] == 1: # Policies
        # Define quantiles for price domains
        price_quantile = np.array([])
        for i in range(len(model_param["price_quantile"])):
            quantile = np.quantile(data_train["lambda_DA_RE"], model_param["price_quantile"][i])
            price_quantile = np.append(price_quantile,quantile)
        price_quantile = np.sort(price_quantile)
        print("Price Domains:", price_quantile)

        # Training results
        x_train = feature_x(data_train)
        t0 = time.time()
        model_train = Model(300,data_train,data_param,model_param,x_train,price_quantile)
        model_train.build_model()
        model_train.run_model()
        t1 = time.time()
        print("Training time: ", t1-t0)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_train_value".format(data_param["save_path"],"policies"), np.array([model_train.results.objective_value]))
            df_results = pd.DataFrame(data=zip(model_train.results.p_DA,model_train.results.p_H,model_train.results.p_IM,model_train.results.p_IM_abs,model_train.results.xi,model_train.results.h_H,model_train.results.E_RE,model_train.results.z_on,model_train.results.z_sb,model_train.results.z_off,model_train.results.z_su,model_train.results.lambda_DA_RE,model_train.results.lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
            df_results.to_csv("{0}{1}.csv".format(data_param["save_path"],"policies_train"),index=False)

        # Validation results
        x_valid = feature_x(data_valid)
        value_valid, df_results = test_policies(model_train.results.q_DA,model_train.results.q_H,data_valid,x_valid,model_param,price_quantile)
        print("ObjVal on validation data: ",value_valid)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_valid_value".format(data_param["save_path"],"policies"), np.array([value_valid]))
            df_results.to_csv("{0}{1}.csv".format(data_param["save_path"],"policies_valid"),index=False)
        
        # Test results
        x_test = feature_x(data_test)
        #plot_bidding(model_train.results.q_DA,model_train.results.q_H,data_test,x_test,model_param,price_quantile)
        value_test, df_results = test_policies(model_train.results.q_DA,model_train.results.q_H,data_test,x_test,model_param,price_quantile)
        print("ObjVal on testing data: ",value_test)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_test_value".format(data_param["save_path"],"policies"), np.array([value_test]))

            df_q_DA = pd.DataFrame(data=zip(model_train.results.q_DA), columns = np.array(["q_DA"]))
            df_q_H = pd.DataFrame(data=zip(model_train.results.q_H), columns = np.array(["q_H"]))
            df_q_DA.to_csv("{0}q_DA.csv".format(data_param["save_path"]),index=False)
            df_q_H.to_csv("{0}q_H.csv".format(data_param["save_path"]),index=False)

            df_results.to_csv("{0}{1}.csv".format(data_param["save_path"],"policies_test"),index=False)
    
    elif model_param["model_type"] == 2: # Hindsight
        # Train results
        t0 = time.time()
        model_train = Model(300,data_train,data_param,model_param)
        model_train.build_model()
        model_train.run_model()
        t1 = time.time()
        print("Training time: ", t1-t0)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_train_value".format(data_param["save_path"],"hindsight"), np.array([model_train.results.objective_value]))
            df_results = pd.DataFrame(data=zip(model_train.results.p_DA,model_train.results.p_H,model_train.results.p_IM,model_train.results.p_IM_abs,model_train.results.xi,model_train.results.h_H,model_train.results.E_RE,model_train.results.z_on,model_train.results.z_sb,model_train.results.z_off,model_train.results.z_su,model_train.results.lambda_DA_RE,model_train.results.lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
            df_results.to_csv("{0}{1}_train.csv".format(data_param["save_path"],"hindsight"),index=False)
        
        # Validation results
        t0 = time.time()
        model_valid = Model(300,data_valid,data_param,model_param)
        model_valid.build_model()
        model_valid.run_model()
        t1 = time.time()
        print("Training time: ", t1-t0)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_valid_value".format(data_param["save_path"],"hindsight"), np.array([model_valid.results.objective_value]))
            df_results = pd.DataFrame(data=zip(model_valid.results.p_DA,model_valid.results.p_H,model_valid.results.p_IM,model_valid.results.p_IM_abs,model_valid.results.xi,model_valid.results.h_H,model_valid.results.E_RE,model_valid.results.z_on,model_valid.results.z_sb,model_valid.results.z_off,model_valid.results.z_su,model_valid.results.lambda_DA_RE,model_valid.results.lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
            df_results.to_csv("{0}{1}_valid.csv".format(data_param["save_path"],"hindsight"),index=False)
        
        # Test results
        t0 = time.time()
        model_test = Model(300,data_test,data_param,model_param)
        model_test.build_model()
        model_test.run_model()
        t1 = time.time()
        print("Training time: ", t1-t0)
        if data_param["save_results"]:
            np.savetxt("{0}{1}_test_value".format(data_param["save_path"],"hindsight"), np.array([model_test.results.objective_value]))
            df_results = pd.DataFrame(data=zip(model_test.results.p_DA,model_test.results.p_H,model_test.results.p_IM,model_test.results.p_IM_abs,model_test.results.xi,model_test.results.h_H,model_test.results.E_RE,model_test.results.z_on,model_test.results.z_sb,model_test.results.z_off,model_test.results.z_su,model_test.results.lambda_DA_RE,model_test.results.lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
            df_results.to_csv("{0}{1}_test.csv".format(data_param["save_path"],"hindsight"),index=False)
    else:
        print("No model chosen")

if __name__ == '__main__':
    main()