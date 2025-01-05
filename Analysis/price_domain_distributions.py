import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import colors
from Functions.import_data import import_data

plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 15
c = list(colors.TABLEAU_COLORS)

os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
save_fig = True

data_param = {"file_path": "Data/",
              "file_names": np.array(["2020_data.csv","RegulatingBalancePowerdata.csv"]),
              "save_path": "Plots/",
              "save_results": True
              }

model_param = {"model_type": "hindsight",    # "HAPD" or "hindsight"
               "max_wind_capacity": 10,    # MW
               "max_electrolyzer_capacity": 10,    # MW
               "min_production": 880,     # 880 kg_H2/day (= 5MWh)
               "max_production": 2640,     #  kg_H2/day 
               "price_quantile": np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9]),    # Price quantiles in % for price domains
               "H2_quantile": False,    # Add hydrogen price as domain boundary
               "lambda_H": 2,     # €/kg_H2
               "H2_efficiency_const": False,    # constant efficiency if True, else model of hydrogen production curve
               "eta_storage": 0.88,    # efficiency for compression and additional energy losses when storing hydrogen
               "eta_production": 20,    # kg_h2/MWh_input - production efficiency when assumed constant
               "p_min": 0.15,    # Minimum load of electrolyzer consumption
               "n_segments": 2,    # Number of segments for hydrogen production modelling
               "lambda_s": 10,    # price of slack variable
               }

# Import data
data_train, data_test = import_data(data_param,model_param)

# Combine data to calculate bin edges
combined_data = np.concatenate((data_train['lambda_DA_RE'], data_test['lambda_DA_RE']))

# Calculate bin edges
bin_width = 2  # Set your desired bin width
bins = np.arange(min(combined_data), max(combined_data) + bin_width, bin_width)

# Plot the distribution of "lambda_DA_RE"
plt.figure(figsize=(10, 6))
plt.hist(data_train['lambda_DA_RE'], bins=bins, color=c[0], alpha=0.7, label='Train Data')
plt.hist(data_test['lambda_DA_RE'], bins=bins, color=c[1], alpha=0.7, label='Test Data')
plt.title('Distribution of lambda_DA_RE', fontsize=25)
plt.xlabel('lambda_DA_RE', fontsize=20)
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()

if save_fig:
    plt.savefig(os.path.join(data_param['save_path'], 'lambda_DA_RE_distribution.png'))
plt.show()
