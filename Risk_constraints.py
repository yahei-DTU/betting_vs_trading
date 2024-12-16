import pandas as pd
import numpy as np
import os

# Define the path to the CSV file
os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")

# Load the data into a DataFrame
data = pd.read_csv("Results/p_HAPD_train.csv")
data1 = pd.read_csv("Results/p_hindsight_test.csv")
# Add the absolute value of p_IM to data1
data1['p_IM_abs'] = np.abs(data1['p_IM'])

# Calculate the expected value of p_IM_abs
expected_value_p_IM_abs = data['p_IM_abs'].mean()
expected_value_p_IM_abs1 = data1['p_IM_abs'].mean()

# Display the expected value
print(f"Expected value of p_IM_abs: {expected_value_p_IM_abs}")
print(f"Expected value of p_IM: {expected_value_p_IM_abs1}")

# Calculate the maximum value of p_IM_abs
max_p_IM_abs = data['p_IM_abs'].max()
max_p_IM_abs1 = data1['p_IM_abs'].max()

# Display the maximum value
print(f"Maximum value of p_IM_abs: {max_p_IM_abs}")
print(f"Maximum value of p_IM_abs: {max_p_IM_abs1}")

# Calculate the Value at Risk (VaR) at the 95% confidence level
VaR_95 = np.percentile(data['p_IM_abs'], 95)
VaR_951 = np.percentile(data1['p_IM_abs'], 95)

# Calculate the Conditional Value at Risk (CVaR) at the 95% confidence level
CVaR_95 = data[data['p_IM_abs'] >= VaR_95]['p_IM_abs'].mean()
CVaR_951 = data1[data1['p_IM_abs'] >= VaR_951]['p_IM_abs'].mean()

# Display the CVaR
print(f"CVaR at 95% confidence level: {CVaR_95}")
print(f"CVaR at 95% confidence level: {CVaR_951}")

#np.savetxt("Results/mean(p_IM_abs)", np.array([expected_value_p_IM_abs]))
#np.savetxt("Results/max(p_IM_abs)", np.array([max_p_IM_abs]))
#np.savetxt("Results/CVaR_95", np.array([CVaR_95]))
