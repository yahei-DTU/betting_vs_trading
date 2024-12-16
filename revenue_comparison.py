import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import colors

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
c = list(colors.TABLEAU_COLORS)

os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
save_fig = True
data_files = [f for f in os.listdir('Results/Revenue comparison') if f.endswith('.csv')]
data = {}

for file in data_files:
    file_path = os.path.join('Results/Revenue comparison', file)
    if os.path.isfile(file_path):
        data[file[:-4]] = pd.read_csv(file_path)

d = np.arange(365)



# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for i, key in enumerate(data.keys()):
    rev = []
    for day in range(365):
        rev.append(np.sum(data[key]['revenue'].iloc[day*24:(day+1)*24])) 
    ax.plot(d, rev, color=c[i], label="{} E(Rev) = {} €/day".format(key[7:],round(np.mean(rev)),0),alpha=0.5)
ax.set_xlabel('Day')
ax.set_ylabel('Revenue')
ax.legend(loc='lower left')
if save_fig:
    plt.savefig('Plots/Revenue_comparison.png', dpi=300)