import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import colors

plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
c = list(colors.TABLEAU_COLORS)

os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
save_fig = True
data_files = [f for f in os.listdir('Results/PD selection valid')]
data = {}

for file in data_files:
    file_path = os.path.join('Results/PD selection valid', file)
    if os.path.isfile(file_path):
        data[file] = np.loadtxt(file_path)

for i in range(len(data.values())):
    plt.bar(list(data.keys())[i], list(data.values())[i]/list(data.values())[-1], color=c[0])

plt.ylabel('Revenue/Revenue_max')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

if save_fig:
    plt.savefig('Plots/price_domains_valid_bar.png')