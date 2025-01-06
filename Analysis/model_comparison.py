import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import colors
import matplotlib as mpl

mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
c = list(colors.TABLEAU_COLORS)

cwd = os.path.dirname(os.path.dirname(__file__))
os.chdir(cwd)
save_fig = True
data_files = [f for f in os.listdir('Results/Model comparison') if not f.endswith('.csv')]
data = {}

for file in data_files:
    file_path = os.path.join('Results/Model comparison', file)
    if os.path.isfile(file_path):
        data[file] = np.loadtxt(file_path)

# Plotting
names= ["Mean","CVaR","Ext"]
x_values = np.arange(1, 4)
ratios_test = []

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

for i in x_values:
    for j in x_values:
        name_HAPD_test = "HAPD_test_value_{}_{}".format(i,j)
        name_hindsight_test = "hindsight_value_test_{}".format(i)
        
        ratio_test = data[name_HAPD_test] / data[name_hindsight_test]
        
        ratios_test.append(ratio_test)
        
        ax1.bar(i+(j-1)*0.3, ratio_test, color=c[j-1], width=0.3, label=names[j-1] if i == x_values[0] else "")

ax1.axhline(y=1, color='black', linestyle='--', linewidth=1)
# Set labels and limits
ax1.set_ylabel('Profit Model / Profit Hindsight')
ax1.set_yticks([0.6,0.7, 0.8,0.9, 1])
ax1.set_xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],[r'$\mathcal{T}_{\rm{mean}}$',r'$\mathcal{T}_{\rm{CVaR}}$',r'$\mathcal{T}_{\rm{ext}}$', r'$\mathcal{T}_{\rm{mean}}^{\rm{res}}$',r'$\mathcal{T}_{\rm{CVaR}}^{\rm{res}}$',r'$\mathcal{T}_{\rm{ext}}^{\rm{res}}$',r'$\mathcal{T}_{\rm{mean}}^{\rm{cond}}$',r'$\mathcal{T}_{\rm{CVaR}}^{\rm{cond}}$',r'$\mathcal{T}_{\rm{ext}}^{\rm{cond}}$'])
plt.xticks(rotation=60, ha='right', fontsize=9, rotation_mode='anchor', verticalalignment='top')
ax1.set_ylim(0.6, 1)
#ax1.legend(loc='upper right', ncol=3,frameon=False,columnspacing=0.8)
plt.tight_layout()

if save_fig:
    plt.savefig('Plots/Model_comparison.png', dpi=300,bbox_inches='tight')

print(ratios_test)