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

os.chdir("C:/Users/yahei/OneDrive - Danmarks Tekniske Universitet/Code/PtX_model")
save_fig = True
data_files = [f for f in os.listdir('Results/H2 price sensitivity') if not f.endswith('.csv')]
data = {}

for file in data_files:
    file_path = os.path.join('Results/H2 price sensitivity', file)
    if os.path.isfile(file_path):
        data[file] = np.loadtxt(file_path)

# Plotting

x_values = np.arange(2, 7)
ratios_test = []
ratios_train = []
discrepancies = []

fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

for i in x_values:
    name_HAPD_test = "HAPD_test_value_{}€".format(i)
    name_HAPD_train = "HAPD_train_value_{}€".format(i)
    name_hindsight_test = "hindsight_value_test_{}€".format(i)
    name_hindsight_train = "hindsight_value_train_{}€".format(i)
    
    ratio_test = data[name_HAPD_test] / data[name_hindsight_test]
    ratio_train = data[name_HAPD_train] / data[name_hindsight_train]
    discrepancy = np.abs(ratio_test - ratio_train)
    
    ratios_test.append(ratio_test)
    ratios_train.append(ratio_train)
    discrepancies.append(discrepancy)
    
    ax1.bar(i - 0.15, ratio_test, color=c[0], width=0.3, label='Test' if i == x_values[0] else "")
    ax1.bar(i + 0.15, ratio_train, color=c[1], width=0.3, label='Train' if i == x_values[0] else "")

# Create a secondary y-axis for the discrepancy line plot
ax2 = ax1.twinx()
ax2.plot(x_values, discrepancies, color='k', marker='o', linestyle='-', linewidth=1, markersize=3, label='Discrepancy')

# Set labels and limits
ax1.set_ylabel('Profit Model / Profit Hindsight', labelpad=0)
ax1.set_xlabel(r'H$_2$ price [€/kg]',labelpad=0)
ax1.set_ylim(0, 1)
ax1.set_xticks([2, 3, 3, 4, 5, 6])
ax2.set_ylim(0, 0.5)
ax2.set_ylabel('Discrepancy')
ax2.yaxis.set_label_coords(1.13, 0.2)
ax2.set_yticks([0, 0.1, 0.2])

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=False, ncol=2,bbox_to_anchor=(0, 1.05),columnspacing=0.5)

plt.tight_layout()

if save_fig:
    plt.savefig('Plots/H2_prices_bar.png', dpi=300, bbox_inches='tight')

print("ratio_test: ", ratios_test)
print("discrepancy: ", discrepancies)