import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import colors, patches
import matplotlib as mpl

mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
c= ["c","y","m"]

cwd = os.path.dirname(os.path.dirname(__file__))
os.chdir(cwd)
save_fig = True
data_files = [f for f in os.listdir('Results/Risk sensitivity') if f.endswith('.csv')]
data = {}

for file in data_files:
    file_path = os.path.join('Results/Risk sensitivity', file)
    if os.path.isfile(file_path):
        data[file[:-4]] = pd.read_csv(file_path)

# Plotting

for i, risk in enumerate([100, 30]):
    keys_HAPD = [f for f in data.keys() if f.startswith(f'p_HAPD_{risk}')]
    keys_HAPD.reverse()
    keys_hindsight = [f for f in data.keys() if f.startswith(f'p_hindsight_{risk}')]
    keys_hindsight.reverse()

    fig_HAPD, ax_HAPD = plt.subplots(figsize=(3.5, 2.5))
    fig_hindsight, ax_hindsight = plt.subplots(figsize=(3.5, 2.5))

    legend_handles_HAPD = []
    legend_handles_hindsight = []

    for j, key in enumerate(keys_HAPD):
        df = data[key]
        # Calculate histogram data
        counts, bin_edges = np.histogram(df['p_DA'], bins=30)
        # Apply a small shift for i == 100
        if risk == 100:
            shift = j * 0.1  # Adjust the shift value as needed
            bin_edges = bin_edges + shift
            # Plot step histogram
            ax_HAPD.hist(df['p_DA']+j*0.1, bins=bin_edges, color=c[j], edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}", histtype='step')
        else:
            ax_HAPD.hist(df['p_DA'], bins=bin_edges, color=c[j], edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}", histtype='step')
        # Fill area beneath the step histogram
        counts = np.append(counts, counts[-1])
        ax_HAPD.fill_between(bin_edges, counts, step="post", color=colors.to_rgba(c[j], alpha=0.1))
        # Create custom legend handle with increased edge size
        legend_handles_HAPD.append(patches.Patch(facecolor=colors.to_rgba(c[j], alpha=0.1), edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}"))
    for j, key in enumerate(keys_hindsight):
        df = data[key]
        # Calculate histogram data
        counts, bin_edges = np.histogram(df['p_DA'], bins=30)        
        if risk == 100:
            shift = j * 0.1  # Adjust the shift value as needed
            bin_edges = bin_edges + shift
            # Plot step histogram
            ax_hindsight.hist(df['p_DA']+j*0.1, bins=bin_edges, color=c[j], edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}", histtype='step')
        else:
            ax_hindsight.hist(df['p_DA'], bins=bin_edges, color=c[j], edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}", histtype='step')
        # Fill area beneath the step histogram
        counts = np.append(counts, counts[-1])
        ax_hindsight.fill_between(bin_edges, counts, step="post", color=colors.to_rgba(c[j], alpha=0.1))
        # Create custom legend handle with increased edge size
        legend_handles_hindsight.append(patches.Patch(facecolor=colors.to_rgba(c[j], alpha=0.1), edgecolor=colors.to_rgba(c[j], alpha=0.8), linewidth=1, label=f"{key.split('_')[-1]}"))

    for axis in [ax_HAPD, ax_hindsight]:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.set_yticks([])
        axis.set_xticks([-10, -5, 0, 5, 10])  # Set x-axis ticks
        
    
    ax_HAPD.set_xlabel('Power Bid [MW]')
    #ax_HAPD.set_xlabel(r'Power Bid [MW] ($\mathcal{T_{\rm{CVaR}}}$)')
    ax_hindsight.set_xlabel('Power Bid [MW]')
    #ax_hindsight.text(0.15, 0.95, r"(c) Betting $\mathcal{B}$ Hindsight", verticalalignment='top', horizontalalignment='left', transform=ax_hindsight.transAxes)
    #ax_HAPD.text(0.15, 0.95, r"(a) Betting $\mathcal{B}$", verticalalignment='top', horizontalalignment='left', transform=ax_HAPD.transAxes)

    # Move legend to be between the subplots and remove the box
    # Combine legends from both axes
    combined_handles, combined_labels = ax_HAPD.get_legend_handles_labels()
    fig_hindsight.legend(combined_handles, combined_labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), title=r"$\lambda^{\rm{H}}$(€/kg)", frameon=False)
    fig_HAPD.tight_layout()
    fig_hindsight.tight_layout()

    if save_fig:
        fig_HAPD.savefig(f'Plots/histogram_HAPD_risk_{risk}.png', dpi=300, bbox_inches='tight')
        fig_hindsight.savefig(f'Plots/histogram_hindsight_risk_{risk}.png', dpi=300, bbox_inches='tight')




    





