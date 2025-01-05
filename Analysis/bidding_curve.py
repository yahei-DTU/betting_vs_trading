import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator

mpl.use('pgf')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 5,
    "legend.title_fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})
c = list(colors.TABLEAU_COLORS)

cwd = os.path.dirname(os.path.dirname(__file__))
os.chdir(cwd)

def plot_bidding(x,y):
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax1.axvline(x=0, color='k', linestyle='--',lw=0.5)
    ax1.axvline(x=-10, color='k', linestyle='--',lw=0.5)
    ax1.axvline(x=10, color='k', linestyle='--',lw=0.5)
    ax1.plot(x, y,"^",color="gray",lw=0.1,markersize=3)
    ax1.step(x, y, where='post', color=c[1], label='Bidding Curve')
    ax1.text(5,45,"Selling",verticalalignment='center',horizontalalignment='center')
    ax1.text(-5,45,"Buying",verticalalignment='center',horizontalalignment='center')
    ax1.set_ylabel('Day-Ahead Price Bid [€/MWh]')
    ax1.set_xlabel('Day-Ahead Power Bid [MW]')
    ax1.set_ylim([0,50])
    ax1.set_xlim([-11.5,11])
    names = ["A","B","C","D","E"]
    for i, y_val in enumerate(reversed([5, 15, 25, 35, 45])):
        #ax1.scatter(-11.5, y_val, color=c[0], marker='x',zorder=5, s=30)
        ax1.text(-11.2, y_val, f'{names[i]}',color = c[0] ,fontweight="bold", verticalalignment='center', horizontalalignment='left')
    ax1.text(-13,-10,"(b)",fontweight="bold", verticalalignment='center', horizontalalignment='left')
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))  # 5 minor ticks per major tick interval
    ax1.yaxis.set_major_locator(plt.MultipleLocator(10))  # Major ticks every 10
    ax1.tick_params(axis='y', which='minor', direction='inout',color = c[0],length=6)  # Customize tick appearance
x = np.array([-10,-10,-10,-10,-10,-10,-5,-5,-5,-5,-5,0,0,0,0,0,5,5,5,5,5,10,10,10,10,10])
y = np.arange(0,51,2)

def plot_feasibility(x,y):
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax1.axvline(x=0, color='k', linestyle='--',lw=0.5)
    ax1.axvline(x=-10, color='k', linestyle='--',lw=0.5)
    ax1.axvline(x=10, color='k', linestyle='--',lw=0.5)
    x_orig = x.copy()
    for i in range(1,len(x)):
        if x[i]<x[i-1]:
            x[i] = x[i-1]
    ax1.step(np.minimum(np.maximum(-10,x),10)-0.15, y, where='post', color=c[0], linestyle='-',label=r'Restoration for $\mathcal{T}$',alpha=0.8)
    ax1.step(np.minimum(np.maximum(0,x),10), y, where='post', color=c[2], linestyle='-', label=r'Restoration for $\mathcal{T}^{\rm{res}}$',alpha=0.8)
    x1 = np.zeros(len(y))
    for i in range(len(y)):
        if y[i] > 20:
            x1[i] = np.minimum(np.maximum(0,x[i]),10)
        else:
            x1[i] = np.minimum(np.maximum(-10,x[i]),10)
    ax1.step(x1+0.15, y, where='post', color=c[1], linestyle='-', label=r'Restoration for $\mathcal{T}^{\rm{cond}}$',alpha=0.8)    
    ax1.step(x_orig, y, where='post', color="black", label='Original bidding curve',lw=0.5)
    ax1.text(5,55,"Selling",verticalalignment='center',horizontalalignment='center')
    ax1.text(-5,55,"Buying",verticalalignment='center',horizontalalignment='center')
    ax1.text(-18,-10,"(c)",fontweight="bold", verticalalignment='center', horizontalalignment='left')
    ax1.set_ylabel('Day-Ahead Price Bid [€/MWh]')
    ax1.set_xlabel('Day-Ahead Power Bid [MW]')
    ax1.set_ylim([0,60])
    ax1.set_xlim([-16,16])
    #ax1.yaxis.set_major_locator(plt.MultipleLocator(10))  # Major ticks every 10
    ax1.set_yticks([20])
    ax1.set_xticks([-10,0,10],labels=[r"$P^{\rm{buy}^{\rm{max}}}$","0",r"$P^{\rm{sell}^{\rm{max}}}$"])
    ax1.legend(framealpha=1)

def q_constraints_plot():
    x1 = np.linspace(0,5,100)
    x2 = np.linspace(5,10,100)
    y = lambda x,a,b: a*x+b
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax1.plot(x1, y(x1, 1, 5), color=c[1], lw=1)
    ax1.plot(x2, y(x2, 0.5, 12.5), color=c[1], lw=1)
    ax1.plot(x2, y(x2, 0.5, 1), color=c[1], lw=1)
    ax1.plot(x2, y(x2, -0.5, 15), color=c[1], lw=1)
    ax1.text(2.5, y(2.5, 1, 5) + 0.5, '1', color="k", verticalalignment='bottom', horizontalalignment='center')
    ax1.text(8, y(8, 0.5, 12.5) + 0.5, '2', color="k", verticalalignment='bottom', horizontalalignment='center')
    ax1.text(8, y(8, 0.5, 12.5) - 1.5, 'accepted', color="k", verticalalignment='bottom', horizontalalignment='center', fontsize=8)
    ax1.text(8, y(8, -0.5, 15) + 0.5, '3', color="k", verticalalignment='bottom', horizontalalignment='center')
    ax1.text(8, y(8, -0.5, 15) - 2, 'eliminated by (12a)', color="k", verticalalignment='bottom', horizontalalignment='center', fontsize=8)
    ax1.text(8, y(8, 0.5, 1) + 0.5, '4', color="k", verticalalignment='bottom', horizontalalignment='center')
    ax1.text(8, y(8, 0.5, 1) - 2.5, 'eliminated by (12b)', color="k", verticalalignment='bottom', horizontalalignment='center', fontsize=8)
    ax1.vlines(5,0,20,linestyles='dashed',color="k",lw=0.5)
    ax1.set_xlim([0,10])
    ax1.set_ylim([0,20])
    ax1.set_yticks([])
    ax1.set_xticks([5],labels=[r"$\lambda_k$"])
    ax1.set_xlabel('Day-Ahead Price [€/MWh]')
    ax1.set_ylabel('Day-Ahead Power Bid [MW]')
    #ax1.text(7.5,-1.5,r"${q}_{j,k+1}^{\rm{DA}}$",verticalalignment='center',horizontalalignment='center')
    #ax1.text(2.5,-1.5,r"${q}_{j,k}^{\rm{DA}}$",verticalalignment='center',horizontalalignment='center')
    ax1.text(7.5,-1.5,r"Price domain $k+1$",verticalalignment='center',horizontalalignment='center')
    ax1.text(2.5,-1.5,r"Price domain $k$",verticalalignment='center',horizontalalignment='center')
    ax1.plot([7.25, 7.75], [4, 5.5], color='r', linestyle='-', lw=2)
    ax1.plot([7.25, 7.75], [5.5, 4], color='r', linestyle='-', lw=2)
    ax1.plot([7.25, 7.75], [10.5, 12], color='r', linestyle='-', lw=2)
    ax1.plot([7.25, 7.75], [12, 10.5], color='r', linestyle='-', lw=2)
    ax1.text(-0.5,-4,"(a)",fontweight="bold", verticalalignment='center', horizontalalignment='left')

def q_constraints_plot1():
    x1 = np.linspace(0,5,100)
    x2 = np.linspace(5,10,100)
    y = lambda x,a,b: a*x+b
    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax1.plot(y(x1, 1, 5),x1, color=c[1], lw=1)
    ax1.plot(y(x2, 0.5, 12.5),x2, color=c[1], lw=1)
    ax1.plot(y(x2, 0.5, 1),x2, color=c[1], lw=1)
    ax1.plot(y(x2, -0.5, 15),x2, color=c[1], lw=1)
    #ax1.text(2.5, y(2.5, 1, 5) + 0.5, '1', color=c[1], verticalalignment='bottom', horizontalalignment='center')
    #ax1.text(8, y(8, 0.5, 12.5) + 0.5, '2', color=c[1], verticalalignment='bottom', horizontalalignment='center')
    #ax1.text(8, y(8, -0.5, 15) + 0.5, '3', color=c[1], verticalalignment='bottom', horizontalalignment='center')
    #ax1.text(8, y(8, 0.5, 1) + 0.5, '4', color=c[1], verticalalignment='bottom', horizontalalignment='center')
    ax1.hlines(5,0,20,linestyles='dashed',color="k",lw=0.5)
    ax1.set_ylim([0,10])
    ax1.set_xlim([0,20])
    ax1.set_xticks([])
    ax1.set_yticks([5],labels=[r"$\lambda_k$"])
    ax1.set_ylabel('Day-Ahead Price [€/MWh]')
    ax1.set_xlabel('Day-Ahead Power Bid [MW]')
    #ax1.text(7.5,-1.5,r"${q}_{j,k+1}^{\rm{DA}}$",verticalalignment='center',horizontalalignment='center')
    #ax1.text(2.5,-1.5,r"${q}_{j,k}^{\rm{DA}}$",verticalalignment='center',horizontalalignment='center')
    #ax1.text(7.5,-1.5,r"$k+1$",verticalalignment='center',horizontalalignment='center')
    #ax1.text(2.5,-1.5,r"$k$",verticalalignment='center',horizontalalignment='center')
    #ax1.plot([7.25, 7.75], [4, 5.5], color='r', linestyle='-', lw=2)
    #ax1.plot([7.25, 7.75], [5.5, 4], color='r', linestyle='-', lw=2)
    #ax1.plot([7.25, 7.75], [10.5, 12], color='r', linestyle='-', lw=2)
    #ax1.plot([7.25, 7.75], [12, 10.5], color='r', linestyle='-', lw=2)
    #ax1.text(-0.5,-4,"(a)",fontweight="bold", verticalalignment='center', horizontalalignment='left')


# bidding plot
x = np.array([-10,-10,-10,-10,-10,-10,-5,-5,-5,-5,-5,0,0,0,0,5,5,5,5,5,10,10,10,10,10,10])
y = np.arange(0,51,2)
plot_bidding(x,y)
plt.tight_layout()
plt.savefig('Plots/bidding_curve.png', dpi=300, bbox_inches='tight')

# feasibility plot
x = np.array([-15,-15,-15,-10,-10,-10,-10,-10,-5,-5,-5,-5,-5,-5,-5,-5,0,0,0,-1,-1,-1,-1,0,0,0,10,10,15,15,15])
y = np.arange(0,61,2)
plot_feasibility(x,y)
plt.tight_layout()
plt.savefig('Plots/feasibility_curve.png', dpi=300, bbox_inches='tight')

# q constraints plot
q_constraints_plot()
plt.tight_layout()
plt.savefig('Plots/q_constraints.png', dpi=300, bbox_inches='tight')