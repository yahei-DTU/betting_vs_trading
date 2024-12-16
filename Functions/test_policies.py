import numpy as np
import pandas as pd
from Functions import electrolyzer_efficiency
import matplotlib.pyplot as plt
import gurobipy as gp
from matplotlib import colors
from gurobipy import GRB
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
c = list(colors.TABLEAU_COLORS)
markers = ["^","o","s"]

def test_policies(q_DA, q_H, data,x,model_param,price_quantile):
    """
    test q-policies on testing data
    --------------------------------
    param: (arraylike) q_DA: trained q-policy for day-ahead trading
    param: (arraylike) q_H: trained q-policy for electrolyzer
    param: (df) data: testing data
    param: (df) x: feature vector of testing data
    param: (dict) model_param: model parameters
    param: (float) price_quantile: price quantiles
    return (float) value: objective value for testing period, (arraylike) p_DA: day-ahead bid, (arraylike) p_H: electrolyzer production
    """
    lambda_max = 300 # max lambda for bidding curve
    lambda_step = 2  # step size of bids
    bid_lambda_DA = np.arange(0,lambda_max,lambda_step) # Discretization for bidding curve, actual bids follow from curve
    bid_p_DA = np.zeros((len(data),lambda_max))
    bid_p_H = np.zeros((len(data),lambda_max))
    p_DA = np.zeros(len(data))
    p_H = np.zeros(len(data))
    p_IM = np.zeros(len(data))
    h_H = np.zeros(len(data))
    s1 = np.zeros(len(data))
    s2 = np.zeros(len(data))
    p_DA_final = np.zeros(len(data))
    p_H_final = np.zeros(len(data))
    s = np.zeros(int(len(p_DA)/24))
    n_features = len(x.columns)
    lambda_DA_RE = data["lambda_DA_RE"].to_numpy()
    lambda_IM = data["lambda_IM"].to_numpy()
    E_FC = data["E_FC"].to_numpy()
    E_RE = data["E_RE"].to_numpy()
    S = np.array([t for t in range(0, model_param['N_s'])])


    T_op = 90
    Pr = 30
    n_price_domains = len(price_quantile)+1

    for t in range(len(data)):
        h = t%24
        for k in range(len(bid_lambda_DA)):
            p_domain = 0
            for i in range(n_price_domains-1):
                if bid_lambda_DA[k] >= price_quantile[i]:
                    p_domain = i+1
            a_DA = q_DA[h:h+1,0:1,p_domain:p_domain+1].reshape((1,1))[0][0]
            b_DA = (q_DA[h:h+1,1:,p_domain:p_domain+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            a_H = q_H[h:h+1,0:1,p_domain:p_domain+1].reshape((1,1))[0][0]
            b_H = (q_H[h:h+1,1:,p_domain:p_domain+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            if bid_lambda_DA[k] > lambda_DA_RE[t]:
                p_DA[t] = bid_p_DA[t,k-1]
                p_H[t] = bid_p_H[t,k-1]
                break

            if model_param["model_type"] == 1:
                bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),-model_param["max_electrolyzer_capacity"])
            elif model_param["model_type"] == 2:
                bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),0)
            elif model_param["model_type"] == 3:
                if lambda_DA_RE[t] > 20: # if price higher than 20, only sell and correct buying to 0
                    bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),0)
                else:
                    bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),-model_param["max_electrolyzer_capacity"])
            bid_p_H[t,k] = np.maximum(np.minimum(a_H * bid_lambda_DA[k] + b_H,model_param["max_electrolyzer_capacity"]),model_param["p_min"]*model_param["max_electrolyzer_capacity"])
            # Feasbility for q-policies at price domain boundary
            if k>0 and bid_p_DA[t,k] < bid_p_DA[t,k-1]:
                bid_p_DA[t,k] = bid_p_DA[t,k-1]
            if k>0 and bid_p_DA[t,k] - bid_p_DA[t,k-1]<-0.01:
                print("Bidding curve not increasing at time:",t)
                print("bid_lambda_DA[k]",bid_lambda_DA[k], "bid_p_DA[t,k]",bid_p_DA[t,k], "bid_p_DA[t,k-1]",bid_p_DA[t,k-1])
        if h==-1: # set to 23 for daily optimization to fullfill cosntraints
            E_FC_day = np.sum(E_FC[t-h:t+1])
            model = gp.Model("test_policies")
            # Variables
            p_DA_gp = model.addMVar(shape=24,lb=-model_param["max_electrolyzer_capacity"],ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="Day-ahead market bid")
            p_H_gp = model.addMVar(shape=24,lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="Electrolyzer consumption DA in MW")
            p_IM_gp = model.addMVar(shape=24,lb=-float('inf'),vtype=GRB.CONTINUOUS, name="over-under-production")
            h_H_gp = model.addMVar(shape=24,lb=0, vtype=GRB.CONTINUOUS,name="Hydrogen production DA in kg")
            p_IM_abs_gp = model.addMVar(shape=24,lb=0 ,vtype=GRB.CONTINUOUS, name="absolut values of p_IM")

            # Model objective
            quad_expr = gp.QuadExpr()  # Initialize a quadratic expression

            # Loop over each period to add the quadratic terms
            for i in range(24):
                quad_expr += (p_DA[t-h+i] - p_DA_gp[i]) * (p_DA[t-h+i] - p_DA_gp[i])
                quad_expr += (p_H[t-h+i] - p_H_gp[t]) * (h_H[t-h+i] - p_H_gp[t])

            # Set the objective to minimize the quadratic expression
            model.setObjective(quad_expr, GRB.MINIMIZE)
            
            # Model Constraints
            model.addConstr(E_FC_day >= p_DA_gp + p_H_gp + p_IM_gp,name="c0")
            model.addConstrs(gp.quicksum(h_H_gp) >= model_param["min_production"],name="c1")
            model.addConstr(p_H_gp >= model_param["p_min"] * model_param["max_electrolyzer_capacity"],name="c2")
            model.addConstr(p_H_gp <= model_param["max_electrolyzer_capacity"],name="c3")          
            for i in range(24):
                model.addConstrs(((h_H[i] <= (p_H_gp[t]*model_param["a"][s] + model_param["b"][s]) * model_param["eta_storage"]) for s in S),name="c4")
            # Constraint on p_IM
            model.addConstr(p_IM_abs_gp >= p_IM, f"AbsPos_{t}",name="c13")
            model.addConstr(p_IM_abs_gp >= -p_IM, f"AbsNeg_{t}",name="c14")
            model.addConstr(gp.quicksum(p_IM_abs_gp) <= model_param["balance_limit"]*24,name="c15")

            # Solve model    
            model.optimize()
            model.write("Functions/test_policies.lp")

            # write results
            p_DA_final[t-h:t+1] = p_DA_gp.X
            p_H_final[t-h:t+1] = p_H_gp.X
            p_IM[t-h:t+1] = E_RE[t-h:t+1] - p_DA_gp.X - p_H_gp.X
            h_H[t-h:t+1] = h_H_gp.X

            if np.sum(h_H[t-h:t+1]) < model_param["min_production"]:
                print("Minimum production not reached at day:",(t-h)/24)
        p_IM[t] = E_RE[t] - p_DA[t] - p_H[t]
        if model_param["model_type"] == 2: 
            if p_IM[t] < 0:
                s1[t] = -p_IM[t]
                p_IM[t] = 0
        if model_param["model_type"] == 3:
            if lambda_DA_RE[t] > 20:
                if p_IM[t] < 0:
                    s1[t] = -p_IM[t]
                    p_IM[t] = 0 
        AA = electrolyzer_efficiency.area(model_param)
        i = electrolyzer_efficiency.find_i_from_p(np.array([p_H[t]]),AA,T_op)
        h_H[t] = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]
    # Slackness for Hydrogen requirement
    for d in range(len(s)):
        h_produced = np.sum(h_H[d*24:(d+1)*24])
        h_offset = h_produced - model_param["min_production"]
        if h_offset < 0:
            s[d] = -h_offset
            s2[d*24] = -h_offset
            print("Day {} missing {} kg_H2".format(d,-h_offset))
    
    value = np.sum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - 100*s1-100*s2)

    df_results = pd.DataFrame(data=zip(p_DA,p_H,h_H,p_IM,s1,s2,E_RE,E_FC,lambda_DA_RE,lambda_IM,lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM), columns = np.array(["p_DA","p_H","h_H","p_IM","s1","s2","E_RE","E_FC","lambda_DA_RE","lambda_IM","revenue"]))

    return value, df_results


def plot_bidding(q_DA,q_H, data, x, model_param,price_quantile):
    """
    plot bidding curves for specific hour
    --------------------------------
    param: (arraylike) q_DA: trained q-policy for day-ahead trading
    param: (arraylike) q_H: trained q-policy for electrolyzer
    param: (df) data: testing data
    param: (df) x: feature vector of testing data
    param: (dict) model_param: model parameters
    param: (float) price_quantile: price quantiles
    return (float) value: objective value for testing period, (arraylike) p_DA: day-ahead bid, (arraylike) p_H: electrolyzer production
    """
    lambda_max = 300 # max lambda for bidding curve
    lambda_step = 2  # step size of bids
    bid_lambda_DA = np.arange(0,lambda_max,lambda_step) # Price bids
    bid_p_DA = np.zeros((len(data),len(bid_lambda_DA)))
    bid_p_H = np.zeros((len(data),len(bid_lambda_DA)))
    p_DA = np.zeros(len(data))
    p_H = np.zeros(len(data))
    p_IM = np.zeros(len(data))
    h_H = np.zeros(len(data))
    p_DA_final = np.zeros(len(data))
    p_H_final = np.zeros(len(data))
    s = np.zeros(int(len(p_DA)/24))
    n_features = len(x.columns)
    lambda_DA_RE = data["lambda_DA_RE"].to_numpy()
    lambda_IM = data["lambda_IM"].to_numpy()
    E_FC = data["E_FC"].to_numpy()
    E_RE = data["E_RE"].to_numpy()
    S = np.array([t for t in range(0, model_param['N_s'])])

    n_price_domains = len(price_quantile)+1
    hours = [8346]

    fig1, ax1 = plt.subplots(1, 1, figsize=(3.5,2.5))
    ax1.axvline(x=0, color='k', linestyle='--',lw=0.5)
    #ax1.axvline(x=-10, color='k', linestyle='--',lw=0.5)
    #ax1.axvline(x=10, color='k', linestyle='--',lw=0.5)
    #ax1.text(2.5,10,"Selling",fontsize=10,verticalalignment='center',horizontalalignment='center')
    #ax1.text(-2.5,10,"Buying",fontsize=10,verticalalignment='center',horizontalalignment='center')
    ax1.set_ylabel('Day-Ahead Price Bid')
    #ax1.set_xlabel('Day-Ahead Power Bid')
    ax1.set_ylim([0,300])
    ax1.set_xlim([-11,11])
    ax1.set_xticks([])
    #ax1.set_ylabel('')

    fig2, ax2 = plt.subplots(1, 1, figsize=(3.5,2.5))
    ax2.axvline(x=0, color='k', linestyle='--',lw=0.5)
    ax2.axvline(x=10, color='k', linestyle='--',lw=0.5)
    ax2.set_ylabel('Day-Ahead Price Bid')
    ax2.set_xlabel('Scheduled Électrolyzer Production')
    ax2.set_ylim([0,300])
    ax2.set_xlim([0,11])

    for i in range(len(hours)):
        marker = markers[i]
        colors = c[i+1]
        #name = names[i]
        t = hours[i]
        h = t%24
        for k in range(len(bid_lambda_DA)):
            p_domain = 0
            for i in range(n_price_domains-1):
                if bid_lambda_DA[k] >= price_quantile[i]:
                    p_domain = i+1
            a_DA = q_DA[h:h+1,0:1,p_domain:p_domain+1].reshape((1,1))[0][0]
            b_DA = (q_DA[h:h+1,1:,p_domain:p_domain+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            a_H = q_H[h:h+1,0:1,p_domain:p_domain+1].reshape((1,1))[0][0]
            b_H = (q_H[h:h+1,1:,p_domain:p_domain+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            if model_param["model_type"] == 1:
                bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),-model_param["max_electrolyzer_capacity"])
            elif model_param["model_type"] == 2:
                bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),0)
            elif model_param["model_type"] == 3:
                if bid_lambda_DA[k] <= 20:
                    bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),-model_param["max_electrolyzer_capacity"])
                else:
                    bid_p_DA[t,k] = np.maximum(np.minimum(a_DA * bid_lambda_DA[k] + b_DA,model_param["max_wind_capacity"]),0)
            bid_p_H[t,k] = np.maximum(np.minimum(a_H * bid_lambda_DA[k] + b_H,model_param["max_electrolyzer_capacity"]),model_param["p_min"]*model_param["max_electrolyzer_capacity"])
            # Feasbility for q-policies at price domain boundary
            if k>0 and bid_p_DA[t,k] < bid_p_DA[t,k-1]:
                bid_p_DA[t,k] = bid_p_DA[t,k-1]
            if k>0 and bid_p_DA[t,k] - bid_p_DA[t,k-1]<-0.01:
                print("Bidding curve not increasing at lambda=",bid_lambda_DA[k])
                print("p_DA_k",bid_p_DA[t,k], "p_DA_k-1",bid_p_DA[t,k-1])
        # plot bidding curve
        ax1.plot(bid_p_DA[t,:],bid_lambda_DA,marker,color="gray",lw=0.5,markersize=1)
        ax1.step(bid_p_DA[t,:],bid_lambda_DA,where="post",color= colors)
        #ax1.axvline(x=lambda_DA_RE[t], color= colors, linestyle='--',lw=0.5)


        ax2.plot(bid_p_H[t,:],bid_lambda_DA,marker,color="gray",lw=0.5)
        ax2.step(bid_p_H[t,:],bid_lambda_DA,where="mid",color= colors)
        #ax2.axvline(x=lambda_DA_RE[t], color=colors, linestyle='--',lw=0.5)
    
    #ax1.legend(loc="best")
    #ax2.legend(loc="best")
    fig1.savefig("Plots/bidding_curve_DA_{}_{}.png".format(model_param["model_type"],model_param["risk_model"]))
    fig2.savefig("Plots/bidding_curve_H_{}_{}.png".format(model_param["model_type"],model_param["risk_model"]))
