import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Functions import electrolyzer_efficiency


def HAPD1(data,model_param,x,price_quantile):
    """
    model (buying and selling) with linear policies
    --------------------------------
    param: (df) data: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    param: (df) x: dataframe conataining feature values
    param: (arraylike) price_quantile: list of price quantiles 
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data["lambda_DA_RE"].to_numpy()
    lambda_IM = data["lambda_IM"].to_numpy()
    E_RE = data["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    n_features = len(x.columns)
    n_price_domains = len(price_quantile)+1
    S = np.array([t for t in range(0, model_param['N_s'])])

    if (len(periods) % 24 != 0):
        print("Error: Training length must be multiple of 24h!")
        #return

    # Model
    model = gp.Model("day_ahead_HAPD")

    #model.Params.TimeLimit = 300 


    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=-model_param["max_electrolyzer_capacity"],ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="p_DA")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="h_H")
    p_s =  model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="p_s")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="p_H")
    p_IM = model.addMVar(shape=len(periods),lb=-float('inf'),vtype=GRB.CONTINUOUS, name="p_IM")
    p_IM_abs = model.addMVar(shape=len(periods),lb=0 ,vtype=GRB.CONTINUOUS, name="p_IM_abs")
    z_su =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_su")
    z_on =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_on")
    z_off =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_off")
    z_sb =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_sb")
    q_DA = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_DA")
    q_H = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_H")
    VaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="VaR")
    CVaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
    xi = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS, name="xi")
    u = model.addMVar(shape=len(periods),lb=0, ub=1, vtype=GRB.BINARY, name="u")

    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - z_su*model_param["K_su"]*model_param["max_electrolyzer_capacity"]), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H,name="c0")
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"],name="c1")
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_s[t]*model_param["a"][s] + z_on[t]*model_param["b"][s]) * model_param["eta_storage"]) for s in S),name="c3")
    model.addConstr(p_s >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on,name="c4")
    model.addConstr(p_s <= model_param["max_electrolyzer_capacity"] * z_on,name="c5")
    model.addConstr(p_H == p_s + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c6")
    # Electrolyzer state
    model.addConstr(z_off + z_on + z_sb == 1,name="c7")
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["max_electrolyzer_capacity"]*model_param["p_sb"],name="c8")
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c9")
    model.addConstr(z_on == 1,name="c10") # Remove this line if electrolyzer should have states
    for t in periods:
        if t == periods[0]:
            model.addConstr(z_su[t] == 0,"c11")
        else:
            model.addConstr(z_su[t] >= z_off[t-1] + z_on[t] + z_sb[t] - 1,name="c12")
    model.addConstr(p_IM_abs >= p_IM, f"AbsPos_{t}",name="c13")
    model.addConstr(p_IM_abs >= -p_IM, f"AbsNeg_{t}",name="c14")
    if model_param["risk_model"] == 1:
        for d in days:
            model.addConstr(gp.quicksum(p_IM_abs[t] for t in range(int(24*d),int(24*(d+1)))) <= model_param["p_IM_abs_mean"]*24,name="c15")
    elif model_param["risk_model"] == 2:
        model.addConstr(CVaR == VaR + (1/(len(periods)*(1-model_param["alpha"])))*gp.quicksum(xi[t] for t in periods),name="c16")
        model.addConstr(xi >= p_IM_abs - VaR,name="c17")
        model.addConstr(CVaR <= model_param["CVaR_95"],name="c18")
    elif model_param["risk_model"] == 3:
        model.addConstr(p_IM_abs <= model_param["p_IM_abs_max"],name="c19")
    else:
        for t in periods:
            model.addConstr(p_IM_abs[t] == u[t]*p_IM[t]-(1-u[t])*p_IM[t],name="c20")
        print("No risk model implemented!")

    # Policy constraints
    print("Price Domains:", price_quantile)
    for t in periods:
        h = t%24
        p_domain = 0
        for i in range(n_price_domains-1):
            if lambda_DA_RE[t] >= price_quantile[i]:
                p_domain = i+1
            a1 = q_DA[h:h+1,0:1,i:i+1].reshape((1,1))[0][0]
            b1 = (q_DA[h:h+1,1:,i:i+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            a2 = q_DA[h:h+1,0:1,i+1:i+2].reshape((1,1))[0][0]
            b2 = (q_DA[h:h+1,1:,i+1:i+2].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            model.addConstr((a2-a1)*price_quantile[i]+b2-b1 >= 0,name="c16")
        model.addConstr(p_DA[t] == q_DA[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c17")
        model.addConstr(p_H[t] == q_H[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c18")
               
    model.addConstr(q_DA[:,0:1,:] >= 0,name="c19")

    # Solve model    
    model.optimize()
    model.write("Models/HAPD.lp")

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check
    print("Var: ", VaR.x)
    print("CVaR: ", CVaR.x)
    print("max p_IM_ABS: ", np.max(p_IM_abs.x))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,p_IM_abs.x,xi.x,h_H.x,E_RE,z_on.x,z_sb.x,z_off.x,z_su.x,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
    df_results.to_csv("{0}p_{1}.csv".format("Results/","HAPD_train"),index=False)

    return q_DA.x, q_H.x, value

def HAPD2(data,model_param,x,price_quantile):
    """
    model (only selling) with linear policies
    --------------------------------
    param: (df) data: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    param: (df) x: dataframe conataining feature values
    param: (arraylike) price_quantile: list of price quantiles 
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data["lambda_DA_RE"].to_numpy()
    lambda_IM = data["lambda_IM"].to_numpy()
    E_RE = data["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    n_features = len(x.columns)
    n_price_domains = len(price_quantile)+1
    S = np.array([t for t in range(0, model_param['N_s'])])

    # Constraint for s2
    T_op = 90
    Pr = 30
    AA = electrolyzer_efficiency.area(model_param)
    i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
    s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]

    if (len(periods) % 24 != 0):
        print("Error: Training length must be multiple of 24h!")
        #return

    # Model
    model = gp.Model("day_ahead_HAPD")

    #model.Params.TimeLimit = 300 


    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="p_DA")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="h_H")
    p_s =  model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="p_s")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="p_H")
    p_IM = model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="p_IM")
    p_IM_abs = model.addMVar(shape=len(periods),lb=0 ,vtype=GRB.CONTINUOUS, name="p_IM_abs")
    z_su =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_su")
    z_on =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_on")
    z_off =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_off")
    z_sb =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_sb")
    q_DA = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_DA")
    q_H = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_H")
    VaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="VaR")
    CVaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
    xi = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS, name="xi")
    u = model.addMVar(shape=len(periods),lb=0, ub=1, vtype=GRB.BINARY, name="u")
    s1 = model.addMVar(shape=len(periods),lb=0,ub=model_param["p_min"] * model_param["max_electrolyzer_capacity"],vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    s2 = model.addMVar(shape=len(periods),lb=0,ub = s2_max,vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")

    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - z_su*model_param["K_su"]*model_param["max_electrolyzer_capacity"]- 100*s1 - 100*s2), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H - s1,name="c0")
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] + s2[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"],name="c1")
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_s[t]*model_param["a"][s] + z_on[t]*model_param["b"][s]) * model_param["eta_storage"]) for s in S),name="c3")
    model.addConstr(p_s >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on,name="c4")
    model.addConstr(p_s <= model_param["max_electrolyzer_capacity"] * z_on,name="c5")
    model.addConstr(p_H == p_s + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c6")
    # Electrolyzer state
    model.addConstr(z_off + z_on + z_sb == 1,name="c7")
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["max_electrolyzer_capacity"]*model_param["p_sb"],name="c8")
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c9")
    model.addConstr(z_on == 1,name="c10") # Remove this line if electrolyzer should have states
    for t in periods:
        if t == periods[0]:
            model.addConstr(z_su[t] == 0,"c11")
        else:
            model.addConstr(z_su[t] >= z_off[t-1] + z_on[t] + z_sb[t] - 1,name="c12")
    model.addConstr(p_IM_abs >= p_IM, f"AbsPos_{t}",name="c13")
    model.addConstr(p_IM_abs >= -p_IM, f"AbsNeg_{t}",name="c14")
    if model_param["risk_model"] == 1:
        for d in days:
            model.addConstr(gp.quicksum(p_IM_abs[t] for t in range(int(24*d),int(24*(d+1)))) <= model_param["p_IM_abs_mean"]*24,name="c15")
    elif model_param["risk_model"] == 2:
        model.addConstr(CVaR == VaR + (1/(len(periods)*(1-model_param["alpha"])))*gp.quicksum(xi[t] for t in periods),name="c16")
        model.addConstr(xi >= p_IM_abs - VaR,name="c17")
        model.addConstr(CVaR <= model_param["CVaR_95"],name="c18")
    elif model_param["risk_model"] == 3:
        model.addConstr(p_IM_abs <= model_param["p_IM_abs_max"],name="c19")
    else:
        for t in periods:
            model.addConstr(p_IM_abs[t] == u[t]*p_IM[t]-(1-u[t])*p_IM[t],name="c20")
        print("No risk model implemented!")

    # Policy constraints
    print("Price Domains:", price_quantile)
    for t in periods:
        h = t%24
        p_domain = 0
        for i in range(n_price_domains-1):
            if lambda_DA_RE[t] >= price_quantile[i]:
                p_domain = i+1
            a1 = q_DA[h:h+1,0:1,i:i+1].reshape((1,1))[0][0]
            b1 = (q_DA[h:h+1,1:,i:i+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            a2 = q_DA[h:h+1,0:1,i+1:i+2].reshape((1,1))[0][0]
            b2 = (q_DA[h:h+1,1:,i+1:i+2].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            model.addConstr((a2-a1)*price_quantile[i]+b2-b1 >= 0,name="c16")
        model.addConstr(p_DA[t] == q_DA[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c17")
        model.addConstr(p_H[t] == q_H[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c18")
               
    model.addConstr(q_DA[:,0:1,:] >= 0,name="c19")



    # Solve model    
    model.optimize()
    model.write("Models/HAPD.lp")

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check
    print("Var: ", VaR.x)
    print("CVaR: ", CVaR.x)
    print("max p_IM_ABS: ", np.max(p_IM_abs.x))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,p_IM_abs.x,xi.x,h_H.x,E_RE,z_on.x,z_sb.x,z_off.x,z_su.x,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
    df_results.to_csv("{0}p_{1}.csv".format("Results/","HAPD_train"),index=False)

    return q_DA.x, q_H.x, value
    
def HAPD3(data,model_param,x,price_quantile):
    """
    model (conditional buying) with linear policies
    --------------------------------
    param: (df) data: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    param: (df) x: dataframe conataining feature values
    param: (arraylike) price_quantile: list of price quantiles 
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data["lambda_DA_RE"].to_numpy()
    lambda_IM = data["lambda_IM"].to_numpy()
    E_RE = data["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    n_features = len(x.columns)
    n_price_domains = len(price_quantile)+1
    S = np.array([t for t in range(0, model_param['N_s'])])
    lambda_green = 20
    M = np.max(lambda_DA_RE)
    eps = 0.0001

    # Constraint for s2
    T_op = 90
    Pr = 30
    AA = electrolyzer_efficiency.area(model_param)
    i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
    s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]

    if (len(periods) % 24 != 0):
        print("Error: Training length must be multiple of 24h!")
        #return

    # Model
    model = gp.Model("day_ahead_HAPD")

    #model.Params.TimeLimit = 300 


    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=-model_param["max_electrolyzer_capacity"],ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="p_DA")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="h_H")
    p_s =  model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="p_s")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="p_H")
    p_IM = model.addMVar(shape=len(periods),lb=-float('inf'),vtype=GRB.CONTINUOUS, name="p_IM")
    p_IM_abs = model.addMVar(shape=len(periods),lb=0 ,vtype=GRB.CONTINUOUS, name="p_IM_abs")
    z_su =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_su")
    z_on =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_on")
    z_off =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_off")
    z_sb =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="z_sb")
    q_DA = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_DA")
    q_H = model.addMVar(shape=(24,n_features,n_price_domains),lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_H")
    VaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="VaR")
    CVaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
    xi = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS, name="xi")
    u = model.addMVar(shape=len(periods),lb=0, ub=1, vtype=GRB.BINARY, name="u")
    s1 = model.addMVar(shape=len(periods),lb=0,ub=model_param["p_min"] * model_param["max_electrolyzer_capacity"],vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    s2 = model.addMVar(shape=len(periods),lb=0,ub = s2_max,vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    b = model.addMVar(shape=len(periods),lb=0,ub=1,vtype=GRB.BINARY, name="binary variable for buying")


    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - z_su*model_param["K_su"]*model_param["max_electrolyzer_capacity"]-100*s1 - 100*s2), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H - s1,name="c0")
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] + s2[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"],name="c1")
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_s[t]*model_param["a"][s] + z_on[t]*model_param["b"][s]) * model_param["eta_storage"]) for s in S),name="c3")
    model.addConstr(p_s >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on,name="c4")
    model.addConstr(p_s <= model_param["max_electrolyzer_capacity"] * z_on,name="c5")
    model.addConstr(p_H == p_s + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c6")
    # Electrolyzer state
    model.addConstr(z_off + z_on + z_sb == 1,name="c7")
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["max_electrolyzer_capacity"]*model_param["p_sb"],name="c8")
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"],name="c9")
    model.addConstr(z_on == 1,name="c10") # Remove this line if electrolyzer should have states
    for t in periods:
        if t == periods[0]:
            model.addConstr(z_su[t] == 0,"c11")
        else:
            model.addConstr(z_su[t] >= z_off[t-1] + z_on[t] + z_sb[t] - 1,name="c12")
    model.addConstr(p_IM_abs >= p_IM, f"AbsPos_{t}",name="c13")
    model.addConstr(p_IM_abs >= -p_IM, f"AbsNeg_{t}",name="c14")
    if model_param["risk_model"] == 1:
        for d in days:
            model.addConstr(gp.quicksum(p_IM_abs[t] for t in range(int(24*d),int(24*(d+1)))) <= model_param["p_IM_abs_mean"]*24,name="c15")
    elif model_param["risk_model"] == 2:
        model.addConstr(CVaR == VaR + (1/(len(periods)*(1-model_param["alpha"])))*gp.quicksum(xi[t] for t in periods),name="c16")
        model.addConstr(xi >= p_IM_abs - VaR,name="c17")
        model.addConstr(CVaR <= model_param["CVaR_95"],name="c18")
    elif model_param["risk_model"] == 3:
        model.addConstr(p_IM_abs <= model_param["p_IM_abs_max"],name="c19")
    else:
        for t in periods:
            model.addConstr(p_IM_abs[t] == u[t]*p_IM[t]-(1-u[t])*p_IM[t],name="c20")
        print("No risk model implemented!")
    # Conditional buying
    model.addConstr(b >= (lambda_green -  lambda_DA_RE) / M, name="c21")
    model.addConstr(b <= (lambda_green - lambda_DA_RE + M - eps) / M, name="c22")
    model.addConstr(p_IM >= (-model_param["max_wind_capacity"]-model_param["max_electrolyzer_capacity"])*b, name="c23")
    model.addConstr(p_DA >= -model_param["max_electrolyzer_capacity"]*b, name="c24")

    # Policy constraints
    print("Price Domains:", price_quantile)
    for t in periods:
        h = t%24
        p_domain = 0
        for i in range(n_price_domains-1):
            if lambda_DA_RE[t] >= price_quantile[i]:
                p_domain = i+1
            a1 = q_DA[h:h+1,0:1,i:i+1].reshape((1,1))[0][0]
            b1 = (q_DA[h:h+1,1:,i:i+1].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            a2 = q_DA[h:h+1,0:1,i+1:i+2].reshape((1,1))[0][0]
            b2 = (q_DA[h:h+1,1:,i+1:i+2].reshape((1,n_features-1))@np.array([x.iloc[t]])[:,1:].T)[0][0]
            model.addConstr((a2-a1)*price_quantile[i]+b2-b1 >= 0,name="c16")
        model.addConstr(p_DA[t] == q_DA[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c17")
        model.addConstr(p_H[t] == q_H[h:h+1,:,p_domain:p_domain+1].reshape((1,n_features))@np.array([x.iloc[t]]).T,name="c18")
               
    model.addConstr(q_DA[:,0:1,:] >= 0,name="c19")



    # Solve model    
    model.optimize()
    model.write("Models/HAPD.lp")

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check
    print("Var: ", VaR.x)
    print("CVaR: ", CVaR.x)
    print("max p_IM_ABS: ", np.max(p_IM_abs.x))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,p_IM_abs.x,b.x,s1.x,s2.x,xi.x,h_H.x,E_RE,z_on.x,z_sb.x,z_off.x,z_su.x,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","p_IM_abs","b","s1","s2","xi","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))
    df_results.to_csv("{0}p_{1}.csv".format("Results/","HAPD_train"),index=False)

    return q_DA.x, q_H.x, value