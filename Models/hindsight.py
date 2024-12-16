import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Functions import electrolyzer_efficiency


def hindsight_with_everything(data_test,model_param):
    """
    hindsight model with full information
    --------------------------------
    param: (df) data_test: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data_test["lambda_DA_RE"].to_numpy()
    lambda_IM = data_test["lambda_IM"].to_numpy()
    E_RE = data_test["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(len(periods)/24)
    S = np.array([t for t in range(0, model_param['N_s'])])

    # Model
    model = gp.Model("day_ahead_hindsight")

    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS, name="Day-ahead market bid")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="Hydrogen production DA in kg")
    p_H = model.addMVar(shape=len(periods),lb=0, ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="Electrolyzer consumption DA in MW")
    p_s =  model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="power consumption of electrolyzer in on state")
    p_IM = model.addMVar(shape=len(periods),lb= 0,vtype=GRB.CONTINUOUS, name="over-under-production")
    p_BS =  model.addMVar(shape=len(periods),lb=-model_param["bs_capacity"],ub=model_param["bs_capacity"],vtype=GRB.CONTINUOUS, name="power consumption of battery storage")    
    b_BS = model.addMVar(shape=len(periods),lb=0,ub=model_param["bs_capacity"],vtype=GRB.CONTINUOUS, name="battery storage charge level")
    z_su =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary start up cost")
    z_on =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer on")
    z_off =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer off")
    z_sb =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer standby")

    # Model objective

    model.setObjective(gp.quicksum(lambda_DA_RE[t]*p_DA[t] + model_param["lambda_H"]*h_H[t] + lambda_IM[t]*p_IM[t] - z_su[t]*model_param["K_su"]*model_param["max_electrolyzer_capacity"] for t in periods), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstrs((p_IM[t] == E_RE[t] - p_DA[t] - p_H[t] - p_BS[t]) for t in periods)
    # Day-ahead bidding constraint
    model.addConstrs((p_DA[t] <= model_param["max_wind_capacity"] - p_BS[t]) for t in periods)
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"])
        #model.addConstr(gp.quicksum(h_H[t] for t in range(int(24*d),int(24*(d+1)))) <= model_param["max_production"])
    # Hydrogen Production
    for t in periods:
        model.addConstrs((h_H[t] <= (p_s[t]*model_param["a"][s] + z_on[t]*model_param["b"][s]) * model_param["eta_storage"]) for s in S)
    model.addConstrs((p_s[t] >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on[t]) for t in periods)
    model.addConstrs((p_s[t] <= model_param["max_electrolyzer_capacity"] * z_on[t]) for t in periods)
    model.addConstrs((p_H[t] == p_s[t] + z_sb[t]*model_param["p_sb"]*model_param["max_electrolyzer_capacity"]) for t in periods)
    # Electrolyzer state
    model.addConstrs((z_off[t] + z_on[t] + z_sb[t] == 1) for t in periods)
    model.addConstrs((p_H[t] <= model_param["max_electrolyzer_capacity"] * z_on[t] + z_sb[t]*model_param["max_electrolyzer_capacity"]*model_param["p_sb"]) for t in periods)
    model.addConstrs((p_H[t] >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on[t] + z_sb[t]*model_param["p_sb"]*model_param["max_electrolyzer_capacity"]) for t in periods)
    for t in periods:
        if t == periods[0]:
            model.addConstr(z_su[t] == 0)
            model.addConstr(b_BS[t] == 0)
            model.addConstr(p_BS[t] >= 0)
        else:
            model.addConstr(z_su[t] >= z_off[t-1] + z_on[t] + z_sb[t] - 1)
            # Ramping
            model.addConstr(p_H[t] <= p_H[t-1] + model_param["el_ramping"]*model_param["max_electrolyzer_capacity"])
            model.addConstr(p_H[t] >= p_H[t-1] - model_param["el_ramping"]*model_param["max_electrolyzer_capacity"])
            # Battery constraints
            model.addConstr(b_BS[t] == b_BS[t-1] + p_BS[t])
            model.addConstr(p_BS[t] <= model_param["bs_capacity"] - b_BS[t-1])
            model.addConstr(p_BS[t] >= -b_BS[t-1])
    # Battery charging/discharging constraint
    model.addConstrs((p_BS[t] <= model_param["bs_charge"]*model_param["bs_capacity"]) for t in periods)
    model.addConstrs((p_BS[t] >= -model_param["bs_charge"]*model_param["bs_capacity"]) for t in periods)

    # Solve model    
    model.optimize()

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check
    print("p_DA: ", np.unique(p_DA.x))
    print("len(p_DA):", len(np.unique(p_DA.x)))
    print("len(p_H):", len(np.unique(p_H.x)))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_s.x,p_BS.x,b_BS.x,p_IM.x,E_RE,lambda_DA_RE,lambda_IM,z_sb.x,z_off.x,z_on.x,z_su.x), columns = np.array(["p_DA","p_H","p_s","p_BS","b_BS","p_IM","E_RE","lambda_DA_RE","lambda_IM","z_sb","z_off","z_on","z_su"]))
    df_results.to_csv("{0}p_{1}.csv".format("Results/","hindsight"),index=False)

    return value

def hindsight1(data_test,model_param):
    """
    hindsight model with full information, selling + buying
    --------------------------------
    param: (df) data_test: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data_test["lambda_DA_RE"].to_numpy()
    lambda_IM = data_test["lambda_IM"].to_numpy()
    E_RE = data_test["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    S = np.array([t for t in range(0, model_param['N_s'])])

    # Constraint for s2
    T_op = 90
    Pr = 30
    AA = electrolyzer_efficiency.area(model_param)
    i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
    s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]

    # Model
    model = gp.Model("day_ahead_hindsight")

    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=-model_param["max_electrolyzer_capacity"],ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="Day-ahead market bid")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="Hydrogen production DA in kg")
    p_s =  model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="power consumption of electrolyzer in on state")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="Electrolyzer consumption DA in MW")
    p_IM = model.addMVar(shape=len(periods),lb=-float('inf'),vtype=GRB.CONTINUOUS, name="over-under-production")
    p_IM_abs = model.addMVar(shape=len(periods),lb=0 ,vtype=GRB.CONTINUOUS, name="p_IM_abs")
    z_su =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary start up cost")
    z_on =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer on")
    z_off =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer off")
    z_sb =  model.addMVar(shape=len(periods),lb=0, ub=1,vtype=GRB.BINARY, name="binary electrolyzer standby")
    VaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="VaR")
    CVaR = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
    xi = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS, name="xi")
    s1 = model.addMVar(shape=len(periods),lb=0,ub=model_param["p_min"] * model_param["max_electrolyzer_capacity"],vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    s2 = model.addMVar(shape=len(periods),lb=0,ub = s2_max,vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")

    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - z_su*model_param["K_su"]*model_param["max_electrolyzer_capacity"] - 100*s1-100*s2), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H - s1)
    model.addConstr(s2 == 0)
    model.addConstr(s1 == 0)
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] + s2[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"])
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_s[t]*model_param["a"][s] + z_on[t]*model_param["b"][s]) * model_param["eta_storage"]) for s in S),name="c3")
    model.addConstr(p_s >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on)
    model.addConstr(p_s <= model_param["max_electrolyzer_capacity"] * z_on)
    model.addConstr(p_H == p_s + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"])
    # Electrolyzer state
    model.addConstr(z_off + z_on + z_sb == 1)
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["max_electrolyzer_capacity"]*model_param["p_sb"])
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"] * z_on + z_sb*model_param["p_sb"]*model_param["max_electrolyzer_capacity"])
    model.addConstr(z_on == 1)
    for t in periods:
        if t == periods[0]:
            model.addConstr(z_su[t] == 0)
        else:
            model.addConstr(z_su[t] >= z_off[t-1] + z_on[t] + z_sb[t] - 1)
    #model.addConstr(p_IM_abs >= p_IM, f"AbsPos_{t}",name="c13")
    #model.addConstr(p_IM_abs >= -p_IM, f"AbsNeg_{t}",name="c14")
    #model.addConstr(CVaR == VaR + (1/(len(periods)*(1-model_param["alpha"])))*gp.quicksum(xi[t] for t in periods),name="c16")
    #model.addConstr(xi >= p_IM_abs - VaR,name="c17")
    #model.addConstr(CVaR <= model_param["CVaR_95"],name="c18")


    # Solve model    
    model.optimize()

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check results
    print("CVaR: ", CVaR.x)
    print("p_DA: ", np.unique(p_DA.x))
    print("len(p_DA):", len(np.unique(p_DA.x)))
    print("len(p_H):", len(np.unique(p_H.x)))
    print("p_H: ", np.unique(p_H.x))
    print("E(p_IM): ", np.sum(np.abs(p_IM.x))/len(periods))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,h_H.x,E_RE,z_on.x,z_sb.x,z_off.x,z_su.x,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","h_H","E_RE","z_on","z_sb","z_off","z_su","lambda_DA_RE","lambda_IM"]))

    return value, df_results

def hindsight2(data_test,model_param):
    """
    hindsight model with full information for only selling
    --------------------------------
    param: (df) data_test: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data_test["lambda_DA_RE"].to_numpy()
    lambda_IM = data_test["lambda_IM"].to_numpy()
    E_RE = data_test["E_RE"].to_numpy()

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    S = np.array([t for t in range(0, model_param['N_s'])])

    # Constraint for s2
    T_op = 90
    Pr = 30
    AA = electrolyzer_efficiency.area(model_param)
    i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
    s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]

    # Model
    model = gp.Model("day_ahead_hindsight")

    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="Day-ahead market bid")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="Hydrogen production DA in kg")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="Electrolyzer consumption DA in MW")
    p_IM = model.addMVar(shape=len(periods),lb=0,vtype=GRB.CONTINUOUS, name="over-under-production")
    s1 = model.addMVar(shape=len(periods),lb=0,ub=model_param["p_min"] * model_param["max_electrolyzer_capacity"],vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    s2 = model.addMVar(shape=len(periods),lb=0,ub = s2_max,vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")

    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - 100*s1 - 100*s2), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H - s1)
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] + s2[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"])
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_H[t]*model_param["a"][f] + model_param["b"][f]) * model_param["eta_storage"]) for f in S),name="c3")
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"])
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"])

    # Solve model    
    model.optimize()
    model.write("Models/Hindsight.lp")

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check results
    print("p_DA: ", np.unique(p_DA.x))
    print("len(p_DA):", len(np.unique(p_DA.x)))
    print("len(p_H):", len(np.unique(p_H.x)))
    print("p_H: ", np.unique(p_H.x))
    print("E(p_IM): ", np.sum(np.abs(p_IM.x))/len(periods))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,h_H.x,s1.X,s2.X,E_RE,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","h_H","s1","s2","E_RE","lambda_DA_RE","lambda_IM"]))

    return value, df_results

def hindsight3(data_test,model_param):
    """
    hindsight model with full information for conditionally buying
    --------------------------------
    param: (df) data_test: testing data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
    param: (dict) model_param: dict containing model parameters such as hydrogen price, max wind capacity, max electrolyzer capacity, min daily production requirement for electrolyzer
    return (float) value: objective value
    """
    # Historic data
    lambda_DA_RE = data_test["lambda_DA_RE"].to_numpy()
    lambda_IM = data_test["lambda_IM"].to_numpy()
    E_RE = data_test["E_RE"].to_numpy()
    lambda_green = 20

    periods = np.arange(len(lambda_DA_RE))
    days = np.arange(int(len(periods)/24))
    S = np.array([t for t in range(0, model_param['N_s'])])
    M = np.max(lambda_DA_RE)
    eps = 0.0001

    # Constraint for s2
    T_op = 90
    Pr = 30
    AA = electrolyzer_efficiency.area(model_param)
    i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
    s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]

    # Model
    model = gp.Model("day_ahead_hindsight")

    # Variables
    p_DA = model.addMVar(shape=len(periods),lb=-model_param["max_electrolyzer_capacity"],ub=model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="Day-ahead market bid")
    h_H = model.addMVar(shape=len(periods),lb=0, vtype=GRB.CONTINUOUS,name="Hydrogen production DA in kg")
    p_H = model.addMVar(shape=len(periods),lb=0,ub=model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS,name="Electrolyzer consumption DA in MW")
    p_IM = model.addMVar(shape=len(periods),lb=-float('inf'),vtype=GRB.CONTINUOUS, name="over-under-production")
    s1 = model.addMVar(shape=len(periods),lb=0,ub=model_param["p_min"] * model_param["max_electrolyzer_capacity"],vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    s2 = model.addMVar(shape=len(periods),lb=0,ub=s2_max,vtype=GRB.CONTINUOUS, name="slack variable for electrolyzer consumption")
    b = model.addMVar(shape=len(periods),lb=0,ub=1,vtype=GRB.BINARY, name="binary variable for buying")

    # Model objective
    model.setObjective(gp.quicksum(lambda_DA_RE*p_DA + model_param["lambda_H"]*h_H + lambda_IM*p_IM - 200*s1 - 200*s2), GRB.MAXIMIZE)

    # Model Constraints
    # Power imbalance settled in real time as difference between realized production E_RE and power scheduled in day-ahead bidding
    model.addConstr(E_RE >= p_DA + p_IM + p_H - s1)
    # Daily hydrogen requirement
    for d in days:
        model.addConstr(gp.quicksum(h_H[t] + s2[t] for t in range(int(24*d),int(24*(d+1)))) >= model_param["min_production"])
    # Hydrogen Production
    for t in periods:
        model.addConstrs(((h_H[t] <= (p_H[t]*model_param["a"][f] + model_param["b"][f]) * model_param["eta_storage"]) for f in S),name="c3")
    model.addConstr(p_H >= model_param["p_min"] * model_param["max_electrolyzer_capacity"])
    model.addConstr(p_H <= model_param["max_electrolyzer_capacity"])
    
    # Conditional buying
    model.addConstr(b >= (lambda_DA_RE - lambda_green) / M, name="c4")
    model.addConstr(b <= (lambda_DA_RE - lambda_green + M - eps) / M, name="c5")
    model.addConstr(p_IM >= (-model_param["max_wind_capacity"]-model_param["max_electrolyzer_capacity"])*(1-b), name="c6")
    model.addConstr(p_DA >= -model_param["max_wind_capacity"]*(1-b), name="c7")

    # Solve model    
    model.optimize()
    model.write("Models/Hindsight.lp")

    # Print objective value
    value = model.ObjVal
    print('Objective value: %g' % value)

    # Check results
    print("p_DA: ", np.unique(p_DA.x))
    print("len(p_DA):", len(np.unique(p_DA.x)))
    print("len(p_H):", len(np.unique(p_H.x)))
    print("p_H: ", np.unique(p_H.x))
    print("E(p_IM): ", np.sum(np.abs(p_IM.x))/len(periods))
    df_results = pd.DataFrame(data=zip(p_DA.x,p_H.x,p_IM.x,b.X,h_H.x,s1.X,s2.X,E_RE,lambda_DA_RE,lambda_IM), columns = np.array(["p_DA","p_H","p_IM","b","h_H","s1","s2","E_RE","lambda_DA_RE","lambda_IM"]))

    return value, df_results