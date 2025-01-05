import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Functions import electrolyzer_efficiency

class expando(object):
    '''
        A small class which can have attributes set
    '''
    pass

class Model:
    def __init__(self, 
                 timelimit:float = 100,
                 data:pd.DataFrame = None, 
                 data_param:dict = None,
                 model_param:dict = None, 
                 x:pd.DataFrame = None, 
                 price_quantile:np.array = None):
        self.data = data # training data for realized day-ahead prices, realized upward balancing prices, realized downward balancing prices, realized wind production
        self.data_param = data_param # data parameters
        self.model_param = model_param # model parameters
        self.x = x # feature vector
        self.price_quantile = price_quantile # price quantiles
        self.timelimit = timelimit # set time limit for optimization to 100 seconds (default)

        # Historic data
        self.lambda_DA_RE = self.data["lambda_DA_RE"].to_numpy()
        self.lambda_IM = self.data["lambda_IM"].to_numpy()
        self.E_RE = self.data["E_RE"].to_numpy()

        # Parameters
        self.periods = np.arange(len(self.data))
        self.days = np.arange(int(len(self.periods)/24))
        try:
            self.n_features = len(self.x.columns)
        except:
            self.n_features = 0
        try:
            self.n_price_domains = len(self.price_quantile)+1
        except:
            self.n_price_domains = 0
        self.S = np.array([t for t in range(0, self.model_param['N_s'])])
        self.lambda_green = 20
        self.M = np.max(self.lambda_DA_RE)
        self.eps = 0.0001
        # Parameters for s2
        T_op = 90
        Pr = 30
        AA = electrolyzer_efficiency.area(model_param)
        i = electrolyzer_efficiency.find_i_from_p(np.array([model_param["max_electrolyzer_capacity"]]),AA,T_op)
        self.s2_max = electrolyzer_efficiency.h_prod(i,T_op,Pr,AA)*model_param["eta_storage"]
        
        if (len(self.periods) % 24 != 0):
            print("Error: Training length must be multiple of 24h!")

        self.variabels = expando()
        self.constraints = expando()
        self.results = expando()
        self.results.objective_value = None # set objective value to None until optimization is performed

    def build_model(self):
        self.model = gp.Model("model_DA")
        self.model.Params.TimeLimit = self.timelimit # set time limit for optimization
        self.model.Params.Seed = 42 # set seed for reproducibility

        # Initialize variables
        self._initialize_variables()
        self.model.update()

        # Initialize objective to maximize profit
        self.model.setObjective(
            gp.quicksum(
            self.lambda_DA_RE[t] * self.variabels.p_DA[t] +
            self.model_param["lambda_H"] * self.variabels.h_H[t] +
            self.lambda_IM[t] * self.variabels.p_IM[t] -
            self.variabels.z_su[t] * self.model_param["K_su"] * self.model_param["max_electrolyzer_capacity"] -
            100 * self.variabels.s1[t] -
            100 * self.variabels.s2[t]
            for t in self.periods
            ), GRB.MAXIMIZE
        )
        self.model.update()

        # Initialize constraints
        self._set_constraints()
        self.model.update()

        # Add model type constraints
        model_type = "_set_model_type_constraints_" + str(self.model_param["model_type"])
        add_constraints = getattr(self, model_type)
        add_constraints()
        self.model.update()

        # Add model conditions constraints
        model_conditions = "_set_model_conditions_constraints_" + str(self.model_param["model_conditions"])
        add_constraints = getattr(self,model_conditions)
        add_constraints()
        self.model.update()

        # Add risk model constraints
        risk_model = "_set_model_risk_constraints_" + str(self.model_param["model_risk"])
        add_constraints = getattr(self,risk_model)
        add_constraints()
        self.model.update()

    def run_model(self):
        self._solve_model()
        self._save_results()

    def _initialize_variables(self):
        self.variabels.p_DA = self.model.addMVar(shape=len(self.periods),lb=-self.model_param["max_electrolyzer_capacity"],ub=self.model_param["max_wind_capacity"], vtype=GRB.CONTINUOUS, name="p_DA")
        self.variabels.h_H = self.model.addMVar(shape=len(self.periods),lb=0, vtype=GRB.CONTINUOUS,name="h_H")
        self.variabels.p_s = self.model.addMVar(shape=len(self.periods), lb=0, vtype=GRB.CONTINUOUS, name="p_s")
        self.variabels.p_H = self.model.addMVar(shape=len(self.periods), lb=0, ub=self.model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS, name="p_H")
        self.variabels.p_IM = self.model.addMVar(shape=len(self.periods), lb=-float('inf'), vtype=GRB.CONTINUOUS, name="p_IM")
        self.variabels.p_IM_abs = self.model.addMVar(shape=len(self.periods), lb=0, vtype=GRB.CONTINUOUS, name="p_IM_abs")
        self.variabels.z_su = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="z_su")
        self.variabels.z_on = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="z_on")
        self.variabels.z_off = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="z_off")
        self.variabels.z_sb = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="z_sb")
        self.variabels.q_DA = self.model.addMVar(shape=(24, self.n_features, self.n_price_domains), lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_DA")
        self.variabels.q_H = self.model.addMVar(shape=(24, self.n_features, self.n_price_domains), lb=-float('inf'), vtype=GRB.CONTINUOUS, name="q_H")
        self.variabels.VaR = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="VaR")
        self.variabels.CVaR = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="CVaR")
        self.variabels.xi = self.model.addMVar(shape=len(self.periods), lb=0, vtype=GRB.CONTINUOUS, name="xi")
        self.variabels.u = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="u")
        self.variabels.s1 = self.model.addMVar(shape=len(self.periods), lb=0, ub=self.model_param["p_min"] * self.model_param["max_electrolyzer_capacity"], vtype=GRB.CONTINUOUS, name="slack_variable_electrolyzer_consumption")
        self.variabels.s2 = self.model.addMVar(shape=len(self.periods), lb=0, ub=self.s2_max, vtype=GRB.CONTINUOUS, name="slack_variable_electrolyzer_consumption")
        self.variabels.b = self.model.addMVar(shape=len(self.periods), lb=0, ub=1, vtype=GRB.BINARY, name="b")

    def _set_constraints(self):
        # (3c) Power imbalance
        self.constraints.c0 = self.model.addConstr(self.E_RE >= self.variabels.p_DA + self.variabels.p_IM + self.variabels.p_H - self.variabels.s1, name="c0")
        # (3e) power consumption of electrolyzer
        self.constraints.c1 = self.model.addConstr(self.variabels.p_H <= self.model_param["max_electrolyzer_capacity"] * self.variabels.z_on + self.variabels.z_sb*self.model_param["max_electrolyzer_capacity"]*self.model_param["p_sb"],name="c1")
        self.constraints.c2 = self.model.addConstr(self.variabels.p_H >= self.model_param["p_min"] * self.model_param["max_electrolyzer_capacity"] * self.variabels.z_on + self.variabels.z_sb*self.model_param["p_sb"]*self.model_param["max_electrolyzer_capacity"],name="c2")
        # Electrolyzer state
        self.constraints.c3 = self.model.addConstr(self.variabels.z_off + self.variabels.z_on + self.variabels.z_sb == 1, name="c3")
        # Assumption: electrolyzer is always on
        self.constraints.c4 = self.model.addConstr(self.variabels.z_on == 1, name="c4")
        # Constraints on z_su
        self.constraints.c5 = self.model.addConstr(self.variabels.z_su[0] == 0, name="c5")
        self.constraints.c6 = self.model.addConstrs((self.variabels.z_su[t] >= self.variabels.z_off[t-1] + self.variabels.z_on[t] + self.variabels.z_sb[t] - 1 for t in self.periods[1:]), name="c6")
        # (3f) Hydrogen production
        self.constraints.c7 = self.model.addConstrs(((self.variabels.h_H[t] <= (self.variabels.p_s[t]*self.model_param["a"][s] + self.variabels.z_on[t]*self.model_param["b"][s]) * self.model_param["eta_storage"]) for t in self.periods for s in self.S), name="c7")
        self.constraints.c8 = self.model.addConstr(self.variabels.p_s >= self.model_param["p_min"] * self.model_param["max_electrolyzer_capacity"] * self.variabels.z_on, name="c8")
        self.constraints.c9 = self.model.addConstr(self.variabels.p_s <= self.model_param["max_electrolyzer_capacity"] * self.variabels.z_on, name="c9")
        self.constraints.c10 = self.model.addConstr(self.variabels.p_H == self.variabels.p_s + self.variabels.z_sb*self.model_param["p_sb"]*self.model_param["max_electrolyzer_capacity"], name="c10")
        # (3g) Daily hydrogen requirement
        self.constraints.c11 = self.model.addConstrs((gp.quicksum(self.variabels.h_H[t] for t in range(int(24*d), int(24*(d+1)))) >= self.model_param["min_production"] for d in self.days), name="c11")
        # (7a) and (7b) absolute imbalance
        self.constraints.c12 = self.model.addConstr(self.variabels.p_IM_abs >= self.variabels.p_IM, name="c12")
        self.constraints.c13 = self.model.addConstr(self.variabels.p_IM_abs >= -self.variabels.p_IM, name="c13")

    def _set_model_type_constraints_1(self):
        print("Model type: policies")    
        # (12a) Linear policy constraints
        self.constraints.c14 = self.model.addConstr(self.variabels.q_DA[:,0:1,:] >= 0, name="c14")
        # (2a), (2b) and (12b) Linear policy constraints
        for t in self.periods:
            h = t%24
            p_domain = 0
            for i in range(self.n_price_domains-1):
                if self.lambda_DA_RE[t] >= self.price_quantile[i]:
                    p_domain = i+1
                a1 = self.variabels.q_DA[h:h+1,0:1,i:i+1].reshape((1,1))[0][0]
                b1 = (self.variabels.q_DA[h:h+1,1:,i:i+1].reshape((1,self.n_features-1))@np.array([self.x.iloc[t]])[:,1:].T)[0][0]
                a2 = self.variabels.q_DA[h:h+1,0:1,i+1:i+2].reshape((1,1))[0][0]
                b2 = (self.variabels.q_DA[h:h+1,1:,i+1:i+2].reshape((1,self.n_features-1))@np.array([self.x.iloc[t]])[:,1:].T)[0][0]
                self.constraints.c15 = self.model.addConstr((a2-a1)*self.price_quantile[i]+b2-b1 >= 0, name="c15") # (12b)
            self.constraints.c16 = self.model.addConstr(self.variabels.p_DA[t] == self.variabels.q_DA[h:h+1,:,p_domain:p_domain+1].reshape((1,self.n_features))@np.array([self.x.iloc[t]]).T, name="c16") # (2a)
            self.constraints.c17 = self.model.addConstr(self.variabels.p_H[t] == self.variabels.q_H[h:h+1,:,p_domain:p_domain+1].reshape((1,self.n_features))@np.array([self.x.iloc[t]]).T, name="c17") # (2b)

    def _set_model_type_constraints_2(self):
        print("Model type: hindsight")

    def _set_model_conditions_constraints_1(self):
        print("Model conditions: Selling+Buying")
        # (4) already implemented in _initialize_variables
        # set s1 and s2 to 0 as they are not needed here
        self.constraints.c18 = self.model.addConstr(self.variabels.s1 == 0, name="c18")
        self.constraints.c19 = self.model.addConstr(self.variabels.s2 == 0, name="c19")

    def _set_model_conditions_constraints_2(self):
        print("Model conditions: Only selling")
        # (5a) lower bound on p_DA
        self.constraints.c20 = self.model.addConstr(self.variabels.p_DA >= 0, name="c20")
        # (5b) lower bound on p_IM
        self.constraints.c21 = self.model.addConstr(self.variabels.p_IM >= 0, name="c21")
    
    def _set_model_conditions_constraints_3(self):
        print("Model conditions: Conditional buying")
        # (6a) Green electricity grid requirement
        self.constraints.c22 = self.model.addConstr(self.variabels.b >= (self.lambda_green - self.lambda_DA_RE) / self.M, name="c22")
        self.constraints.c23 = self.model.addConstr(self.variabels.b <= (self.lambda_green - self.lambda_DA_RE + self.M - self.eps) / self.M, name="c23")
        # (6b)
        self.constraints.c24 = self.model.addConstr(self.variabels.p_IM >= (-self.model_param["max_wind_capacity"] - self.model_param["max_electrolyzer_capacity"]) * self.variabels.b, name="c24")
        # (6c)
        self.constraints.c25 = self.model.addConstr(self.variabels.p_DA >= -self.model_param["max_electrolyzer_capacity"] * self.variabels.b, name="c25")

    def _set_model_risk_constraints_0(self):
        print("No risk model implemented!")
        # Make sure p_IM_abs is equal to p_IM or -p_IM and not greater
        self.constraints.c26 = self.model.addConstrs((self.variabels.p_IM_abs[t] == self.variabels.u[t] * self.variabels.p_IM[t] - (1 - self.variabels.u[t]) * self.variabels.p_IM[t] for t in self.periods), name="c26")

    def _set_model_risk_constraints_1(self):
        print("Risk: Mean imbalance")
        # (8) Expected value of absolute imbalance
        self.constraints.c27 = self.model.addConstrs((gp.quicksum(self.variabels.p_IM_abs[t] for t in range(int(24*d), int(24*(d+1)))) <= self.model_param["p_IM_abs_mean"]*24 for d in self.days), name="c27")
    
    def _set_model_risk_constraints_2(self):
        print("Risk: CVaR")
        # (9) CVaR constraint
        self.constraints.c28 = self.model.addConstr(self.variabels.CVaR == self.variabels.VaR + (1/(len(self.periods)*(1-self.model_param["alpha"])))*gp.quicksum(self.variabels.xi[t] for t in self.periods), name="c28")
        self.constraints.c29 = self.model.addConstrs((self.variabels.xi[t] >= self.variabels.p_IM_abs[t] - self.variabels.VaR for t in self.periods), name="c29")
        self.constraints.c30 = self.model.addConstr(self.variabels.CVaR <= self.model_param["CVaR_95"], name="c30")
    
    def _set_risk_constraints_3(self):
        print("Risk: Maximum imbalance")
        # (10) Maximum imbalance constraint
        self.constraints.c31 = self.model.addConstr(self.variabels.p_IM_abs <= self.model_param["p_IM_abs_max"], name="c31")

    def _solve_model(self):
        self.model.optimize()
        self.model.write("Models/model.lp")

    def _save_results(self):
        solution = False # Whether a solution was found
        # Check if a solution was found
        if self.model.status == GRB.Status.OPTIMAL:
            self.results.status = 'Optimal'
            solution = True
        elif self.model.status == GRB.Status.INFEASIBLE:
            self.results.status = 'Infeasible'
        elif self.model.status == GRB.Status.TIME_LIMIT:
            self.results.status = 'Time limit reached'
            if self.model.SolCount > 0:
                solution = True
        elif self.model.status == GRB.Status.UNBOUNDED:
            self.results.status = 'Unbounded'
        else:
            self.results.status = 'Other'
            if self.model.SolCount > 0:
                solution = True

        if solution:
            # Save objective value
            self.results.objective_value = self.model.ObjVal
            print('Objective value: %g' % self.results.objective_value)


            # Save results
            self.results.q_DA = self.variabels.q_DA.x
            self.results.q_H = self.variabels.q_H.x
            self.results.p_DA = self.variabels.p_DA.x
            self.results.p_H = self.variabels.p_H.x
            self.results.p_IM = self.variabels.p_IM.x
            self.results.p_IM_abs = self.variabels.p_IM_abs.x
            self.results.xi = self.variabels.xi.x
            self.results.h_H = self.variabels.h_H.x
            self.results.E_RE = self.E_RE
            self.results.z_on = self.variabels.z_on.x
            self.results.z_sb = self.variabels.z_sb.x
            self.results.z_off = self.variabels.z_off.x
            self.results.z_su = self.variabels.z_su.x
            self.results.lambda_DA_RE = self.lambda_DA_RE
            self.results.lambda_IM = self.lambda_IM
            self.results.CVaR = self.variabels.CVaR.x
            self.results.VaR = self.variabels.VaR.x
            self.results.u = self.variabels.u.x
            self.results.s1 = self.variabels.s1.x
            self.results.s2 = self.variabels.s2.x
            self.results.b = self.variabels.b.x

            # Check results
            print("mean p_IM_ABS: ", np.mean(self.results.p_IM_abs))
            print("Var: ", self.results.VaR)
            print("CVaR: ", self.results.CVaR)
            print("max p_IM_ABS: ", np.max(self.results.p_IM_abs))

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.dirname(__file__))
    os.chdir(cwd)
    # test = Model()
    # test.build_model()
    # test.run_model()