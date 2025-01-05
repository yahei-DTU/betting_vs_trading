import pandas as pd
import numpy as np
import os

class RiskConstraintLimits:
    def __init__(self, data):
        self.data = data
        self._add_absolute_value()
    
    def _add_absolute_value(self):
        self.data['p_IM_abs'] = np.abs(self.data['p_IM'])

    def expected_value(self):
        return self.data['p_IM_abs'].mean()
    
    def extreme_value(self):
        return self.data['p_IM_abs'].max()
    
    def CVaR(self, alpha=0.95):
        Var = np.percentile(self.data['p_IM_abs'], 100*alpha)
        return self.data[self.data['p_IM_abs'] >= Var]['p_IM_abs'].mean()


def main():
    cwd = os.path.dirname(os.path.dirname(__file__))
    os.chdir(cwd)

    risk_limits = RiskConstraintLimits(pd.read_csv("Results/policies_train.csv"))

    # Save results
    np.savetxt("Results/mean(p_IM_abs)", np.array([risk_limits.expected_value()]))
    np.savetxt("Results/max(p_IM_abs)", np.array([risk_limits.extreme_value()]))
    np.savetxt("Results/CVaR_95", np.array([risk_limits.CVaR(risk_limits.alpha)]))

if __name__ == '__main__':
    main()
