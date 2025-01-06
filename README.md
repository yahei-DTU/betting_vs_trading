## Betting vs. Trading: Learning a Linear Decision Policy for Selling Wind Power and Hydrogen

This repository contains all data and code to run the same model simulations and analysis as presented in the article ["Betting vs. Trading: Learning a Linear Decision Policy for Selling Wind Power and Hydrogen"](https://arxiv.org/abs/2412.18479) by Heiser et al. (2024).

The required packages are `time`, `numpy`, `pandas`, `os`, `matplotlib`and `gurobipy`.

### Overview of files
- "main.py": This file is the main file to run the model and perform a simulation. The model setup can be adjusted by changing the parameters in `data_param`and `model_param`. The file loads the `Model` class from the file "model.py" in the `Models/` directory together with other helping functions from the `Functions/` directory.
  
- `Data/`: Contains the data used for the analysis in the paper.
  - "2020_data.csv" constains the realized power prices and wind production as well as forecasts is taken from [Helgren et al. (2024)](https://www.sciencedirect.com/science/article/pii/S0378779624006734) and covers the years 2019 and 2020.
  - "RegulatingBalancePowerdata.csv" is taken from Energinet and provides the single balancing prices for the same years.
    
- `Functions/`: The files in this folde provide some helping functions, that are used by the main file.
  - "electrolyzer_efficiency.py" is taken from [Raheli et al. (2023)](https://www.sciencedirect.com/science/article/pii/S0098135423003204) and provides the methods to calculate the linear segments when modelling the electrolyzers efficiency.
  - "feature_x.py": Contains a function to return the feature vector for the given input data.
  - "import_data.py": Takes the raw data files as input to return the training, validation and testing sets.
  - "insert_row.py": Is used by `import_data` to adjust the hours in 2020 to account for the leap year.
  - "test_policies.py": Tests the given policies using the testing data.

- `Models/`: Contains the model formulation.
  - "model.py": formulation of the linear policies and hindsight models

- `Analysis/`: Contains the code files to analyze the model results and generate the paper plots.
  - "bidding_curve.py": Create the bidding curve plots used in the paper.
  - "H2 price sensitivity.py": Create the paper plot showing the price sensitivity.
  - "model_comparison.py": Create the model comparison plot in the paper.
  - "risk_constraints.py": Calculate the risk limits for the risk implementation in the models.
  - "risk_sensitivity.py": Generate the histogram plots from the paper to show the effect of the risk constraints.
  
- `Plots/`: Contains the plots used in the paper.

- `Results/`: Contains the model output files used for the analysis.

