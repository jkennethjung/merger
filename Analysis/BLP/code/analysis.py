#!/usr/bin/env python
# coding: utf-8

import pyblp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pyblp.options.digits = 2
pyblp.options.verbose = False 


# ### Loading the simulated data

import csv
data = list(csv.reader(open('../temp/zeta_1000.csv')))
column_name = ['product_ids', 'market_ids', 'quality', 'satellite', 'wired', 'prices', 'obs_cost','unobs_demand','unobs_cost','shares', 'marginal_cost','price_elasticity','D1','D2','D3','D4']
product_data = pd.DataFrame(data, columns = column_name)
product_data = product_data.astype('float')
product_data['firm_ids'] = product_data['product_ids']

# ## 5 Estimate the Correctly Specified Model
# ### 5 (8) Report a table with the estimates of the demand parameters and standard error
# #### (a) When estimating demand alone

#demand_instruments.shape

# blp instrument
demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('quality + satellite + wired'), product_data)
pd.DataFrame(demand_instruments).describe()

demand_instruments = demand_instruments[:,5]
demand_instruments = demand_instruments.reshape((len(demand_instruments),1))
product_data['demand_instruments0'] = demand_instruments[:,0]
# product_data['demand_instruments1'] = demand_instruments[:,1]

supply_instruments = pyblp.build_blp_instruments(pyblp.Formulation('obs_cost'), product_data)
supply_instruments = supply_instruments[:,3]
supply_instruments.reshape((len(supply_instruments),1))
product_data['supply_instruments0'] = supply_instruments

product_data.head()
product_data.describe()

# product_formulation
X1_formulation = pyblp.Formulation('0 + quality + prices')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
product_formulations = (X1_formulation, X2_formulation)

# integration
integration = pyblp.Integration('product', size= 17)
problem = pyblp.Problem(product_formulations, product_data, integration=integration)
opti = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-20})
results = problem.solve(sigma=1*np.eye(2), optimization=opti)
print(results)

# update the results with optimal instruments
instrument_results = results.compute_optimal_instruments(method='approximate')
updated_problem = instrument_results.to_problem()
updated_results = updated_problem.solve(
    results.sigma,
    optimization=opti,
    method='1s'
)
print(updated_results)
results = updated_results

'''
# #### (b) When estimating jointly with supply

# instruments
local_instruments = pyblp.build_differentiation_instruments(
    pyblp.Formulation('1 + obs_cost'),
    product_data
)
# pd.DataFrame(local_instruments).describe()

supply_instruments = local_instruments[:,3]
supply_instruments.reshape((len(demand_instruments),1))
product_data['supply_instruments0'] = supply_instruments

# product_formulation
X1_formulation = pyblp.Formulation('0 + quality + prices + satellite + wired')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
X3_formulation = pyblp.Formulation('1 + obs_cost')
product_formulations = (X1_formulation, X2_formulation, X3_formulation)

mc_integration = pyblp.Integration('monte_carlo', size=300, specification_options={'seed': 0})

problem = pyblp.Problem(product_formulations, product_data, integration=mc_integration, 
                        costs_type='log')

initial_sigma = np.diag([1, 1])

# estimate the model
results_supply = problem.solve(
    initial_sigma,
    beta = results.beta, # use the estimates from above as the initial value
    costs_bounds=(0.001, None),
    initial_update=True
)
print(results_supply)
'''

'''
# update the results with optimal instruments
instrument_results = results_supply.compute_optimal_instruments(method='approximate')
updated_problem = instrument_results.to_problem()

updated_results_supply = problem.solve(
    initial_sigma,
    beta = results.beta, # use the estimates from above as the initial value
    costs_bounds=(0.001, None),
    initial_update=True
)
print(updated_results_supply)
'''

'''
# ### 5 (9)

# estimated own price elasticity
elasticities = results.compute_elasticities(name = 'prices')
e_estimated = np.empty(2400,)
for i in range(600):
    e_estimated[4*i:4*i+4] = np.diag(elasticities[4*i:4*i+4, :]);

# A table comparing the estimated own-price elasticities to the true own-price elasticities
elasticity_table = pd.DataFrame()
elasticity_table['true_elasticity'] = product_data['price_elasticity']
elasticity_table['estimated_elasticity'] = e_estimated
print("elasticity_table:")
print(elasticity_table)

# true diversion
diversion_true = pd.DataFrame()
diversion_true = product_data[['D1','D2','D3','D4']]
print("true diversion ratio:")
print(diversion_true)

# estimated diversion
d = results.compute_diversion_ratios(name='prices') 
diversion_estimated = pd.DataFrame({'D1': d[:,0], 'D2': d[:,1], 'D3': d[:,2], 'D4': d[:,3]})
print("estimated diversion ratio:")
print(diversion_estimated)

# ### 6(11) simulation of merger between 1 and 2

# assume unchanged marginal costs
costs = results.compute_costs()
costs = costs.reshape((2400,))

product_data['merger_ids_12'] = product_data['firm_ids'].replace(2, 1)
# post-merger equilibrium prices
changed_prices_12 = results.compute_prices(
    firm_ids=product_data['merger_ids_12'],
    costs=costs
)
changed_prices_12 = changed_prices_12.reshape((2400,))

# ### 6(12) simulation of merger between 1 and 3

product_data['merger_ids_13'] = product_data['firm_ids'].replace(3, 1)
# post-merger equilibrium prices 
changed_prices_13 = results.compute_prices(
    firm_ids=product_data['merger_ids_13'],
    costs=costs
)
changed_prices_13 = changed_prices_13.reshape((2400,))


# a table comparing average prices after the two mergers
merger_price = pd.DataFrame()

average_price = []
for j in range(4):
    price_j = np.mean( [changed_prices_12[4*t+j] for t in range(600)] )
    average_price.append(price_j)
merger_price['merge 1 and 2'] = average_price

average_price = []
for j in range(4):
    price_j = np.mean( [changed_prices_13[4*t+j] for t in range(600)] )
    average_price.append(price_j)
merger_price['merge 1 and 3'] = average_price

print(merger_price)

# ### 6(14) Merger with cost reduction

# marginal costs of product 1 and 2 reduced by 15%
costs = results.compute_costs()
costs_reduced = costs.reshape((600,4))
costs_reduced[:,0] = 0.85*costs_reduced[:,0]
costs_reduced[:,1] = 0.85*costs_reduced[:,1]
costs_reduced = costs_reduced.reshape((2400,1))

# post-merger equilibrium prices with cost change
price_postmerger = results.compute_prices(
    firm_ids=product_data['merger_ids_12'],
    costs=costs_reduced
)
price_postmerger = price_postmerger.reshape((2400,))

# pre-merger consumer surplus
cs_pre = results.compute_consumer_surpluses()
cs_post = results.compute_consumer_surpluses(price_postmerger)
plt.hist(cs_post - cs_pre, bins=50);
plt.legend(["Consumer Surplus Changes"]);

'''
