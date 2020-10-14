#!/usr/bin/env python
# coding: utf-8

import pyblp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

pyblp.options.digits = 2
pyblp.options.verbose = False 

# import the csv file
import csv
data = list(csv.reader(open('../temp/zeta_1000.csv')))
product_data = pd.DataFrame()
column_name = ['product_ids', 'market_ids', 'quality', 'satellite', 'wired', 'prices', 'obs_cost','unobs_demand','unobs_cost','shares', 'marginal_cost','price_elasticity','D1','D2','D3','D4']
product_data = pd.DataFrame(data, columns = column_name)
product_data = product_data.astype('float')
product_data['firm_ids'] = product_data['product_ids']


# ## 5 Estimate the Correctly Specified Model
# ### 5 (8) Report a table with the estimates of the demand parameters and standard error
# 
# #### (a) When estimating demand alone

# ### blp instrument
# demand_instruments = pyblp.build_blp_instruments(pyblp.Formulation('quality + satellite + wired'), product_data)
# pd.DataFrame(demand_instruments).describe()
# 
# demand_instruments = demand_instruments[:,5]
# demand_instruments = demand_instruments.reshape((len(demand_instruments),1))
# product_data['demand_instruments0'] = demand_instruments[:,0]
# 
# supply_instruments = pyblp.build_blp_instruments(pyblp.Formulation('obs_cost'), product_data)
# supply_instruments = supply_instruments[:,3]
# supply_instruments.reshape((len(supply_instruments),1))
# product_data['supply_instruments0'] = supply_instruments

# In[3]:


# quad diff instruments
demand_instruments = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + quality + obs_cost'), product_data, version='quadratic')
demand_instruments = demand_instruments[:,2:4]
demand_instruments = demand_instruments.reshape((len(demand_instruments),2))
product_data['demand_instruments0'] = demand_instruments[:,0]
product_data['demand_instruments1'] = demand_instruments[:,1]

supply_instruments = pyblp.build_differentiation_instruments(pyblp.Formulation('0 + obs_cost'), product_data, version='quadratic')
supply_instruments = supply_instruments[:,1]
supply_instruments.reshape((len(supply_instruments),1))
product_data['supply_instruments0'] = supply_instruments
product_data_diff = product_data.copy()


# In[4]:


product_data.head()


# In[5]:


product_data.describe()


# In[6]:


X1_formulation = pyblp.Formulation('0 + quality + prices + satellite + wired')
X2_formulation = pyblp.Formulation('0 + satellite + wired')
product_formulations = (X1_formulation, X2_formulation)

SIGMA0 = np.eye(2)
SIGMA_BOUNDS = ([[-1e2, -1e2], [-1e2, -1e2]], [[1e2, 1e2], [1e2, 1e2]])
BETA_BOUNDS = ([1e-3, -1e2, 1e-3, 1e-3], [1e2, -1e-3, 1e2, 1e2])
INTEGRATION = pyblp.Integration('product', size = 17)
OPTI = pyblp.Optimization('l-bfgs-b', {'gtol': 1e-6})


# In[7]:


# estimate the model
problem = pyblp.Problem(product_formulations, product_data, integration=INTEGRATION)
results_demand = problem.solve(sigma = SIGMA0, optimization=OPTI,                         sigma_bounds = SIGMA_BOUNDS, beta_bounds = BETA_BOUNDS)


# In[8]:


# update the results with optimal instruments
instrument_results = results_demand.compute_optimal_instruments(method='approximate')
updated_problem = instrument_results.to_problem()

updated_demand = updated_problem.solve(
    results_demand.sigma,
    optimization=OPTI,
    method='1s',
    sigma_bounds = SIGMA_BOUNDS,
    beta_bounds = BETA_BOUNDS 
)


# #### (b) When estimating jointly with supply

# In[9]:


# product_formulation
X3_formulation = pyblp.Formulation('1 + obs_cost')
product_formulations = (X1_formulation, X2_formulation, X3_formulation)
problem = pyblp.Problem(product_formulations, product_data, integration=INTEGRATION, costs_type='log')

results_supply = problem.solve(
    results_demand.sigma,
    beta = results_demand.beta,
    costs_bounds=(1e-4, None),
    sigma_bounds = SIGMA_BOUNDS,
    beta_bounds = BETA_BOUNDS 
)


# In[10]:


# update the results with optimal instruments
instrument_results = results_supply.compute_optimal_instruments(method='approximate')
updated_problem = instrument_results.to_problem()

updated_supply = updated_problem.solve(
    results_supply.sigma,
    beta = results_supply.beta,
    costs_bounds=(1e-4, None),
    sigma_bounds = SIGMA_BOUNDS,
    beta_bounds = BETA_BOUNDS 
)


# ### print TeX table

# In[11]:


def resultstotex(r):
    
    label = r.beta_labels + ['sigma2_1','sigma2_2'] + r.gamma_labels 
    esti = np.append(r.beta.reshape(-1), np.diag(r.sigma_squared) )
    esti = np.append(esti, r.gamma.reshape(-1))                  
    esti = np.around(esti,4)
    
    se = np.append(r.beta_se.reshape(-1), np.diag(r.sigma_squared_se)) 
    se = np.append(se, r.gamma_se.reshape(-1))               
    se = np.around(se,4)
    
    index1 = []
    for i in range(len(label)):
        index1.append(label[i])
        index1.append(" ")
    value = []
    for i in range(len(label)):
        value.append(str(esti[i]))
        value.append("("+str(se[i])+")")
        
    df = pd.DataFrame(value,index=index1)

    return df


# In[12]:


df1 = resultstotex(results_demand);
df2 = resultstotex(updated_demand);
df3 = resultstotex(results_supply);
df4 = resultstotex(updated_supply);

for i in range(4):
    df1 = df1.append(pd.Series([""]), ignore_index=True)
    df2 = df2.append(pd.Series([""]), ignore_index=True)

df = pd.DataFrame(index = df4.index)
df['demand_diff'] = df1[0].values
df['demand_opti'] = df2[0].values
df['supply_diff'] = df3[0]
df['supply_opti'] = df4[0]

df = df.rename(index={'1': 'cost_const'})
print(df.to_latex(index=True))


# ### 5 (9)

# In[13]:


results = updated_supply


# In[14]:


# estimated own price elasticity
elasticities = results.compute_elasticities(name = 'prices')
e_estimated = np.empty(2400,)
for i in range(600):
    e_estimated[4*i:4*i+4] = np.diag(elasticities[4*i:4*i+4, :]);

# A table comparing the estimated own-price elasticities to the true own-price elasticities
elasticity_table = pd.DataFrame()
elasticity_table['true_elasticity'] = product_data['price_elasticity']
elasticity_table['estimated_elasticity'] = e_estimated

# true diversion
diversion_true = pd.DataFrame()
diversion_true = product_data[['D1','D2','D3','D4']]

# estimated diversion
d = results.compute_diversion_ratios(name='prices') 
diversion_estimated = pd.DataFrame({'D1': d[:,0], 'D2': d[:,1], 'D3': d[:,2], 'D4': d[:,3]})


# #### average values

# In[15]:


df1 = pd.DataFrame()
df1['product'] = [1,2,3,4]
df1['elasticity_true'] = elasticity_table['true_elasticity'].                             values.reshape((600,4)).mean(axis=0)

for i in np.arange(1,5):
    df1['D'+str(i)+'_true'] = diversion_true['D'+str(i)].                                values.reshape((600,4)).mean(axis=0)
print(df1.to_latex(index=False)) 

df2 = pd.DataFrame()
df2['product'] = [1,2,3,4]
df2['elasticity_esti'] = elasticity_table['estimated_elasticity'].values.reshape((600,4)).mean(axis=0)

for i in np.arange(1,5):
    df2['D'+str(i)+'_esti'] = diversion_estimated['D'+str(i)].values.reshape((600,4)).mean(axis=0)

print(df2.to_latex(index=False)) 


# #### correlation between true and estimated columns

# In[16]:


df_corr = pd.DataFrame()
df_corr['product'] = [1,2,3,4]

price = []
for i in np.arange(1,5):
    c1 = product_data[product_data['product_ids']==i]['price_elasticity'].values
    c2 = e_estimated.reshape((600,4))[:,i-1]
    price.append(np.corrcoef(c1,c2)[0,1])
df_corr['own price elasticity'] = price

for i in np.arange(1,5):
    l=[]
    for j in np.arange(1,5):
        c1 = diversion_true['D'+str(i)].values.reshape((600,4))[:,j-1] 
        c2 = diversion_estimated['D'+str(i)].values.reshape((600,4))[:,j-1] 
        l.append( np.corrcoef(c1,c2)[0,1] )
    df_corr['D'+str(i)] = l

print(df_corr.to_latex(index=False)) 


# ### 6(11) simulation of merger between 1 and 2

# In[17]:


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

# In[18]:


product_data['merger_ids_13'] = product_data['firm_ids'].replace(3, 1)
# post-merger equilibrium prices 
changed_prices_13 = results.compute_prices(
    firm_ids=product_data['merger_ids_13'],
    costs=costs
)
changed_prices_13 = changed_prices_13.reshape((2400,))


# #### LaTeX output

# In[19]:


merger_price = pd.DataFrame()
merger_price['product'] = [1,2,3,4]

average_price = []
for i in range(4):
    average_price.append( product_data['prices'].values.reshape((600,4))[:,i].mean() )
merger_price['premerger'] = np.around(average_price,4)

average_price = []
for j in range(4):
    price_j = np.mean( [changed_prices_12[4*t+j] for t in range(600)] )
    average_price.append(price_j)
merger_price['merge 1 and 2'] = np.around(average_price,4)

average_price = []
for j in range(4):
    price_j = np.mean( [changed_prices_13[4*t+j] for t in range(600)] )
    average_price.append(price_j)
merger_price['merge 1 and 3'] = np.around(average_price,4)

print(merger_price.to_latex(index=False)) 


# ### 6(14) Merger with cost reduction

# In[20]:


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


# In[24]:


average_price = []
for j in range(4):
    price_j = np.mean( [price_postmerger[4*t+j] for t in range(600)] )
    average_price.append(price_j)
merger_price['cost reduction'] = np.around(average_price,4)


# In[26]:


print(merger_price.to_latex(index=False)) 


# In[21]:


# consumer surplus
cs_pre = results.compute_consumer_surpluses()
cs_post = results.compute_consumer_surpluses(price_postmerger)
plt.hist(cs_post - cs_pre, bins=50);
plt.legend(["Consumer Surplus Changes"]);


# In[41]:


# change in total welfare
profits_pre = results.compute_profits()
profits_pre = profits_pre.reshape((600,4)).sum(axis=1).reshape((600,1))
changed_shares = results.compute_shares(price_postmerger)
profits_post = results.compute_profits(price_postmerger, changed_shares, costs_reduced)
profits_post = profits_post.reshape((600,4)).sum(axis=1).reshape((600,1))

changed_totalwelfare = profits_post + cs_post - profits_pre - cs_pre
plt.hist(changed_totalwelfare, bins=50);
plt.legend(["Total Welfare Changes"]);


# In[ ]:




