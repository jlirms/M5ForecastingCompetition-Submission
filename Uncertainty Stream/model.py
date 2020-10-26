# -*- coding: utf-8 -*-
"""
Probablistic model
Goal is to generate uncertainty quantiles around these predictions.

This was my first project working with prediction intervals so I chose a simple strategy of:
    a centered normal distribution around the point forecasts made in the accuracy competition
    the variance of the distribution used is wider for item level forecasts and narrower for state level
With help from the kaggle community

The biggest drawback with this method is that the intervals only change from item level to state level etc
Significant improvement can be made by having wider/narrower intervals for certain products, certain dates etc
A students t distribution would likely work better, or a beta distribution as many products show mostly level sales and sporadic peaks

Currently exploring possibilities with quantile regressor, ARIMA and state space models based on exponential smoothing
Backtesting on Walmart data and before working on new project for a local clothing business in GTA area


"""


import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import scipy.stats  as stats
from util import M5Data

#%%Get data
data = M5Data(data_path = 'data/')
data.get_salesdf()
data.get_accdf()
df_merged = data.get_merge_acc()

#%% Get Intervals


qs = np.array([0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995])

def get_ratios(coef=0.15, zeros = False):
    '''returns ratios given scale for each level, taken from normal cdf of quantiles and scale
    uses a logit link function '''
    qs2 = np.log(qs/(1-qs))*coef ## logit transformation probability-> real numbers
    ratios = stats.norm.cdf(qs2) ## real numbers -> cdf 0<p<1
    ratios /= ratios[4] ## divide to center at 1
    ratios = pd.Series(ratios, index=qs) ## returns a series
    return ratios.round(3)

level_coef_dict = {"id": get_ratios(coef=0.3), "item_id": get_ratios(coef=0.15),
                   "dept_id": get_ratios(coef=0.08), "cat_id": get_ratios(coef=0.07),
                   "store_id": get_ratios(coef=0.08), "state_id": get_ratios(coef=0.07), "_all_": get_ratios(coef=0.05),
                   ("state_id", "item_id"): get_ratios(coef=0.19),  ("state_id", "dept_id"): get_ratios(coef=0.1),
                    ("store_id","dept_id") : get_ratios(coef=0.11), ("state_id", "cat_id"): get_ratios(coef=0.08),
                    ("store_id","cat_id"): get_ratios(coef=0.1)
                  }

"""## Functions to make predictions"""

cols = [f"F{i}" for i in range(1, 29)]

def quantile_coefs(q, level):
    '''locating scale from get_ratios() series'''
    ratios = level_coef_dict[level]
    return ratios.loc[q].values

def get_group_preds(pred, level):
    '''make predictions from a single level'''
    df = pred.groupby(level)[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q, level)[:, None]
    if level != "id":
        df["id"] = [f"{lev}_X_{q:.3f}_evaluation" for lev, q in zip(df[level].values, q)]
    else:
        df["id"] = [f"{lev.replace('_evaluation', '')}_{q:.3f}_evaluation" for lev, q in zip(df[level].values, q)]
    df = df[["id"]+list(cols)]
    print(df.id[0])
    return df

def get_couple_group_preds(pred, level1, level2):
    '''make predictions for series involving multiple levels'''
    df = pred.groupby([level1, level2])[cols].sum()
    q = np.repeat(qs, len(df))
    df = pd.concat([df]*9, axis=0, sort=False)
    df.reset_index(inplace = True)
    df[cols] *= quantile_coefs(q, (level1, level2))[:, None]
    df["id"] = [f"{lev1}_{lev2}_{q:.3f}_evaluation" for lev1,lev2, q in 
                zip(df[level1].values,df[level2].values, q)]
    df = df[["id"]+list(cols)]
    print(df.id[0])
    return df


#%% Make interval predictions


df_sub = []
for level in data.levels :
    print('\n',level, end = " ")
    print(df_merged.groupby(level)[cols].sum().shape)
    df_sub.append(get_group_preds(df_merged, level))

print('\n')

for level1,level2 in data.couples:
    print('\n',level1, level2, end = " ")
    print(df_merged.groupby([level1, level2])[cols].sum().shape)
    df_sub.append(get_couple_group_preds(df_merged, level1, level2))
  
df_sub = pd.concat(df_sub, axis=0, sort=False)


#%%
df_sub.reset_index(drop=True, inplace=True)

df_sub2 = df_sub.copy()
df_sub2['id'] = df_sub2['id'].str.replace("_evaluation", "_validation")
df_sub = pd.concat([df_sub,df_sub2] , axis=0, sort=False) ## stack on self for evaluation and validation 
del df_sub2
df_sub.set_index('id', drop = True, inplace = True)

print("Submission dataframe created, size: ", end ="")
print(df_sub.shape)


#%% Adjust for zeros based on sales history for specific items, ids and state_id, item_id couple
print("Adjusting for zeros based on sales histroy: ", end = "")
print("per item (all stores), ", end = "")
for thresh in [0.025, 0.21]:
  itemsonly = data.get_salesdf().groupby(['item_id']).sum()
  item_zeros = (itemsonly==0).sum(axis = 1)/1913
  item_zind = item_zeros[item_zeros > thresh].index
  if thresh == 0.025: 
    item_z = item_zind + '_X_0.005_evaluation'
    item_z2 = item_zind + '_X_0.025_evaluation'
  else: 
    item_z = item_zind + '_X_0.165_evaluation'
    item_z2 = item_zind + '_X_0.165_evaluation'

  df_sub.loc[item_z] = 0
  df_sub.loc[item_z2] = 0

print("per item (each store), ", end = "")
for thresh in [0.025, 0.21]:
  uniqueid = data.get_salesdf().groupby(['id']).sum()
  uniqueid_zeros = (uniqueid == 0).sum(axis = 1) / 1913
  unique_zind = uniqueid_zeros[uniqueid_zeros > thresh].index
  if thresh == 0.025: 
      unique_z = unique_zind.str[:-11] + '_0.005_evaluation'
      unique_z2 = unique_zind.str[:-11] + '_0.025_evaluation'
  else: 
      unique_z = unique_zind.str[:-11] + '_0.165_evaluation'
      unique_z2 = unique_zind.str[:-11] + '_0.165_evaluation'

  df_sub.loc[unique_z] = 0
  df_sub.loc[unique_z2] = 0

print("per item (each state). ", end = "")
for thresh in [0.025, 0.21]:
  stateitem = data.get_salesdf().groupby(['state_id','item_id']).sum()
  stateitem_zeros = (stateitem == 0).sum(axis = 1)/1913
  statei_zind = stateitem_zeros[stateitem_zeros > 0.025].index
  if thresh == 0.025: 
    uniques_z = [state+'_' + item + '_0.005_evaluation'  for state, item in statei_zind] 
    uniques_z2 = [state+'_' + item + '_0.025_evaluation'  for state, item in statei_zind] 
  else: 
    uniques_z = [state+'_' + item + '_0.165_evaluation'  for state, item in statei_zind] 
    uniques_z2 = [state+'_' + item + '_0.165_evaluation'  for state, item in statei_zind] 

  df_sub.loc[uniques_z] = 0
  df_sub.loc[uniques_z2] = 0
  
  
#%%Create submission file

df_sub.to_csv("submission_uc_joshli.csv", float_format='%.4g')















