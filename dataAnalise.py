#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 23:11:01 2018

@author: douglas
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import copy

from fastFM.datasets import make_user_item_regression
from fastFM import als
from sklearn.metrics import mean_squared_log_error
import scipy.sparse as sp
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt

# data analysis

def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


# load or create your dataset
print('Load data...')
df_train = pd.read_csv("train_GzS76OK/train.csv")
print(df_train.head())
df_test = pd.read_csv('test_QoiMO9B.csv')

#-------------- load support ------------------------------------------
df_center = pd.read_csv("train_GzS76OK/fulfilment_center_info.csv")
df_meal = pd.read_csv("train_GzS76OK/meal_info.csv")


# drop duplicate data if it exists
df_train.drop_duplicates(keep='first', inplace=True)

# print data size
print('Data size', len(df_train))

for x in  list(df_train):
    print (x, len(df_train[x].unique()) )
    
# count instances for each center
t = df_train['center_id'].value_counts()

# join data frames 
df_train2 = pd.merge(df_train, df_center, on=['center_id'], how='left')
# second join
df_train3 = pd.merge(df_train2, df_meal, on=['meal_id'], how='left')

# -------------joint test data set---------------------------
df_test = pd.merge(df_test, df_center, on=['center_id'], how='left')
df_test = pd.merge(df_test, df_meal, on=['meal_id'], how='left')

for x in  list(df_train3):
    print (x, len(df_train3[x].unique()) )



# num orders mean of each variable
    
df_train3.groupby(['center_id'])['num_orders'].mean()
# plot histgram
#plt.hist(df_train3.groupby(['center_id'])['num_orders'].mean() - df_train3['num_orders'].mean() )
stdData = df_train3['num_orders'].std()

centerMean = df_train3.groupby(['center_id'])['num_orders'].mean().reset_index(name='centerMean') 
centerMean['centerMean'] =  (centerMean['centerMean'] - df_train3['num_orders'].mean() )/stdData

# num orders based on products for each value
df_train3.groupby(['meal_id'])['num_orders'].mean()
# plot histgram
#plt.hist(df_train3.groupby(['meal_id'])['num_orders'].mean() - df_train3['num_orders'].mean() )

mealMean = df_train3.groupby(['meal_id'])['num_orders'].mean().reset_index(name='mealMean') 
mealMean['mealMean'] = (mealMean.mealMean - df_train3['num_orders'].mean()) /stdData

# mean of each sample
f = df_train3.groupby(['center_id','meal_id'])['base_price'].mean().reset_index(name='valueMean')

#adding the new features

df_train3 = pd.merge(df_train3, mealMean, on=['meal_id'], how='left')
df_train3 = pd.merge(df_train3, centerMean, on=['center_id'], how='left')

df_train3 = pd.merge(df_train3, f, on=['center_id','meal_id'], how='left')
df_train3['checkout_price'] = df_train3['checkout_price']/df_train3['valueMean']
df_train3['base_price'] = df_train3['base_price']/df_train3['valueMean']


#---------------joint test data set--------
df_test = pd.merge(df_test, mealMean, on=['meal_id'], how='left')
df_test = pd.merge(df_test, centerMean, on=['center_id'], how='left')

df_test = pd.merge(df_test, f, on=['center_id','meal_id'], how='left')
#pos = np.isnan(df_test['valueMean'])
df_test['valueMean'].loc[np.isnan(df_test['valueMean']) == 1] =  df_test['base_price'].loc[np.isnan(df_test['valueMean']) == 1]

df_test['checkout_price'] = df_test['checkout_price']/df_test['valueMean']
df_test['base_price'] = df_test['base_price']/df_test['valueMean']


#plot weekly demand
#vals = df_train3.groupby(['week'])['num_orders'].mean()
#plt.plot(np.correlate(vals,vals,mode='full'))
#plt.plot(vals)

# using  FM to predict data



df_train4 = pd.get_dummies(df_train3, columns=["cuisine"], prefix=["cui"])
df_train4 = pd.get_dummies(df_train4, columns=["category"], prefix=["cat"])
df_train4 = pd.get_dummies(df_train4, columns=["region_code"], prefix=["reg"])
df_train4 = pd.get_dummies(df_train4, columns=["center_type"], prefix=["cen"])
#df_train4 = pd.get_dummies(df_train4, columns=["emailer_for_promotion"], prefix=["ema"])
#df_train4 = pd.get_dummies(df_train4, columns=["homepage_featured"], prefix=["hom"])
df_train4 = pd.get_dummies(df_train4, columns=["center_id"], prefix=["hom"])

# new variables creation
df_train4['diffPrice'] = (df_train4['base_price'] - df_train4['checkout_price'])/ df_train4['base_price']


# --------------test dataset ------------------------------
df_test = pd.get_dummies(df_test, columns=["cuisine"], prefix=["cui"])
df_test = pd.get_dummies(df_test, columns=["category"], prefix=["cat"])
df_test = pd.get_dummies(df_test, columns=["region_code"], prefix=["reg"])
df_test = pd.get_dummies(df_test, columns=["center_type"], prefix=["cen"])
df_test = pd.get_dummies(df_test, columns=["center_id"], prefix=["hom"])
df_test['diffPrice'] = (df_test['base_price'] - df_test['checkout_price'])/ df_test['base_price']

x_data_test = copy.deepcopy( df_test.drop(['id', 'week', 'city_code', 'op_area', 'meal_id','valueMean'], axis=1).values)

log = 0
if (log):
    y = np.log(df_train4['num_orders'])
else:
    #y = (df_train4['num_orders'] - df_train4['num_orders'].mean())/ df_train4['num_orders'].std()
    y = df_train4['num_orders']
    
x_data = copy.deepcopy( df_train4.drop(['id', 'week', 'num_orders','city_code', 'op_area', 'meal_id','valueMean'], axis=1).values)


if (True):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y, test_size=0.33, random_state=42)
    X_train = sp.csc_matrix(X_train)
    X_test = sp.csc_matrix(X_test)
    
    n_iter = 100
    rank = 17 # 11
    seed = 333
    step_size = 1
    l2_reg_w = 0
    l2_reg_V = 0
    
    if (True):
        fm = als.FMRegression(n_iter=0, l2_reg_w=l2_reg_w, l2_reg_V=l2_reg_V, rank=rank, random_state=seed)
    
    # initalize coefs
    fm.fit(X_train, y_train)
    
    #pp = fm.predict(X_train)
    
    rmsle_train = []
    rmsle_test = []
    for i in range(1, n_iter):
        fm.fit(X_train, y_train, n_more_iter=step_size)
        
        y_predT = fm.predict(X_train)
        if (log):
            y_predT[y_predT > 10.098190476218488] = 10.098190476218488
            y_predT = np.exp(y_predT)
        else:
            #y_predT = (y_predT * df_train4['num_orders'].std() ) + df_train4['num_orders'].mean()
            pass
        
        y_predT[y_predT <= 0] = 13
        
        
        y_pred = fm.predict(X_test)
        if (log):
            y_pred[y_pred > 10.098190476218488] = 10.098190476218488
            y_pred = np.exp(y_pred)
        else:
             #y_pred = (y_pred * df_train4['num_orders'].std() ) + df_train4['num_orders'].mean()
             pass
         
        y_pred[y_pred <= 0] = 13
        
        if (log):
            rmsle_train.append(100*np.sqrt(mean_squared_log_error(np.exp(y_train),y_predT)))
            rmsle_test.append(100*np.sqrt(mean_squared_log_error(np.exp(y_test),y_pred )))
        else:
            #rmsle_train.append(100*np.sqrt(mean_squared_log_error(((y_train*df_train4['num_orders'].std() ) + df_train4['num_orders'].mean()),y_predT)))
            #rmsle_test.append(100*np.sqrt(mean_squared_log_error(((y_test*df_train4['num_orders'].std() ) + df_train4['num_orders'].mean()),y_pred )))
            rmsle_train.append(100*np.sqrt(mean_squared_log_error(y_train,y_predT)))
            rmsle_test.append(100*np.sqrt(mean_squared_log_error(y_test,y_pred )))
            
    x = np.arange(1, n_iter) * step_size
    
    with plt.style.context('fivethirtyeight'):
        plt.plot(x, rmsle_train, label='train')
        plt.plot(x, rmsle_test, label='test')
            #plt.plot(values, rmse_train_re, label='train re', linestyle='--')
            #plt.plot(values, rmse_test_re, label='test re', ls='--')
        plt.legend()
    plt.show()
    
    
    
    # final prediction
    fim = []
    x_comp = sp.csc_matrix(x_data_test)
    for num,p in enumerate(x_comp):
        try:
            finalPred = fm.predict(p)
            if (finalPred) < 0:
                finalPred = 13
            
            finalPred = np.round(finalPred)
            
            fim.append(copy.deepcopy(finalPred))
        except:
            print('No possible to predicit')
            print(num)
            fim.append(copy.deepcopy(-1))
    
    fim[fim == -1] = 255 # sets the unpredicted values to the mean
    fim = np.array(fim)
    fim[fim < 0] = 13
    
    df_test['num_orders'] = np.round(fim)
    
    df_test[['id','num_orders']].to_csv("previsao.csv",index=False)
    '''
    category 14
    cuisine 4
    region_code 8
    center_type 3
    emailer_for_promotion 2
    homepage_featured 2
    checkout_price 1992
    base_price 1907
    emailer_for_promotion 2
    homepage_featured 2
    '''