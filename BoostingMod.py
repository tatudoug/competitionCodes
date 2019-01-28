#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 12:41:23 2019

@author: douglas
"""
import lightgbm as lgb
import xgboost
import numpy as np
import pandas as pd
import copy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.linear_model import LinearRegression

class boostingMod():
    '''This class implements a diffetent boosting where diferent models are used at
    each step'''
    
    #logistic,neural network, FM,SVM, lgbm, xgbm - (catboost)     
    
    def __init__(self, X_train, y_train,n_models = 100,learning_rate = 0.01):
        
        self.train_x = X_train
        self.train_y = y_train
        self.alpha = []
        self.test_x = None
        self.test_y = None
        self.model = []
        # number of models
        self.n_models = n_models
        # learning rate
        self.learning_rate = []
        
    def train(self):
        '''train the models '''  
        
        # first model is a linear regression
        reg = LinearRegression().fit(self.train_x, np.transpose(self.train_y))
        
        self.model.append(reg) # append the first model
        self.alpha.append(1) # the weight of the first model is 1
        self.learning_rate.append(1)
        
        modSel = 1
        
        for num_model in np.arange(self.n_models):
            
           
            if(modSel==1):
           
            
                lgb_train = lgb.Dataset(self.train_x, self.train_y-self.predict(self.train_x))
                
                params = {
                'boosting_type': 'gbdt',
                'objective': 'regression'
                }       
                
                 
                gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=1)
                
                self.grad(gbm.predict(self.train_x))
                self.learning_rate.append(0.1)
                self.model.append(gbm)
                
                modSel = 2
            
            else:
                xgb = xgboost.XGBRegressor(n_estimators=1, gamma=0,max_depth=20,learning_rate=0.1)
                xgb.fit(self.train_x, self.train_y-self.predict(self.train_x) ) # y - f(x)
                
                self.grad(xgb.predict(self.train_x))
                self.learning_rate.append(0.1)
                self.model.append(xgb)
                
                modSel = 1
            
        # calls gradient descendent function
        

        
        
    def grad(self,newVals):
        '''calculates the descendent gradient for the optimization'''
        # to be improved
                # solve the one dimensional gradient problem 
        
        
        predsTemp = self.predict(self.train_x)# sum new vals
        
        alpha = 0 # [-1,1]
        error = mean_squared_error( self.train_y , predsTemp + (newVals*alpha) )
        step = 0.1
        
        print('Error', error )
        # https://en.wikipedia.org/wiki/Gradient_boosting
        tol=1
        #------------------ check gradient algorithm to improve this part of the code---------------------
        n = 1
        while( np.abs(tol) > 0.00000000001 and n < 10000):
            
            #if (n%100 == 0):
            #print('n', n )
            
            
            alpha_i = copy.deepcopy(alpha)
            
            alpha = alpha - step
            #print('alpha', alpha )
            
            error_mod = mean_squared_error( self.train_y , predsTemp+ (newVals  * alpha) ) 
            #print('Error_mod', error_mod )
            
            tol = error_mod-error
            #print('tol', tol )
            
            step = tol/(alpha-alpha_i)
            
                
            if( abs(tol) < 0.0001):
                tol = np.sign(tol)*0.001
            n = n+1
            
            error = copy.deepcopy(error_mod)
            
        
        self.alpha.append(alpha)
        
        
    def predict(self,data):
        '''output the predicted value of the model '''
        y=0.0          
        for pos,modelo in enumerate (self.model): # varre todos os modelos
            #print(pos)    
            y += modelo.predict(data) * self.alpha[pos]* self.learning_rate[pos] #* self.learning_rate
            #print('y',y)
        return(y) # returns model output
    
    def predict_proba():
        ''' for classification '''          
        pass
    
    def plot():
        '''training values '''   
        pass
    
if __name__ == "__main__":
    
    # get the cal_housing dataset to test the new boosting algoritm
    cal_housing = fetch_california_housing()
        
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                                cal_housing.target,
                                                                test_size=0.3,
                                                                random_state=1)
    firstTry = boostingMod(X_train, y_train,1000)
    firstTry.train()
    
    a = firstTry.model[0].predict(X_test)
    b = firstTry.model[1].predict(X_test)
    
    #print('lgbm',mean_squared_error(y_test, a))
    #print('xg boost',mean_squared_error(y_test, b))
    #print('union a+alpha*b', mean_squared_error(y_test, a+ firstTry.alpha*b))
    
    print('lr', mean_squared_error(y_test, firstTry.model[0].predict(X_test)))
    print('l1', mean_squared_error(y_test, firstTry.model[0].predict(X_test)+firstTry.model[1].predict(X_test)*10 ))
    #print('union a+alpha*b', mean_squared_error(y_test, firstTry.predict(X_test)))
    print('Mod', mean_squared_error(y_test, firstTry.predict(X_test)))
    
    # references to compare the algorithms
    if(True):
         #xgb  base line    
        xgb = xgboost.XGBRegressor(n_estimators=1000, gamma=0,max_depth=20,learning_rate=0.1)
        xgb.fit(X_train, y_train ) # y - f(x)
        print('xgb', mean_squared_error(y_test, xgb.predict(X_test)))
        
        #lgbm baseline
        lgb_train = lgb.Dataset(X_train, y_train)
                    
        params = {
                   'boosting_type': 'gbdt',
                   'objective': 'regression',
                   'learning_rate': '0.1'
                  }       
                    
                     
        gbm = lgb.train(params,lgb_train,num_boost_round=1000)
        print('lgb', mean_squared_error(y_test, gbm.predict(X_test)))
        