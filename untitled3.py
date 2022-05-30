#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:58:40 2022

@author: zhye
"""
#Import all needed packages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from scipy import stats
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr as pearsonr
import pickle
import os
from UKB_graph_metrics import *

#Read GMV and phenotype info
df_gmv_SCE = pd.read_csv('SCE_gmv_updated.csv')
df_genetic_noNAN = pd.read_csv('df_genetic.csv')
gmv = np.genfromtxt('GMV.txt', dtype='str')

#Randomly sample 2000 subjects, random_state can be any seed
df_SCE_gmv_2000 = df_gmv_SCE.sample(n = 2000, random_state = 16)
df_SCE_gmv_2000.reset_index(drop = True, inplace = True)
#Z-score GMV data
df_SCE_gmv_2000[gmv] = df_SCE_gmv_2000[gmv].apply(stats.zscore)

#Cross-validation function
def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        r2test.append(r2_score(y_test, y_pred))
        
    if (n_beta):
        return r2train, r2test, beta
    else:
        return r2train, r2test

#Set parameters of cross-validation
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
rand = 9

#Read ICA data
os.chdir('ICA100_corr')
IC55 = []
for i in range(0, df_SCE_gmv_2000.shape[0]):
    tem = np.loadtxt(df_SCE_gmv_2000['file'].loc[i])
    IC55.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")

#Compute graph theory metrics of ICA
IC_Graph = []
for i in range(df_SCE_gmv_2000.shape[0]):
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")

#Save IC_Graph as list
#with open("IC_Graph", "wb") as fp:   #Pickling
#    pickle.dump(IC_Graph, fp)
os.chdir('../')
#with open("IC_Graph", "rb") as fp:   # Unpickling
#    IC_Graph = pickle.load(fp)

#Energy/WHR ~ GMV + Control
XE1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GMV_train_E, energy_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE1, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA + Control
XE2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_IC_train_E, energy_IC_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE2, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
XE3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_IC_gmv_train_E, energy_IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
XE_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GT_GMV_train_E, energy_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

#Energy/WHR ~ ICA_GT + Control
XE_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GT_train_E, energy_GT_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 110, rand = 9)

#Energy/WHR ~ BMI + Control
XE_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
energy_BMI_train_E, energy_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_BMI, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_BMI_train_E, WHR_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

#Energy/WHR ~ Control
XE_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
energy_Control_train_E, energy_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_Control, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_Control_train_E, WHR_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T


