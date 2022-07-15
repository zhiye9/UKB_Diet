# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 13:42:14 2021

@author: fly98
"""

from importlib.resources import read_text
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
import os
from numpy import loadtxt
from UKB_graph_metrics import *

os.chdir('/home/ubuntu/UKB/UK_Biobank_diet')
os.getcwd()

np.set_printoptions(suppress = True)     

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')

df_SCE = pd.read_csv('X_value_control.csv', index_col=0)
df_SCE = df_SCE.rename(columns={"subjects": "eid"})
df_SCE['eid'] = df_SCE['eid'].astype(str)

gmv = np.genfromtxt('GMV.txt', dtype='str')
gmv_l = np.char.add(gmv, '-')
df_gmv = df.filter(regex = convstr(gmv_l, '|'))
df_gmv['eid'] = df['eid']
df_gmv_noNAN = df_gmv.dropna(how='all', subset = df_gmv.columns[:-1])
df_gmv_noNAN.reset_index(drop = True, inplace = True)

df_gmv_mean = pd.DataFrame(0, index = np.arange(df_gmv_noNAN.shape[0]), columns = gmv)

for j in range(len(gmv_l)): 
    phe = [col for col in df_gmv_noNAN.columns if gmv_l[j] in col]
    #df_gmv_noNAN[gmv[j]] = 0
    for i in range(df_gmv_noNAN.shape[0]):
        df_gmv_mean.loc[i, gmv[j]] = np.nanmean(df_gmv_noNAN.loc[i, phe])
    print(j)

df_gmv_mean['eid'] = df_gmv_noNAN['eid']
df_gmv_mean.to_csv('GMV_mean_unnormalized.csv', index = False)
df_gmv_mean = pd.read_csv('GMV_mean_unnormalized.csv')

df_gmv_total = df.filter(regex = '25006|eid')
df_gmv_total_noNAN = df_gmv_total.dropna(how='all', subset = df_gmv_total.columns[1:])
df_gmv_total_noNAN.reset_index(drop = True, inplace = True)

df_gmv_total_mean = pd.DataFrame(0, index = np.arange(df_gmv_total_noNAN.shape[0]), columns = ['25006'])

for i in range(df_gmv_total_noNAN.shape[0]):
     df_gmv_total_mean.loc[i] = np.nanmean(df_gmv_total_noNAN.loc[i, df_gmv_total.columns[1:]])
    print("\r Process{0}%".format(round((i+1)*100/df_gmv_total_noNAN.shape[0])), end="")
    
df_gmv_total_mean['eid'] = df_gmv_total_noNAN['eid']

df_gmv_withtotal = pd.merge(df_gmv_total_mean, df_gmv_mean, on = 'eid')

for i in range(df_gmv_withtotal.shape[0]):
    df_gmv_withtotal.loc[i, gmv] = df_gmv_withtotal.loc[i, gmv]/df_gmv_withtotal.loc[i, ['25006']].values
    print("\r Process{0}%".format(round((i+1)*100/df_gmv_withtotal.shape[0])), end="")

df_gmv_mean_normalized = df_gmv_withtotal[df_gmv_withtotal.columns[1:]]
df_gmv_mean_normalized['eid'] = df_gmv_mean_normalized['eid'].astype(str)
df_gmv_mean_normalized.to_csv('GMV_mean_normalized.csv', index = False)
df_gmv_mean_normalized = pd.read_csv('GMV_mean_normalized.csv')
df_gmv_mean_normalized['eid'] = df_gmv_mean_normalized['eid'].astype(str)

df_SCE_update1 = pd.read_csv('df_SCE_update1.csv')

like_high = pd.read_csv('like_high.csv')
like_low = pd.read_csv('like_low.csv')

like_high = like_high.rename(columns={"liking": "liking_high"})
like_low = like_low.rename(columns={"liking": "liking_low"})

df_SCE_update1 = pd.merge(df_SCE_update1, like_high, on = 'eid')
df_SCE_update1 = pd.merge(df_SCE_update1, like_low, on = 'eid')
df_SCE_update1['eid'] = df_SCE_update1['eid'].astype(str)

#df_gmv_SCE = pd.merge(df_gmv_mean_normalized, df_SCE, on = 'eid')
df_gmv_SCE = pd.merge(df_gmv_mean_normalized, df_SCE_update1, on = 'eid')
df_gmv_SCE.to_csv('SCE_gmv_updated.csv', index = False)
#df_SCE_gmv_2000 = df_gmv_SCE.sample(n = 2000, random_state = 600)
df_SCE_gmv_2000 = df_gmv_SCE.sample(n = 2000, random_state = 16)
df_SCE_gmv_2000.reset_index(drop = True, inplace = True)
df_SCE_gmv_2000[gmv] = df_SCE_gmv_2000[gmv].apply(stats.zscore)
                           
df_genetic_noNAN = pd.read_csv('df_genetic.csv')

def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #find optim paramater setting in the inner cv
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
    
        #predict labels on the test set
        y_pred = clf.predict(x_train)
        #print(r2_score(y_train, y_pred))
        #print(y_pred)
        #calculate metrics
        #r2train.append(mean_squared_error(y_train, y_pred))
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #print(r2_score(y_test, y_pred))
        #print(y_pred)
        #calculate metrics
        #r2test.append(mean_squared_error(y_test, y_pred))
        r2test.append(r2_score(y_test, y_pred))
        
    return r2train, r2test, beta

pca_n = 72
#par_grid = {'alpha': [1e-2, 1e-1, 1e-8, 1e-4, 1e-6]}
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 0.3]}
p_gridsvr = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}
p_gridrf = {'max_depth': [2, 3, 4, 5, 6]}
rand = 9

X = stats.zscore(np.array(df_SCE_gmv_2000[gmv]))
#y = stats.zscore(df_SCE_gmv_2000['SCE'])
y = stats.zscore(df_SCE_gmv_2000['energy'])
y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])

#XG1 = np.concatenate((stats.zscore(np.array(df_SCE_gmv_2000[gmv])), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)

GMV_train_E, GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XG1, y = stats.zscore(df_SCE_gmv_2000['SCE']))
GMV_train_SVR , GMV_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'rbf'), X = XG1, y = stats.zscore(df_SCE_gmv_2000['SCE']))
#GMV_train_RF , GMV_test_RF = CV(p_grid = p_gridrf, out_fold = 5, in_fold = 5, model = RandomForestRegressor(), X = stats.zscore(np.array(df_SCE_gmv_2000[gmv])), y = stats.zscore(df_SCE_gmv_2000['SCE']))

#energy_GMV_train_E, energy_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.concatenate((stats.zscore(np.array(df_SCE_gmv_2000[gmv])), np.array(df_SCE_gmv_2000[['sex', 'slope']])), axis = 1), y = stats.zscore(df_SCE_gmv_2000['energy']))
#energy_GMV_train_SVR , energy_GMV_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = np.concatenate((stats.zscore(np.array(df_SCE_gmv_2000[gmv])), np.array(df_SCE_gmv_2000[['sex', 'slope']])), axis = 1), y = stats.zscore(df_SCE_gmv_2000['energy']))
XE1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GMV_train_E, energy_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE1, y = stats.zscore(df_SCE_gmv_2000['energy']), n_beta = 139, rand = 9)
#energy_GMV_train_SVR , energy_GMV_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'rbf'), X = XE1, y = stats.zscore(df_SCE_gmv_2000['energy']))

X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR1 = np.concatenate((stats.zscore(df_SCE_gmv_2000[gmv], axis = 1), np.array(df_SCE_gmv_2000[list(df_genetic_noNAN.columns[1:]) + ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 139, rand = 9)

pd.DataFrame(beta_GMV).T.to_csv('beta_GMV.csv')
pd.DataFrame(beta_GT).T.to_csv('beta_GT.csv')

IC55 = []
for i in range(0, df_SCE_gmv_2000.shape[0]):
    tem = np.loadtxt(df_SCE_gmv_2000['file'].loc[i])
    IC55.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")

#XG2 = np.concatenate((stats.zscore(np.array(IC55), axis = 1), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#IC_train_E, IC_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XG2, y = stats.zscore(df_SCE_gmv_2000['SCE']))
#IC_train_SVR, IC_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = XG2, y = stats.zscore(df_SCE_gmv_2000['SCE']))
#IC_train_RF, IC_test_RF = CV(p_grid = p_gridrf, out_fold = 5, in_fold = 5, model = RandomForestRegressor(), X = stats.zscore(np.array(IC55)), y = stats.zscore(df_SCE_gmv_2000['SCE']))

XE2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_IC_train_E, energy_IC_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE2, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#energy_IC_train_SVR , energy_IC_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = XE2, y = stats.zscore(df_SCE_gmv_2000['energy']))

X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

IC55_gmv = []
for i in range(0, df_SCE_gmv_2000.shape[0]):
    tem = np.append(np.loadtxt(df_SCE_gmv_2000['file'].loc[i]), np.array(df_SCE_gmv_2000[gmv].loc[i]))
    IC55_gmv.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")
   
XG3 = np.concatenate((stats.zscore(np.array(IC55_gmv)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XG3, y = stats.zscore(df_SCE_gmv_2000['SCE']))
IC_gmv_train_SVR, IC_gmv_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = XG3, y = stats.zscore(df_SCE_gmv_2000['SCE']))                                 

#XE3 = np.concatenate((stats.zscore(np.array(IC55_gmv)), np.array(df_SCE_gmv_2000[['sex', 'slope', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_IC_gmv_train_E, energy_IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#energy_IC_gmv_train_SVR, energy_IC_gmv_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']))                                 

#X_WHR3 = np.concatenate((stats.zscore(np.array(IC55_gmv)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)
XE3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_IC_gmv_train_E, energy_IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)

X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

X = df_SCE_gmv_2000
y = stats.zscore(df_SCE_gmv_2000['SCE'])

def PCA_CV(pca_n, p_grid, out_fold, in_fold, engy, model, X, y):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = 20)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = 20)
    r2train = []
    r2test = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train_all, x_test_all = X.loc[train], X.loc[test]
        x_train_all.reset_index(drop = True, inplace = True)
        x_test_all.reset_index(drop = True, inplace = True)
        y_train, y_test = y[train], y[test]
        #find optim paramater setting in the inner cv
        
        pca = PCA(n_components = pca_n)
        gmv_pca_train = pca.fit_transform(preprocessing.scale(x_train_all[gmv]))
        x_train = []
        if (engy == True):   
            for i in range(0, x_train_all.shape[0]):
                item = np.append(stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_train.append(item)
                print("\r enegy Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")
        else:
            for i in range(0, x_train_all.shape[0]):
                item = np.append(stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_train.append(item)
                print("\r Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")            
                
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(np.array(x_train), y_train)
        print(j)
        
        #predict labels on the test set
        y_pred = clf.predict(x_train)
        #print(r2_score(y_train, y_pred))
        #print(y_pred)
        #calculate metrics
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        gmv_pca_test = pca.transform(preprocessing.scale(x_test_all[gmv]))
        x_test = []
        if (engy == True): 
            for i in range(0, x_test_all.shape[0]):
                item = np.append(stats.zscore(gmv_pca_test[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_test.append(item)
                print("\r energy test Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")
        else:
            for i in range(0, x_test_all.shape[0]):
                item = np.append(stats.zscore(gmv_pca_test[i]), df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_test.append(item)
                print("\r test Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")            
             
        y_pred = clf.predict(np.array(x_test))
        #print(r2_score(y_test, y_pred))
        #print(y_pred)
        #calculate metrics
        r2test.append(r2_score(y_test, y_pred))
        
    return r2train, r2test

def PCA_ICA_CV(pca_n, p_grid, out_fold, in_fold, engy, model, X, y):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = 20)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = 20)
    r2train = []
    r2test = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train_all, x_test_all = X.loc[train], X.loc[test]
        x_train_all.reset_index(drop = True, inplace = True)
        x_test_all.reset_index(drop = True, inplace = True)
        y_train, y_test = y[train], y[test]
        #find optim paramater setting in the inner cv
        
        pca = PCA(n_components = pca_n)
        gmv_pca_train = pca.fit_transform(preprocessing.scale(x_train_all[gmv]))
        x_train = []
        if (engy == True):   
            for i in range(0, x_train_all.shape[0]):
                item = np.append(stats.zscore(np.append(stats.zscore(np.loadtxt(x_train_all['file'].loc[i])), stats.zscore(gmv_pca_train[i]))), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_train.append(item)
                print("\r energy Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")
        else:
            for i in range(0, x_train_all.shape[0]):
                item = np.append(stats.zscore(np.append(stats.zscore(np.loadtxt(x_train_all['file'].loc[i])), stats.zscore(gmv_pca_train[i]))), df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_train.append(item)
                print("\r Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")
           
         
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(np.array(x_train), y_train)
        print(j)
        
        #predict labels on the test set
        y_pred = clf.predict(np.array(x_train))
        #print(r2_score(y_train, y_pred))
        #print(y_pred)
        #calculate metrics
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        gmv_pca_test = pca.transform(preprocessing.scale(x_test_all[gmv]))
        x_test = []
        if (engy == True):  
            for i in range(0, x_test_all.shape[0]):
                item = np.append(stats.zscore(np.append(stats.zscore(np.loadtxt(x_test_all['file'].loc[i])), stats.zscore(gmv_pca_test[i]))), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_test.append(item)
                print("\r energy test Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")
        else:
            for i in range(0, x_test_all.shape[0]):
                item = np.append(stats.zscore(np.append(stats.zscore(np.loadtxt(x_test_all['file'].loc[i])), stats.zscore(gmv_pca_test[i]))), df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]])
                x_test.append(item)
                print("\r test Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")
             
        y_pred = clf.predict(np.array(x_test))
        #print(r2_score(y_test, y_pred))
        #print(y_pred)
        #calculate metrics
        r2test.append(r2_score(y_test, y_pred))
        print(j)
        
    return r2train, r2test

energy_IC_GMVPCA_train_E, energy_IC_GMVPCA_test_E = PCA_ICA_CV(pca_n = 72, p_grid = par_grid, out_fold = 5, in_fold = 5, engy = True, model = ElasticNet(max_iter = 1000000), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['energy']))
energy_IC_GMVPCA_train_SVR, energy_IC_GMVPCA_test_SVR = PCA_ICA_CV(pca_n = 72, p_grid = p_gridsvr, out_fold = 5, in_fold = 5, engy = True, model = SVR(kernel='poly'), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['energy'])) 

IC_GMVPCA_train_E, IC_GMVPCA_test_E = PCA_ICA_CV(pca_n = 72, p_grid = par_grid, out_fold = 5, in_fold = 5, engy = False, model = ElasticNet(max_iter = 1000000), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['SCE']))
IC_GMVPCA_train_SVR, IC_GMVPCA_test_SVR = PCA_ICA_CV(pca_n = 72, p_grid = p_gridsvr, out_fold = 5, in_fold = 5, engy = False, model = SVR(kernel='poly'), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['SCE'])) 

energy_GMVPCA_train_E, energy_GMVPCA_test_E = PCA_CV(pca_n = 72, p_grid = par_grid, out_fold = 5, in_fold = 5, engy = True, model = ElasticNet(max_iter = 1000000), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['energy']))
energy_GMVPCA_train_SVR, energy_GMVPCA_test_SVR = PCA_CV(pca_n = 72, p_grid = p_gridsvr, out_fold = 5, in_fold = 5, engy = True, model = SVR(kernel='poly'), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['energy'])) 

GMVPCA_train_E, GMVPCA_test_E = PCA_CV(pca_n = 72, p_grid = par_grid, out_fold = 5, in_fold = 5, engy = False, model = ElasticNet(max_iter = 1000000), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['SCE']))
GMVPCA_train_SVR, GMVPCA_test_SVR = PCA_CV(pca_n = 72, p_grid = p_gridsvr, out_fold = 5, in_fold = 5, engy = False, model = SVR(kernel='poly'), X = df_SCE_gmv_2000, y = stats.zscore(df_SCE_gmv_2000['SCE'])) 

#df_gmvwithSCE = df_gmv_noNAN.loc[df_gmv_noNAN['eid'].isin(df_SCE['eid'])]
#df_gmv_noNAN_scaled = preprocessing.scale(df_gmvwithSCE[gmv_l])
#pca = PCA(n_components = 'mle')
#pca = PCA(n_components = 72)
#gmv_pca = pca.fit_transform(df_gmv_noNAN_scaled[5:])

#ICGM = []
#for i in range(0, df_SCE_gmv_2000.shape[0]):
#    tem = np.append(stats.zscore(weight_central(df_SCE_gmv_2000['file'].loc[i], 55)), stats.zscore(np.array(df_SCE_gmv_2000[gmv].loc[i])))
#    ICGM.append(tem)
#    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")
   
#CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 100000), X = np.array(ICGM), y = y)
#CV(p_grid = p_gridsvr, out_fold = 5, in_fold = np.array(ICGM), model = SVR(), X = X, y = y)


 
IC_Graph_df = pd.DataFrame(np.array(IC_Graph))
IC_Graph_df['eid'] = df_SCE_gmv_2000['eid']
IC_Graph_df.to_csv('IC_graphmetrics_2000.csv', index = False)

IC_Graph_df = pd.read_csv('IC_graphmetrics_2000.csv')
IC_Graph_df['eid'] = IC_Graph_df['eid'].astype(str)

df_Graph_gmv_SCE_2000 = pd.merge(IC_Graph_df, df_SCE_gmv_2000, on = 'eid')

IC_Graph_binary = []

for i in range(0, df_SCE_gmv_2000.shape[0]):
    IC_Graph_binary.append(Graph_binary_metrics(df_SCE_gmv_2000['file'].loc[i], 55))
    if IC_Graph_binary[i].shape[0] != 114:
        print(i, 'id is', df_SCE_gmv_2000['file'].loc[i], 'shape is', IC_Graph_binary[i].shape[0])

    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")
    
#IC_Graph_binary_df = pd.DataFrame(np.array(IC_Graph_binary))
#IC_Graph_binary_df['eid'] = df_SCE_gmv_2000['eid']
#IC_Graph_binary_df.to_csv('IC_binary_graphmetrics_2000.csv', index = False)

#IC_Graph_df = pd.read_csv('IC_graphmetrics_2000.csv')
#IC_Graph_df['eid'] = IC_Graph_df['eid'].astype(str)

#df_Graph_binary_gmv_SCE_2000 = pd.merge(IC_Graph_binary_df, df_SCE_gmv_2000, on = 'eid')

def Graph_CV(p_grid, out_fold, pca_n, engy, in_fold, model, X, y, rand):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train_all, x_test_all = X.loc[train], X.loc[test]
        #x_train_all.reset_index(drop = True, inplace = True)
        #x_test_all.reset_index(drop = True, inplace = True)
        y_train, y_test = y[train], y[test]
        #find optim paramater setting in the inner cv
        
        pca = PCA(n_components = pca_n)
        gmv_pca_train = pca.fit_transform(preprocessing.scale(x_train_all[gmv]))
        x_train = []
        if (engy == True):
            for i in range(0, x_train_all.shape[0]):
                std_scaler = StandardScaler()
                #item = np.append(np.append(np.append(np.append(std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), 
                                           #std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel()), stats.zscore(gmv_pca_train[i])), df_SCE_gmv_2000['sex'].loc[x_train_all.index[i]]), df_SCE_gmv_2000['slope'].loc[x_train_all.index[i]])
                item = np.concatenate((stats.zscore(x_train_all[gmv].loc[x_train_all.index[i]]), x_train_all[list(df_genetic_noNAN.columns[1:]) + ['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio']].loc[x_train_all.index[i]]))           
                #item = np.concatenate((stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))           
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_trXain_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))           
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), stats.zscore(x_train_all[gmv].loc[x_train_all.index[i]]), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))           
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))           
                x_train.append(item)
                print("\r energy Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")
        else:
            for i in range(0, x_train_all.shape[0]):
                std_scaler = StandardScaler()
                #item = np.append(np.append(std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), 
                                           #std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel()), stats.zscore(gmv_pca_train[i]))                
                item = np.concatenate((stats.zscore(x_train_all[gmv].loc[x_train_all.index[i]]), x_train_all[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))           
                #item = np.concatenate((stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), stats.zscore(gmv_pca_train[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), stats.zscore(x_train_all[gmv].loc[x_train_all.index[i]]), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))            
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_train_all.loc[x_train_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_train_all.index[i]]))            
                x_train.append(item)
                print("\r Process{}%".format(round((i+1)*100/x_train_all.shape[0])), end="")
        clf = GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(np.array(x_train), y_train)
        print(j)
            
        #predict labels on the test set
        y_pred_train = clf.predict(np.array(x_train))
        #print(r2_score(y_train, y_pred))
        #print(y_pred)
        #calculate metrics
        r2train.append(r2_score(y_train, y_pred_train))
        
        #predict labels on the test set
        gmv_pca_test = pca.transform(preprocessing.scale(x_test_all[gmv]))
        x_test = []
        if (engy == True):
            for i in range(0, x_test_all.shape[0]):
                #item = np.append(np.append(np.append(np.append(std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), 
                                       #std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel()), stats.zscore(gmv_pca_test[i])), df_SCE_gmv_2000['sex'].loc[x_test_all.index[i]]), df_SCE_gmv_2000['slope'].loc[x_test_all.index[i]])
                item = np.concatenate((stats.zscore(x_test_all[gmv].loc[x_test_all.index[i]]), x_test_all[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((stats.zscore(gmv_pca_test[i]), x_test_all[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), stats.zscore(gmv_pca_test[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), stats.zscore(x_test_all[gmv].loc[x_test_all.index[i]]), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                x_test.append(item)
                print("\r energy Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")
        else:
            for i in range(0, x_test_all.shape[0]):
                #item = np.append(np.append(std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), 
                                       #std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel()), stats.zscore(gmv_pca_test[i]))
                item = np.concatenate((stats.zscore(x_test_all[gmv].loc[x_test_all.index[i]]), x_test_all[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((stats.zscore(gmv_pca_test[i]), x_test_all[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))                               
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), stats.zscore(gmv_pca_test[i]), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), stats.zscore(x_test_all[gmv].loc[x_test_all.index[i]]), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                #item = np.concatenate((std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][0:110]).reshape(-1, 1)).ravel(), std_scaler.fit_transform(np.array(x_test_all.loc[x_test_all.index[i]][110:114]).reshape(-1, 1)).ravel(), df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])].loc[x_test_all.index[i]]))
                x_test.append(item)
                print("\r Process{}%".format(round((i+1)*100/x_test_all.shape[0])), end="")
        y_pred = clf.predict(np.array(x_test))
        #print(r2_score(y_test, y_pred))
        #print(y_pred)
        #calculate metrics
        r2test.append(r2_score(y_test, y_pred))
        
    return r2train, r2test

GT_GMV_train_E, GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['SCE']))
#GT_GMV_train_SVR, GT_GMV_test_SVR = Graph_CV(p_grid = p_gridsvr, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = SVR(), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['SCE']))       

energy_GT_GMV_train_E, energy_GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']))
#energy_GT_GMV_train_SVR, energy_GT_GMV_test_SVR = Graph_CV(p_grid = p_gridsvr, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = SVR(), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']))       

GT_binary_GMV_train_E, GT_binary_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_binary_gmv_SCE_2000, y = stats.zscore(df_Graph_binary_gmv_SCE_2000['SCE']))
GT_binary_GMV_train_SVR, GT_binary_GMV_test_SVR = Graph_CV(p_grid = p_gridsvr, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = SVR(), X = df_Graph_binary_gmv_SCE_2000, y = stats.zscore(df_Graph_binary_gmv_SCE_2000['SCE']))       

energy_GT_binary_GMV_train_E, energy_GT_binary_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_binary_gmv_SCE_2000, y = stats.zscore(df_Graph_binary_gmv_SCE_2000['energy']))
energy_GT_binary_GMV_train_SVR, energy_GT_binary_GMV_test_SVR = Graph_CV(p_grid = p_gridsvr, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = SVR(), X = df_Graph_binary_gmv_SCE_2000, y = stats.zscore(df_Graph_binary_gmv_SCE_2000['energy']))       

#energy_GT_GMV_train_E, energy_GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']), rand = 9)
XE_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GT_GMV_train_E, energy_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

energy_GMV_train_E12, energy_GMV_test_E12 = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']), rand = 9)
GMV_train_WHR12, GMV_test_WHR12 = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)

#energy_GT_train_E, energy_GT_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']), rand = 9)
XE_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
energy_GT_train_E, energy_GT_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), n_beta = 110, rand = 9)
#WHR_GT_train_E, WHR_GT_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)
X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 110, rand = 9)

pd.DataFrame(beta_GMV).T.to_csv('beta_GMV.csv')

energy_GT_GMVPCA_train_E, energy_GT_GMVPCA_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']), rand = 9)
WHR_GT_GMVPCA_train_E, WHR_GT_GMVPCA_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)

energy_GMVPCA_train_E, energy_GMVPCA_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = True, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['energy']), rand = 9)
WHR_GMVPCA_train_E, WHR_GMVPCA_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
XE_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
energy_BMI_train_E, energy_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_BMI, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
#WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)
WHR_BMI_train_E, WHR_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

XE_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
energy_Control_train_E, energy_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_Control, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
X_WHR_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
#WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = Graph_CV(p_grid = par_grid, out_fold = 5, pca_n = 72, engy = False, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = df_Graph_gmv_SCE_2000, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)
WHR_Control_train_E, WHR_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)


------------------------------------------------------------------------------------------------------------------------------------------------------------------------
X_WHR_ICA_GT = np.concatenate((stats.zscore(np.array(df_SCE_gmv_2000[gmv])), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_ICA_GT, y = stats.zscore(df_Graph_gmv_SCE_2000['waist_hip_ratio']), rand = 9)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

labels = ['SCE_Elastic', 'SCE_SVR', 'energy_Elsatic', 'energy_SVR']

x = np.arange(len(labels)) 
train_mean = [np.mean(GT_GMV_train_E), np.mean(GT_GMV_train_SVR), np.mean(energy_GT_GMV_train_E), np.mean(energy_GT_GMV_train_SVR)]
test_mean = [np.mean(GT_GMV_test_E), np.mean(GT_GMV_test_SVR), np.mean(energy_GT_GMV_test_E), np.mean(energy_GT_GMV_test_SVR)]
train_std = [np.std(GT_GMV_train_E), np.std(GT_GMV_train_SVR), np.std(energy_GT_GMV_train_E), np.std(energy_GT_GMV_train_SVR)]
test_std = [np.std(GT_GMV_test_E), np.std(GT_GMV_test_SVR), np.std(energy_GT_GMV_test_E), np.std(energy_GT_GMV_test_SVR)]
    
# Save the figure and show
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png', dpi = 4000)
#plt.show()


fig, ax = plt.subplots( dpi = 1600)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2 Scores')
ax.set_title('Regression results of graph SCE energy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
plt.ylim([0, 1])
fig.tight_layout()

plt.show()

# Build the plot
labels = ['GMV_Elastic', 'GMV_SVR', 'ICA_Elastic', 'ICA_SVR', 'GMV_ICA_Elastic', 'GMV_ICA_SVR', 'GMVPCA_ICAGT_Elastic', 'GMVPCA_ICAGT_SVR']

x = np.arange(len(labels)) 
train_mean = [np.mean(GMV_train_E), np.mean(GMV_train_SVR), np.mean(IC_train_E), np.mean(IC_train_SVR), 
              np.mean(IC_gmv_train_E), np.mean(IC_gmv_train_SVR), np.mean(GT_GMV_train_E), np.mean(GT_GMV_train_SVR)]
test_mean = [np.mean(GMV_test_E), np.mean(GMV_test_SVR), np.mean(IC_test_E), np.mean(IC_test_SVR),
             np.mean(IC_gmv_test_E), np.mean(IC_gmv_test_SVR), np.mean(GT_GMV_test_E), np.mean(GT_GMV_test_SVR)]
train_std = [np.std(GMV_train_E), np.std(GMV_train_SVR), np.std(IC_train_E), np.std(IC_train_SVR),
             np.std(IC_gmv_train_E), np.std(IC_gmv_train_SVR), np.std(GT_GMV_train_E), np.std(GT_GMV_train_SVR)]
test_std = [np.std(GMV_test_E), np.std(GMV_test_SVR), np.std(IC_test_E), np.std(IC_test_SVR),
           np.std(IC_gmv_test_E), np.std(IC_gmv_test_SVR), np.std(GT_GMV_test_E), np.std(GT_GMV_test_SVR)]
    
# Save the figure and show
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png', dpi = 4000)
#plt.show()


fig, ax = plt.subplots( dpi = 1600)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2 Scores')
ax.set_title('Regression results of SCE')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
plt.savefig('energy_bar_plot_with_error_bars.png')


labels = ['GMV_Elastic', 'GMV_SVR', 'ICA_Elastic', 'ICA_SVR', 'GMV_ICA_Elastic', 'GMV_ICA_SVR', 'GMVPCA_ICA_Elastic', 'GMVPCA_ICA_SVR', 'GMVPCA_ICAGT_Elastic', 'GMVPCA_ICAGT_SVR']

x = np.arange(len(labels)) 
train_mean = [np.mean(energy_GMV_train_E), np.mean(energy_GMV_train_SVR), np.mean(energy_GMVPCA_train_E), np.mean(energy_GMVPCA_train_SVR), np.mean(energy_IC_train_E), np.mean(energy_IC_train_SVR), 
              np.mean(energy_IC_gmv_train_E), np.mean(energy_IC_gmv_train_SVR), np.mean(energy_IC_GMVPCA_train_E), np.mean(energy_IC_GMVPCA_train_SVR), np.mean(energy_GT_GMV_train_E), np.mean(energy_GT_GMV_train_SVR)]
test_mean = [np.mean(energy_GMV_test_E), np.mean(energy_GMV_test_SVR), np.mean(energy_IC_test_E), np.mean(energy_IC_test_SVR),
             np.mean(energy_IC_gmv_test_E), np.mean(energy_IC_gmv_test_SVR), np.mean(energy_GT_GMV_test_E), np.mean(energy_GT_GMV_test_SVR)]
train_std = [np.std(energy_GMV_train_E), np.std(energy_GMV_train_SVR), np.std(energy_GMVPCA_train_E), np.std(energy_GMVPCA_train_SVR), np.std(energy_IC_train_E), np.std(energy_IC_train_SVR),
             np.std(energy_IC_gmv_train_E), np.std(energy_IC_gmv_train_SVR), np.std(energy_IC_GMVPCA_train_E), np.std(energy_IC_GMVPCA_train_SVR), np.std(energy_GT_GMV_train_E), np.std(energy_GT_GMV_train_SVR)]
test_std = [np.std(energy_GMV_test_E), np.std(energy_GMV_test_SVR), np.std(energy_GMVPCA_test_E), np.std(energy_GMVPCA_test_SVR), np.std(energy_IC_test_E), np.std(energy_IC_test_SVR),
           np.std(energy_IC_gmv_test_E), np.std(energy_IC_gmv_test_SVR), np.std(energy_IC_GMVPCA_test_E), np.std(energy_IC_GMVPCA_test_SVR), np.std(energy_GT_GMV_test_E), np.std(energy_GT_GMV_test_SVR)]
    
# Save the figure and show
#plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars.png', dpi = 4000)
#plt.show()

fig, ax = plt.subplots( dpi = 1600)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2 Scores')
ax.set_title('Regression results of energy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

#labels = ['Energy_GMV', 'WHR_GMV', 'Energy_ICA', 'WHR_ICA', 'Energy_GMV_ICA', 'WHR_GMV_ICA', 'Energy_GT', 'WHR_GT', 'Energy_GMV_GT', 'WHR_GMV_GT', 'Energy_BMI', 'WHR_BMI', 'Energy_Control', 'WHR_Control']
labels = ['WHR_GMV', 'WHR_ICA', 'WHR_GMV_ICA', 'WHR_GT', 'WHR_GMV_GT', 'WHR_BMI', 'WHR_Control']
#labels = ['WHR_GMV', 'WHR_ICA', 'WHR_GMV_ICA', 'WHR_GT', 'WHR_GMV_GT']
dpi = 1600
title = 'Regression results of WHR'

x = np.arange(len(labels)) 
#train_mean = [np.mean(energy_GMV_train_E), np.mean(GMV_train_WHR), np.mean(energy_IC_train_E), np.mean(IC_train_WHR), np.mean(energy_IC_gmv_train_E), np.mean(IC_gmv_train_E), np.mean(energy_GT_train_E), np.mean(WHR_GT_train_E), np.mean(energy_GT_GMV_train_E), np.mean(WHR_GT_GMV_train_E), np.mean(energy_BMI_train_E), np.mean(WHR_BMI_train_E), np.mean(energy_Control_train_E), np.mean(WHR_Control_train_E)]
#test_mean = [np.mean(energy_GMV_test_E), np.mean(GMV_test_WHR), np.mean(energy_IC_test_E), np.mean(IC_test_WHR), np.mean(energy_IC_gmv_test_E), np.mean(IC_gmv_test_E), np.mean(energy_GT_test_E), np.mean(WHR_GT_test_E), np.mean(energy_GT_GMV_test_E), np.mean(WHR_GT_GMV_test_E), np.mean(energy_BMI_test_E), np.mean(WHR_BMI_test_E), np.mean(energy_Control_test_E), np.mean(WHR_Control_test_E)]
#train_std = [np.std(energy_GMV_train_E), np.std(GMV_train_WHR), np.std(energy_IC_train_E), np.std(IC_train_WHR), np.std(energy_IC_gmv_train_E), np.std(IC_gmv_train_E), np.std(energy_GT_train_E), np.std(WHR_GT_train_E), np.std(energy_GT_GMV_train_E), np.std(WHR_GT_GMV_train_E), np.std(energy_BMI_train_E), np.std(WHR_BMI_train_E), np.std(energy_Control_train_E), np.std(WHR_Control_train_E)]
#test_std = [np.std(energy_GMV_test_E), np.std(GMV_test_WHR), np.std(energy_IC_test_E), np.std(IC_test_WHR), np.std(energy_IC_gmv_test_E), np.std(IC_gmv_test_E), np.std(energy_GT_test_E), np.std(WHR_GT_test_E), np.std(energy_GT_GMV_test_E), np.std(WHR_GT_GMV_test_E), np.std(energy_BMI_test_E), np.std(WHR_BMI_test_E), np.std(energy_Control_test_E), np.std(WHR_Control_test_E)]    

#train_mean = [np.mean(GMV_train_WHR), np.mean(IC_train_WHR), np.mean(IC_gmv_train_E), np.mean(WHR_GT_train_E), np.mean(WHR_GT_GMV_train_E)]
#test_mean = [np.mean(GMV_test_WHR), np.mean(IC_test_WHR), np.mean(IC_gmv_test_E), np.mean(WHR_GT_test_E), np.mean(WHR_GT_GMV_test_E)]
#train_std = [np.std(GMV_train_WHR), np.std(IC_train_WHR), np.std(IC_gmv_train_E), np.std(WHR_GT_train_E), np.std(WHR_GT_GMV_train_E)]
#test_std = [np.std(GMV_test_WHR), np.std(IC_test_WHR), np.std(IC_gmv_test_E), np.std(WHR_GT_test_E), np.std(WHR_GT_GMV_test_E)]    

train_mean = [np.mean(GMV_train_WHR), np.mean(IC_train_WHR), np.mean(IC_gmv_train_E), np.mean(WHR_GT_train_E), np.mean(WHR_GT_GMV_train_E), np.mean(WHR_BMI_train_E), np.mean(WHR_Control_train_E)]
test_mean = [np.mean(GMV_test_WHR), np.mean(IC_test_WHR), np.mean(IC_gmv_test_E), np.mean(WHR_GT_test_E), np.mean(WHR_GT_GMV_test_E), np.mean(WHR_BMI_test_E), np.mean(WHR_Control_test_E)]
train_std = [np.std(GMV_train_WHR), np.std(IC_train_WHR), np.std(IC_gmv_train_E), np.std(WHR_GT_train_E), np.std(WHR_GT_GMV_train_E), np.std(WHR_BMI_train_E), np.std(WHR_Control_train_E)]
test_std = [np.std(GMV_test_WHR), np.std(IC_test_WHR), np.std(IC_gmv_test_E), np.std(WHR_GT_test_E), np.std(WHR_GT_GMV_test_E), np.std(WHR_BMI_test_E), np.std(WHR_Control_test_E)]    

fig, ax = plt.subplots(dpi = dpi)

width = 0.35
rects1 = ax.bar(x - width/2, train_mean, width, yerr = train_std, label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, test_mean, width, yerr = test_std, label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('R2 Scores')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1, 1))
plt.xticks(rotation=60)
plt.yticks(np.arange(0, 0.9, step=0.1))
#plt.yticks(np.linspace(0.0,0.7,0.1))
ax.set_xticklabels(labels, fontsize = 9)
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()                                                                                                                   

percent = []
for i in range(1, df_SCE.shape[0]):
    IC = np.loadtxt(df_SCE['file'].loc[i])
    size = 55
    corr_matrix = np.zeros((size,size))
    corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC
    corr_matrix = corr_matrix + corr_matrix.T
    G1 = nx.from_numpy_array(corr_matrix)
    pos_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']>0]
    neg_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']<0]
    percent.append(len(pos_edges)/1485)
    print("\r Process{}%".format(round((i+1)*100/df_SCE.shape[0])), end="")

ax = plt.plot(dpi = 1600)
ax.set_title('Postive edges ratio')
ax.hist(percent)
--------------------------------------------------------------------------------------------------

df_gmvwithSCE = df_gmv_noNAN.loc[df_gmv_noNAN['eid'].isin(df_SCE['eid'])]
#df_SCEwithgmv = df_SCE.loc[df_SCE['eid'].isin(df_gmv_noNAN['eid'])]

df_gmv_noNAN_scaled = preprocessing.scale(df_gmvwithSCE[gmv_l])
pca = PCA(n_components = 'mle')
pca = PCA(n_components = 72)
gmv_pca = pca.fit_transform(df_gmv_noNAN_scaled[5:])
print(pca.explained_variance_ratio_)
len(pca.explained_variance_ratio_)

def plotCumSumVariance(var=None):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save file
    cumvar = var.cumsum()

    plt.figure()
    plt.bar(np.arange(len(var)), cumvar, width = 1.0)
    plt.axhline(y=0.9, color='r', linestyle='-')

plotCumSumVariance(pca.explained_variance_ratio_)

df_SCE_2000 = df_SCEwithgmv.sample(n = 2000,      = 99)
df_SCE_2000.reset_index(drop = True, inplace = True)
IC55 = []
for i in range(0, df_SCE_2000.shape[0]):
    tem = stats.zscore(np.loadtxt(df_SCE_2000['file'].loc[i]))
    IC55.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_2000.shape[0])), end="")
    
data1 = {"a":[1.,3.,5.,2.],
         "b":[4.,8.,3.,7.],
         "c":[5.,45.,67.,34]}
data2 = {"a":[4., 2., 1., 10.]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2) 

for i in range(4):
    df1.loc[i, ['b', 'c']] = df1.loc[i, ['b', 'c']]/df1.loc[i, ['a']].values

IC_Graph = []
for i in range(df_SCE_gmv_2000.shape[0]):
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")
    
IC_Graph_df = pd.DataFrame(np.array(IC_Graph))
IC_Graph_df['eid'] = df_SCE_gmv_2000['eid']
IC_Graph_df.to_csv('IC_graphmetrics_2000.csv', index = False)
t = pd.read_csv('IC_graphmetrics_2000.csv')
t == IC_Graph_df
type(IC_Graph_df.columns[0])
IC_Graph_df.columns = IC_Graph_df.columns.astype(str)


df_check = df.filter(regex = '31|21003|21001|6138||eid')
df_height = df.filter(regex = '12144|eid')
df_height_noNAN = df_height.dropna(subset = df_height.columns[1:],how = 'all')
df_height_noNAN['eid'] = df_height_noNAN['eid'].astype(str)

X_WHR1 = np.concatenate((stats.zscore(np.array(df_SCE_gmv_2000[gmv])), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
GMV_train_WHR, GMV_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']), rand = 9)

mu, sigma = 0, 2
s = np.random.normal(mu, sigma, 100)
plt.ylim([-5, 5])
#plt.scatter(range(100), s)
plt.scatter(range(100), stats.zscore(s))

plt.scatter(df_gmv_SCE[['liking_low']], df_gmv_SCE[['waist_hip_ratio']], s = 1)
plt.xlabel('liking of low calorie food')
plt.ylabel('waist-hip-ratio')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

plt.scatter(df_gmv_SCE[['liking_high']], df_gmv_SCE[['waist_hip_ratio']], s = 1)
plt.xlabel('liking of high calorie food')
plt.ylabel('waist-hip-ratio')
plt.xlim(-5, 5)
plt.ylim(-5, 5)

import scipy
np.corrcoef(np.array(df_gmv_SCE['liking_low']), np.array(df_gmv_SCE['waist_hip_ratio']))
np.corrcoef(np.array(df_gmv_SCE['liking_high']), np.array(df_gmv_SCE['waist_hip_ratio']))
stats.pearsonr(df_gmv_SCE['liking_high'], df_gmv_SCE['waist_hip_ratio'])
stats.pearsonr(df_gmv_SCE['liking_low'], df_gmv_SCE['waist_hip_ratio'])

df_gmv_SCE['liking_high'].corr(df_gmv_SCE['waist_hip_ratio'])

-------------------------------------------------------------------------------------------------
#from nilearn.plotting import plot_prob_atlas
import nilearn.plotting as plotting
import nilearn as nl
import nibabel as nib
from nilearn.input_data import NiftiMasker

#importlib.reload(nilearn)
from nilearn import plotting

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
ica100_np = np.array(ica100_template.dataobj)

n = 2
plotting.plot_roi(ica100_template.slicer[..., n])
plotting.plot_img(ica100_template.slicer[..., n])
plotting.plot_prob_atlas(ica100_template)

from nilearn.image import get_data
fmri_data = get_data(ica100_template.slicer[..., n])

fmri_data[fmri_data > 10] = 10
fmri_data[fmri_data < 10] = 0

from nilearn.image import new_img_like
new_fmri_img = new_img_like(ica100_template.slicer[..., n], fmri_data)
plotting.plot_roi(new_fmri_img)

from nilearn.datasets import load_mni152_template, load_mni152_gm_template

template = load_mni152_gm_template(resolution = 2)
print(template.shape)
print(template.header.get_zooms())
plotting.plot_img(template)

a1 = nib.load('Cerebellum-MNInorm-prob.nii')
a1.shape
a2 = nib.load('Cerebellum-MNIflirt-prob.nii')
a2.shape
a34 = nib.load('Cerebellum-MNIfnirt-prob.nii')
a3 = nib.load('Cerebellum-MNIfnirt-maxprob-thr25.nii')
a3.shape
a4 = nib.load('Cerebellum-SUIT-prob.nii')
a4.shape

Cerebellum_label = pd.read_csv('Cerebellum_MNIfnirt.csv', header = None, usecols=[6], names = ['atlas'])['atlas'].values.tolist()[2:]

from nilearn import datasets

#dataset_ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
harvard_oxford_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr0-2mm')
harvard_oxford_sub1 = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')
h_o_cort_s = nib.load(harvard_oxford_s.filename)
h_o_cort_s.shape
harvard_oxford.labels
h_o_sub = nib.load(harvard_oxford_sub.filename)
h_o_sub.shape

h_o_sub1 = nib.load(harvard_oxford_sub1.filename)

from nilearn.masking import intersect_masks
from nilearn.image import new_img_like
import nibabel.processing

'''
def extract_atlas(file, n):
    GMatlas = nib.load(file)
    data = get_data(GMatlas)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(GMatlas, data)
    resampled_GM = resample_to_img(new_img, nib.load(harvard_oxford_s.filename))
    return resampled_GM
'''

GMatlas = nib.load('GMatlas.nii.gz')
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_sub.filename))
new_GM_id = [x+1 for x in GM_id]
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id)] = 1
GM_template_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask, cut_coords = [-1, -44, 12])
nib.save(GM_template_mask, 'GM_mask_all34.nii')
print(np.count_nonzero(get_data(GM_template_mask)))

#def extract_GMV(atlas, n):
#    data = get_data(atlas)
 #   data[data != n] = 0
 #   data[data == n] = 1
 #   new_img = new_img_like(atlas, data)
 #  return new_img

GM = extract_atlas('GMatlas.nii.gz', 1)
d1 = extract_atlas(harvard_oxford_sub.filename, 1)
d2 = extract_atlas(harvard_oxford_s.filename, 2)
cd1 = extract_atlas(harvard_oxford_s.filename, 1)
cd2 = extract_atlas(harvard_oxford_s.filename, 2)
c1 = extract_atlas('Cerebellum-MNIfnirt-maxprob-thr25.nii', 1)
t1 = intersect_masks([d2,hh], threshold = 1)
ct1 = intersect_masks([cd2,hh], threshold = 1)

def extract_IC(IC, n):
    data = get_data(IC.slicer[..., n])
    data[data < 5] = 0
    data[data >= 5] = 1
    new_img = new_img_like(IC.slicer[..., n], data)
    return new_img

IC_template_mask = []
for i in range(len(GT_id)):
    IC_template_mask.append(extract_IC(ica100_template, GT_id[i] - 1))

new_IC_template_mask = intersect_masks(IC_template_mask, threshold = 0)
plotting.plot_roi(new_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask)))
nib.save(new_IC_template_mask, 'IC_mask_all24.nii')

GM_IC_template_mask = intersect_masks([new_IC_template_mask, GM_template_mask], threshold = 1)
print(np.count_nonzero(get_data(GM_IC_template_mask)))
plotting.plot_roi(GM_IC_template_mask, cut_coords = [-1, -44, 12])
nib.save(GM_IC_template_mask, 'GM34_IC24_mask_all.nii')


plotting.plot_roi(extract_atlas('GMatlas.nii.gz', GM_id[30] + 1), cut_coords = [-1, -44, 12])
plotting.plot_roi(extract_IC(ica100_template, 23), cut_coords = [-1, -44, 12])
plotting.plot_roi(intersect_masks([extract_atlas('GMatlas.nii.gz', GM_id[30] + 1), extract_IC(ica100_template, 23)], threshold = 1), cut_coords = [-1, -44, 12])
np.count_nonzero(get_data(intersect_masks([extract_atlas('GMatlas.nii.gz', GM_id[30] + 1), extract_IC(ica100_template, 23)], threshold = 1)))
np.count_nonzero(get_data(extract_IC(ica100_template, 23)))
np.count_nonzero(get_data(extract_atlas('GMatlas.nii.gz', GM_id[30] + 1)))
        
#for n in range(10):
#    plotting.plot_roi(extract_IC(ica100_template, GT_id[n]))

ho_data = get_data(h_o_cort_s)

#ho_data[ho_data == 2] = 2
ho_data[ho_data != 2] = 0

new_ho_img = new_img_like(h_o_cort_s, ho_data)
plotting.plot_roi(new_ho_img)

atlas_ho_filename = dataset_ho.filename
H1 = nib.load(atlas_ho_filename)
H1.shape
nn = np.unique(get_data(H1))

beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')
beta_GMV_id.to_csv('beta_GMV_id.csv', index = False)
beta_GMV_id = pd.read_csv('beta_GMV_id.csv')
beta_GMV_id_nonzero = beta_GMV_id[(beta_GMV_id != 0).all(1)]
beta_GMV_id_nonzero.reset_index(drop = True, inplace = True)
beta_GMV_id_nonzero.to_csv('beta_GMV_id_nonzero.csv', index = False)
beta_GMV_id_nonzero = pd.read_csv('beta_GMV_id_nonzero.csv')

beta_GT_df = pd.read_csv('beta_GT.csv')
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i-1]}' for i in range(1, 56)] + [f'IC_neg {ICA_good_100[i-1]}' for i in range(1, 56)]
beta_GT_nonzero = beta_GT_df[(beta_GT_df != 0).all(1)]

GT_id = np.sort(np.unique(beta_GT_nonzero['ICs'].str[7:], return_counts = True)[0].astype(int))

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
ica100_np = np.array(ica100_template.dataobj)

GMatlas = nib.load('GMatlas.nii.gz')
GMatlas1 = nib.load('GMatlas_v_1.0__until_11_Dec_2016.nii.gz')
GMatlasTemplate = nib.load('template_GM.nii.gz')
GMatlasTemplate.shape
ica100_template.shape

from nilearn.image import resample_to_img

resampled_GM = resample_to_img(GMatlas, h_o_cort_s)
resampled_GM1 = resample_to_img(GMatlas, a3)

from numpy import loadtxt
GM_labels = loadtxt("GMatlas_name.txt", dtype=str, delimiter="\t", unpack=False).tolist()

GM_id = []
for i in range(beta_GMV_id_nonzero.shape[0]):
    if any(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id.append(indices[0])

ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

IC_mask_id = []
for per in range(2, 10, 2):
    IC_mask_id_run = []
    for k in range(len(GT_id)):
        IC_mask = extract_IC(ica100_template, GT_id[k] - 1)
        GMV_mask_id = []
        if (len(GM_id) != 0):
            for m in range(len(GM_id)):
                GMV_mask = extract_atlas('GMatlas.nii.gz', GM_id[m] + 1)
                inter_mask = intersect_masks([IC_mask, GMV_mask], threshold = 1)
                if (np.count_nonzero(get_data(inter_mask)) > 0.1*per*min(np.count_nonzero(get_data(IC_mask)), np.count_nonzero(get_data(GMV_mask)))):
                    GMV_mask_id.append(m)
        IC_mask_id_run.append(GMV_mask_id)         
        print("\r Process{}%".format(round((k+1)*100/len(GT_id))), end="") 
    IC_mask_id.append(IC_mask_id_run)
    print(per)

l = 3
for j in range(len(IC_mask_id[l])):
    for i in range(len(IC_mask_id[l][j])):
        print(GM_labels[GM_id[IC_mask_id[l][j][i]]])
    print(GT_id[j])

ff = pd.DataFrame(IC_mask_id[0])[pd.DataFrame(IC_mask_id[0]).astype(bool)]
ff_nonan = ff[ff.notnull().any(1)]
for v in range(len(ff_nonan.index)):
    print(GT_id[ff_nonan.index[v]])

#ff_fillnan = ff_nonan.fillna(0)
f_flatten = ff_nonan.to_numpy().flatten()
f_withounana = [x for x in f_flatten if str(x) != 'nan']
f_withounana.astype()
result = [x for l in f_withounana for x in l]

---------------------------------------------------------------------------------------------
temp = [s for s in harvard_oxford_s.labels if ("Superior Temporal Gyrus")  in s]
harvard_oxford_s.labels.index(temp)

indices = [i for i, elem in enumerate(harvard_oxford_s.labels) if any(a in elem for a in ["Inferior Frontal Gyrus, pars triangularis", "right"])]

indices = [i for i, s in enumerate(harvard_oxford_s.labels) if 'Right Inferior Frontal Gyrus, pars triangularis' in s]

h_id = []
h_s_id = []
c_id =[]

for i in range(beta_GMV_id_nonzero.shape[0]):
    if any(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in harvard_oxford_s.labels):
        if (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(harvard_oxford_s.labels) if ('Right ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(harvard_oxford_s.labels) if ('Left ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        h_id.append(indices[0])
        #print(harvard_oxford_s.labels[indices[0]])
    elif any(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in harvard_oxford_sub.labels):
        if (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(harvard_oxford_s.labels) if ('Right ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(harvard_oxford_s.labels) if ('Left ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        h_s_id.append(indices[0])
        #print(harvard_oxford_sub.labels[indices[0]])
    if any('Cerebellum' in s for s in beta_GMV_id_nonzero[['GMV_name']].loc[i]):
        if (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(Cerebellum_label) if ('Right ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-19].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(Cerebellum_label) if ('Left ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-18].values[0])) in s]
        elif (beta_GMV_id_nonzero[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(Cerebellum_label) if ('vermis ' + str(beta_GMV_id_nonzero[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        c_id.append(indices[0])
        #print(Cerebellum_label[indices[0]])
    print(i)

IC_mask_id20 = []

for k in range(len(GT_id)):
    IC_mask = extract_IC(ica100_template, GT_id[k])
    h_mask_id = []
    h_s_mask_id = []
    c_mask_id = []
    if (len(h_id) != 0):
        for j in range(len(h_id)):
            h_mask = extract_atlas(harvard_oxford_s.filename, h_id[j])
            inter_mask = intersect_masks([IC_mask, h_mask], threshold = 1)
            if (np.count_nonzero(get_data(inter_mask)) > 0.2*min(np.count_nonzero(get_data(IC_mask)), np.count_nonzero(get_data(h_mask)))):
                h_mask_id.append(j)
    if (len(h_s_id) != 0):
        for l in range(len(h_s_id)):
            h_s_mask = extract_atlas(harvard_oxford_sub.filename, h_s_id[l])
            inter_mask = intersect_masks([IC_mask, h_mask], threshold = 1)
            if (np.count_nonzero(get_data(inter_mask)) > 0.2*min(np.count_nonzero(get_data(IC_mask)), np.count_nonzero(get_data(h_s_mask)))):
                h_s_mask_id.append(l)
    if (len(c_id) != 0):
        for m in range(len(c_id)):
            c_mask = extract_atlas('Cerebellum-MNIfnirt-maxprob-thr25.nii', c_id[m])
            inter_mask = intersect_masks([IC_mask, c_mask], threshold = 1)
            if (np.count_nonzero(get_data(inter_mask)) > 0.2*min(np.count_nonzero(get_data(IC_mask)), np.count_nonzero(get_data(c_mask)))):
                c_mask_id.append(m)
    IC_mask_id20.append([h_mask_id, h_s_mask_id, c_mask_id])
    print(k)           
    print("\r Process{}%".format(round((k+1)*100/len(GT_id))), end="")
    
ff = pd.DataFrame(IC_mask_id20)[pd.DataFrame(IC_mask_id20).astype(bool)]
ff_nonan = ff[ff.notnull().any(1)]
for v in range(len(ff_nonan.index)):
    print(GT_id[ff_nonan.index[v]])

#ff_fillnan = ff_nonan.fillna(0)
f_flatten = ff_nonan.to_numpy().flatten()
f_withounana = [x for x in f_flatten if str(x) != 'nan']
result = [x for l in f_withounana for x in l]

--------------------------------------------------------------------------------------------------
import nilearn.plotting as plotting
import nilearn as nl
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import get_data
from numpy import loadtxt

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
fmri_data = get_data(ica100_template)
good_IC = loadtxt('UKBiobank_BrainImaging_GroupMeanTemplates/rfMRI_GoodComponents_d100_v1.txt', dtype = int)
good_IC_minus1 = (good_IC -1).tolist()
fmri_data_new = fmri_data[:,:,:, good_IC_minus1]
new_fmri_img = new_img_like(ica100_template, fmri_data_new)
plotting.plot_prob_atlas(new_fmri_img)


plotting.plot_roi(ica100_template.slicer[..., n])
plotting.plot_img(ica100_template.slicer[..., n])
plotting.plot_prob_atlas(ica100_template)

