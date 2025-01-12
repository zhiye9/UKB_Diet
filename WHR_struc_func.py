#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:58:40 2022

@author: zhye
"""
#Import all needed packages
#from tkinter import N
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
from matplotlib import colors
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy import stats
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import pearsonr as pearsonr
import pickle
import os
from numpy import loadtxt
import nilearn.plotting as plotting
import nilearn as nl
import nibabel as nib
from nilearn.image import get_data
from nilearn.input_data import NiftiMasker
from nilearn.image import new_img_like
from nilearn.image import resample_to_img
import nibabel.processing
from nilearn import datasets
from nilearn.masking import intersect_masks
from nilearn import image
import matplotlib.markers as markers
from UKB_graph_metrics import *

from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    column_or_1d,
)

from sklearn.metrics._regression import _check_reg_targets

import warnings
warnings.filterwarnings('ignore')

os.chdir('/home/ubuntu/UKB/UK_Biobank_diet')

#df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')
df_copy = copy.copy(df)
df_2000 = df_copy[df_copy['eid'].isin(df_SCE_gmv_2000['eid'].tolist())]

df_hip = df_2000[['eid', '49-0.0', '49-1.0', '49-2.0', '49-3.0']]
df_hip_noNeg = df_hip.mask(df_hip < 0).dropna(subset = df_hip.columns[1:], how = 'all')
df_hip_noNeg.reset_index(drop = True, inplace = True)
df_hip_noNeg['eid'] = df_hip_noNeg['eid'].astype(str)
df_hip_noNeg['49'] = 0
for i in range(df_hip_noNeg.shape[0]):
    df_hip_noNeg['49'].loc[i] = np.nanmean(df_hip_noNeg[df_hip_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_hip_noNeg.shape[0])), end="")

df_waist = df_2000[['eid', '48-0.0', '48-1.0', '48-2.0', '48-3.0']]
df_waist_noNeg = df_waist.mask(df_waist < 0).dropna(subset = df_waist.columns[1:], how = 'all')
df_waist_noNeg.reset_index(drop = True, inplace = True)
df_waist_noNeg['eid'] = df_waist_noNeg['eid'].astype(str)
df_waist_noNeg['48'] = 0
for i in range(df_waist_noNeg.shape[0]):
    df_waist_noNeg['48'].loc[i] = np.nanmean(df_waist_noNeg[df_waist_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_waist_noNeg.shape[0])), end="")

df_waist_noNeg = df_waist_noNeg.drop(columns = df_waist.columns[1:])
df_hip_noNeg = df_hip_noNeg.drop(columns = df_hip.columns[1:])
df_hipwaist = pd.merge(df_waist_noNeg, df_hip_noNeg, on = 'eid')

df_hipwaist['waist_hip_ratio_raw'] = df_hipwaist['48']/df_hipwaist['49']
df_hipwaist_mean = df_hipwaist.drop(columns=['48', '49'])

df_SCE_gmv_2000 = pd.merge(df_SCE_gmv_2000, df_hipwaist_mean, on = 'eid')

#Read GMV and phenotype info
df_gmv_SCE = pd.read_csv('SCE_gmv_updated.csv')
df_genetic_noNAN = pd.read_csv('df_genetic.csv')
gmv = np.genfromtxt('GMV.txt', dtype='str')

#Randomly sample 2000 subjects, random_state can be any seed
df_SCE_gmv_2000 = df_gmv_SCE.sample(n = 2000, random_state = 16)
df_SCE_gmv_2000.reset_index(drop = True, inplace = True)
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)
#Z-score GMV data
df_SCE_gmv_2000[gmv] = df_SCE_gmv_2000[gmv].apply(stats.zscore)
#df_SCE_gmv_2000.to_csv('df_SCE_gmv_2000.csv', index = False)
df_SCE_gmv_2000 = pd.read_csv('df_SCE_gmv_2000.csv')
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)

#Cross-validation function
def CV(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    models = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #clf = GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        clf = GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        #r2train.append(mean_squared_error(y_train, y_pred))
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #r2test.append(mean_squared_error(y_test, y_pred))
        r2test.append(r2_score(y_test, y_pred))
        print(r2test)

        models.append(clf)
        
    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models

#Set parameters of cross-validation
par_grid = {'alpha': [1e-2, 1e-1, 1, 1e-3, 10]}
#par_grid = {'alpha': [1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 3, 5, 7, 10]}
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
    #IC_Graph.append(np.log(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55)))
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")

#Save IC_Graph as list
#with open("IC_Graph", "wb") as fp:   #Pickling
#    pickle.dump(IC_Graph, fp)
os.chdir('../.')
with open("IC_Graph", "rb") as fp:   # Unpickling
    IC_Graph = pickle.load(fp)

#Read GNN embeddings
#with open("/home/ubuntu/UKB_GNN/UK_Biobank_diet/emd_p_3GATConv_lr1e-4_4000_match_corrup", "rb") as fp:   # Unpickling
with open("/home/ubuntu/UKB_GNN/UK_Biobank_diet/emd_p_6GCN32_64_128_lr1e-4_600", "rb") as fp:   # Unpickling
    emb_p = pickle.load(fp)

with open("/home/ubuntu/UKB_GNN/UK_Biobank_diet/emd_n_6GCN32_64_128_lr1e-4_600", "rb") as fp:   # Unpickling
    emb_n = pickle.load(fp)

#Energy/WHR ~ GMV + Control
#XE1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GMV_train_E, energy_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE1, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
#X_WHR11 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1 = np.array(df_SCE_gmv_2000[gmv])
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 148, rand = 9)
#GMV_train_WHR1, GMV_test_WHR1, beta_GMV1, model_GMV1 = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR11, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA + Control
#XE2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_IC_train_E, energy_IC_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE2, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR2 = stats.zscore(np.array(IC55))
IC_raw_train_WHR, IC_raw_test_WHR, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
#XE3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_IC_gmv_train_E, energy_IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
#XE_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GT_GMV_train_E, energy_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.concatenate((emb_p, emb_n), axis = 1)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph))), axis = 1)
X_WHR_GMV_GT1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array([i[:-4] for i in IC_Graph])), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph))), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.concatenate((emb_p, emb_n), axis = 1)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.concatenate((emb_p, emb_n), axis = 1))), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E, beta_GMV_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 249,rand = 9)
WHR_6GCN_GMV_train_E1, WHR_6GCN_GMV_test_E1, beta_GMV_6GCN1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 249,rand = 9)
WHR_6GAT_GMV_train_E1, WHR_6GAT_GMV_test_E1, beta_GMV_6GAT1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 249,rand = 9)
WHR_3GCN_GMV_train_E1, WHR_3GCN_GMV_test_E1, beta_GMV_3GCN1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 249,rand = 9)
WHR_3GAT_GMV_train_E1, WHR_3GAT_GMV_test_E1, beta_GMV_3GAT1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 249,rand = 9)

#WHR_GT_GMV_train_E1, WHR_GT_GMV_test_E1, WHR_GT_GMV_model1 = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ ICA_GT + Control
#XE_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GT_train_E, energy_GT_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GT1 = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GT1 = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT1 = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT11 = np.concatenate((stats.zscore(np.array([i[:-4] for i in IC_Graph])), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
#X_WHR_GT = stats.zscore(np.array(IC_Graph))
X_WHR_GT1 = np.concatenate((stats.zscore(np.concatenate((emb_p, emb_n), axis = 1)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT1 = stats.zscore(np.array(IC_Graph))
X_WHR_GT1 = stats.zscore(np.array([i[:-4] for i in IC_Graph1]))
X_WHR_GT1 = stats.zscore(np.array(np.concatenate((emb_p, emb_n), axis = 1)))
WHR_GT_train_E, WHR_GT_test_E, beta_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 110, rand = 9)
WHR_GT_train_E1, WHR_GT_test_E1, beta_GT1, GT_model1 = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT11, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 110, rand = 9)
#WHR_GT_train_E1, WHR_GT_test_E1, beta_GT1, GT_model1 = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 110, rand = 9)

#Energy/WHR ~ BMI + Control
#XE_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
#energy_BMI_train_E, energy_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_BMI, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_BMI = np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])
#X_WHR_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
X_WHR_BMI = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])
WHR_BMI_train_E, WHR_BMI_test_E, WHR_BMI_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ Control
#XE_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])
#energy_Control_train_E, energy_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_Control, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])
#X_WHR_Control = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])
X_WHR_Control = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])
X_WHR_Control = np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])
WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 9, rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T
beta_GMV_GT_df = pd.DataFrame(beta_GMV_GT).T
beta_control_df = pd.DataFrame(beta_control).T

beta_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
WHR_Control_test_E.append('R2')
beta_control_df.loc[9] = WHR_Control_test_E

beta_GMV_control_df = beta_GMV_df.loc[139:]
beta_GMV_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
beta_GMV_control_df = beta_GMV_control_df.rename(columns = {'control': 'GMV_name'})
GMV_test_WHR.append('R2')
beta_GMV_control_df = pd.concat([pd.DataFrame([GMV_test_WHR], columns = beta_GMV_control_df.columns), beta_GMV_control_df])
beta_GMV_df = beta_GMV_df.loc[:138]

beta_GT_control_df = beta_GT_df.loc[110:]
beta_GT_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
beta_GT_df = beta_GT_df.loc[:109]

#Get beta coefficients
beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')
#beta_GMV_id.to_csv('beta_GMV_id.csv', index = False)
#beta_GMV_id = pd.read_csv('beta_GMV_id.csv')
beta_GMV_id = beta_GMV_id.drop(columns = ['eid'])
#beta_GMV_control_id = pd.concat([beta_GMV_control_df, beta_GMV_id])
#beta_GMV_control_id_nonzero = beta_GMV_control_id[(beta_GMV_control_id != 0).all(1)]
#beta_GMV_control_id_nonzero.reset_index(drop = True, inplace = True)

beta_GMV_id_nonzero = beta_GMV_id[(beta_GMV_id != 0).all(1)]
beta_GMV_id_nonzero.reset_index(drop = True, inplace = True)
#beta_GMV_id_nonzero.to_csv('beta_GMV_id_nonzero.csv', index = False)
#beta_GMV_id_nonzero = pd.read_csv('beta_GMV_id_nonzero.csv')

#Read GM labels
GM_labels = loadtxt("GMatlas_name.txt", dtype=str, delimiter="\t", unpack=False).tolist()

#Get predictive GM features index.
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

#Get positive/negative beta GMs
beta_GMV_positive = beta_GMV_id_nonzero[beta_GMV_id_nonzero.index.isin([i for i in range(beta_GMV_id_nonzero.shape[0]) if sum(beta_GMV_id_nonzero.loc[i][:5].gt(0)) > 2])]
beta_GMV_positive.reset_index(drop = True, inplace = True)
beta_GMV_negative = beta_GMV_id_nonzero[beta_GMV_id_nonzero.index.isin([i for i in range(beta_GMV_id_nonzero.shape[0]) if sum(beta_GMV_id_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GMV_negative.reset_index(drop = True, inplace = True)

#Get positive/negative beta predictive GM features index.
GM_id_pos = []
for i in range(beta_GMV_positive.shape[0]):
    if any(beta_GMV_positive[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_positive[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_positive[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_positive[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id_pos.append(indices[0])

GM_id_neg = []
for i in range(beta_GMV_negative.shape[0]):
    if any(beta_GMV_negative[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_negative[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_negative[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_negative[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id_neg.append(indices[0])

#Extract IC and GM
def extract_IC(IC, n):
    data = get_data(IC.slicer[..., n])
    data[data < 5] = 0
    data[data >= 5] = 1
    new_img = new_img_like(IC.slicer[..., n], data)
    return new_img

def extract_IC_withoutprob(file, n):
    IC = nib.load(file)
    data = get_data(IC)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(IC, data)
    return new_img

def extract_atlas(file, n):
    GMatlas = nib.load(file)
    resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
    data = get_data(resampled_GM)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(resampled_GM, data)
    return new_img

#Extract IC with color
def extract_IC_color(file, n, colo):
    IC = nib.load(file)
    data = get_data(IC)
    data[data != n] = 0
    data[data == n] = colo
    new_img = new_img_like(IC, data)
    return new_img

#The atlas start from 1, 0 is background, but GM_id start from 0 
new_GM_id = [x+1 for x in GM_id]
new_GM_id_pos = [x+1 for x in GM_id_pos]
new_GM_id_neg = [x+1 for x in GM_id_neg]

#Resample GM atlas to (91, 109, 91)
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))

#Find intersection of 34 predictive GMs
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id)] = 1
GM_template_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask)))

#Find intersection of 18 positive predictive GMs
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id_pos)] = 0
data[np.isin(data, new_GM_id_pos)] = 1
GM_template_mask_pos = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask_pos, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask_pos)))

#Find intersection of 16 negative predictive GMs
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id_neg)] = 0
data[np.isin(data, new_GM_id_neg)] = 1
GM_template_mask_neg = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask_neg, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask_neg)))

#Plotting positive and neagtive GMs together
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.1
data[np.isin(data, new_GM_id_neg)] = -1.1
GM_template_mask_pos_neg = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask_pos_neg, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
print(np.count_nonzero(get_data(GM_template_mask_pos_neg)))

overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in GM_id_pos]
overlap_per_labels_names_GM_neg = [i for i in GM_labels if GM_labels.index(i) in GM_id_neg]

#Read 55 ICA good compoents, start from 1
ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

#Drop ICs with beta = 0 in any round
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_nonzero = beta_GT_df[(beta_GT_df != 0).all(1)]
beta_GT_nonzero.reset_index(drop = True, inplace = True)

#Find postive (16) and negative (14) correlation ICs separately
beta_GT_pos_corr = beta_GT_df.loc[:55]
beta_GT_neg_corr = beta_GT_df.loc[55:]
beta_GT_pos_corr_nonzero = beta_GT_pos_corr[(beta_GT_pos_corr != 0).all(1)]
beta_GT_pos_corr_nonzero.reset_index(drop = True, inplace = True)
beta_GT_neg_corr_nonzero = beta_GT_neg_corr[(beta_GT_neg_corr != 0).all(1)]
beta_GT_neg_corr_nonzero.reset_index(drop = True, inplace = True)

#Get 24 predictive ICs
GT_id = np.sort(np.unique(beta_GT_nonzero['ICs'].str[7:], return_counts = True)[0].astype(int))

#Get positive/negative beta ICs in positive/negative correlation ICs separately
beta_GT_pos_corr_positive = beta_GT_pos_corr_nonzero[beta_GT_pos_corr_nonzero.index.isin([i for i in range(beta_GT_pos_corr_nonzero.shape[0]) if sum(beta_GT_pos_corr_nonzero.loc[i][:-1].gt(0)) > 2])]
beta_GT_pos_corr_positive.reset_index(drop = True, inplace = True)
beta_GT_pos_corr_negative = beta_GT_pos_corr_nonzero[beta_GT_pos_corr_nonzero.index.isin([i for i in range(beta_GT_pos_corr_nonzero.shape[0]) if sum(beta_GT_pos_corr_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GT_pos_corr_negative.reset_index(drop = True, inplace = True)

beta_GT_neg_corr_positive = beta_GT_neg_corr_nonzero[beta_GT_neg_corr_nonzero.index.isin([i for i in range(beta_GT_neg_corr_nonzero.shape[0]) if sum(beta_GT_neg_corr_nonzero.loc[i][:-1].gt(0)) > 2])]
beta_GT_neg_corr_positive.reset_index(drop = True, inplace = True)
beta_GT_neg_corr_negative = beta_GT_neg_corr_nonzero[beta_GT_neg_corr_nonzero.index.isin([i for i in range(beta_GT_neg_corr_nonzero.shape[0]) if sum(beta_GT_neg_corr_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GT_neg_corr_negative.reset_index(drop = True, inplace = True)

#Get positive/negative beta ICs id
GT_id_pos_corr_pos = np.sort(np.unique(beta_GT_pos_corr_positive['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_pos_corr_neg = np.sort(np.unique(beta_GT_pos_corr_negative['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_neg_corr_pos = np.sort(np.unique(beta_GT_neg_corr_positive['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_neg_corr_neg = np.sort(np.unique(beta_GT_neg_corr_negative['ICs'].str[7:], return_counts = True)[0].astype(int))

#Get ICs predictive both in postive/negative beta and positive/negative correlations 
GT_id_pos_corr_pos_neg_corr_pos = [i for i in GT_id_pos_corr_pos if i in GT_id_neg_corr_pos]
GT_id_pos_corr_pos_neg_corr_neg = [i for i in GT_id_pos_corr_pos if i in GT_id_neg_corr_neg]
GT_id_pos_corr_neg_neg_corr_pos = [i for i in GT_id_pos_corr_neg if i in GT_id_neg_corr_pos]
GT_id_pos_corr_neg_neg_corr_neg = [i for i in GT_id_pos_corr_neg if i in GT_id_neg_corr_neg]

#Get ICs predictive only in one postive/negative beta and positive/negative correlations 
GT_id_pos_corr_pos_only = GT_id_pos_corr_pos[~np.isin(GT_id_pos_corr_pos, GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg)] 
GT_id_pos_corr_neg_only = GT_id_pos_corr_neg[~np.isin(GT_id_pos_corr_neg, GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg)] 
GT_id_neg_corr_pos_only = GT_id_neg_corr_pos[~np.isin(GT_id_neg_corr_pos, GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_pos)] 
GT_id_neg_corr_neg_only = GT_id_neg_corr_neg[~np.isin(GT_id_neg_corr_neg, GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_neg)] 

#np.unique(GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg + GT_id_pos_corr_pos_only.tolist() + GT_id_pos_corr_neg_only.tolist() + GT_id_neg_corr_pos_only.tolist() + GT_id_neg_corr_neg_only.tolist(), return_counts = True)
#np.unique((GT_id_pos_corr_pos.tolist() + GT_id_pos_corr_neg.tolist()  + GT_id_neg_corr_pos.tolist()  + GT_id_neg_corr_neg.tolist()), return_counts = True)

#Read IC templates
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
#ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
#ica100_np = np.array(ica100_template.dataobj)
t1 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.1) for i in GT_id_pos_corr_pos_neg_corr_pos]
t2 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.2) for i in GT_id_pos_corr_pos_neg_corr_neg]
t3 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in GT_id_pos_corr_neg_neg_corr_pos]
t4 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.4) for i in GT_id_pos_corr_neg_neg_corr_neg]
t5 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in GT_id_pos_corr_pos_only]
t6 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.6) for i in GT_id_pos_corr_neg_only]
t7 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.7) for i in GT_id_neg_corr_pos_only]
t8 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.8) for i in GT_id_neg_corr_neg_only]
IC_img = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
#IC_img = t7
#atlas = image.math_img('np.sum(img, axis = -1)', img = IC_img)
##plotting.plot_roi(atlas, cut_coords = [1, -71, 13], cmap = "tab20", colorbar = True)
#oo = np.unique(atlas.get_data(), return_counts = True)
'''
def avoid_IC_overlap(IC1, IC2, t):
    data = np.sum([IC1.get_data(), IC2.get_data()], axis = 0)
    data[data > t] = np.unique(IC1.get_data())[-1]
    return new_img_like(IC1, data)

last_IC = IC_img[0]
for i in range(len(IC_img) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC, IC_img[i + 1], 2)
    last_IC = new_IC_overlap
    #print(i)
    #print(np.unique(last_IC.get_data(), return_counts = True))

last_IC_data = get_data(last_IC)
last_IC_data_10 = np.multiply(last_IC_data, np.full(last_IC_data.shape, 10))
last_IC_10 = new_img_like(last_IC, last_IC_data_10)
''''

IC_img_value = [IC_img[i].get_fdata() for i in range(len(IC_img))]
IC_data = np.sum(IC_img_value, axis = 0)
last_IC_data_10 = np.multiply(IC_data, np.full(IC_data.shape, 10))
last_IC_10 = new_img_like(IC_img[0], last_IC_data_10)

#plotting.plot_roi(last_IC_10, cut_coords = [1, -71, 13], cmap = "Set1", colorbar = True)
plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

#t1 = extract_IC_color(ica100_template, 1)
#atlas = image.math_img('np.sum(img, axis=-1)', img=[extract_IC_color(ica100_template, 12, 1.1), extract_IC_color(ica100_template, 9, 1.2)])

#Find intersection of 24 predictive GT
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1
new_IC_template_mask = new_img_like(ica100_template, data)
plotting.plot_roi(new_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask)))

#Compute intersection of predicve GMs and ICs
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = int(0)
data[np.isin(data, new_GM_id)] = int(1)
GM_template_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask)))

GM_IC_template_mask = intersect_masks([new_IC_template_mask, GM_template_mask], threshold = 1, connected = False)
plotting.plot_roi(GM_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_IC_template_mask)))

ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
GM_IC_template_mask_sum = np.multiply(GM_template_mask.get_fdata(), new_IC_template_mask.get_fdata())
GM_IC_template_mask_sum_mri = new_img_like(ica100_template, GM_IC_template_mask_sum) 
plotting.plot_roi(GM_IC_template_mask_sum_mri, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_IC_template_mask_sum_mri)))

#Find predictive GMs in the overlap of predictive GMs and ICs
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1
new_IC_template_mask = new_img_like(ica100_template, data)

overlap_voxel_per_pred_GM_pos = []
for i in range(len(new_GM_id_pos)):
    GM_atlas_pred_extracted_pos = extract_atlas('GMatlas.nii.gz', new_GM_id_pos[i])
    int_IC_GM_pred_pos = intersect_masks([new_IC_template_mask, GM_atlas_pred_extracted_pos], threshold = 1)
    overlap_voxel_per_pred_GM_pos.append(np.count_nonzero(get_data(int_IC_GM_pred_pos))/np.count_nonzero(get_data(GM_atlas_pred_extracted_pos)))
    print("\r Process{}%".format(round((i+1)*100/len(new_GM_id_pos))), end="")

overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in GM_id_pos]
df_overlap_GM_pos = pd.DataFrame(data = {'GM_name': overlap_per_labels_names_GM_pos, 'Overlap_percentage': overlap_voxel_per_pred_GM_pos})
#overlap_per_labels = [j for j, s in enumerate(overlap_voxel_per) if (s < 0.5 and s >= 0.4)]
#overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in overlap_per_labels_GM_pos]

overlap_voxel_per_pred_GM_neg = []
for i in range(len(new_GM_id_neg)):
    GM_atlas_pred_extracted_neg = extract_atlas('GMatlas.nii.gz', new_GM_id_neg[i])
    int_IC_GM_pred_neg = intersect_masks([new_IC_template_mask, GM_atlas_pred_extracted_neg], threshold = 1)
    overlap_voxel_per_pred_GM_neg.append(np.count_nonzero(get_data(int_IC_GM_pred_neg))/np.count_nonzero(get_data(GM_atlas_pred_extracted_neg)))
    print("\r Process{}%".format(round((i+1)*100/len(new_GM_id_neg))), end="")

overlap_per_labels_names_GM_neg = [i for i in GM_labels if GM_labels.index(i) in GM_id_neg]
df_overlap_GM_neg = pd.DataFrame(data = {'GM_name': overlap_per_labels_names_GM_neg, 'Overlap_percentage': overlap_voxel_per_pred_GM_neg})
#overlap_per_labels = [j for j, s in enumerate(overlap_voxel_per) if (s < 0.5 and s >= 0.4)]
#overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in overlap_per_labels_GM_pos]

#Find intersection of 17 positive predictive ICs
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1
new_IC_template_mask = new_img_like(ica100_template, data)
plotting.plot_roi(new_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask)))

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
IC_template_mask_pos = []
for i in range(len(GT_id_pos)):
    IC_template_mask_pos.append(extract_IC(ica100_template, GT_id_pos[i] - 1))
new_IC_template_mask_pos = intersect_masks(IC_template_mask_pos, threshold = 0)
plotting.plot_roi(new_IC_template_mask_pos, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask_pos)))

#Find intersection of 10 negative predictive ICs
ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
IC_template_mask_neg = []
for i in range(len(GT_id_neg)):
    IC_template_mask_neg.append(extract_IC(ica100_template, GT_id_neg[i] - 1))
new_IC_template_mask_neg = intersect_masks(IC_template_mask_neg, threshold = 0)
plotting.plot_roi(new_IC_template_mask_neg, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask_neg)))

#Plot postive and negative ICs in postive network
'''
last_IC_p = t_p_p[0]
for i in range(len(t_p_p) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_p, t_p_p[i + 1], 2)
    last_IC_p = new_IC_overlap
    #print(i)
    #print(np.unique(last_IC.get_data(), return_counts = True))

def avoid_IC_overlap_neg(IC1, IC2, t):
    data = np.sum([IC1.get_data(), IC2.get_data()], axis = 0)
    data[data < t] = np.unique(IC1.get_data())[-1]
    return new_img_like(IC1, data)

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
last_IC_n = t_p_n[0]
for i in range(len(t_p_n) -1):
    new_IC_overlap = avoid_IC_overlap_neg(last_IC_n, t_p_n[i + 1], -2)
    last_IC_n = new_IC_overlap

data_p_n = np.sum([last_IC_p.get_data(), last_IC_n.get_data()], axis = 0)
data_p_n[data_p_n == np.unique(data_p_n)[1]] = np.unique(last_IC_p.get_data())[-1]
last_IC_p_n = new_img_like(last_IC_p, data_p_n)
plotting.plot_roi(last_IC_p_n, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar= True)
print(np.unique(get_data(last_IC_p_n), return_counts = True))
'''
IC_pos_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_pos_only.tolist()
IC_pos_net_neg = GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg + GT_id_pos_corr_neg_only.tolist()
IC_neg_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_pos + GT_id_neg_corr_pos_only.tolist()
IC_neg_net_neg = GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_neg + GT_id_neg_corr_neg_only.tolist()

t_p_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.1) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, -1.2) for i in IC_pos_net_neg]

IC_img_p = t_p_p + t_p_n
IC_img_value_p = [IC_img_p[i].get_fdata() for i in range(len(IC_img_p))]
IC_data = np.sum(IC_img_value_p, axis = 0)
#last_IC_data_10 = np.multiply(IC_data, np.full(IC_data.shape, 10))
last_IC_10 = new_img_like(IC_img[0], IC_data)

#plotting.plot_roi(last_IC_10, cut_coords = [1, -71, 13], cmap = "Set1", colorbar = True)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

'''
last_IC_p = t_n_p[0]
for i in range(len(t_n_p) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_p, t_n_p[i + 1], 2)
    last_IC_p = new_IC_overlap
    #print(i)
    #print(np.unique(last_IC.get_data(), return_counts = True))

last_IC_n = t_n_n[0]
for i in range(len(t_n_n) -1):
    new_IC_overlap = avoid_IC_overlap_neg(last_IC_n, t_n_n[i + 1], -2)
    last_IC_n = new_IC_overlap

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')

last_IC_p_n = new_img_like(last_IC_p, np.sum([last_IC_p.get_data(), last_IC_n.get_data()], axis = 0))
plotting.plot_roi(last_IC_p_n, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar= True)
print(np.unique(get_data(last_IC_p_n), return_counts = True))
''''
t_n_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.1) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, -1.1) for i in IC_neg_net_neg]

IC_img_n = t_n_p + t_n_n
IC_img_value_n = [IC_img_n[i].get_fdata() for i in range(len(IC_img_n))]
IC_data = np.sum(IC_img_value_n, axis = 0)
#last_IC_data_10 = np.multiply(IC_data, np.full(IC_data.shape, 10))
last_IC_10 = new_img_like(IC_img[0], IC_data)

plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

#Check pos/neg in overlap region
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.2
data[np.isin(data, new_GM_id_neg)] = 1.1
GM_intersec_pos_neg = new_img_like(resampled_GM, data)

t_p_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_pos_net_neg]

IC_value_p_p = [t_p_p[i].get_fdata() for i in range(len(t_p_p))]
IC_p_p_data = np.sum(IC_value_p_p, axis = 0)

IC_value_p_n = [t_p_n[i].get_fdata() for i in range(len(t_p_n))]
IC_p_n_data = np.sum(IC_value_p_n, axis = 0)

IC_p_data = np.sum([IC_p_p_data, IC_p_n_data], axis = 0)
last_IC_10_p = new_img_like(t_p_p[0], IC_p_data)

plotting.plot_roi(last_IC_10_p, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

GM_IC_pos_net_data = np.sum([GM_intersec_pos_neg.get_fdata(), last_IC_10_p.get_fdata()], axis = 0)
GM_IC_pos_net_data_sub = np.subtract(np.multiply(GM_IC_pos_net_data, np.full(GM_IC_pos_net_data.shape, 10)), np.full(GM_IC_pos_net_data.shape, 23))
GM_IC_pos_net_data_sub[GM_IC_pos_net_data_sub < 0] = 0
GM_IC_pos_net = new_img_like(GM_intersec_pos_neg, np.around(GM_IC_pos_net_data_sub))
plotting.plot_roi(GM_IC_pos_net, cut_coords = [-1, -44, 12], cmap = cmap4, colorbar = True)
print(np.unique(get_data(GM_IC_pos_net), return_counts = True))

GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.2
data[np.isin(data, new_GM_id_neg)] = 1.1
GM_intersec_pos_neg = new_img_like(resampled_GM, data)

t_n_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_neg_net_neg]

IC_value_n_p = [t_n_p[i].get_fdata() for i in range(len(t_n_p))]
IC_n_p_data = np.sum(IC_value_n_p, axis = 0)

IC_value_n_n = [t_n_n[i].get_fdata() for i in range(len(t_n_n))]
IC_n_n_data = np.sum(IC_value_n_n, axis = 0)

IC_n_data = np.sum([IC_n_p_data, IC_n_n_data], axis = 0)
last_IC_10_n = new_img_like(t_n_p[0], IC_n_data)

GM_IC_pos_net_data = np.sum([GM_intersec_pos_neg.get_fdata(), last_IC_10_n.get_fdata()], axis = 0)
GM_IC_pos_net_data_sub = np.subtract(np.multiply(GM_IC_pos_net_data, np.full(GM_IC_pos_net_data.shape, 10)), np.full(GM_IC_pos_net_data.shape, 23))
GM_IC_pos_net_data_sub[GM_IC_pos_net_data_sub < 0] = 0
GM_IC_pos_net = new_img_like(GM_intersec_pos_neg, np.around(GM_IC_pos_net_data_sub))
plotting.plot_roi(GM_IC_pos_net, cut_coords = [-1, -44, 12], cmap = cmap4, colorbar = True)
print(np.unique(get_data(GM_IC_pos_net), return_counts = True))
----------------------------------------------------------------------------------------------------------------

#Find intersection of all GMs
GMatlas = nib.load('GMatlas.nii.gz')
t_id = np.unique(GMatlas.get_data()).tolist()[1:]
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data1 = get_data(resampled_GM)
data1[~np.isin(data1, t_id)] = 0
data1[np.isin(data1, t_id)] = 1
GM_atlas_mask = new_img_like(resampled_GM, data1)
plotting.plot_roi(GM_atlas_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_atlas_mask)))

#Find intersection of all ICs
good_IC = loadtxt('UKBiobank_BrainImaging_GroupMeanTemplates/rfMRI_GoodComponents_d100_v1.txt', dtype = int)
good_IC_minus1 = (good_IC -1).tolist()
IC_atlas_mask = []
for i in range(len(good_IC_minus1)):
    IC_atlas_mask.append(extract_IC(ica100_template, good_IC_minus1[i]))
new_IC_atlas_mask = intersect_masks(IC_atlas_mask, threshold = 0)
plotting.plot_roi(new_IC_atlas_mask, cut_coords = [-20, -44, 12])
plotting.plot_roi(new_IC_atlas_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_atlas_mask)))

#Intersection of all GMs and ICs
GM_IC_atlas_mask = intersect_masks([new_IC_atlas_mask, GM_atlas_mask], threshold = 1)
print(np.count_nonzero(get_data(GM_IC_atlas_mask)))
plotting.plot_roi(GM_IC_atlas_mask, cut_coords = [-1, -44, 12])

#Compute overlaps between all GM and all IC atlas
overlap_voxel_per =[]
for i in range(len(t_id)):
    GM_atlas_extracted = extract_atlas('GMatlas.nii.gz', t_id[i])
    int_IC_GM = intersect_masks([new_IC_atlas_mask, GM_atlas_extracted], threshold = 1)
    overlap_voxel_per.append(np.count_nonzero(get_data(int_IC_GM))/np.count_nonzero(get_data(GM_atlas_extracted)))
    print("\r Process{}%".format(round((i+1)*100/len(t_id))), end="")

#Compute overlaps between predictive GM and all IC atlas
overlap_voxel_per_pred =[]
for i in range(len(new_GM_id)):
    GM_atlas_pred_extracted = extract_atlas('GMatlas.nii.gz', new_GM_id[i])
    int_IC_GM_pred = intersect_masks([new_IC_atlas_mask, GM_atlas_pred_extracted], threshold = 1)
    overlap_voxel_per_pred.append(np.count_nonzero(get_data(int_IC_GM_pred ))/np.count_nonzero(get_data(GM_atlas_pred_extracted)))
    print("\r Process{}%".format(round((i+1)*100/len(new_GM_id))), end="")

#Find GMs with overlap degree less than 0.5 or 0.4
overlap_per_labels = [j for j, s in enumerate(overlap_voxel_per) if s < 0.5]
#overlap_per_labels = [j for j, s in enumerate(overlap_voxel_per) if (s < 0.5 and s >= 0.4)]
overlap_per_labels_names = [i for i in GM_labels if GM_labels.index(i) in overlap_per_labels]
pd.DataFrame(overlap_per_labels_names)

#Find predictive GMs with overlap degree less than 0.5 or 0.4
GM_pred_labels = [i for i in GM_labels if GM_labels.index(i) in [x - 1 for x in new_GM_id]]
GM_pred_pos_labels = [i for i in GM_labels if GM_labels.index(i) in [x - 1 for x in new_GM_id_pos]]
GM_pred_neg_labels = [i for i in GM_labels if GM_labels.index(i) in [x - 1 for x in new_GM_id_neg]]
overlap_per_pred_labels = [j for j, s in enumerate(overlap_voxel_per_pred) if s < 0.5]
overlap_per_pred_labels_names = [i for i in GM_pred_labels if GM_pred_labels.index(i) in overlap_per_pred_labels]
pd.DataFrame(overlap_per_pred_labels_names)

##Plot GMs with low overlap degree
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))

#Get all low overlap degree GM ids
new_overlap_per_labels = [x+1 for x in overlap_per_labels]

#Get predictive low overlap degree GM ids
new_overlap_per_pred_labels = [i for i in new_GM_id if new_GM_id.index(i) in overlap_per_pred_labels]
#[GM_labels[i] for i in [i-1 for i in new_GM_id if new_GM_id.index(i) in overlap_per_pred_labels]]

#Plot GMs with low overlap degree
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
#data[~np.isin(data, new_overlap_per_labels)] = 0
#data[np.isin(data, new_overlap_per_labels)] = 1
data[~np.isin(data, new_overlap_per_pred_labels)] = 0
data[np.isin(data, new_overlap_per_pred_labels)] = 1
GM_overlap_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_overlap_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_overlap_mask)))

#Get a resampled GMatlas
GMatlas = nib.load('GMatlas.nii.gz')
t_id = np.unique(GMatlas.get_data()).tolist()[1:]
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
GM_atlas_resampled = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_atlas_resampled, cut_coords = [-1, -44, 12])

#Get postive and neagetive beta GMatlas
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.2
data[np.isin(data, new_GM_id_neg)] = 1.1
GM_intersec_pos_neg = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_intersec_pos_neg, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
###################################################################################################################
#Get a 3D IC atlas
ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')

t_p_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_pos_net_neg]

t_n_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_neg_net_neg]

def avoid_IC_overlap(IC1, IC2, t):
    data = np.sum([IC1.get_data(), IC2.get_data()], axis = 0)
    data[data > t] = np.unique(IC1.get_data())[-1]
    return new_img_like(IC1, data)

last_IC_p = t_p_p[0]
for i in range(len(t_p_p) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_p, t_p_p[i + 1], 2)
    last_IC_p = new_IC_overlap

last_IC_n = t_p_n[0]
for i in range(len(t_p_n) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_n, t_p_n[i + 1], 2)
    last_IC_n = new_IC_overlap

last_IC_p = t_n_p[0]
for i in range(len(t_n_p) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_p, t_n_p[i + 1], 2)
    last_IC_p = new_IC_overlap

last_IC_n = t_n_n[0]
for i in range(len(t_n_n) -1):
    new_IC_overlap = avoid_IC_overlap(last_IC_n, t_n_n[i + 1], 2)
    last_IC_n = new_IC_overlap

last_IC_p_n = avoid_IC_overlap(last_IC_p, last_IC_n, 2)
data11 = get_data(last_IC_p_n)
data11[data11 == np.unique(get_data(last_IC_p_n))[1]] = -1
data11[data11 == np.unique(get_data(last_IC_p_n))[2]] = 1
plotting.plot_roi(last_IC_p_n, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar= True)
print(np.unique(get_data(last_IC_p_n), return_counts = True))

import matplotlib
from matplotlib import colors
list(matplotlib.colors.ListedColormap('list-of-colours'))

cmap4 = colors.ListedColormap(['tab:red', 'tab:orange', 'tab:brown', 'tab:purple', 'tab:blue'])

GM_IC_pos_net_data = np.sum([GM_intersec_pos_neg.get_data(), last_IC_p_n.get_data()], axis = 0)
GM_IC_pos_net_data_sub = np.subtract(np.multiply(GM_IC_pos_net_data, np.full(GM_IC_pos_net_data.shape, 10)), np.full(GM_IC_pos_net_data.shape, 23))
GM_IC_pos_net_data_sub[GM_IC_pos_net_data_sub < 0] = 0
GM_IC_pos_net = new_img_like(GM_intersec_pos_neg, np.around(GM_IC_pos_net_data_sub))
plotting.plot_roi(GM_IC_pos_net, cut_coords = [-1, -44, 12], cmap = cmap4, colorbar = True)
print(np.unique(get_data(GM_IC_pos_net), return_counts = True))

######################################################################################################
t_p_p = [extract_IC_color(ica100_template, i - 1, 1.1) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color(ica100_template, i - 1, -1.2) for i in IC_pos_net_neg]

def avoid_IC_overlap_neg(IC1, IC2, t):
    data = np.sum([IC1.get_data(), IC2.get_data()], axis = 0)
    data[data < t] = np.unique(IC1.get_data())[-1]
    return new_img_like(IC1, data)

last_IC_n = t_p_n[0]
for i in range(len(t_p_n) -1):
    new_IC_overlap = avoid_IC_overlap_neg(last_IC_n, t_p_n[i + 1], -2)
    last_IC_n = new_IC_overlap

from nilearn.maskers import NiftiMapsMasker
rest_dataset = datasets.fetch_development_fmri(n_subjects=30)
func_filenames = rest_dataset.func  # list of 4D nifti files for each subject

from nilearn.decomposition import CanICA

canica = CanICA(n_components=20,
                memory="nilearn_cache", memory_level=2,
                verbose=10,
                mask_strategy='whole-brain-template',
                random_state=0)
canica.fit(func_filenames)

# Retrieve the independent components in brain space. Directly
# accessible through attribute `components_img_`.
canica_components_img = canica.components_img_

from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show

for i, cur_img in enumerate(iter_img(new_fmri_img)):
    plot_stat_map(cur_img, display_mode="z", title="IC %d" % good_IC[i],
                  cut_coords=1, colorbar=False)
    print(i)

-----------------------------------------------------------------------------------------------------------------------------

#Test overlap between male and female

np.var(df_SCE_gmv_2000_female['waist_hip_ratio'])

for i in ['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']:
    pcr = pearsonr(df_SCE_gmv_2000_female[i], df_SCE_gmv_2000_female['waist_hip_ratio'])
    print('Correlatio betweeb %s and WHR is  %.9f and p value is  %.9f' % (i, pcr[0], pcr[1]))
    print(pcr[0])
    print(pcr[1])

corr_gmv = []
p_gmv = []
for i in gmv:
    r, p = pearsonr(df_SCE_gmv_2000_female[i], df_SCE_gmv_2000_female['waist_hip_ratio'])
    corr_gmv.append(r)
    p_gmv.append(p)
np.mean(corr_gmv)
np.mean(p_gmv)

corr_gt = []
p_gt = []
for i in range(len(IC_Graph[0])):
    r, p = pearsonr(np.array([IC_Graph[j] for j in df_SCE_gmv_2000_female.index])[:, i], df_SCE_gmv_2000_female['waist_hip_ratio'])
    corr_gt.append(r)
    p_gt.append(p)
np.mean(corr_gt)
np.mean(p_gt)

#Select female
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]
#Select male
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]

#WHR ~ GMV + Control
#X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
#GMV_train_WHR, GMV_test_WHR, beta_GMV, GM_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 139, rand = 9)
GMV_train_WHR, GMV_test_WHR, beta_GMV, GM_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 139, rand = 9)
GMV_train_WHR, GMV_test_WHR, beta_GMV, GM_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 146, rand = 9)

#Energy/WHR ~ ICA_GT + Control
#X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
#WHR_GT_train_E, WHR_GT_test_E, beta_GT, GT_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 110, rand = 9)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, GT_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 110, rand = 9)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, GT_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 117, rand = 9)

X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E, beta_GMV_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 249, rand = 9)
#WHR_GT_GMV_train_E, WHR_GT_GMV_test_E, beta_GMV_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 146, rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

beta_GMV_control_df = beta_GMV_df.loc[139:]
beta_GMV_control_df['control'] = ['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']
beta_GT_control_df = beta_GT_df.loc[110:]
beta_GT_control_df['control'] = ['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']

beta_GMV_control_df = beta_GMV_df.loc[139:]
beta_GMV_control_df['control'] = ['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']
beta_GT_control_df = beta_GT_df.loc[110:]
beta_GT_control_df['control'] = ['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']

#Get beta coefficients
beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')
#beta_GMV_id.to_csv('beta_GMV_id.csv', index = False)
#beta_GMV_id = pd.read_csv('beta_GMV_id.csv')

#beta_GMV_id_nonzero = beta_GMV_id[((beta_GMV_id != 0)[beta_GMV_id.columns[:5]]).sum(axis = 1) >= 3]
beta_GMV_id_nonzero = beta_GMV_id[(beta_GMV_id != 0).all(1)]
beta_GMV_id_nonzero.reset_index(drop = True, inplace = True)
#beta_GMV_id_nonzero.to_csv('beta_GMV_id_nonzero.csv', index = False)
#beta_GMV_id_nonzero = pd.read_csv('beta_GMV_id_nonzero.csv')

#Read GM labels
GM_labels = loadtxt("GMatlas_name.txt", dtype=str, delimiter="\t", unpack=False).tolist()

#Get predictive GM features index.
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

#Get positive/negative beta GMs
beta_GMV_positive = beta_GMV_id_nonzero[beta_GMV_id_nonzero.index.isin([i for i in range(beta_GMV_id_nonzero.shape[0]) if sum(beta_GMV_id_nonzero.loc[i][:5].gt(0)) > 2])]
beta_GMV_positive.reset_index(drop = True, inplace = True)
beta_GMV_negative = beta_GMV_id_nonzero[beta_GMV_id_nonzero.index.isin([i for i in range(beta_GMV_id_nonzero.shape[0]) if sum(beta_GMV_id_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GMV_negative.reset_index(drop = True, inplace = True)

#Get positive/negative beta predictive GM features index.
GM_id_pos = []
for i in range(beta_GMV_positive.shape[0]):
    if any(beta_GMV_positive[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_positive[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_positive[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_positive[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_positive[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id_pos.append(indices[0])

GM_id_neg = []
for i in range(beta_GMV_negative.shape[0]):
    if any(beta_GMV_negative[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_negative[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_negative[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_negative[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_negative[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id_neg.append(indices[0])

#Extract IC and GM
def extract_IC(IC, n):
    data = get_data(IC.slicer[..., n])
    data[data < 5] = 0
    data[data >= 5] = 1
    new_img = new_img_like(IC.slicer[..., n], data)
    return new_img

def extract_IC_withoutprob(file, n):
    IC = nib.load(file)
    data = get_data(IC)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(IC, data)
    return new_img

def extract_atlas(file, n):
    GMatlas = nib.load(file)
    resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
    data = get_data(resampled_GM)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(resampled_GM, data)
    return new_img

#Extract IC with color
def extract_IC_color(file, n, colo):
    IC = nib.load(file)
    data = get_data(IC)
    data[data != n] = 0
    data[data == n] = colo
    new_img = new_img_like(IC, data)
    return new_img

#The atlas start from 1, 0 is background, but GM_id start from 0 
new_GM_id = [x+1 for x in GM_id]
new_GM_id_pos = [x+1 for x in GM_id_pos]
new_GM_id_neg = [x+1 for x in GM_id_neg]

#Plotting positive and neagtive GMs together
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.1
data[np.isin(data, new_GM_id_neg)] = -1.1
GM_template_mask_pos_neg = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask_pos_neg, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
print(np.count_nonzero(get_data(GM_template_mask_pos_neg)))

#Get predictive GM labels
overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in GM_id_pos]
overlap_per_labels_names_GM_neg = [i for i in GM_labels if GM_labels.index(i) in GM_id_neg]

#Read 55 ICA good compoents, start from 1
ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

#Drop ICs with beta = 0 in any round
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_nonzero = beta_GT_df[(beta_GT_df != 0).all(1)]
#beta_GT_nonzero = beta_GT_df[((beta_GT_df != 0)[beta_GT_df.columns[:5]]).sum(axis = 1) >= 3]
beta_GT_nonzero.reset_index(drop = True, inplace = True)

#Find postive (16) and negative (14) correlation ICs separately
beta_GT_pos_corr = beta_GT_df.loc[:55]
beta_GT_neg_corr = beta_GT_df.loc[55:]

#beta_GT_pos_corr_nonzero = beta_GT_pos_corr[((beta_GT_pos_corr != 0)[beta_GT_pos_corr.columns[:5]]).sum(axis = 1) >= 3]
beta_GT_pos_corr_nonzero = beta_GT_pos_corr[(beta_GT_pos_corr != 0).all(1)]
beta_GT_pos_corr_nonzero.reset_index(drop = True, inplace = True)
#beta_GT_neg_corr_nonzero = beta_GT_neg_corr[((beta_GT_neg_corr != 0)[beta_GT_neg_corr.columns[:5]]).sum(axis = 1) >= 3]
beta_GT_neg_corr_nonzero = beta_GT_neg_corr[(beta_GT_neg_corr != 0).all(1)]
beta_GT_neg_corr_nonzero.reset_index(drop = True, inplace = True)

#Get 24 predictive ICs
GT_id = np.sort(np.unique(beta_GT_nonzero['ICs'].str[7:], return_counts = True)[0].astype(int))

#Get positive/negative beta ICs in positive/negative correlation ICs separately
beta_GT_pos_corr_positive = beta_GT_pos_corr_nonzero[beta_GT_pos_corr_nonzero.index.isin([i for i in range(beta_GT_pos_corr_nonzero.shape[0]) if sum(beta_GT_pos_corr_nonzero.loc[i][:-1].gt(0)) > 2])]
beta_GT_pos_corr_positive.reset_index(drop = True, inplace = True)
beta_GT_pos_corr_negative = beta_GT_pos_corr_nonzero[beta_GT_pos_corr_nonzero.index.isin([i for i in range(beta_GT_pos_corr_nonzero.shape[0]) if sum(beta_GT_pos_corr_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GT_pos_corr_negative.reset_index(drop = True, inplace = True)

beta_GT_neg_corr_positive = beta_GT_neg_corr_nonzero[beta_GT_neg_corr_nonzero.index.isin([i for i in range(beta_GT_neg_corr_nonzero.shape[0]) if sum(beta_GT_neg_corr_nonzero.loc[i][:-1].gt(0)) > 2])]
beta_GT_neg_corr_positive.reset_index(drop = True, inplace = True)
beta_GT_neg_corr_negative = beta_GT_neg_corr_nonzero[beta_GT_neg_corr_nonzero.index.isin([i for i in range(beta_GT_neg_corr_nonzero.shape[0]) if sum(beta_GT_neg_corr_nonzero.loc[i][:-1].gt(0)) < 3])]
beta_GT_neg_corr_negative.reset_index(drop = True, inplace = True)

#Get positive/negative beta ICs id
GT_id_pos_corr_pos = np.sort(np.unique(beta_GT_pos_corr_positive['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_pos_corr_neg = np.sort(np.unique(beta_GT_pos_corr_negative['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_neg_corr_pos = np.sort(np.unique(beta_GT_neg_corr_positive['ICs'].str[7:], return_counts = True)[0].astype(int))
GT_id_neg_corr_neg = np.sort(np.unique(beta_GT_neg_corr_negative['ICs'].str[7:], return_counts = True)[0].astype(int))

#Get ICs predictive both in postive/negative beta and positive/negative correlations 
GT_id_pos_corr_pos_neg_corr_pos = [i for i in GT_id_pos_corr_pos if i in GT_id_neg_corr_pos]
GT_id_pos_corr_pos_neg_corr_neg = [i for i in GT_id_pos_corr_pos if i in GT_id_neg_corr_neg]
GT_id_pos_corr_neg_neg_corr_pos = [i for i in GT_id_pos_corr_neg if i in GT_id_neg_corr_pos]
GT_id_pos_corr_neg_neg_corr_neg = [i for i in GT_id_pos_corr_neg if i in GT_id_neg_corr_neg]

#Get ICs predictive only in one postive/negative beta and positive/negative correlations 
GT_id_pos_corr_pos_only = GT_id_pos_corr_pos[~np.isin(GT_id_pos_corr_pos, GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg)] 
GT_id_pos_corr_neg_only = GT_id_pos_corr_neg[~np.isin(GT_id_pos_corr_neg, GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg)] 
GT_id_neg_corr_pos_only = GT_id_neg_corr_pos[~np.isin(GT_id_neg_corr_pos, GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_pos)] 
GT_id_neg_corr_neg_only = GT_id_neg_corr_neg[~np.isin(GT_id_neg_corr_neg, GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_neg)] 

#np.unique(GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg + GT_id_pos_corr_pos_only.tolist() + GT_id_pos_corr_neg_only.tolist() + GT_id_neg_corr_pos_only.tolist() + GT_id_neg_corr_neg_only.tolist(), return_counts = True)
#np.unique((GT_id_pos_corr_pos.tolist() + GT_id_pos_corr_neg.tolist()  + GT_id_neg_corr_pos.tolist()  + GT_id_neg_corr_neg.tolist()), return_counts = True)

#Read IC templates
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
#ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
#ica100_np = np.array(ica100_template.dataobj)
t1 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.1) for i in GT_id_pos_corr_pos_neg_corr_pos]
t2 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.2) for i in GT_id_pos_corr_pos_neg_corr_neg]
t3 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in GT_id_pos_corr_neg_neg_corr_pos]
t4 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.4) for i in GT_id_pos_corr_neg_neg_corr_neg]
t5 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in GT_id_pos_corr_pos_only]
t6 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.6) for i in GT_id_pos_corr_neg_only]
t7 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.7) for i in GT_id_neg_corr_pos_only]
t8 = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.8) for i in GT_id_neg_corr_neg_only]
IC_img = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8

IC_img_value = [IC_img[i].get_fdata() for i in range(len(IC_img))]
IC_data = np.sum(IC_img_value, axis = 0)
last_IC_data_10 = np.multiply(IC_data, np.full(IC_data.shape, 10))
last_IC_10 = new_img_like(IC_img[0], last_IC_data_10)

#plotting.plot_roi(last_IC_10, cut_coords = [1, -71, 13], cmap = "Set1", colorbar = True)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = colors.ListedColormap(['red', 'blue', 'green', 'purple', 'yellow', 'brown', 'pink', 'gray']), colorbar = True, vmin = 10, vmax = 19)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

#Find intersection of 24 predictive GT
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1
new_IC_template_mask = new_img_like(ica100_template, data)
plotting.plot_roi(new_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask)))

#Compute intersection of predicve GMs and ICs
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = int(0)
data[np.isin(data, new_GM_id)] = int(1)
GM_template_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask)))

GM_IC_template_mask = intersect_masks([new_IC_template_mask, GM_template_mask], threshold = 1, connected = False)
plotting.plot_roi(GM_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_IC_template_mask)))

#Check pos/neg in overlap region
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.2
data[np.isin(data, new_GM_id_neg)] = 1.1
GM_intersec_pos_neg = new_img_like(resampled_GM, data)

IC_pos_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_pos_only.tolist()
IC_pos_net_neg = GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg + GT_id_pos_corr_neg_only.tolist()
IC_neg_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_pos + GT_id_neg_corr_pos_only.tolist()
IC_neg_net_neg = GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_neg + GT_id_neg_corr_neg_only.tolist()

t_p_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_pos_net_neg]

IC_value_p_p = [t_p_p[i].get_fdata() for i in range(len(t_p_p))]
IC_p_p_data = np.sum(IC_value_p_p, axis = 0)

IC_value_p_n = [t_p_n[i].get_fdata() for i in range(len(t_p_n))]
IC_p_n_data = np.sum(IC_value_p_n, axis = 0)

if (len(IC_value_p_n) > 0 and len(IC_value_p_p) > 0):
    print(1)
    IC_p_data = np.sum([IC_p_p_data, IC_p_n_data], axis = 0)
elif (len(IC_value_p_n) > 0 and len(IC_value_p_p) == 0):
    print(2)
    IC_p_data = IC_p_n_data
elif (len(IC_value_p_n) == 0 and len(IC_value_p_p) > 0):
    print(3)
    IC_p_data = IC_p_p_data
last_IC_10_p = new_img_like(t_p_p[0], IC_p_data)

GM_IC_pos_net_data = np.sum([GM_intersec_pos_neg.get_fdata(), last_IC_10_p.get_fdata()], axis = 0)
GM_IC_pos_net_data_sub = np.subtract(np.multiply(GM_IC_pos_net_data, np.full(GM_IC_pos_net_data.shape, 10)), np.full(GM_IC_pos_net_data.shape, 23))
GM_IC_pos_net_data_sub[GM_IC_pos_net_data_sub < 0] = 0
GM_IC_pos_net = new_img_like(GM_intersec_pos_neg, np.around(GM_IC_pos_net_data_sub))
plotting.plot_roi(GM_IC_pos_net, cmap = cmap4, colorbar = True, cut_coords = [-1, -44, 12])
plotting.plot_roi(GM_IC_pos_net, cmap = colors.ListedColormap(['blue']), cut_coords = [-1, -44, 12], colorbar = True)
print(np.unique(get_data(GM_IC_pos_net), return_counts = True))

GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id_pos)] = 1.2
data[np.isin(data, new_GM_id_neg)] = 1.1
GM_intersec_pos_neg = new_img_like(resampled_GM, data)

t_n_p = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.5) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color('IC_55_atlas_withoutprob.nii.gz', i - 1, 1.3) for i in IC_neg_net_neg]

IC_value_n_p = [t_n_p[i].get_fdata() for i in range(len(t_n_p))]
IC_n_p_data = np.sum(IC_value_n_p, axis = 0)

IC_value_n_n = [t_n_n[i].get_fdata() for i in range(len(t_n_n))]
IC_n_n_data = np.sum(IC_value_n_n, axis = 0)

if (len(IC_value_n_n) > 0 and len(IC_value_n_p) > 0):
    print(1)
    IC_n_data = np.sum([IC_n_p_data, IC_n_n_data], axis = 0)
elif (len(IC_value_n_n) > 0 and len(IC_value_n_p) == 0):
    print(2)
    IC_n_data = IC_n_n_data
elif (len(IC_value_n_n) == 0 and len(IC_value_n_p) > 0):
    print(3)
    IC_n_data = IC_n_p_data
last_IC_10_n = new_img_like(t_n_p[0], IC_n_data)

GM_IC_pos_net_data = np.sum([GM_intersec_pos_neg.get_fdata(), last_IC_10_n.get_fdata()], axis = 0)
GM_IC_pos_net_data_sub = np.subtract(np.multiply(GM_IC_pos_net_data, np.full(GM_IC_pos_net_data.shape, 10)), np.full(GM_IC_pos_net_data.shape, 23))
GM_IC_pos_net_data_sub[GM_IC_pos_net_data_sub < 0] = 0
GM_IC_pos_net = new_img_like(GM_intersec_pos_neg, np.around(GM_IC_pos_net_data_sub))
plotting.plot_roi(GM_IC_pos_net, cut_coords = [-1, -44, 12], cmap = colors.ListedColormap(['purple']), colorbar = True)
plotting.plot_roi(GM_IC_pos_net, cmap = cmap4, colorbar = True, cut_coords = [-1, -44, 12])
print(np.unique(get_data(GM_IC_pos_net), return_counts = True))

'''
#Energy/WHR ~ ICA + Control
X_WHR2 = np.concatenate((stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ BMI + Control
X_WHR_BMI = np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_BMI_train_E, WHR_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ Control
X_WHR_Control = np.array(df_SCE_gmv_2000_female[['age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_Control_train_E, WHR_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)
'''

---------------------------------------------------------------------------------------------------------------------------------------------------
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[stats.zscore(df_SCE_gmv_2000['IQ']) > 0]
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[stats.zscore(df_SCE_gmv_2000['IQ']) < 0]

df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] > 0]
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] < 0]

#Use median of slope instead of 0
median = np.median(df_SCE_gmv_2000['slope'])
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] > median]
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] < median]


df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['SCE'] > 0]
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['SCE'] < 0]

#Energy/WHR ~ GMV + Control
#X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR11 = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), np.array(df_SCE_gmv_2000_SCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 139, rand = 9)
GMV_train_WHR1, GMV_test_WHR1, beta_GMV1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR11, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA_GT + Control
#X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT1 = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] ])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 110, rand = 9)
WHR_GT_train_E1, WHR_GT_test_E1, beta_GT1, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 110, rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

beta_GMV_df = pd.DataFrame(beta_GMV1).T
beta_GT_df = pd.DataFrame(beta_GT1).T
'''
#Energy/WHR ~ ICA + Control
X_WHR2 = np.concatenate((stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ ICA_GT + Control
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 110, rand = 9)

#Energy/WHR ~ BMI + Control
X_WHR_BMI = np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_BMI_train_E, WHR_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ Control
X_WHR_Control = np.array(df_SCE_gmv_2000_SCE[['sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_Control_train_E, WHR_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), rand = 9)
'''

----------------------------------------------------------------------------------------------------------------------------------------------------------
df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')
df['eid'] = df['eid'].astype(str)

df_SCE_gmv_WHR = pd.merge(df_SCE_gmv_2000, df[['eid', '48-0.0', '48-1.0', '48-2.0', '48-3.0', '49-0.0', '49-1.0', '49-2.0', '49-3.0']], on = 'eid')

df_waist = df_SCE_gmv_WHR[['eid', '48-0.0', '48-1.0', '48-2.0', '48-3.0']]
df_waist_noNeg = df_waist[df_waist.columns[1:]].mask(df_waist[df_waist.columns[1:]] < 0).dropna(subset = df_waist.columns[1:], how = 'all')
df_waist_noNeg.reset_index(drop = True, inplace = True)
df_waist_noNeg['eid'] = df_waist['eid'].astype(str)
#df_waist_noNeg_noMRI = pd.merge(df_BMI_liking_energy_mri_age_sex, df_waist_noNeg, on = 'eid')
df_waist_noNeg['48'] = 0
for i in range(df_waist_noNeg.shape[0]):
    df_waist_noNeg['48'].loc[i] = np.nanmean(df_waist_noNeg[df_waist.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_waist_noNeg.shape[0])), end="")

df_waist_mean = df_waist_noNeg.drop(columns = df_waist.columns[1:])

df_hip = df_SCE_gmv_WHR[['eid', '49-0.0', '49-1.0', '49-2.0', '49-3.0']]
df_hip_noNeg = df_hip[df_hip.columns[1:]].mask(df_hip[df_hip.columns[1:]] < 0).dropna(subset = df_hip.columns[1:], how = 'all')
df_hip_noNeg.reset_index(drop = True, inplace = True)
df_hip_noNeg['eid'] = df_hip['eid'].astype(str)
df_hip_noNeg['49'] = 0
for i in range(df_hip_noNeg.shape[0]):
    df_hip_noNeg['49'].loc[i] = np.nanmean(df_hip_noNeg[df_hip.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_hip_noNeg.shape[0])), end="")

df_hip_mean = df_hip_noNeg.drop(columns = df_hip.columns[1:])

df_waist_hip = pd.merge(df_waist_mean, df_hip_mean, on = 'eid')

df_waist_hip['waist_hip_ratio_nonZscore'] = df_waist_hip['48']/df_waist_hip['49']

df_SCE_gmv_WHR_ratio = pd.merge(df_SCE_gmv_2000, df_waist_hip, on = 'eid')

df_SCE_gmv_WHR_ratio[df_SCE_gmv_WHR_ratio['waist_hip_ratio_nonZscore'] == 1][['eid', 'waist_hip_ratio', 'waist_hip_ratio_nonZscore']]
df_SCE_gmv_WHR_ratio[df_SCE_gmv_WHR_ratio['waist_hip_ratio_nonZscore'] == 0.85][['eid', 'waist_hip_ratio', 'waist_hip_ratio_nonZscore']]
df_SCE_gmv_WHR_ratio[df_SCE_gmv_WHR_ratio['waist_hip_ratio_nonZscore'] == 0.9][['eid', 'waist_hip_ratio', 'waist_hip_ratio_nonZscore']]
#WHR = 1., zscored WHR = 1.716036  WHR = 0.85, zscored WHR = -0.139564    WHR = 0.9, zscored WHR = 0.478969 

df_SCE_gmv_2000_WHR = df_SCE_gmv_2000.loc[((df_SCE_gmv_2000['waist_hip_ratio'] >= -0.139564	) & (df_SCE_gmv_2000['sex'] == 0)) | ((df_SCE_gmv_2000['waist_hip_ratio'] >= 0.478969) & (df_SCE_gmv_2000['sex'] == 1))]
df_SCE_gmv_2000_WHR = df_SCE_gmv_2000.loc[((df_SCE_gmv_2000['waist_hip_ratio'] < -0.139564	) & (df_SCE_gmv_2000['sex'] == 0)) | ((df_SCE_gmv_2000['waist_hip_ratio'] < 0.478969) & (df_SCE_gmv_2000['sex'] == 1))]

#Energy/WHR ~ GMV + Control
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_WHR[gmv]), np.array(df_SCE_gmv_2000_WHR[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_WHR[gmv]), np.array(df_SCE_gmv_2000_WHR[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, GMV_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_WHR['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA_GT + Control
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_WHR.index])), np.array(df_SCE_gmv_2000_WHR[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_WHR.index])), np.array(df_SCE_gmv_2000_WHR[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, GT_models = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_WHR['waist_hip_ratio'])), n_beta = 110, rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

df_test = pd.read_csv('whr_test.csv')
df_test['eid'] = df_test['eid'].astype(str)

------------------------------------------------------------------------------------------------------------------------------------------
def mean_error(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    #check_consistent_length(y_true, y_pred, sample_weight)
    return np.average(y_true - y_pred, weights=sample_weight, axis=0)

def mean_error_score():
    return make_scorer(mean_error, greater_is_better=False)

def TrainBoth_TestOne(p_grid, out_fold, in_fold, model, X, y, gender, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2train1 = []
    r2train0 = []
    r2test = []
    r2test1 = []
    r2test0 = []
    beta = []
    models = []

    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        gender_train = np.array(gender)[train]
        gender_test = np.array(gender)[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = make_scorer(mean_error, greater_is_better=False))
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_absolute_error")
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        y_pred1 = clf.predict(x_train[gender_train == 1])
        y_pred0 = clf.predict(x_train[gender_train == 0])
        #r2train.append(r2_score(y_train, y_pred))
        #r2train1.append(r2_score(y_train[gender_train == 1], y_pred1))
        #r2train0.append(r2_score(y_train[gender_train == 0], y_pred0))
        #r2train.append(mean_squared_error(y_train, y_pred))
        #r2train1.append(mean_squared_error(y_train[gender_train == 1], y_pred1))
        #r2train0.append(mean_squared_error(y_train[gender_train == 0], y_pred0))
        #r2train1.append(1-(1-r2_score(y_train[gender_train == 1], y_pred1))*((x_train[gender_train == 1].shape[0]-1)/(x_train[gender_train == 1].shape[0]-x_train[gender_train == 1].shape[1]-1)))
        #r2train0.append(1-(1-r2_score(y_train[gender_train == 0], y_pred0))*((x_train[gender_train == 0].shape[0]-1)/(x_train[gender_train == 0].shape[0]-x_train[gender_train == 0].shape[1]-1)))
        #r2train.append(mean_absolute_error(y_train, y_pred))
        #r2train1.append(mean_absolute_error(y_train[gender_train == 1], y_pred1))
        #r2train0.append(mean_absolute_error(y_train[gender_train == 0], y_pred0))
        r2train.append(mean_error(y_train, y_pred))
        r2train1.append(mean_error(y_train[gender_train == 1], y_pred1))
        r2train0.append(mean_error(y_train[gender_train == 0], y_pred0))

        #predict labels on the test set
        y_pred = clf.predict(x_test)
        y_pred1 = clf.predict(x_test[gender_test == 1])
        y_pred0 = clf.predict(x_test[gender_test == 0])
        #r2test.append(r2_score(y_test, y_pred))
        #r2test1.append(r2_score(y_test[gender_test == 1], y_pred1))
        #r2test0.append(r2_score(y_test[gender_test == 0], y_pred0))
        #r2test.append(mean_squared_error(y_test, y_pred))
        #r2test1.append(mean_squared_error(y_test[gender_test == 1], y_pred1))
        #r2test0.append(mean_squared_error(y_test[gender_test == 0], y_pred0))
        #r2test1.append(1-(1-r2_score(y_test[gender_test == 1], y_pred1))*((x_test[gender_test == 1].shape[0]-1)/(x_test[gender_test == 1].shape[0]-x_test[gender_test == 1].shape[1]-1)))
        #r2test0.append(1-(1-r2_score(y_test[gender_test == 0], y_pred0))*((x_test[gender_test == 0].shape[0]-1)/(x_test[gender_test == 0].shape[0]-x_test[gender_test == 0].shape[1]-1)))
        #r2test.append(mean_absolute_error(y_test, y_pred))
        #r2test1.append(mean_absolute_error(y_test[gender_test == 1], y_pred1))
        #r2test0.append(mean_absolute_error(y_test[gender_test == 0], y_pred0))
        r2test.append(mean_error(y_test, y_pred))
        r2test1.append(mean_error(y_test[gender_test == 1], y_pred1))
        r2test0.append(mean_error(y_test[gender_test == 0], y_pred0))
        models.append(clf)
        
    if (n_beta):
        return r2train, r2train1, r2train0, r2test, r2test1, r2test0, beta, models
    else:
        return r2train, r2train1, r2train0, r2test, r2test1, r2test0, models

X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)

X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)

gender = df_SCE_gmv_2000['sex']
model = ElasticNet(max_iter = 1000000)
X = X_WHR1
y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']))
n_beta = 139

GMV_train_WHR, GMV_train_WHR1, GMV_train_WHR0, GMV_test_WHR, GMV_test_WHR1, GMV_test_WHR0, beta_GMV, _= TrainBoth_TestOne(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = y, gender = gender, n_beta = 139, rand = 9)

X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)

X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_train, WHR_GT_train1, WHR_GT_train0, WHR_GT_test, WHR_GT_test1, WHR_GT_test0, beta_GT, _ = TrainBoth_TestOne(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = y, gender = gender, n_beta = 139, rand = 9)

X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)

X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_GMV_train, WHR_GT_GMV_train1, WHR_GT_GMV_train0, WHR_GT_GMV_test, WHR_GT_GMV_test1, WHR_GT_GMV_test0, WHR_GT_GMV_model = TrainBoth_TestOne(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = y, gender = gender, rand = 9)

scatter = plt.scatter(np.arange(2000), df_SCE_gmv_2000[['waist_hip_ratio']], c = np.array(df_SCE_gmv_2000[['sex']]), marker = markers.MarkerStyle(marker='.'), facecolors='none', cmap='Spectral')
plt.legend(*scatter.legend_elements())

#Select female
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]
#Select male
df_SCE_gmv_2000_male = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]

def Pseudo_CV(p_grid, out_fold, in_fold, model, X_m, X_f, y_m, y_f, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2train1 = []
    r2train0 = []
    r2test = []
    r2test1 = []
    r2test0 = []
    beta = []
    models = []

    for j, (train, test) in enumerate(outer_cv.split(X_m, y_m)):
        #split dataset to decoding set and test set
        x_train_m, x_test_m = X_m[train], X_m[test]
        y_train_m, y_test_m = y_m[train], y_m[test]
        x_train_f, x_test_f = X_f[train], X_f[test]
        y_train_f, y_test_f = y_f[train], y_f[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train_m, y_train_m)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train_m)
        #r2train.append(r2_score(y_train_m, y_pred))
        r2train.append(mean_squared_error(y_train_m, y_pred))
         
        #predict labels on the test set
        y_pred = clf.predict(x_test_f)
        #r2test.append(r2_score(y_test_f, y_pred))
        r2test.append(mean_squared_error(y_test_f, y_pred))
        models.append(clf)
        
    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models

X_WHR1_m = np.concatenate((np.array(df_SCE_gmv_2000_male[gmv]), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1_f = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)

y_m = np.array(stats.zscore(df_SCE_gmv_2000_male['waist_hip_ratio']))
y_f = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio']))

GMV_train_WHR, GMV_test_WHR, beta_GMV, GMV_models= Pseudo_CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR1_m, X_f = X_WHR1_f, y_m = y_m, y_f = y_f, n_beta = 139, rand = 9)

X_WHR_GT_m = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] ])), axis = 1)
X_WHR_GT_f = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']] )), axis = 1)

WHR_GT_train, WHR_GT_test, beta_GT, GT_models = Pseudo_CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR_GT_m, X_f = X_WHR_GT_f, y_m = y_m, y_f = y_f, n_beta = 139, rand = 9)

X_WHR_GMV_GT_m = np.concatenate((np.array(df_SCE_gmv_2000_male[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT_f = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)

WHR_GT_GMV_train, WHR_GT_GMV_test, WHR_GT_GMV_model = Pseudo_CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR_GMV_GT_m, X_f = X_WHR_GMV_GT_f, y_m = y_m, y_f = y_f, rand = 9)

def Pseudo_CV_f(p_grid, out_fold, in_fold, model, X_m, X_f, y_m, y_f, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2train1 = []
    r2train0 = []
    r2test = []
    r2test1 = []
    r2test0 = []
    beta = []
    models = []

    for j, (train, test) in enumerate(outer_cv.split(X_f, y_f)):
        #split dataset to decoding set and test set
        x_train_m, x_test_m = X_m[train], X_m[test]
        y_train_m, y_test_m = y_m[train], y_m[test]
        x_train_f, x_test_f = X_f[train], X_f[test]
        y_train_f, y_test_f = y_f[train], y_f[test]
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train_m, y_train_m)
        if (n_beta):
            beta.append(clf.best_estimator_.coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train_m)
        r2train.append(r2_score(y_train_m, y_pred))
        #r2train.append(mean_squared_error(y_train_m, y_pred))
         
        #predict labels on the test set
        y_pred = clf.predict(x_test_f)
        r2test.append(r2_score(y_test_f, y_pred))
        #r2test.append(mean_squared_error(y_test_f, y_pred))
        models.append(clf)
        
    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models


X_WHR1_m = np.concatenate((np.array(df_SCE_gmv_2000_male[gmv]), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR1_f = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)

y_m = np.array(stats.zscore(df_SCE_gmv_2000_male['waist_hip_ratio']))
y_f = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio']))

GMV_train_WHR, GMV_test_WHR, beta_GMV, GMV_models= Pseudo_CV_f(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR1_f, X_f = X_WHR1_m, y_m = y_f, y_f = y_m, n_beta = 139, rand = 9)

X_WHR_GT_m = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] ])), axis = 1)
X_WHR_GT_f = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']] )), axis = 1)

WHR_GT_train, WHR_GT_test, beta_GT, GT_models = Pseudo_CV_f(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR_GT_f, X_f = X_WHR_GT_m, y_m = y_f, y_f = y_m, n_beta = 139, rand = 9)

X_WHR_GMV_GT_m = np.concatenate((np.array(df_SCE_gmv_2000_male[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR_GMV_GT_f = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)

WHR_GT_GMV_train, WHR_GT_GMV_test, WHR_GT_GMV_model = Pseudo_CV_f(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X_m = X_WHR_GMV_GT_f, X_f = X_WHR_GMV_GT_m, y_m = y_f, y_f = y_m, rand = 9)

----------------------------------------------------------------------------------------------------------------------------------------------

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')

beta_GMV_id_nonzero = beta_GMV_id[(beta_GMV_id != 0).all(1)]
beta_GMV_id_nonzero.reset_index(drop = True, inplace = True)

gmv_pred = np.array(beta_GMV_id_nonzero['eid'].astype(str))

ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

IC_Graph_df = pd.DataFrame(IC_Graph)
IC_Graph_df.columns = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]

beta_GT_nonzero = beta_GT_df[(beta_GT_df != 0).all(1)]
beta_GT_nonzero.reset_index(drop = True, inplace = True)

GT_pred = np.array(beta_GT_nonzero['ICs'])

X_WHR = np.concatenate((np.array(df_SCE_gmv_2000[gmv_pred]), np.array(IC_Graph_df[GT_pred])), axis = 1)
y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio']))

def OLS(out_fold, in_fold, X, y, rand):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2test = []
    
    tn = []
    tt = []
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        clf = LinearRegression()
        clf.fit(x_train, y_train)
        #print(clf1.best_score_)
        #print(j)
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #r2test.append(mean_squared_error(y_test, y_pred))
        r2test.append(r2_score(y_test, y_pred))
        #print(r2test)
    
    return np.mean(r2test)

OLS_all = OLS(out_fold = 5, in_fold = 5, X = X_WHR, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)
OLS_diff = []

for i in range(X_WHR.shape[1]):
    OLS_loocv = OLS(out_fold = 5, in_fold = 5, X = np.delete(X_WHR, i, 1), y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)   
    OLS_diff.append(OLS_all - OLS_loocv)
    print("\r Process{}%".format(round((i+1)*100/X_WHR.shape[1])), end="")

gmv_pred_name = np.array(beta_GMV_id_nonzero[beta_GMV_id_nonzero['eid'].astype(str) == gmv_pred]['GMV_name'])

OLS_df = pd.DataFrame(columns = ['Diff', 'Name'])
OLS_df['Name'] = np.concatenate((gmv_pred_name, GT_pred), axis = 0)
OLS_df['Diff'] = np.array(OLS_diff)
OLS_diff_sort = OLS_df.sort_values('Diff', ascending = False)
OLS_diff_sort.reset_index(drop = True, inplace = True)
OLS_diff_sort.to_csv('OLS_diff_sort.csv', index = False)

dpi = 1600
title = 'Leave-one out CV of WHR using all predictive GMV and IC'
fig, ax = plt.subplots(dpi = dpi)
ax.set_ylabel('R2 %')
ax.set_title(title)
plt.bar(x = np.concatenate((gmv_pred_name, GT_pred), axis = 0), height = OLS_diff)
plt.xticks(rotation=90)
ax.set_xticklabels(np.concatenate((gmv_pred_name, GT_pred), axis = 0), fontsize = 3)
fig.tight_layout()

plt.show()

#clf1 =  GridSearchCV(estimator = ElasticNet(max_iter = 1000000), param_grid = par_grid, cv = KFold(n_splits = 5, shuffle = True, random_state = 9), scoring = "r2")
clf1 =  GridSearchCV(estimator = LinearRegression(), param_grid = par_grid, cv = KFold(n_splits = 5, shuffle = True, random_state = 9), scoring = "r2")
clf1.fit(X_WHR, y)
all_r2 = clf1.best_score_

r2_diff = []

for i in range(X_WHR.shape[1]):
    #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
    clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "r2")
    clf.fit(np.delete(X_WHR, i, 1), y)
    
    r2_diff.append(all_r2 - clf.best_score_)
    print("\r Process{}%".format(round((i+1)*100/X_WHR.shape[1])), end="")

-------------------------------------------------------------------------------------------
beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T
beta_GMV_GT_df = pd.DataFrame(beta_GMV_GT).T
beta_control_df = pd.DataFrame(beta_control).T

beta_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
WHR_Control_test_E.append('R2')
beta_control_df.loc[9] = WHR_Control_test_E

beta_GMV_control_df = beta_GMV_df.loc[139:]
beta_GMV_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
beta_GMV_control_df = beta_GMV_control_df.rename(columns = {'control': 'GMV_name'})
GMV_test_WHR.append('R2')
beta_GMV_control_df = pd.concat([pd.DataFrame([GMV_test_WHR], columns = beta_GMV_control_df.columns), beta_GMV_control_df])
beta_GMV_df = beta_GMV_df.loc[:138]

#Get beta coefficients
beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')
#beta_GMV_id.to_csv('beta_GMV_id.csv', index = False)
#beta_GMV_id = pd.read_csv('beta_GMV_id.csv')
beta_GMV_id = beta_GMV_id.drop(columns = ['eid'])
beta_GMV_control_id = pd.concat([beta_GMV_control_df, beta_GMV_id])
beta_GMV_control_id_nonzero = beta_GMV_control_id[(beta_GMV_control_id != 0).all(1)]
beta_GMV_control_id_nonzero.reset_index(drop = True, inplace = True)

#beta_GMV_control_id_nonzero.round(5).to_csv('beta_GMV_control_id_nonzero.csv', index = False)

beta_GT_control_df = beta_GT_df.loc[110:]
beta_GT_control_df['control'] = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
beta_GT_control_df = beta_GT_control_df.rename(columns = {'control': 'ICs'})
WHR_GT_test_E.append('R2')
beta_GT_control_df = pd.concat([pd.DataFrame([WHR_GT_test_E], columns = beta_GT_control_df.columns), beta_GT_control_df])
beta_GT_df = beta_GT_df.loc[:109]

ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

#Drop ICs with beta = 0 in any round
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_control_id = pd.concat([beta_GT_control_df, beta_GT_df])
beta_GT_control_id_nonzero = beta_GT_control_id[(beta_GT_control_id != 0).all(1)]
beta_GT_control_id_nonzero.reset_index(drop = True, inplace = True)

#beta_GT_control_id_nonzero.round(5).to_csv('beta_GT_control_id_nonzero.csv', index = False)

----------------------------------------------------------------------------------------------------------------------------------
control = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
#control = ['age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']

llo_reg = {}
for i in range(len(control)):
    llo_ctrl = control[:i] + control[i+1:]
    X_WHR_Control = np.array(df_SCE_gmv_2000[llo_ctrl])
    #WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio_zscore']), n_beta = 8, rand = 9)
    WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 9, rand = 9)
    keys = 'drop_' + control[i] + '_train'
    llo_reg[keys] = WHR_Control_train_E
    keys1 = 'drop_' + control[i] + '_test'
    llo_reg[keys1] = WHR_Control_test_E
    #keys2 = 'drop_' + control[i] + '_beta'
    #llo_reg[keys2] = beta_GMV

X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)
X_WHR1_sex = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex']])), axis = 1)
GMV_sex_train_WHR, GMV_sex_test_WHR, beta_sex_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1_sex, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)
X_WHR1_nosex = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_nosex_train_WHR, GMV_nosex_test_WHR, beta_nosex_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1_nosex, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)

WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[control]), y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 9, rand = 9)
Sex_train_E, Sex_test_E, beta_Sex,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[['sex']]), y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 1, rand = 9)
WHR_Control_nosex_train_E, WHR_Control_nosex_test_E, beta_nosex_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[control[1:]]), y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 9, rand = 9)

X_WHR_GT1 = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 119, rand = 9)
X_WHR_GT1_sex = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex']])), axis = 1)
WHR_GT_sex_train_E, WHR_GT_sex_test_E, beta_sex_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1_sex, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 119, rand = 9)
X_WHR_GT1_nosex = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_nosex_train_E, WHR_GT_nosex_test_E, beta_nosex_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1_nosex, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 119, rand = 9)

#zscore by gender
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(df_SCE_gmv_2000['waist_hip_ratio_zscore']), n_beta = 139, rand = 9)

WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[control]), y = np.array(df_SCE_gmv_2000['waist_hip_ratio_zscore']), n_beta = 9, rand = 9)

X_WHR_GT1 = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, _ = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(df_SCE_gmv_2000['waist_hip_ratio_zscore']), n_beta = 119, rand = 9)

----------------------------------------------------------------------------------------------------------------------------------
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]
df_SCE_gmv_2000_female['waist_hip_ratio_zscore'] = stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])
df_SCE_gmv_2000_male = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]
df_SCE_gmv_2000_male['waist_hip_ratio_zscore'] = stats.zscore(df_SCE_gmv_2000_male['waist_hip_ratio'])

df_SCE_gmv_2000 = pd.concat([df_SCE_gmv_2000_female, df_SCE_gmv_2000_male]).sort_index()
----------------------------------------------------------------------------------------------------------------------------------
#labels = ['all controls except sex', 'GMV + controls except sex', 'GT + controls except sex']
labels = ['all controls', 'controls drop sex', 'sex only', 'GMV + controls', 'GMV + sex', 'GMV + controls drop sex', 'GT + controls', 'GT + sex', 'GT + controls drop sex']
#labels = ['all controls'] + ['drop_' + i + '_test' for i in control]
dpi = 1600
title = 'Regression results of WHR with control/sex/brain features'
#title = 'Regression results of WHR with leave-one-control-out'

x = np.arange(len(labels)) 

#train_mean = [np.mean(WHR_Control_train_E)] + [np.mean(llo_reg['drop_' + i + '_train']) for i in control]
#test_mean = [np.mean(WHR_Control_test_E)] + [np.mean(llo_reg['drop_' + i + '_test']) for i in control]
#train_std = [np.std(WHR_Control_train_E)] + [np.std(llo_reg['drop_' + i + '_train']) for i in control]
#test_std = [np.std(WHR_Control_test_E)] + [np.std(llo_reg['drop_' + i + '_test']) for i in control]

train_mean = [np.mean(WHR_Control_train_E), np.mean(WHR_Control_nosex_train_E), np.mean(Sex_train_E), np.mean(GMV_train_WHR), np.mean(GMV_sex_train_WHR), np.mean(GMV_nosex_train_WHR), np.mean(WHR_GT_train_E), np.mean(WHR_GT_sex_train_E), np.mean(WHR_GT_nosex_train_E)]
test_mean = [np.mean(WHR_Control_test_E), np.mean(WHR_Control_nosex_test_E), np.mean(Sex_test_E), np.mean(GMV_test_WHR), np.mean(GMV_sex_test_WHR), np.mean(GMV_nosex_test_WHR), np.mean(WHR_GT_test_E), np.mean(WHR_GT_sex_test_E), np.mean(WHR_GT_nosex_test_E)] 
train_std = [np.std(WHR_Control_train_E), np.std(WHR_Control_nosex_train_E), np.std(Sex_train_E), np.std(GMV_train_WHR), np.std(GMV_sex_train_WHR), np.std(GMV_nosex_train_WHR), np.std(WHR_GT_train_E), np.std(WHR_GT_sex_train_E), np.std(WHR_GT_nosex_train_E)]
test_std = [np.std(WHR_Control_test_E), np.std(WHR_Control_nosex_test_E), np.std(Sex_test_E), np.std(GMV_test_WHR), np.std(GMV_sex_test_WHR), np.std(GMV_nosex_test_WHR), np.std(WHR_GT_test_E), np.std(WHR_GT_sex_test_E), np.std(WHR_GT_nosex_test_E)]

#train_mean = [np.mean(WHR_Control_train_E), np.mean(GMV_train_WHR), np.mean(WHR_GT_train_E)]
#test_mean = [np.mean(WHR_Control_test_E), np.mean(GMV_test_WHR),np.mean(WHR_GT_test_E)] 
#train_std = [np.std(WHR_Control_train_E), np.std(GMV_train_WHR), np.std(WHR_GT_train_E)]
#test_std = [np.std(WHR_Control_test_E), np.std(GMV_test_WHR), np.std(WHR_GT_test_E)]


fig, ax = plt.subplots(dpi = dpi)

width = 0.4

rects1 = ax.bar(x - width/2, [round(i, 4) for i in train_mean], width, yerr = [i for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4) for i in test_mean], width, yerr = [i for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE')
#ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
#plt.ylim(0, 1)
#plt.yticks(np.arange(0, 85, step=10))
#plt.yticks(np.linspace(0.0,0.7,0.1))
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
#for i, v in enumerate(train_mean):
   # rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
#for i, v in enumerate(test_mean):
    #rects1.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    
fig.tight_layout()

plt.show()          


#Z-score Cross-validation function
def CV_zscore(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    models = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        pipeline = Pipeline([('scaler', StandardScaler()),
            ('regressor', model)])
        clf =  GridSearchCV(estimator = pipeline, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.steps[1][1].coef_[:n_beta])
        print(j)
        
        #predict labels on the train set
        y_pred = clf.best_estimator_.steps[1][1].predict(stats.zscore(x_train))
        #r2train.append(mean_squared_error(y_train, y_pred))
        r2train.append(r2_score(stats.zscore(y_train), y_pred))
        
        #predict labels on the test set
        y_pred = clf.best_estimator_.steps[1][1].predict(stats.zscore(x_test))
        #r2test.append(mean_squared_error(y_test, y_pred))
        r2test.append(r2_score(stats.zscore(y_test), y_pred))
        print(r2test)

        models.append(clf)
        
    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models

#Set parameters of cross-validation
par_grid = {'regressor__alpha': [1e-2, 1e-1, 1, 1e-3, 10]}
#par_grid = {'alpha': [1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 3, 5, 7, 10]}
rand = 9

WHR_Control_train_E, WHR_Control_test_E, beta_control,_ = CV_zscore(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[control]), y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 9, rand = 9)
Sex_train_E, Sex_test_E, beta_Sex,_ = CV_zscore(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[['sex']]), y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 1, rand = 9)
WHR_Control_nosex_train_E, WHR_Control_nosex_test_E, beta_nosex_control,_ = CV_zscore(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_SCE_gmv_2000[control[1:]]), y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 9, rand = 9)

X_WHR_GT1 = np.concatenate((np.array(IC_Graph), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, _ = CV_zscore(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT1, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 119, rand = 9)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, _ = CV_zscore(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), n_beta = 139, rand = 9)

#PCR Cross-validation function
def CV_pcr(p_grid, out_fold, in_fold, model, X, y, rand, n_beta = False, n_preprocess1 = 0,  n_preprocess2 = 1485):
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    beta = []
    models = []
    
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        scale = StandardScaler()
        scale_by_column = ColumnTransformer(transformers=[
            ('scale', scale, list(range(n_preprocess1, n_preprocess2)))
            ],
            remainder='passthrough')

        pca = PCA(n_components=0.75)
        pca_by_column = ColumnTransformer(transformers=[
            ('pca', pca, list(range(n_preprocess1, n_preprocess2)))
            ],
            remainder='passthrough')

        pipeline = Pipeline(steps = [('scaler', scale_by_column), ('pca', pca_by_column),
            ('regressor', model)])
        clf =  GridSearchCV(estimator = pipeline, param_grid = p_grid, cv = inner_cv, scoring = "r2")
        clf.fit(x_train, y_train)
        if (n_beta):
            beta.append(clf.best_estimator_.steps[2][1].coef_)
        print(j)
        
        #predict labels on the train set
        y_pred = clf.best_estimator_.predict(x_train)
        #r2train.append(mean_squared_error(y_train, y_pred))
        r2train.append(r2_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.best_estimator_.predict(x_test)
        #r2test.append(mean_squared_error(y_test, y_pred))
        r2test.append(r2_score(y_test, y_pred))
        print(r2test)

        models.append(clf)
        
    if (n_beta):
        return r2train, r2test, beta, models
    else:
        return r2train, r2test, models

#Set parameters of cross-validation
par_grid = {'regressor__alpha': [1e-2, 1e-1, 1, 1e-3, 10]}
#par_grid = {'alpha': [1e-2, 3e-2, 5e-2, 7e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 3, 5, 7, 10]}
rand = 9

X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_WHR21 = stats.zscore(np.array(IC55))
IC_train_WHR, IC_test_WHR, beta_IC_PCR, _ = CV_pcr(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = True, rand = 9)
IC_train_WHR1, IC_test_WHR1, _ = CV_pcr(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR21, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E, _ = CV_pcr(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_preprocess1 = 0, n_preprocess2 = 1624, rand = 9)
