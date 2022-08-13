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
from UKB_graph_metrics import *

import warnings
warnings.filterwarnings('ignore')

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
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))
    print("\r Process{}%".format(round((i+1)*100/df_SCE_gmv_2000.shape[0])), end="")

#Save IC_Graph as list
#with open("IC_Graph", "wb") as fp:   #Pickling
#    pickle.dump(IC_Graph, fp)
os.chdir('../')
with open("IC_Graph", "rb") as fp:   # Unpickling
    IC_Graph = pickle.load(fp)

#Energy/WHR ~ GMV + Control
#XE1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GMV_train_E, energy_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE1, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV, model_GMV = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA + Control
#XE2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_IC_train_E, energy_IC_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE2, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR2 = np.concatenate((stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR, IC_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
#XE3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_IC_gmv_train_E, energy_IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE3, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC55)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E, IC_gmv_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
#XE_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GT_GMV_train_E, energy_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GMV_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E, WHR_GT_GMV_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ ICA_GT + Control
#XE_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'waist_hip_ratio'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#energy_GT_train_E, energy_GT_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = XE_GT, y = stats.zscore(df_SCE_gmv_2000['energy']), rand = 9)
#X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
#X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
X_WHR_GT = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT, GT_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), n_beta = 110, rand = 9)

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
X_WHR_Control = np.array(df_SCE_gmv_2000[['BMI', 'sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household'] + list(df_genetic_noNAN.columns[1:])])
WHR_Control_train_E, WHR_Control_test_E, WHR_Control_model = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000['waist_hip_ratio'])), rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

#Get beta coefficients
beta_GMV_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
beta_GMV_id = pd.merge(beta_GMV_df, gmv_id, on = 'eid')
#beta_GMV_id.to_csv('beta_GMV_id.csv', index = False)
#beta_GMV_id = pd.read_csv('beta_GMV_id.csv')

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
    ICatlas = nib.load(file)
    data = get_data(resampled_GM)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(ICatlas, data)
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
def extract_IC_color(IC, n, colo):
    data = get_data(IC.slicer[..., n])
    data[data < 5] = 0
    data[data >= 5] = colo
    new_img = new_img_like(IC.slicer[..., n], data)
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

#Read 55 ICA good compoents, start from 1
ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()

#Drop ICs with beta = 0 in any round
beta_GT_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_nonzero = beta_GT_df[(beta_GT_df != 0).all(1)]
beta_GT_nonzero.reset_index(drop = True, inplace = True)

#Find postive (16) and negative (14) correlation ICs separately
beta_GT_pos_corr = beta_GT_df.loc[:54]
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
ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
#ica100_np = np.array(ica100_template.dataobj)
t1 = [extract_IC_color(ica100_template, i - 1, 1.1) for i in GT_id_pos_corr_pos_neg_corr_pos]
t2 = [extract_IC_color(ica100_template, i - 1, 1.2) for i in GT_id_pos_corr_pos_neg_corr_neg]
t3 = [extract_IC_color(ica100_template, i - 1, 1.3) for i in GT_id_pos_corr_neg_neg_corr_pos]
t4 = [extract_IC_color(ica100_template, i - 1, 1.4) for i in GT_id_pos_corr_neg_neg_corr_neg]
t5 = [extract_IC_color(ica100_template, i - 1, 1.5) for i in GT_id_pos_corr_pos_only]
t6 = [extract_IC_color(ica100_template, i - 1, 1.6) for i in GT_id_pos_corr_neg_only]
t7 = [extract_IC_color(ica100_template, i - 1, 1.7) for i in GT_id_neg_corr_pos_only]
t8 = [extract_IC_color(ica100_template, i - 1, 1.8) for i in GT_id_neg_corr_neg_only]
IC_img = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
IC_img = t5 + t6 + t7 + t8
#atlas = image.math_img('np.sum(img, axis = -1)', img = IC_img)
##plotting.plot_roi(atlas, cut_coords = [1, -71, 13], cmap = "tab20", colorbar = True)
#oo = np.unique(atlas.get_data(), return_counts = True)

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

plotting.plot_roi(last_IC_10, cut_coords = [1, -71, 13], cmap = "Set1", colorbar = True)
plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
#plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Spectral", colorbar = True)
np.unique(last_IC_10.get_data(), return_counts = True)

#t1 = extract_IC_color(ica100_template, 1)
#atlas = image.math_img('np.sum(img, axis=-1)', img=[extract_IC_color(ica100_template, 12, 1.1), extract_IC_color(ica100_template, 9, 1.2)])

#Find intersection of 24 predictive GT
ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
IC_template_mask = []
for i in range(len(GT_id)):
    IC_template_mask.append(extract_IC(ica100_template, GT_id[i] - 1))
new_IC_template_mask = intersect_masks(IC_template_mask, threshold = 0)
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

#Find intersection of 17 positive predictive ICs
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
IC_pos_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_pos_only.tolist()
IC_pos_net_neg = GT_id_pos_corr_neg_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_neg + GT_id_pos_corr_neg_only.tolist()
IC_neg_net_pos = GT_id_pos_corr_pos_neg_corr_pos + GT_id_pos_corr_neg_neg_corr_pos + GT_id_neg_corr_pos_only.tolist()
IC_neg_net_neg = GT_id_pos_corr_pos_neg_corr_neg + GT_id_pos_corr_neg_neg_corr_neg + GT_id_neg_corr_neg_only.tolist()

t_p_p = [extract_IC_color(ica100_template, i - 1, 1.1) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color(ica100_template, i - 1, -1.2) for i in IC_pos_net_neg]

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

t_n_p = [extract_IC_color(ica100_template, i - 1, 1.1) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color(ica100_template, i - 1, -1.1) for i in IC_neg_net_neg]

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

t_p_p = [extract_IC_color(ica100_template, i - 1, 1.5) for i in IC_pos_net_pos]
t_p_n = [extract_IC_color(ica100_template, i - 1, 1.3) for i in IC_pos_net_neg]

t_n_p = [extract_IC_color(ica100_template, i - 1, 1.5) for i in IC_neg_net_pos]
t_n_n = [extract_IC_color(ica100_template, i - 1, 1.3) for i in IC_neg_net_neg]

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

#Select female
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]
#Select male
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]

#WHR ~ GMV + Control
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA + Control
X_WHR2 = np.concatenate((stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_train_WHR, IC_test_WHR = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR2, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA + Control
X_WHR3 = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
IC_gmv_train_E, IC_gmv_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR3, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ GMV + ICA_GT + Control
X_WHR_GMV_GT = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]), stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_GMV_train_E, WHR_GT_GMV_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GMV_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ ICA_GT + Control
X_WHR_GT = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
WHR_GT_train_E, WHR_GT_test_E, beta_GT = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_GT, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), n_beta = 110, rand = 9)

#Energy/WHR ~ BMI + Control
X_WHR_BMI = np.array(df_SCE_gmv_2000_female[['BMI', 'age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_BMI_train_E, WHR_BMI_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_BMI, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

#Energy/WHR ~ Control
X_WHR_Control = np.array(df_SCE_gmv_2000_female[['age', 'slope', 'height', 'hand', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])
WHR_Control_train_E, WHR_Control_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR_Control, y = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand = 9)

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T

df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['SCE'] > 0]
df_SCE_gmv_2000_SCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['SCE'] < 0]

#Energy/WHR ~ GMV + Control
X_WHR1 = np.concatenate((np.array(df_SCE_gmv_2000_SCE[gmv]), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
GMV_train_WHR, GMV_test_WHR, beta_GMV = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = X_WHR1, y = np.array(stats.zscore(df_SCE_gmv_2000_SCE['waist_hip_ratio'])), n_beta = 139, rand = 9)

#Energy/WHR ~ ICA + Control
X_WHR2 = np.concatenate((stats.zscore(np.array([IC55[i] for i in df_SCE_gmv_2000_SCE.index])), np.array(df_SCE_gmv_2000_SCE[['BMI', 'sex', 'age', 'slope', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household', 'liking_low', 'liking_high', 'energy'] + list(df_genetic_noNAN.columns[1:])])), axis = 1)
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

beta_GMV_df = pd.DataFrame(beta_GMV).T
beta_GT_df = pd.DataFrame(beta_GT).T
