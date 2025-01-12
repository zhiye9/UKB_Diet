import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import colors
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from scipy import stats
import networkx as nx
import pickle
import os
from knockpy.knockoffs import GaussianSampler
from numpy import loadtxt
import time
from joblib import Parallel, delayed
from UKB_graph_metrics import *
from utils import OLS, OLS_diff

# Load data
os.chdir('/home/ubuntu/UK_Biobank_diet')
df_SCE_gmv_2000 = pd.read_csv('df_SCE_gmv_2000.csv')
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)

# Load the selected GMV
beta_GMV_selected = pd.read_csv('beta_GMV_selected.csv')
gmv_pred = np.array(beta_GMV_selected['eid'].astype(str))

# Load the selected IC
beta_GT_selected = pd.read_csv('beta_GT_selected.csv')
ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()
IC_Graph_df = pd.DataFrame([i[:-4] for i in IC_Graph])
IC_Graph_df.columns = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
GT_pred = np.array(beta_GT_selected['ICs'])

# Load the SCE GMV
X_pred = np.concatenate((np.array(df_SCE_gmv_2000[gmv_pred]), np.array(IC_Graph_df[GT_pred])), axis = 1)

# Repeat the OLS with different random seeds
np.random.seed(42)
rand_idx_OLS = np.random.randint(0, 1000, 100)

print("LOOCV starts")
results_loocv = Parallel(n_jobs=-1, verbose = 5)(
    delayed(OLS_diff)(
        out_fold=10,
        in_fold=10,
        X=X_pred,
        y=np.array(df_SCE_gmv_2000['waist_hip_ratio']),
        rand=i,
    ) for i in rand_idx_OLS
)

# Get the results
OLS_all, OLS_diff = zip(*results_loocv)
OLS_diff_array = np.array(OLS_diff)
OLS_diff_feature = np.mean(OLS_diff_array, axis=0)

# Match the names and the differences
gmv_pred_name = np.array(beta_GMV_selected[beta_GMV_selected['eid'].astype(str) == gmv_pred]['GMV_name'])

OLS_df = pd.DataFrame(columns = ['Diff', 'Name'])
OLS_df['Name'] = np.concatenate((gmv_pred_name, GT_pred), axis = 0)
OLS_df['Diff'] = np.array(OLS_diff_feature)
OLS_diff_sort = OLS_df.sort_values('Diff', ascending = False)
OLS_diff_sort.reset_index(drop = True, inplace = True)

# plot the OLS differences
dpi = 1600
title = 'Leave-one out CV of WHR using all predictive GMV and IC'
fig, ax = plt.subplots(dpi = dpi)
ax.set_ylabel('R2 %')
ax.set_title(title)
plt.bar(x = np.concatenate((gmv_pred_name, GT_pred), axis = 0), height = OLS_diff_feature*100)
plt.xticks(rotation=90)
ax.set_xticklabels(np.concatenate((gmv_pred_name, GT_pred), axis = 0), fontsize = 3)
fig.tight_layout()

plt.show()

