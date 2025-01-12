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
from utils import CV, make_artificial_features, loop_CV

# Load data
df_SCE_gmv_2000 = pd.read_csv('df_SCE_gmv_2000.csv')
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)

# Load the control and GMV id
control = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
gmv = np.genfromtxt('GMV.txt', dtype='str')

# Parameters for the ElasticNet model
par_grid = {"alpha": np.logspace(-2, 1, 15), "l1_ratio": [.3, .5, .7, .9]}

# Parameters for noise injection
X_GMV = np.array(df_SCE_gmv_2000[gmv])
n_injected_noise = 139 

# Generate 50 random seeds
rand_id = np.random.randint(0, 1000, 50)
r_total_GMV = []

print("GMV CV started")

# Loop over the cross-validation function for multiple random nosie injection
for j in range(len(rand_id)):
    random_state = rand_id[j]

    # Inject ramdom permutation noise
    X_GMV_artificial = make_artificial_features(
        X=X_GMV,
        noise=n_injected_noise,
        random_state=random_state
    )

    # Loop over the cross-validation function for multiple random seeds
    X_GMV_artificial_control = np.concatenate((X_GMV_artificial, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
    results_GMV_artificial_control = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_artificial_control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 0, n_beta = 278)
    
    r_total_GMV.append(results_GMV_artificial_control)

# Read ICA data
IC55 = []
for i in range(0, df_SCE_gmv_2000.shape[0]):
    tem = np.loadtxt(df_SCE_gmv_2000['file'].loc[i])
    IC55.append(tem)

# Compute graph theory metrics of ICA
IC_Graph = []
for i in range(df_SCE_gmv_2000.shape[0]):
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))

# Parameters for noise injection
X_GT = stats.zscore(np.array([i[:-4] for i in IC_Graph]))
n_injected_noise = 110 

r_total_GT = []
print("GT CV started")

for j in range(len(rand_id)):
    random_state = rand_id[j]
    # Inject ramdom permutation noise
    X_GT_artificial = make_artificial_features(
        X=X_GT,
        noise=n_injected_noise,
        random_state=random_state
    )

    X_GT_artificial_control = np.concatenate((X_GT_artificial, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
    results_GT_artificial_control = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_artificial_control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), rand_idx_length = 100, n_jobs = 1, verbose = 0, n_beta = 220)

    r_total_GT.append(results_GT_artificial_control)