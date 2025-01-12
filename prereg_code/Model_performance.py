import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import colors
import seaborn as sns
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy import stats
import networkx as nx
import pickle
import os
from knockpy.knockoffs import GaussianSampler
from numpy import loadtxt
import time
from joblib import Parallel, delayed
from utils import  CV, CV_pcr, loop_CV
from UKB_graph_metrics import *

# Load data
os.chdir('/home/ubuntu/UK_Biobank_diet')
df_SCE_gmv_2000 = pd.read_csv('df_SCE_gmv_2000.csv')
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)

# Load the control and GMV id
control = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
gmv = np.genfromtxt('GMV.txt', dtype='str')

# Read ICA data
IC55 = []
for i in range(0, df_SCE_gmv_2000.shape[0]):
    tem = np.loadtxt(df_SCE_gmv_2000['file'].loc[i])
    IC55.append(tem)

# Compute graph theory metrics of ICA
IC_Graph = []
for i in range(df_SCE_gmv_2000.shape[0]):
    IC_Graph.append(Graph_metrics(df_SCE_gmv_2000['file'].loc[i], 55))

# Parameters for the ElasticNet model
par_grid = {"alpha": np.logspace(-2, -1, 5), "l1_ratio": [.7, .9]}

# Loop over the cross-validation function for GMV
X_GMV = np.array(df_SCE_gmv_2000[gmv])
X_GMV_control = np.concatenate((X_GMV, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 139)
r2train_GMV, r2test_GMV, beta_GMV, model_GMV = zip(*results_GMV)

# Loop over the cross-validation function for GT
X_GT = stats.zscore(np.array([i for i in IC_Graph]))
X_GT_control = np.concatenate((X_GT, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GT= loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 110)
r2train_GT, r2test_GT, beta_GT, model_GT = zip(*results_GT)

# Loop over the cross-validation function for GMV and GT
X_GMV_GT = np.concatenate((X_GMV, X_GT), axis = 1)
X_GMV_GT_control = np.concatenate((X_GMV_GT, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV_GT = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_GT_control, y = np.array(df_SCE_gmv_2000['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 249)
r2train_GMV_GT, r2test_GMV_GT, beta_GMV_GT, model_GMV_GT = zip(*results_GMV_GT)    

# Loop over the cross-validation function for IC   
X_IC = stats.zscore(np.array(IC55))
X_IC_control = np.concatenate((X_IC, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)

np.random.seed(42)
rand_idx_pcr = np.random.randint(0, 1000, 10)

print("PCR CV starts")
results_IC_control = Parallel(n_jobs=-1, verbose = 5)(
    delayed(CV_pcr)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=ElasticNet(max_iter = 1000000),
        X=X_IC_control,
        y=np.array(df_SCE_gmv_2000['waist_hip_ratio']),
        rand=i,
    ) for i in rand_idx_pcr
)

# Extract results in r_total_GT
r2_train_IC = []
r2_test_IC = []
model_IC = []

for i in range(len(results_IC_control)):
    r2_train_IC.append(results_IC_control[i][0])
    r2_test_IC.append(results_IC_control[i][1])
    model_IC.append(results_IC_control[i][2])

# Loop over the cross-validation function for IC and GMV
X_GMV = np.array(df_SCE_gmv_2000[gmv])
X_IC = stats.zscore(np.array(IC55))
X_IC_GMV= np.concatenate((X_IC, X_GMV), axis = 1)
X_IC_GMV_control = np.concatenate((X_IC_GMV, np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)

np.random.seed(42)
rand_idx_pcr = np.random.randint(0, 1000, 10)

print("PCR CV starts")
results_IC_GMV_control = Parallel(n_jobs=-1, verbose = 5)(
    delayed(CV_pcr)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=ElasticNet(max_iter = 1000000),
        X=X_IC_GMV_control,
        y=np.array(df_SCE_gmv_2000['waist_hip_ratio']),
        rand=i,
    ) for i in rand_idx_pcr
)

# Extract parameters
r2_train_IC_GMV = []
r2_test_IC_GMV = []
model_IC_GMV = []

# Extract results from each model in r_total_GT
for i in range(len(results_IC_GMV_control)):
    r2_train_IC_GMV.append(results_IC_GMV_control[i][0])
    r2_test_IC_GMV.append(results_IC_GMV_control[i][1])
    model_IC_GMV.append(results_IC_GMV_control[i][2])

# Plot CV results
labels = ['WHR~GMV', 'WHR~IC (PCR)', 'WHR~GT', 'WHR~GMV+IC (PCR)', 'WHR~GMV+GT']
dpi = 1600
title = 'Regression results of GMVIC/GT with control'

x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV), np.mean(r2_train_IC), np.mean(r2train_GT), np.mean(r2_train_IC_GMV), np.mean(r2train_GMV_GT)]
test_mean = [np.mean(r2test_GMV), np.mean(r2_test_IC), np.mean(r2test_GT), np.mean(r2_test_IC_GMV), np.mean(r2test_GMV_GT)]
train_std = [np.std(r2train_GMV), np.std(r2_train_IC), np.std(r2train_GT), np.std(r2_train_IC_GMV), np.std(r2train_GMV_GT)]
test_std = [np.std(r2test_GMV), np.std(r2_test_IC), np.std(r2test_GT), np.std(r2_test_IC_GMV), np.std(r2test_GMV_GT)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# ax.set_ylabel('MSE')
ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Male performance
df_SCE_gmv_2000_male = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]

# Loop over the cross-validation function for GMV
X_GMV_male = np.array(df_SCE_gmv_2000_male[gmv])
X_GMV_control_male = np.concatenate((X_GMV_male, np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV_male = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_control_male, y = stats.zscore(np.array(df_SCE_gmv_2000_male['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 139)
r2train_GMV_male, r2test_GMV_male, beta_GMV_male, model_GMV_male = zip(*results_GMV_male)

# Loop over the cross-validation function for GT
X_GT_male = stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index]))
X_GT_control_male = np.concatenate((X_GT_male, np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GT_male= loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_control_male, y = stats.zscore(np.array(df_SCE_gmv_2000_male['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 110)
r2train_GT_male, r2test_GT_male, beta_GT_male, model_GT_male = zip(*results_GT_male)

# Plot CV results
labels = ['WHR~GMV', 'WHR~GT']
dpi = 1600
title = 'Regression results of GMV/GT in male subjects with control'

x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_male), np.mean(r2train_GT_male)]
test_mean = [np.mean(r2test_GMV_male), np.mean(r2test_GT_male)]
train_std = [np.std(r2train_GMV_male), np.std(r2train_GT_male)]
test_std = [np.std(r2test_GMV_male), np.std(r2test_GT_male)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# ax.set_ylabel('MSE')
ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Female performance
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]

# Loop over the cross-validation function for GMV
X_GMV_female = np.array(df_SCE_gmv_2000_female[gmv])
X_GMV_control_female = np.concatenate((X_GMV_female, np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV_female = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_control_female, y = stats.zscore(np.array(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 139)
r2train_GMV_female, r2test_GMV_female, beta_GMV_female, model_GMV_female = zip(*results_GMV_female)

# Loop over the cross-validation function for GT
X_GT_female = stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index]))
X_GT_control_female = np.concatenate((X_GT_female, np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GT_female= loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_control_female, y = stats.zscore(np.array(df_SCE_gmv_2000_female['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 110)
r2train_GT_female, r2test_GT_female, beta_GT_female, model_GT_female = zip(*results_GT_female)

# Plot CV results
labels = ['WHR~GMV', 'WHR~GT']
dpi = 1600
title = 'Regression results of GMV/GT in female subjects with control'

x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_female), np.mean(r2train_GT_female)]
test_mean = [np.mean(r2test_GMV_female), np.mean(r2test_GT_female)]
train_std = [np.std(r2train_GMV_female), np.std(r2train_GT_female)]
test_std = [np.std(r2test_GMV_female), np.std(r2test_GT_female)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# ax.set_ylabel('MSE')
ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# High SCE performance
median = np.median(df_SCE_gmv_2000['slope'])
df_SCE_gmv_2000_highSCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] > median]

# Loop over the cross-validation function for GMV
X_GMV_highSCE = np.array(df_SCE_gmv_2000_highSCE[gmv])
X_GMV_control_highSCE = np.concatenate((X_GMV_highSCE, np.array(df_SCE_gmv_2000_highSCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV_highSCE = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_control_highSCE, y = stats.zscore(np.array(df_SCE_gmv_2000_highSCE['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 139)
r2train_GMV_highSCE, r2test_GMV_highSCE, beta_GMV_highSCE, model_GMV_highSCE = zip(*results_GMV_highSCE)

# Loop over the cross-validation function for GT
X_GT_highSCE = stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_highSCE.index]))
X_GT_control_highSCE = np.concatenate((X_GT_highSCE, np.array(df_SCE_gmv_2000_highSCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GT_highSCE= loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_control_highSCE, y = stats.zscore(np.array(df_SCE_gmv_2000_highSCE['waist_hip_ratio'])), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 110)
r2train_GT_highSCE, r2test_GT_highSCE, beta_GT_highSCE, model_GT_highSCE = zip(*results_GT_highSCE)

# Plot CV results
labels = ['WHR~GMV', 'WHR~GT']
dpi = 1600
title = 'Regression results of GMV/GT in high beta1 subjects with control'

x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_highSCE), np.mean(r2train_GT_highSCE)]
test_mean = [np.mean(r2test_GMV_highSCE), np.mean(r2test_GT_highSCE)]
train_std = [np.std(r2train_GMV_highSCE), np.std(r2train_GT_highSCE)]
test_std = [np.std(r2test_GMV_highSCE), np.std(r2test_GT_highSCE)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# ax.set_ylabel('MSE')
ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      


# Low SCE performance
median = np.median(df_SCE_gmv_2000['slope'])
df_SCE_gmv_2000_lowSCE = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['slope'] < median]

# Loop over the cross-validation function for GMV
X_GMV_lowSCE = np.array(df_SCE_gmv_2000_lowSCE[gmv])
X_GMV_control_lowSCE = np.concatenate((X_GMV_lowSCE, np.array(df_SCE_gmv_2000_lowSCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GMV_lowSCE = loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GMV_control_lowSCE, y = np.array(df_SCE_gmv_2000_lowSCE['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 139)
r2train_GMV_lowSCE, r2test_GMV_lowSCE, beta_GMV_lowSCE, model_GMV_lowSCE = zip(*results_GMV_lowSCE)

# Loop over the cross-validation function for GT
X_GT_lowSCE = stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_lowSCE.index]))
X_GT_control_lowSCE = np.concatenate((X_GT_lowSCE, np.array(df_SCE_gmv_2000_lowSCE[['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)
results_GT_lowSCE= loop_CV(p_grid = par_grid, out_fold = 10, in_fold = 10, model = ElasticNet(max_iter = 1000000), X = X_GT_control_lowSCE, y = np.array(df_SCE_gmv_2000_lowSCE['waist_hip_ratio']), rand_idx_length = 100, n_jobs = -1, verbose = 5, n_beta = 110)
r2train_GT_lowSCE, r2test_GT_lowSCE, beta_GT_lowSCE, model_GT_lowSCE = zip(*results_GT_lowSCE)

# Plot CV results
labels = ['WHR~GMV', 'WHR~GT']
dpi = 1600
title = 'Regression results of GMV/GT in low beta1 subjects with control'

x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_lowSCE), np.mean(r2train_GT_lowSCE)]
test_mean = [np.mean(r2test_GMV_lowSCE), np.mean(r2test_GT_lowSCE)]
train_std = [np.std(r2train_GMV_lowSCE), np.std(r2train_GT_lowSCE)]
test_std = [np.std(r2test_GMV_lowSCE), np.std(r2test_GT_lowSCE)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

# ax.set_ylabel('MSE')
ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()    