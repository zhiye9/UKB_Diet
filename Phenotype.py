#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 22:02:12 2021

@author: zhye
"""

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
from scipy.stats import zscore

df_income = df.filter(regex = '738|eid').drop(columns = ['20738-0.0'])
df_income_noNeg = df_income.mask(df_income < 0).dropna(subset = df_income.columns[1:], how = 'all')
df_income_noNeg.reset_index(drop = True, inplace = True)
df_income_noNeg['eid'] = df_income_noNeg['eid'].astype(str)
df_income_noNeg_SCE = pd.merge(df_SCE, df_income_noNeg, on = 'eid')
df_income_noNeg_SCE['income'] = 0
for i in range(df_income_noNeg_SCE.shape[0]):
    df_income_noNeg_SCE['income'].loc[i] = np.nanmean(df_income_noNeg_SCE[df_income_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_income_noNeg_SCE.shape[0])), end="")

df_income_noNeg_SCE = df_income_noNeg_SCE.drop(columns = df_income.columns[1:])

df_edu = df.filter(regex = '6138|eid')
df_edu_noNeg = df_edu.mask(df_edu == -3).dropna(subset = df_edu.columns[1:],how = 'all')
df_edu_noNeg.reset_index(drop = True, inplace = True)
df_edu_noNeg['eid'] = df_edu_noNeg['eid'].astype(str)
df_edu_noNeg_SCE = pd.merge(df_income_noNeg_SCE, df_edu_noNeg, on = 'eid')
df_edu_noNeg_SCE['edu'] = 9999
df_edu_noNeg_SCE[df_edu_noNeg.columns[1:]] = df_edu_noNeg_SCE[df_edu_noNeg.columns[1:]].replace(-7, 7)
for i in range(df_edu_noNeg_SCE.shape[0]):      
    df_edu_noNeg_SCE['edu'].loc[i] = np.nanmin(df_edu_noNeg_SCE[df_edu_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_edu_noNeg_SCE.shape[0])), end="")
df_edu_noNeg_SCE = df_edu_noNeg_SCE.drop(columns = df_edu.columns[1:])

df_height_noNAN = df.filter(regex = '12144|eid')
df_height['eid'] = df_height['eid'].astype(str)
#df_height_noNeg = df_height.mask(df_height[df_height.columns[1:]] < 0).dropna(subset = df_height.columns[1:],how = 'all')
df_height_noNAN = df_height.dropna(subset = df_height.columns[1:],how = 'all')
df_height_noNAN.reset_index(drop = True, inplace = True)
df_height_noNAN_SCE = pd.merge(df_edu_noNeg_SCE, df_height_noNAN, on = 'eid')
df_height_noNAN_SCE['height'] = 0
for i in range(df_height_noNAN_SCE.shape[0]):
    df_height_noNAN_SCE['height'].loc[i] = np.nanmean(df_height_noNAN_SCE[df_height_noNAN.columns[1:3]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_height_noNAN_SCE.shape[0])), end="")

df_height_noNAN_SCE = df_height_noNAN_SCE.drop(columns = df_height.columns[1:])

df_check = df_height_noNAN_SCE[['SCE', 'energy', 'BMI',  'age', 'income', 'edu', 'height']]
df_check_zscore = df_check.apply(zscore)
df_check_zscore[['eid', 'sex' ]] = df_height_noNAN_SCE[['eid', 'sex' ]]
SCE_pheno = ['BMI',  'age', 'income', 'edu', 'height', 'sex']
energy_pheno = ['BMI',  'age', 'income', 'edu', 'height', 'sex']

SCE_check_train_E, SCE_check_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = np.array(df_check_zscore[SCE_pheno]), y = df_check_zscore['SCE'])
SCE_check_train_SVR , SCE_check_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = np.array(df_check_zscore[SCE_pheno]), y = df_check_zscore['SCE'])
#SCE_check_train_SVRrbf , SCE_check_test_SVRrbf = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'rbf'), X = np.array(df_check_zscore[SCE_pheno]), y = df_check_zscore['SCE'])

energy_check_train_E, energy_check_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(max_iter = 1000000), X = np.array(df_check_zscore[energy_pheno]), y = df_check_zscore['SCE'])
energy_check_train_SVR , energy_check_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = np.array(df_check_zscore[energy_pheno]), y = df_check_zscore['SCE'])
#energy_check_train_SVRrbf , energy_check_test_SVRrbf = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'rbf'), X = np.array(df_check_zscore[energy_pheno]), y = df_check_zscore['SCE'])


labels = ['SCE_Elastic', 'SCE_SVR', 'energy_Elsatic', 'energy_SVR']

x = np.arange(len(labels)) 
train_mean = [np.mean(SCE_check_train_E), np.mean(SCE_check_train_SVR), np.mean(energy_check_train_E), np.mean(energy_check_train_SVR)]
test_mean = [np.mean(SCE_check_test_E), np.mean(SCE_check_test_SVR), np.mean(energy_check_test_E), np.mean(energy_check_test_SVR)]
train_std = [np.std(SCE_check_train_E), np.std(SCE_check_train_SVR), np.std(energy_check_train_E), np.std(energy_check_train_SVR)]
test_std = [np.std(SCE_check_test_E), np.std(SCE_check_test_SVR), np.std(energy_check_test_E), np.std(energy_check_test_SVR)]
    
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
ax.set_title('Regression results of SCE energy')
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

from scipy.linalg import eigh, cholesky
from scipy.stats import norm, pearsonr
from scipy.stats import multivariate_normal as mvn

from pylab import plot, show, axis, subplot, xlabel, ylabel, grid




# Choice of cholesky or eigenvector method.
method = 'cholesky'
#method = 'eigenvectors'

#num_samples = 400

# The desired covariance matrix.
r = np.array([[1,0.3], [0.3,1]])
#r = 0.3
# Generate samples from three independent normally distributed random
# variables (with mean 0 and std. dev. 1).
xt = np.hstack((np.array(df_SCE_gmv_2000[['SCE']]), norm.rvs(size=(2000, 1))))
x = xt.T

if method == 'cholesky':
    # Compute the Cholesky decomposition.
    c = cholesky(r, lower=True)
else:
    # Compute the eigenvalues and eigenvectors.
    evals, evecs = eigh(r)
    # Construct c, so c*c^T = r.
    c = np.dot(evecs, np.diag(np.sqrt(evals)))

# Convert the data to correlated random variables. 
y = np.dot(c, x)

plot(x[0], y[1], 'b.')
ylabel('Sythetic data')
xlabel('energy')
grid(True)
print(np.corrcoef(x[0], y[1]))
pearsonr(x[0], y[1])

# We need a matrix `c` for which `c*c^T = r`.  We can use, for example,
# the Cholesky decomposition, or the we can construct `c` from the
# eigenvectors and eigenvalues.
def correlation3(r, x):
    if method == 'cholesky':
        # Compute the Cholesky decomposition.
        c = cholesky(r, lower=True)
    else:
        # Compute the eigenvalues and eigenvectors.
        evals, evecs = eigh(r)
        # Construct c, so c*c^T = r.
        c = np.dot(evecs, np.diag(np.sqrt(evals)))
    
    # Convert the data to correlated random variables. 
    y = np.dot(c, x)
    return y

energyt = np.hstack((np.array(df_SCE_gmv_2000[['energy']]), norm.rvs(size=(2000, 1))))
energyx = energyt.T
energy_y = correlation3(r = r, x = energyx)
plot(energyx[0], energy_y[1], 'b.')
ylabel('Sythetic data')
xlabel('energy')
grid(True)
np.corrcoef(energyx[0], energy_y[1])
def syth(x, y):
    df_sytht = np.stack((x[0], y[1]))
    df_syth = df_sytht.T
    df_syth_all = np.hstack((df_syth, norm.rvs(size=(2000, 5))))
    
    df_syth_all_zscore = zscore(df_syth_all, axis = 0)
    return df_syth_all_zscore

energy_df_syth_all_zscore = syth(x = energyx, y = energy_y)
energy_syth_train_E, energy_syth_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = energy_df_syth_all_zscore[:, 1:], y = energy_df_syth_all_zscore[:, 0])
energy_syth_train_SVR , energy_syth_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = energy_df_syth_all_zscore[:, 1:], y = energy_df_syth_all_zscore[:, 0])

df_sytht = np.stack((x[0], y[1]))
df_syth = df_sytht.T
df_syth_all = np.hstack((df_syth, norm.rvs(size=(2000, 5))))

df_syth_all_zscore = zscore(df_syth_all, axis = 0)

SCE_syth_train_E, SCE_syth_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = df_syth_all_zscore[:, 1:], y = df_syth_all_zscore[:, 0])
SCE_syth_train_SVR , SCE_syth_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = df_syth_all_zscore[:, 1:], y = df_syth_all_zscore[:, 0])

np.corrcoef(df_syth_all_zscore[:, 1], np.array(df_syth_all_zscore[:, 0]))

from sklearn.linear_model import LinearRegression

m = LinearRegression().fit(df_syth_all_zscore[:, 1:], np.array(df_syth_all_zscore[:, 0]).T)
ped = m.predict(df_syth_all_zscore[:, 1:])
ped
r2_score(df_syth_all_zscore[:, 0], ped)

plot(x[0], y[1], 'b.')
ylabel('Sythetic data')
xlabel('SCE')
grid(True)
np.corrcoef(x[0], y)

df_sytht = np.stack((x[0], y[1]))
df_syth = df_sytht.T
df_syth_all = np.hstack((df_syth, norm.rvs(size=(2000, 5))))

df_syth_all_zscore = zscore(df_syth_all, axis = 0)

SCE_syth_train_E, SCE_syth_test_E = CV(p_grid = par_grid, out_fold = 5, in_fold = 5, model = ElasticNet(), X = df_syth_all_zscore[:, 1:], y = df_syth_all_zscore[:, 0])
SCE_syth_train_SVR , SCE_syth_test_SVR = CV(p_grid = p_gridsvr, out_fold = 5, in_fold = 5, model = SVR(kernel = 'poly'), X = df_syth_all_zscore[:, 1:], y = df_syth_all_zscore[:, 0])

labels = ['SCE_Elastic', 'SCE_SVR', 'energy_Elsatic', 'energy_SVR']

x = np.arange(len(labels)) 
train_mean = [np.mean(SCE_syth_train_E), np.mean(SCE_syth_train_SVR), np.mean(energy_syth_train_E), np.mean(energy_syth_train_SVR)]
test_mean = [np.mean(SCE_syth_test_E), np.mean(SCE_syth_test_SVR), np.mean(energy_syth_test_E), np.mean(energy_syth_test_SVR)]
train_std = [np.std(SCE_syth_train_E), np.std(SCE_syth_train_SVR), np.std(energy_syth_train_E), np.std(energy_syth_train_SVR)]
test_std = [np.std(SCE_syth_test_E), np.std(SCE_syth_test_SVR), np.std(energy_syth_test_E), np.std(energy_syth_test_SVR)]
    
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
ax.set_title('Regression results of synthetic SCE energy')
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

# Plot various projections of the samples.
#
subplot(2,2,1)
plot(y[0], y[1], 'b.')
ylabel('y[1]')
axis('equal')
grid(True)

subplot(2,2,3)
plot(y[0], y[2], 'b.')
xlabel('y[0]')
ylabel('y[2]')
axis('equal')
grid(True)

subplot(2,2,4)
plot(y[1], y[2], 'b.')
xlabel('y[1]')
axis('equal')
grid(True)
show()