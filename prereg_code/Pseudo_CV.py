import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics._regression import _check_reg_targets
from scipy import stats
import networkx as nx
import pickle
import os
from numpy import loadtxt
from joblib import Parallel, delayed
from UKB_graph_metrics import *

# Load data
os.chdir('/home/ubuntu/UK_Biobank_diet')
df_SCE_gmv_2000 = pd.read_csv('df_SCE_gmv_2000.csv')
df_SCE_gmv_2000['eid'] = df_SCE_gmv_2000['eid'].astype(str)

# Load the control and GMV id
control = ['sex', 'age', 'height', 'hand', 'sexage', 'edu', 'IQ', 'income', 'household']
gmv = np.genfromtxt('GMV.txt', dtype='str')

# Function to perform Pseudo-CV
def Pseudo_CV(p_grid, out_fold, in_fold, model, X_1, X_2, y_1, y_2, rand, score):
    """
    Perform Pseudo-CV to train on a subgroup X_1 and test on another subgroup X_2.
    
    Parameters
    ----------
    p_grid : dict
        The grid of hyperparameters to search

    out_fold : int
        The number of outer folds

    in_fold : int
        The number of inner folds

    model : object
        The model to be used

    X_1 : array 
        The feature matrix of the first group

    X_2 : array
        The feature matrix of the second group

    y_1 : array
        The target of the first group

    y_2 : array
        The target of the second group

    rand : int
        The random seed

    score : str
        The scoring method
    
    Returns
    -------
    r2train : list
        The r2 score of the training set
    r2test : list
        The r2 score of the test set 
    """
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2test = []
    if len(y_1) <= len(y_2):
        X_CV = X_1
        y_CV = y_1
    else:
        X_CV = X_2 
        y_CV = y_2

    for j, (train, test) in enumerate(outer_cv.split(X_CV, y_CV)):
        #split dataset to decoding set and test set
        x_train_1, x_test_1 = X_1[train], X_1[test]
        y_train_1, y_test_1 = y_1[train], y_1[test]
        x_train_2, x_test_2 = X_2[train], X_2[test]
        y_train_2, y_test_2 = y_2[train], y_2[test]
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = score)
        clf.fit(x_train_1, y_train_1)

        #predict labels on the train set
        y_pred = clf.predict(x_train_1)
        if (score == 'r2'):
            r2train.append(r2_score(y_train_1, y_pred))
        elif (score == 'neg_mean_squared_error'):
            r2train.append(mean_squared_error(y_train_1, y_pred))
        else:
            ValueError('Invalid score, must be r2 or neg_mean_squared_error')
         
        #predict labels on the test set
        y_pred = clf.predict(x_test_2)
        if (score == 'r2'):
            r2test.append(r2_score(y_test_2, y_pred))
        elif (score == 'neg_mean_squared_error'):
            r2test.append(mean_squared_error(y_test_2, y_pred))
        else:
            ValueError('Invalid score, must be r2 or neg_mean_squared_error')
   
    return r2train, r2test

#Select female
df_SCE_gmv_2000_female = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 0]
#Select male
df_SCE_gmv_2000_male = df_SCE_gmv_2000.loc[df_SCE_gmv_2000['sex'] == 1]

# Set parameters
par_grid = {"alpha": np.logspace(-2, -1, 5), "l1_ratio": [.7, .9]}
model = ElasticNet(max_iter = 1000000)
np.random.seed(42)
rand_id_psd_CV = np.random.randint(0, 1000, 100)

# Loop over the Pseudo-CV for the GMV
X_GMV_m = np.concatenate((np.array(df_SCE_gmv_2000_male[gmv]), 
                           np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
X_GMV_f = np.concatenate((np.array(df_SCE_gmv_2000_female[gmv]),
                            np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']])), axis = 1)
y_m = np.array(stats.zscore(df_SCE_gmv_2000_male['waist_hip_ratio']))
y_f = np.array(stats.zscore(df_SCE_gmv_2000_female['waist_hip_ratio']))

# Train on male and test on female
results_GMV_m2f = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GMV_m, 
        X_2 = X_GMV_f, 
        y_1 = y_m,
        y_2 = y_f, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GMV_m2f, r2test_GMV_m2f = zip(*results_GMV_m2f) 

# Train on female and test on male
results_GMV_f2m = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GMV_f, 
        X_2 = X_GMV_m, 
        y_1 = y_f,
        y_2 = y_m, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GMV_f2m, r2test_GMV_f2m = zip(*results_GMV_f2m) 

# Loop over the Pseudo-CV for the GT
X_GT_m = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), 
                             np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household'] ])), axis = 1)
X_GT_f = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), 
                             np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand', 'edu', 'IQ', 'income', 'household']] )), axis = 1)

# Train on male and test on female
results_GT_m2f = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GT_m, 
        X_2 = X_GT_f, 
        y_1 = y_m,
        y_2 = y_f, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_m2f, r2test_GT_m2f = zip(*results_GT_m2f)

# Train on female and test on male
results_GT_f2m = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GT_f, 
        X_2 = X_GT_m, 
        y_1 = y_f,
        y_2 = y_m, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_f2m, r2test_GT_f2m = zip(*results_GT_f2m)

# Loop over the Pseudo-CV for the GT and GMV
X_GT_GMV_m = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_male.index])), 
                                np.array(df_SCE_gmv_2000_male[gmv]), 
                                np.array(df_SCE_gmv_2000_male[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)
X_GT_GMV_f = np.concatenate((stats.zscore(np.array([IC_Graph[i] for i in df_SCE_gmv_2000_female.index])), 
                                np.array(df_SCE_gmv_2000_female[gmv]),                                   
                                np.array(df_SCE_gmv_2000_female[['age', 'height', 'hand','edu', 'IQ', 'income', 'household']])), axis = 1)

# Train on male and test on female
results_GT_GMV_m2f = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GT_GMV_m, 
        X_2 = X_GT_GMV_f, 
        y_1 = y_m,
        y_2 = y_f, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_GMV_m2f, r2test_GT_GMV_m2f = zip(*results_GT_GMV_m2f)

# Train on female and test on male
results_GT_GMV_f2m = Parallel(n_jobs=-1, verbose = 5)(
    delayed(Pseudo_CV)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X_1 = X_GT_GMV_f, 
        X_2 = X_GT_GMV_m, 
        y_1 = y_f,
        y_2 = y_m, 
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_GMV_f2m, r2test_GT_GMV_f2m = zip(*results_GT_GMV_f2m)

# Plot pseudo CV results train on male and test on female
labels = ['WHR~GMV', 'WHR~GT', 'WHR~GMV+GT']
dpi = 1600
title = 'Regression results of GMV/GT using pseudo CV train on male and test on female with control'
 
x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_m2f), np.mean(r2train_GT_m2f), np.mean(r2train_GT_GMV_m2f)]
test_mean = [np.mean(r2test_GMV_m2f), np.mean(r2test_GT_m2f), np.mean(r2test_GT_GMV_m2f)]
train_std = [np.std(r2train_GMV_m2f), np.std(r2train_GT_m2f), np.std(r2train_GT_GMV_m2f)]
test_std = [np.std(r2test_GMV_m2f), np.std(r2test_GT_m2f), np.std(r2test_GT_GMV_m2f)]

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
# plt.yticks(np.arange(0, 79, 10))  
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Plot pseudo CV results train on female and test on male
labels = ['WHR~GMV', 'WHR~GT', 'WHR~GMV+GT']
dpi = 1600
title = 'Regression results of GMV/GT using pseudo CV train on female and test on male with control'
 
x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_f2m), np.mean(r2train_GT_f2m), np.mean(r2train_GT_GMV_f2m)]
test_mean = [np.mean(r2test_GMV_f2m), np.mean(r2test_GT_f2m), np.mean(r2test_GT_GMV_f2m)]
train_std = [np.std(r2train_GMV_f2m), np.std(r2train_GT_f2m), np.std(r2train_GT_GMV_f2m)]
test_std = [np.std(r2test_GMV_f2m), np.std(r2test_GT_f2m), np.std(r2test_GT_GMV_f2m)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Psuedo CV train on both genders and test on one
def mean_error(y_true, y_pred, sample_weight=None, multioutput="uniform_average"):
    """
    Compute the mean error.
    
    Parameters
    ----------
    y_true : array
        The true target values

    y_pred : array
        The predicted target values

    sample_weight : array
        The sample weights

    multioutput : str
        The method to average the error

    Returns
    -------
    error : float
        The mean error  
    """
    y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    return np.average(y_true - y_pred, weights=sample_weight, axis=0)

def mean_error_score():
    """
    Score function for the mean error.

    Returns
    -------
    function
        The mean error score function 
    """
    return make_scorer(mean_error, greater_is_better=False)

def TrainBoth_TestOne(p_grid, out_fold, in_fold, model, X, y, gender, rand, score):
    """
    Function to train on full dataset and test on each gender subgroup
    
    Parameters
    ----------
    p_grid : dict
        The grid of hyperparameters to search
    
    out_fold : int
        The number of outer folds

    in_fold : int
        The number of inner folds

    model : object
        The model to be used

    X : array
        The feature matrix

    y : array   
        The target

    gender : dataframe
        The gender column

    rand : int
        The random seed

    score : str
        The scoring method

    Returns
    -------
    r2train : list
        The r2 score of the training set

    r2train1 : list
        The r2 score of the male training set

    r2train0 : list
        The r2 score of the female training set

    r2test : list
        The r2 score of the test set

    r2test1 : list
        The r2 score of the male test set

    r2test0 : list
        The r2 score of the female test set
    """
    outer_cv = KFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = KFold(n_splits = in_fold, shuffle = True, random_state = rand)
    r2train = []
    r2train1 = []
    r2train0 = []
    r2test = []
    r2test1 = []
    r2test0 = []
    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        gender_train = np.array(gender)[train]
        gender_test = np.array(gender)[test]
        if score == 'mean_error':
            clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, 
                                scoring = make_scorer(mean_error, greater_is_better=False))
        elif score == 'MSE':
            clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, 
                                scoring = "neg_mean_squared_error")
        elif score == 'r2':
            clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, 
                                scoring = "r2")
        else:
            ValueError('Invalid score, must be mean_error, r2 or neg_mean_squared_error')
        clf.fit(x_train, y_train)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        y_pred1 = clf.predict(x_train[gender_train == 1])
        y_pred0 = clf.predict(x_train[gender_train == 0])
        if score == 'mean_error':
            r2train.append(mean_error(y_train, y_pred))
            r2train1.append(mean_error(y_train[gender_train == 1], y_pred1))
            r2train0.append(mean_error(y_train[gender_train == 0], y_pred0))
        elif score == 'MSE':
            r2train.append(mean_squared_error(y_train, y_pred))
            r2train1.append(mean_squared_error(y_train[gender_train == 1], y_pred1))
            r2train0.append(mean_squared_error(y_train[gender_train == 0], y_pred0))
        elif score == 'r2':
            r2train.append(r2_score(y_train, y_pred))
            r2train1.append(r2_score(y_train[gender_train == 1], y_pred1))
            r2train0.append(r2_score(y_train[gender_train == 0], y_pred0))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        y_pred1 = clf.predict(x_test[gender_test == 1])
        y_pred0 = clf.predict(x_test[gender_test == 0])
        if score == 'mean_error':
            r2test.append(mean_error(y_test, y_pred))
            r2test1.append(mean_error(y_test[gender_test == 1], y_pred1))
            r2test0.append(mean_error(y_test[gender_test == 0], y_pred0))
        elif score == 'MSE':
            r2test.append(mean_squared_error(y_test, y_pred))
            r2test1.append(mean_squared_error(y_test[gender_test == 1], y_pred1))
            r2test0.append(mean_squared_error(y_test[gender_test == 0], y_pred0))
        elif score == 'r2':
            r2test.append(r2_score(y_test, y_pred))
            r2test1.append(r2_score(y_test[gender_test == 1], y_pred1))
            r2test0.append(r2_score(y_test[gender_test == 0], y_pred0))
 
    return r2train, r2train1, r2train0, r2test, r2test1, r2test0

gender = df_SCE_gmv_2000['sex']
y = np.array(df_SCE_gmv_2000['waist_hip_ratio'])

# Loop over the TrainBoth Pseudo-CV for the GMV
X_GMV_TrainBoth = np.concatenate((np.array(df_SCE_gmv_2000[gmv]), 
                                  np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 
                                 'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)

# Train with r2
results_GMV_both = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GMV_both, r2train_GMV_both1, r2train_GMV_both0, \
    r2test_GMV_both, r2test_GMV_both1, r2test_GMV_both0 = zip(*results_GMV_both)

# Train with MSE
results_GMV_both_MSE = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "MSE",
        rand=i,
    ) for i in rand_id_psd_CV
)

MSEtrain_GMV_both, MSEtrain_GMV_both1, MSEtrain_GMV_both0, \
    MSEtest_GMV_both, MSEtest_GMV_both1, MSEtest_GMV_both0 = zip(*results_GMV_both_MSE)

# Train with mean_error
results_GMV_both_mean_error = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "mean_error",
        rand=i,
    ) for i in rand_id_psd_CV
)

mean_errortrain_GMV_both, mean_errortrain_GMV_both1, mean_errortrain_GMV_both0, \
    mean_errortest_GMV_both, mean_errortest_GMV_both1, mean_errortest_GMV_both0 = zip(*results_GMV_both_mean_error)

#Loop over the TrainBoth Pseudo-CV for the GT
X_GT_TrainBoth = np.concatenate((stats.zscore(np.array(IC_Graph)), 
                                 np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 
                                'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)

# Train with r2
results_GT_both_r2 = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_TrainBoth, 
        y = y, 
        gender = gender,
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_both, r2train_GT_both1, r2train_GT_both0, \
    r2test_GT_both, r2test_GT_both1, r2test_GT_both0 = zip(*results_GT_both_r2)

# Train with MSE
results_GT_both_MSE = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_TrainBoth, 
        y = y, 
        gender = gender,
        score = "MSE",
        rand=i,
    ) for i in rand_id_psd_CV
)

MSEtrain_GT_both, MSEtrain_GT_both1, MSEtrain_GT_both0, \
    MSEtest_GT_both, MSEtest_GT_both1, MSEtest_GT_both0 = zip(*results_GT_both_MSE)

# Train with mean_error
results_GT_both_mean_error = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_TrainBoth, 
        y = y, 
        gender = gender,
        score = "mean_error",
        rand=i,
    ) for i in rand_id_psd_CV
)

mean_errortrain_GT_both, mean_errortrain_GT_both1, mean_errortrain_GT_both0, \
    mean_errortest_GT_both, mean_errortest_GT_both1, mean_errortest_GT_both0 = zip(*results_GT_both_mean_error)

#Loop over the TrainBoth Pseudo-CV for the GT and GMV
X_GT_GMV_TrainBoth = np.concatenate((stats.zscore(np.array(IC_Graph)), np.array(df_SCE_gmv_2000[gmv]), 
                               np.array(df_SCE_gmv_2000[['sex', 'age', 'height', 'hand', 
                                'sexage', 'edu', 'IQ', 'income', 'household']])), axis = 1)

# Train with r2
results_GT_GMV_both = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "r2",
        rand=i,
    ) for i in rand_id_psd_CV
)

r2train_GT_GMV_both, r2train_GT_GMV_both1, r2train_GT_GMV_both0, \
    r2test_GT_GMV_both, r2test_GT_GMV_both1, r2test_GT_GMV_both0 = zip(*results_GT_GMV_both)

# Train with MSE
results_GT_GMV_both_MSE = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "MSE",
        rand=i,
    ) for i in rand_id_psd_CV
)

MSEtrain_GT_GMV_both, MSEtrain_GT_GMV_both1, MSEtrain_GT_GMV_both0, \
    MSEtest_GT_GMV_both, MSEtest_GT_GMV_both1, MSEtest_GT_GMV_both0 = zip(*results_GT_GMV_both_MSE)

# Train with mean_error
results_GT_GMV_both_mean_error = Parallel(n_jobs=-1, verbose = 5)(
    delayed(TrainBoth_TestOne)(
        p_grid=par_grid,
        out_fold=10,
        in_fold=10,
        model=model,
        X = X_GT_GMV_TrainBoth, 
        y = y, 
        gender = gender,
        score = "mean_error",
        rand=i,
    ) for i in rand_id_psd_CV
)

mean_errortrain_GT_GMV_both, mean_errortrain_GT_GMV_both1, mean_errortrain_GT_GMV_both0, \
    mean_errortest_GT_GMV_both, mean_errortest_GT_GMV_both1, mean_errortest_GT_GMV_both0 = zip(*results_GT_GMV_both_mean_error)

# Plot pseudo CV results train on both and test on one with r2
labels = ['WHR~GMV', 'WHR~GMV (male)', 'WHR~GMV (female)', 
          'WHR~GT', 'WHR~GT (male)', 'WHR~GT (female)',  
          'WHR~GMV+GT', 'WHR~GMV+GT (male)', 'WHR~GMV+GT (female)', ]
dpi = 1600
title = 'Regression results of GMV/GT train on both gender and test on one with control'
 
x = np.arange(len(labels))
train_mean = [np.mean(r2train_GMV_both), np.mean(r2train_GMV_both1), np.mean(r2train_GMV_both0), np.mean(r2train_GT_both), np.mean(r2train_GT_both1), np.mean(r2train_GT_both0), np.mean(r2train_GT_GMV_both), np.mean(r2train_GT_GMV_both1), np.mean(r2train_GT_GMV_both0)]
test_mean = [np.mean(r2test_GMV_both), np.mean(r2test_GMV_both1), np.mean(r2test_GMV_both0), np.mean(r2test_GT_both), np.mean(r2test_GT_both1), np.mean(r2test_GT_both0), np.mean(r2test_GT_GMV_both), np.mean(r2test_GT_GMV_both1), np.mean(r2test_GT_GMV_both0)]
train_std = [np.std(r2train_GMV_both), np.std(r2train_GMV_both1), np.std(r2train_GMV_both0), np.std(r2train_GT_both), np.std(r2train_GT_both1), np.std(r2train_GT_both0), np.std(r2train_GT_GMV_both), np.std(r2train_GT_GMV_both1), np.std(r2train_GT_GMV_both0)]
test_std = [np.std(r2test_GMV_both), np.std(r2test_GMV_both1), np.std(r2test_GMV_both0), np.std(r2test_GT_both), np.std(r2test_GT_both1), np.std(r2test_GT_both0), np.std(r2test_GT_GMV_both), np.std(r2test_GT_GMV_both1), np.std(r2test_GT_GMV_both0)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4)*100 for i in train_mean], width, yerr = [i*100 for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i , 4)*100 for i in test_mean], width, yerr = [i*100 for i in test_std], label='test', align='center', ecolor='black', capsize=2)

ax.set_ylabel('R2 %')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Plot pseudo CV results train on both and test on one with MSE
labels = ['WHR~GMV', 'WHR~GMV (male)', 'WHR~GMV (female)', 
          'WHR~GT', 'WHR~GT (male)', 'WHR~GT (female)',  
          'WHR~GMV+GT', 'WHR~GMV+GT (male)', 'WHR~GMV+GT (female)', ]
dpi = 1600
title = 'Regression results of GMV/GT train on both gender and test on one with control'
 
x = np.arange(len(labels))
train_mean = [np.mean(MSEtrain_GMV_both), np.mean(MSEtrain_GMV_both1), np.mean(MSEtrain_GMV_both0), np.mean(MSEtrain_GT_both), np.mean(MSEtrain_GT_both1), np.mean(MSEtrain_GT_both0), np.mean(MSEtrain_GT_GMV_both), np.mean(MSEtrain_GT_GMV_both1), np.mean(MSEtrain_GT_GMV_both0)]
test_mean = [np.mean(MSEtest_GMV_both), np.mean(MSEtest_GMV_both1), np.mean(MSEtest_GMV_both0), np.mean(MSEtest_GT_both), np.mean(MSEtest_GT_both1), np.mean(MSEtest_GT_both0), np.mean(MSEtest_GT_GMV_both), np.mean(MSEtest_GT_GMV_both1), np.mean(MSEtest_GT_GMV_both0)]
train_std = [np.std(MSEtrain_GMV_both), np.std(MSEtrain_GMV_both1), np.std(MSEtrain_GMV_both0), np.std(MSEtrain_GT_both), np.std(MSEtrain_GT_both1), np.std(MSEtrain_GT_both0), np.std(MSEtrain_GT_GMV_both), np.std(MSEtrain_GT_GMV_both1), np.std(MSEtrain_GT_GMV_both0)]
test_std = [np.std(MSEtest_GMV_both), np.std(MSEtest_GMV_both1), np.std(MSEtest_GMV_both0), np.std(MSEtest_GT_both), np.std(MSEtest_GT_both1), np.std(MSEtest_GT_both0), np.std(MSEtest_GT_GMV_both), np.std(MSEtest_GT_GMV_both1), np.std(MSEtest_GT_GMV_both0)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 2) for i in train_mean], width, yerr = [i for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i, 2) for i in test_mean], width, yerr = [i for i in test_std], label='test', align='center', ecolor='black', capsize=2)

ax.set_ylabel('MSE')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.ylim(0, 1)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      

# Plot pseudo CV results train on both and test on one with mean_error
labels = ['WHR~GMV', 'WHR~GMV (male)', 'WHR~GMV (female)', 
          'WHR~GT', 'WHR~GT (male)', 'WHR~GT (female)',  
          'WHR~GMV+GT', 'WHR~GMV+GT (male)', 'WHR~GMV+GT (female)', ]
dpi = 1600
title = 'Regression results of GMV/GT train on both gender and test on one with control'
 
x = np.arange(len(labels))
train_mean = [np.mean(mean_errortrain_GMV_both), np.mean(mean_errortrain_GMV_both1), np.mean(mean_errortrain_GMV_both0), np.mean(mean_errortrain_GT_both), np.mean(mean_errortrain_GT_both1), np.mean(mean_errortrain_GT_both0), np.mean(mean_errortrain_GT_GMV_both), np.mean(mean_errortrain_GT_GMV_both1), np.mean(mean_errortrain_GT_GMV_both0)]
test_mean = [np.mean(mean_errortest_GMV_both), np.mean(mean_errortest_GMV_both1), np.mean(mean_errortest_GMV_both0), np.mean(mean_errortest_GT_both), np.mean(mean_errortest_GT_both1), np.mean(mean_errortest_GT_both0), np.mean(mean_errortest_GT_GMV_both), np.mean(mean_errortest_GT_GMV_both1), np.mean(mean_errortest_GT_GMV_both0)]
train_std = [np.std(mean_errortrain_GMV_both), np.std(mean_errortrain_GMV_both1), np.std(mean_errortrain_GMV_both0), np.std(mean_errortrain_GT_both), np.std(mean_errortrain_GT_both1), np.std(mean_errortrain_GT_both0), np.std(mean_errortrain_GT_GMV_both), np.std(mean_errortrain_GT_GMV_both1), np.std(mean_errortrain_GT_GMV_both0)]
test_std = [np.std(mean_errortest_GMV_both), np.std(mean_errortest_GMV_both1), np.std(mean_errortest_GMV_both0), np.std(mean_errortest_GT_both), np.std(mean_errortest_GT_both1), np.std(mean_errortest_GT_both0), np.std(mean_errortest_GT_GMV_both), np.std(mean_errortest_GT_GMV_both1), np.std(mean_errortest_GT_GMV_both0)]

fig, ax = plt.subplots(dpi = dpi)
width = 0.4
rects1 = ax.bar(x - width/2, [round(i, 4) for i in train_mean], width, yerr = [i for i in train_std], label='train', align='center', ecolor='black', capsize=2)
rects2 = ax.bar(x + width/2, [round(i, 4) for i in test_mean], width, yerr = [i for i in test_std], label='test', align='center', ecolor='black', capsize=2)

ax.set_ylabel('Mean Error')
ax.set_title(title)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1.22, 1))
plt.xticks(rotation=60)
ax.set_xticklabels(labels, fontsize = 9)
ax.bar_label(rects1, padding=2, fontsize = 5)
ax.bar_label(rects2, padding=2, fontsize = 5)
fig.tight_layout()
plt.show()      