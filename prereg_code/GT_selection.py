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
import nibabel as nib
from nilearn.image import get_data, new_img_like
import nilearn.plotting as plotting
from UKB_graph_metrics import *
from utils import compute_FDD, Get_selected_features

## Select the features

# Load the CV results
with open("r_total_randomper_GT", "rb") as fp:
            r_total_GT = pickle.load(fp)

n_beta = 110
rand_idx_length = 100
CV_folds = 10
method = 'subtract'

feature_counts, all_beta_df = Get_selected_features(r_total_GT, n_beta, rand_idx_length, CV_folds, method)

# Get max frequency of artificial features
max_artificial_freq = np.max(feature_counts[n_beta:])

# Get real features above threshold
GT_real_features = feature_counts[:n_beta]
GT_features_df = pd.DataFrame(GT_real_features, columns = ['Selection_Frequency'])

#Read 55 ICA good compoents, start from 1
ICA_good_100 = loadtxt("rfMRI_GoodComponents_d100_v1.txt", dtype = int, unpack=False).tolist()
GT_features_df['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]

# Get frequencies of selected features
GT_selected_features_df = GT_features_df[GT_features_df['Selection_Frequency'] > max_artificial_freq]

# Sort by frequency
GT_selected_features_df_id_sorted = GT_selected_features_df.sort_values('Selection_Frequency', ascending=False).reset_index(drop=True)
GT_selected_features_df['IC_number'] = GT_selected_features_df['ICs'].str[7:].astype(int)
GT_selected_featuresid_sorted = GT_selected_features_df.sort_values('IC_number').reset_index(drop=True)

## Plot the selected IC regions

#Extract IC with color
def extract_IC_color(file, n, colo):
    IC = nib.load(file)
    data = get_data(IC)
    data[data != n] = 0
    data[data == n] = colo
    new_img = new_img_like(IC, data)
    return new_img

#Find postive and negative correlation ICs separately
all_beta_df_real = all_beta_df.iloc[:110, :]
all_beta_df_real['ICs'] = [f'IC_pos {ICA_good_100[i]}' for i in range(55)] + [f'IC_neg {ICA_good_100[i]}' for i in range(55)]
beta_GT_selected = pd.merge(all_beta_df_real, GT_selected_features_df, on = 'ICs')
beta_GT_selected = beta_GT_selected.drop(columns = ['Selection_Frequency'])
beta_GT_selected.reset_index(drop = True, inplace = True)
# beta_GT_selected.to_csv('beta_GT_selected.csv', index = False)

beta_GT_pos_corr = all_beta_df_real.iloc[:55, :]
beta_GT_neg_corr = all_beta_df_real.iloc[55:, :]
beta_GT_pos_corr_selected = pd.merge(beta_GT_pos_corr, GT_selected_features_df, on = 'ICs')
beta_GT_pos_corr_selected = beta_GT_pos_corr_selected.drop(columns = ['Selection_Frequency'])
beta_GT_pos_corr_selected.reset_index(drop = True, inplace = True)
beta_GT_neg_corr_selected = pd.merge(beta_GT_neg_corr, GT_selected_features_df, on = 'ICs')
beta_GT_neg_corr_selected = beta_GT_neg_corr_selected.drop(columns = ['Selection_Frequency'])
beta_GT_neg_corr_selected.reset_index(drop = True, inplace = True)

#Get predictive ICs
GT_id = np.sort(np.unique(GT_selected_features_df_id_sorted['ICs'].str[7:], return_counts = True)[0].astype(int))

#Get positive/negative beta ICs in positive/negative correlation ICs separately
beta_GT_pos_corr_positive = beta_GT_pos_corr_selected[
    (beta_GT_pos_corr_selected.iloc[:, :-1] > 0).sum(axis=1) > 
    (beta_GT_pos_corr_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)

beta_GT_pos_corr_negative = beta_GT_pos_corr_selected[
    (beta_GT_pos_corr_selected.iloc[:, :-1] > 0).sum(axis=1) < 
    (beta_GT_pos_corr_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)

beta_GT_neg_corr_positive = beta_GT_neg_corr_selected[
    (beta_GT_neg_corr_selected.iloc[:, :-1] > 0).sum(axis=1) > 
    (beta_GT_neg_corr_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)

beta_GT_neg_corr_negative = beta_GT_neg_corr_selected[
    (beta_GT_neg_corr_selected.iloc[:, :-1] > 0).sum(axis=1) < 
    (beta_GT_neg_corr_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)

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

plotting.plot_roi(last_IC_10, cut_coords = [-1, -44, 12], cmap = "Set1", colorbar = True)
np.unique(last_IC_10.get_fdata(), return_counts = True)

#Find intersection of predictive GT
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1
new_IC_template_mask = new_img_like(ica100_template, data)
plotting.plot_roi(new_IC_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(new_IC_template_mask)))
