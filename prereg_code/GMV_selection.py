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
from nilearn.image import get_data, new_img_like, resample_to_img
from nilearn import datasets
import nilearn.plotting as plotting
from UKB_graph_metrics import *
from utils import compute_FDD, Get_selected_features

## Select the features

# Load the CV results
with open("r_total_randomper", "rb") as fp:
            r_total_GMV = pickle.load(fp)

n_beta = 139
rand_idx_length = 100
CV_folds = 10
method = 'subtract'

feature_counts, all_beta_df = Get_selected_features(r_total_GMV, n_beta, rand_idx_length, CV_folds, method)

# Get max frequency of artificial features
max_artificial_freq = np.max(feature_counts[n_beta:])

# Get real features above threshold
GMV_real_features = feature_counts[:n_beta]
GMV_features_df = pd.DataFrame(GMV_real_features, columns = ['Selection_Frequency'])
gmv = np.genfromtxt('GMV.txt', dtype='str')
GMV_features_df['eid'] = gmv.astype(int)
gmv_id = pd.read_csv('GMV_eid.csv')
GMV_features_df_id = pd.merge(GMV_features_df, gmv_id, on = 'eid')
GMV_features_df_id = GMV_features_df_id.drop(columns = ['eid'])

# Get frequencies of selected features
GMV_selected_features_df = GMV_features_df_id[GMV_features_df_id['Selection_Frequency'] > max_artificial_freq]

# Sort by frequency
GMV_selected_features_df_id_sorted = GMV_selected_features_df.sort_values('Selection_Frequency', ascending=False).reset_index(drop=True)

## Plot the selected features

# Filter the selected features
all_beta_df_real = all_beta_df.iloc[:139, :]
all_beta_df_real['eid'] = gmv.astype(int)
all_beta_df_real_eid = pd.merge(all_beta_df_real, gmv_id, on = 'eid')
all_beta_df_real_eid = all_beta_df_real_eid.drop(columns = ['eid'])

beta_GMV_selected = pd.merge(all_beta_df_real_eid, GMV_selected_features_df, on = 'GMV_name')
beta_GMV_selected = beta_GMV_selected.drop(columns = ['Selection_Frequency'])
beta_GMV_selected.reset_index(drop = True, inplace = True)
# beta_GMV_selected.to_csv('beta_GMV_selected.csv', index = False)

#Read GM labels
GM_labels = loadtxt("GMatlas_name.txt", dtype=str, delimiter="\t", unpack=False).tolist()

#Get predictive GM features index.
GM_id = []
for i in range(beta_GMV_selected.shape[0]):
    if any(beta_GMV_selected[['GMV_name']].loc[i].str[25: -10].values[0] in s for s in GM_labels):
        if (beta_GMV_selected[['GMV_name']].loc[i].str[-6:-1].values[0] == 'right'):
            indices = [j for j, s in enumerate(GM_labels) if ('Right ' + str(beta_GMV_selected[['GMV_name']].loc[i].str[25:-8].values[0])) in s]
        elif (beta_GMV_selected[['GMV_name']].loc[i].str[-5:-1].values[0] == 'left'):
            indices = [j for j, s in enumerate(GM_labels) if ('Left ' + str(beta_GMV_selected[['GMV_name']].loc[i].str[25:-7].values[0])) in s]
        elif (beta_GMV_selected[['GMV_name']].loc[i].str[-7:-1].values[0] == 'vermis'):
            indices = [j for j, s in enumerate(GM_labels) if ('Vermis ' + str(beta_GMV_selected[['GMV_name']].loc[i].str[25:-20].values[0])) in s]
        GM_id.append(indices[0])

#Get positive/negative beta GMs
beta_GMV_positive = beta_GMV_selected[
    (beta_GMV_selected.iloc[:, :-1] > 0).sum(axis=1) > 
    (beta_GMV_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)

beta_GMV_negative = beta_GMV_selected[
    (beta_GMV_selected.iloc[:, :-1] > 0).sum(axis=1) < 
    (beta_GMV_selected.iloc[:, :-1] < 0).sum(axis=1)
].reset_index(drop=True)


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

labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in GM_id_pos]
labels_names_GM_neg = [i for i in GM_labels if GM_labels.index(i) in GM_id_neg]
labels_names_GM_pos.extend(labels_names_GM_neg)

#The atlas start from 1, 0 is background, but GM_id start from 0 
new_GM_id = [x+1 for x in GM_id]
new_GM_id_pos = [x+1 for x in GM_id_pos]
new_GM_id_neg = [x+1 for x in GM_id_neg]

#Resample GM atlas to (91, 109, 91)
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))

#Find intersection of predictive GMs
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = 0
data[np.isin(data, new_GM_id)] = 1
GM_template_mask = new_img_like(resampled_GM, data)
plotting.plot_roi(GM_template_mask, cut_coords = [-1, -44, 12])
print(np.count_nonzero(get_data(GM_template_mask)))

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