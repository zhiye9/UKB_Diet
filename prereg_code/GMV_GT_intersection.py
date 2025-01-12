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
from nilearn.masking import intersect_masks
import nilearn.plotting as plotting
from UKB_graph_metrics import *

def extract_atlas(file, n):
    GMatlas = nib.load(file)
    resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
    data = get_data(resampled_GM)
    data[data != n] = 0
    data[data == n] = 1
    new_img = new_img_like(resampled_GM, data)
    return new_img

#Compute intersection of predicve GMs and ICs
GMatlas = nib.load('GMatlas.nii.gz')
harvard_oxford_s = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-2mm', symmetric_split = True)
resampled_GM = resample_to_img(GMatlas, nib.load(harvard_oxford_s.filename))
data = get_data(resampled_GM)
data[~np.isin(data, new_GM_id)] = int(0)
data[np.isin(data, new_GM_id)] = int(1)
GM_template_mask = new_img_like(resampled_GM, data)

ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
GM_IC_template_mask_sum = np.multiply(GM_template_mask.get_fdata(), new_IC_template_mask.get_fdata())
GM_IC_template_mask_sum_mri = new_img_like(ica100_template, GM_IC_template_mask_sum) 
plotting.plot_roi(GM_IC_template_mask_sum_mri)
print(np.count_nonzero(get_data(GM_IC_template_mask_sum_mri)))

#Find predictive GMs in the overlap of predictive GMs and ICs
ica100_template = nib.load('IC_55_atlas_withoutprob.nii.gz')
data = get_data(ica100_template)
data[~np.isin(data, GT_id - 1)] = 0
data[np.isin(data, GT_id - 1)] = 1

new_IC_template_mask = new_img_like(ica100_template, data)

# Find the intersection of the predictive GM and IC
overlap_voxel_per_pred_GM_pos = []
for i in range(len(new_GM_id_pos)):
    GM_atlas_pred_extracted_pos = extract_atlas('GMatlas.nii.gz', new_GM_id_pos[i])
    int_IC_GM_pred_pos = intersect_masks([new_IC_template_mask, GM_atlas_pred_extracted_pos], threshold = 1)
    overlap_voxel_per_pred_GM_pos.append(np.count_nonzero(get_data(int_IC_GM_pred_pos))/np.count_nonzero(get_data(GM_atlas_pred_extracted_pos)))
    print("\r Process{}%".format(round((i+1)*100/len(new_GM_id_pos))), end="")

overlap_per_labels_names_GM_pos = [i for i in GM_labels if GM_labels.index(i) in GM_id_pos]
df_overlap_GM_pos = pd.DataFrame(data = {'GM_name': overlap_per_labels_names_GM_pos, 'Overlap_percentage': overlap_voxel_per_pred_GM_pos})

overlap_voxel_per_pred_GM_neg = []
for i in range(len(new_GM_id_neg)):
    GM_atlas_pred_extracted_neg = extract_atlas('GMatlas.nii.gz', new_GM_id_neg[i])
    int_IC_GM_pred_neg = intersect_masks([new_IC_template_mask, GM_atlas_pred_extracted_neg], threshold = 1)
    overlap_voxel_per_pred_GM_neg.append(np.count_nonzero(get_data(int_IC_GM_pred_neg))/np.count_nonzero(get_data(GM_atlas_pred_extracted_neg)))
    print("\r Process{}%".format(round((i+1)*100/len(new_GM_id_neg))), end="")

overlap_per_labels_names_GM_neg = [i for i in GM_labels if GM_labels.index(i) in GM_id_neg]
df_overlap_GM_neg = pd.DataFrame(data = {'GM_name': overlap_per_labels_names_GM_neg, 'Overlap_percentage': overlap_voxel_per_pred_GM_neg})

df_overlap_GM = pd.concat([df_overlap_GM_pos, df_overlap_GM_neg])
df_overlap_GM.sort_values(by = 'Overlap_percentage', ascending = False)