#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import nilearn.image  as im
import time

import warnings
warnings.filterwarnings('ignore')

from nilearn._utils.extmath import fast_abs_percentile
maps_img = new_fmri_img
n_maps = maps_img.shape[3]
correction_factor = .8
threshold = "%f%%" % (100 * (1 - .2 * correction_factor / n_maps))
threshold = [threshold] * n_maps
cmap=plt.cm.gist_rainbow
color_list = cmap(np.linspace(0, 1, n_maps))

thr_55 = []
for (map_img, color, thr) in zip(nl.image.iter_img(maps_img), color_list,
                                    threshold):
    data_img = get_data(map_img)
    # To threshold or choose the level of the contours
    thr = check_threshold(thr, data_img,
                            percentile_func=fast_abs_percentile,
                            name='threshold')
    #print(thr)
    # Get rid of background values in all cases
    thr = max(thr, 1e-6)
    thr_55.append(thr)

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
fmri_data = get_data(ica100_template)
good_IC = loadtxt('UKBiobank_BrainImaging_GroupMeanTemplates/rfMRI_GoodComponents_d100_v1.txt', dtype = int)
good_IC_minus1 = (good_IC -1).tolist()
fmri_data_new = fmri_data[:,:,:, good_IC_minus1]
new_fmri_img = new_img_like(ica100_template, fmri_data_new)
data = new_fmri_img.get_data()

target = np.zeros(new_fmri_img.shape[:-1])
for i in range(target.shape[0]):
    start_time = time.time()
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            target_ijk = [data[i][j][k][l] if (data[i][j][k][l] >= thr_55[l]) else -1 for l in range(new_fmri_img.shape[3])]
            if (np.unique(target_ijk)[0] == -1.0) and (len(np.unique(target_ijk)) == 1):
                target[i][j][k] = 0
            else:
                target[i][j][k] = good_IC_minus1[np.argmax(target_ijk)]

#np.save('IC_no_overlap', target)

#target = np.load('IC_no_overlap.npy')
IC_atlas = new_img_like(new_fmri_img, target)
plotting.plot_roi(IC_atlas, cut_coords = [-1, -44, 12])

nib.save(IC_atlas, 'IC_55_atlas_withoutprob.nii.gz')

------------------------------------------------------------------------------------------------------------------------

target = np.zeros(new_fmri_img.shape[:-1])
for i in range(target.shape[0]):
    start_time = time.time()
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            target[i][j][k] = np.argmax([data[i][j][k][l] if (data[i][j][k][l] >= thr_55[l]) else -1 for l in range(new_fmri_img.shape[3])])
    print(time.time() - start_time)
        #print("\r Process{}%".format(round((i*j*k)*100/(target.shape[0]*target.shape[1]*target.shape[2]))), end="")

-------------------------------------------------------------------------------------------------------------------------    


target = np.zeros(new_fmri_img.shape[:-1])
ijk_ind = []
for i in range(target.shape[0]):
    #start_time = time.time()
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            if (data[i][j][k][7] >= thr_55[7]):
                ijk_ind.append([i, j, k])
            #target[i][j][k] = [data[i][j][k][7] if (data[i][j][k][7] >= thr_55[7]) else 0][0]
    #print(time.time() - start_time)

lj =[]
for i in range(len(ijk_ind)):
    lj.append(target[ijk_ind[i][0]][ijk_ind[i][1]][ijk_ind[i][2]])

IC_atlas = new_img_like(new_fmri_img, target)
plotting.plot_roi(IC_atlas, cut_coords = [-1, -44, 12])

target = np.zeros(new_fmri_img.shape[:-1])
for i in range(len(ijk_ind)):
    #target[ijk_ind[i][0]][ijk_ind[i][1]][ijk_ind[i][2]] = data[ijk_ind[i][0]][ijk_ind[i][1]][ijk_ind[i][2]][7]
    target[ijk_ind[i][0]][ijk_ind[i][1]][ijk_ind[i][2]] = 1

target[target != 7.] = 0
target[target == 7.] = 1
GM_template_mask = new_img_like(new_fmri_img, target)
