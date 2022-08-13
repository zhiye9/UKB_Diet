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
import time

import warnings
warnings.filterwarnings('ignore')

ica100_template = nib.load('rfMRI_ICA_d100.nii.gz')
fmri_data = get_data(ica100_template)
good_IC = loadtxt('UKBiobank_BrainImaging_GroupMeanTemplates/rfMRI_GoodComponents_d100_v1.txt', dtype = int)
good_IC_minus1 = (good_IC -1).tolist()
fmri_data_new = fmri_data[:,:,:, good_IC_minus1]
new_fmri_img = new_img_like(ica100_template, fmri_data_new)
data = new_fmri_img.get_data()

target = np.zeros(new_fmri_img.shape[:-1], dtype = int)
for i in range(target.shape[0]):
    start_time = time.time()
    for j in range(target.shape[1]):
        for k in range(target.shape[2]):
            target[i][j][k] = np.argmax([data[i][j][k][l] for l in range(new_fmri_img.shape[3])])
    print(time.time() - start_time)
        #print("\r Process{}%".format(round((i*j*k)*100/(target.shape[0]*target.shape[1]*target.shape[2]))), end="")

IC_atlas = new_img_like(new_fmri_img, target)
nib.save(IC_atlas, 'IC_55_atlas_withoutprob.nii')
