#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 03:43:59 2021

@author: zhye
"""

import numpy as np
import pandas as pd


IC1 = np.loadtxt('1000048_25751_2_0.txt')
size = 55
corr_matrix = np.zeros((size,size))
corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC1
corr_matrix = corr_matrix + corr_matrix.T

