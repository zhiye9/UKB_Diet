#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 20:25:05 2021

@author: zhye
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import time

df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')
#intake = [col for col in df.columns if  in col]
#df_intake = df.filter(regex = '1289|1299|1309|1319|1438|1458|1488|1498|1528')
#df_intake = df.filter(regex = '1289|1299')
#df_intake['eid'] = df['eid']
#df_intake_noNeg = df_intake.mask(df_intake < 0)
#df_intake_noNAN_noNeg = df_intake_noNeg.dropna(how='all', subset=intake)
#df_intake_noNAN_noNeg.reset_index(drop = True, inplace = True)

#ntake_ls = ['1289', '1299', '1309', '1319', '1438', '1458','1488', '1498', '1528']
#intake_ls = ['1289', '1299']
'''
for j in range(len(intake_ls)):
    phe = [col for col in df.columns if intake_ls[j] in col]
    df_intake_noNAN_noNeg[intake_ls[j]] = 0
    print(j)
    for i in range(df_intake_noNAN_noNeg.shape[0]):
        df_intake_noNAN_noNeg[intake_ls[j]].loc[i] = np.nanmean(df_intake_noNAN_noNeg[phe].loc[i])
 
df_intake_center = df_intake_noNAN_noNeg[intake_ls]
df_intake_center_nonNan = df_intake_center.dropna()

df_center_scaled = preprocessing.scale(df_intake_center_nonNan)
pca = PCA(n_components = 'mle')
center_pca = pca.fit_transform(df_center_scaled)
print(pca.explained_variance_ratio_)
'''
----------------------------------------------------------------------------------
def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos
--------------------------------------------------------------------------
df_intake_center = df.filter(regex = '1289|1299|1309|1319|1438|1458|1488|1498|1528|1329|1339|1349|1359|1369|1379|1389|1408|1478')
#df_intake = df.filter(regex = '1289|1299')
#df_intake_center['eid'] = df['eid']
df_intake_center_noNeg = df_intake_center.mask(df_intake_center < 0)
df_intake_center_noNAN_noNeg = df_intake_center_noNeg.dropna(how='all', subset=intake)
df_intake_center_noNAN_noNeg.reset_index(drop = True, inplace = True)

intake_center = ['1289', '1299', '1309', '1319', '1438', '1458','1488', '1498', '1528','1329', '1339',
             '1349', '1359', '1369', '1379', '1389', '1408', '1478']
#intake_ls = ['1289', '1299']

for j in range(len(intake_center)):
    phe = [col for col in df_intake_center_noNAN_noNeg.columns if intake_center[j] in col]
    df_intake_center_noNAN_noNeg[intake_center[j]] = 0
    print(j)
    for i in range(df_intake_center_noNAN_noNeg.shape[0]):
        df_intake_center_noNAN_noNeg[intake_center[j]].loc[i] = np.nanmean(df_intake_center_noNAN_noNeg[phe].loc[i])

df_intake_center_all = df_intake_center_noNAN_noNeg[intake_center + ['eid']]
df_intake_center_all_nonNan = df_intake_center_all.dropna()
df_intake_center_all_nonNan.reset_index(drop = True, inplace = True)

df_center_scaled = preprocessing.scale(df_intake_center_all_nonNan[intake_center])
pca = PCA(n_components = 'mle')
pca = PCA(n_components = 0.9)
center_pca = pca.fit_transform(df_center_scaled)
print(pca.explained_variance_ratio_)

def plotCumSumVariance(var=None):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save file
    cumvar = var.cumsum()

    plt.figure()
    plt.bar(np.arange(len(var)), cumvar, width = 1.0)
    plt.axhline(y=0.9, color='r', linestyle='-')

plotCumSumVariance(pca.explained_variance_ratio_)

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loadings, index = food_name)

loading_matrix.to_csv('PCA_center_loading.csv')

food_name = ['Cooked vegetable intake','Salad / raw vegetable intake','Fresh fruit intake','Dried fruit intake','Bread intake','Cereal intake','Tea intake','Coffee intake', 'Water intake','Oily fish intake','Non-oily fish intake', 'Processed meat intake','Poultry intake','Beef intake' ,'Lamb/mutton intake','Pork intake','Cheese intake','Salt added to food']

def myplot(score,coeff,name, size, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    figure(num=None, figsize=size, dpi=400, facecolor='w', edgecolor='k')
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, name[i], color = , ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

#myplot(center_pca[:,0:2],np.transpose(pca.components_[0:2, :]))
myplot(center_pca[:,0:2],loadings[:,0:2], name = food_name, size = (12, 6))
plt.show()

----------------------------------------------------------------------------------
intake_online = pd.read_csv('food_intake.csv', dtype = str, names = ['code', 'name'])
food_online = intake_online['code'].to_list()

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

df_intake_online = df.filter(regex = convstr(food_online, '|'))
df_intake_online['eid'] = df['eid']

#intake = [col for col in df.columns if food_online[i] in col]
df_intake_online = df_intake_online.mask(df_intake_online.applymap(type) == str)
df_intake_online_noNAN = df_intake_online.dropna(how='all', subset = df_intake_online.columns[:-1])
df_intake_online_noNAN.reset_index(drop = True, inplace = True)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].astype(np.float64)

df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(555.0, 0.5)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(444.0, 0.25)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(200.0, 3.0)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(300.0, 4.0)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(400.0, 5.0)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(500.0, 6.0)
df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]] = df_intake_online_noNAN[df_intake_online_noNAN.columns[:-1]].replace(600.0, 7.0)


df_intake_online_mean = pd.DataFrame()
df_intake_online_mean['eid'] = df_intake_online_noNAN['eid']
for j in range(len(food_online)):
    #start_time = time.time()
    phe = [col for col in df_intake_online_noNAN.columns if food_online[j] in col]
    df_intake_online_mean[food_online[j]] = df_intake_online_noNAN[phe].apply(np.nanmean, axis = 1)
    print(j)
    #print(time.time() - start_time)

    '''    
    for i in range(df_intake_online_noNAN.shape[0]):
        df_intake_online_noNAN[food_online[j]].loc[i] = np.nanmean(df_intake_online_noNAN[phe].loc[i])

for j in range(len(food_online)):
    start_time = time.time()
    phe = [col for col in df_intake_online_noNAN.columns if food_online[j] in col]
    df_intake_online_noNAN[food_online[j]] = 0 
    for i in range(df_intake_online_noNAN.shape[0]):
        df_intake_online_noNAN[food_online[j]].loc[i] = np.nanmean(df_intake_online_noNAN[phe].loc[i])
    print(j)
    print(time.time())
'''
df_intake_online_all = df_intake_online_noNAN[food_online + ['eid']]
nan_num = (df_intake_online_all.drop(columns='eid').isna().sum())
nonan_num = np.repeat(df_intake_online_all.shape[0], len(nan_num.to_list())) - nan_num.to_list()

df_intake_online_all.to_csv('intake_online.csv')

plt.figure(dpi = 400)
axes = plt.gca()
#axes.set_ylim([0, 1000])
plt.bar(np.arange(len(nan_num)-1), nonan_num[:-1], width=1.0)

df_intake_online_all_nonNan = df_intake_online_all.dropna()
df_intake_online_all_nonNan.reset_index(drop = True, inplace = True)
-----------------------------------------------------------------------------------
liking = pd.read_csv('liking rating.csv', dtype = str, names = ['code', 'name'])
liking_rating = liking['code'].to_list()

df_liking = df.filter(regex = convstr(liking_rating, '|'))
df_liking['eid'] = df['eid']
df_liking_noNeg = df_liking.mask(df_liking < 0)
df_liking_noNeg_noNAN = df_liking_noNeg.dropna(how='all', subset = df_liking_noNeg.columns[:-1])
df_liking_all_noNAN = df_liking_noNeg_noNAN.dropna()
df_liking_all_noNAN.reset_index(drop = True, inplace = True)

df_liking_scaled = preprocessing.scale(df_liking_all_noNAN[df_liking_noNeg.columns[:-1]])
pca = PCA(n_components = 'mle')
pca = PCA(n_components = 0.9)
liking_pca = pca.fit_transform(df_liking_scaled)
print(pca.explained_variance_ratio_)

plotCumSumVariance(pca.explained_variance_ratio_)

liking_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
liking_loading_matrix = pd.DataFrame(liking_loadings, index = liking['name'].to_list())

liking_loading_matrix.to_csv('PCA_liking_loading.csv')
short_name = [d[11:] for d in liking['name'].to_list()]

liking_color = np.concatenate((np.repeat(0, 10), np.repeat(1, 47), np.repeat(2, 11), np.repeat(3, 29), np.repeat(4, 6), np.repeat(5, 20), np.repeat(6, 11), np.repeat(7, 6), np.repeat(8, 8), np.repeat(9,2)))
color_list = []
for key, value in  matplotlib.colors.TABLEAU_COLORS.items():
    temp = value
    color_list.append(temp)
    
def myplot_color(score,coeff,name, size, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    figure(num=None, figsize=size, dpi=400, facecolor='w', edgecolor='k')
    plt.scatter(xs * scalex,ys * scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = color_list[liking_color[i]], alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, name[i], ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    
myplot_color(liking_pca[:,0:2],liking_loadings[:,0:2], name = short_name, size = (12, 12))
plt.show()
-----------------------------------------------------------------------------------
df_energy = df.filter(regex = '100002|eid')
df_energy_noNeg = df_energy.mask(df_energy < 0).dropna(subset = df_energy.columns[1:],how = 'all')

df_energy_liking = pd.merge(df_liking_all_noNAN, df_energy_noNeg, on = 'eid')

df_fat = df.filter(regex = '100004|eid')
df_fat_noNeg = df_fat.mask(df_fat < 0).dropna(subset = df_fat.columns[1:],how = 'all')

df_fat_liking = pd.merge(df_liking_all_noNAN, df_fat_noNeg, on = 'eid')

df_sugar = df.filter(regex = '100008|eid')
df_sugar_noNeg = df_sugar.mask(df_sugar < 0).dropna(subset = df_sugar.columns[1:],how = 'all')

df_sugar_liking = pd.merge(df_liking_all_noNAN, df_sugar_noNeg, on = 'eid')

df_protein = df.filter(regex = '100003|eid')
df_protein_noNeg = df_protein.mask(df_protein < 0).dropna(subset = df_protein.columns[1:],how = 'all')

df_protein_liking = pd.merge(df_liking_all_noNAN, df_protein_noNeg, on = 'eid')
-----------------------------------------------------------------------------------
corr100 = pd.read_csv('corr100.txt', dtype = str, names = ['file'], header = None)
corr100['eid'] = corr100['file'].str[:7].astype(float)

df_liking_all_noNAN['eid'] = df_liking_all_noNAN['eid'].astype(str) 
df_energy_noNeg['eid'] = df_energy_noNeg['eid'].astype(str) 


df_liking_energy = pd.merge(df_liking_all_noNAN, df_energy_noNeg, on = 'eid')
df_liking_mri = pd.merge(df_liking_all_noNAN, corr100, on = 'eid')
df_energy_mri = pd.merge(df_energy_noNeg, corr100, on = 'eid')

df_energy_liking_mri = pd.merge(df_liking_mri, df_energy_mri, on = 'eid')

----------------------------------------------------------------------------------
df_intake_online_all_10 = df_intake_online_all.dropna(thresh = 11)
df_intake_online_all_10.reset_index(drop = True, inplace = True)
df_liking_noNeg_noNAN_10 = df_liking_noNeg_noNAN.dropna(thresh = 11)
df_liking_noNeg_noNAN_10.reset_index(drop = True, inplace = True)

df_liking_mri_10 = pd.merge(df_liking_noNeg_noNAN_10, corr100, on = 'eid')
df_intake_online_all_10['eid'] = df_intake_online_all_10['eid'].astype(str) 
df_intake_mri_10 = pd.merge(df_intake_online_all_10, corr100, on = 'eid')

df_intake_liking_mri_10 = pd.merge(df_liking_mri_10, df_intake_mri_10, on = 'eid')

intake_overlap = pd.read_csv('intake_overlap.csv', dtype = str, names = ['code', 'name'])
intake_overlap_code = intake_overlap['code'].to_list()

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

df_intake_overlap = df_intake_online_all.filter(regex = convstr(intake_overlap_code, '|'))
df_intake_overlap['eid'] = df_intake_online_all['eid']
df_intake_overlap_10 = df_intake_overlap.dropna(thresh = 11)
df_intake_overlap_10.reset_index(drop = True, inplace = True)

liking_overlap = pd.read_csv('liking_overlap.csv', dtype = str, names = ['code', 'name'])
liking_overlap_code = liking_overlap['code'].to_list()
liking_overlap_code = [s + '-0.0' for s in liking_overlap_code]

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

df_liking_overlap = df_liking_noNeg_noNAN.filter(regex = convstr(liking_overlap_code, '|'))
df_liking_overlap['eid'] = df_liking_noNeg_noNAN['eid']
df_liking_overlap_10 = df_liking_overlap.dropna(thresh = 11)
df_liking_overlap_10.reset_index(drop = True, inplace = True)

df_intake_overlap_10['eid'] = df_intake_overlap_10['eid'].astype(str)
df_intake_liking_overlap_10 = pd.merge(df_intake_overlap_10, df_liking_overlap_10, on = 'eid')

df_intake_over10_TF = df_intake_overlap_10[df_intake_overlap_10.columns[:-1]].notna()
df_intake_over10_TF['eid'] = df_intake_overlap_10['eid']

df_liking_over10_TF = df_liking_overlap_10[df_liking_overlap_10.columns[:-1]].notna()
df_liking_over10_TF['eid'] = df_liking_overlap_10['eid']

df_intake_liking_over10_TF = pd.merge(df_intake_over10_TF, df_liking_over10_TF, on = 'eid')
df_TF = pd.DataFrame(np.zeros([df_intake_liking_over10_TF.shape[0], 41])*np.nan)

for j in range(df_intake_liking_over10_TF.shape[0]):
    for i in range(len(intake_overlap_code)):
        if df_intake_liking_over10_TF.loc[j][intake_overlap_code[i]] & df_intake_liking_over10_TF.loc[j][liking_overlap_code[i]]:
            df_TF.loc[j][i] = 0
    print(j)
df_TF['eid'] = df_intake_liking_over10_TF['eid']
df_TF_10 = df_TF.dropna(thresh = 11)       
