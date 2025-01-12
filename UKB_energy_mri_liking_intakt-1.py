# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 04:38:26 2021

@author: Zhi Ye
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib
import seaborn as sns

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

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

df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')
#intake = [col for col in df.columns if  in col]
df_intake = df.filter(regex = '1289|1299|1309|1319|1438|1458|1488|1498|1528')
#df_intake = df.filter(regex = '1289|1299')
#df_intake['eid'] = df['eid']
df_intake_noNeg = df_intake.mask(df_intake < 0)
df_intake_noNAN_noNeg = df_intake_noNeg.dropna(how='all', subset=intake)
df_intake_noNAN_noNeg.reset_index(drop = True, inplace = True)

intake_ls = ['1289', '1299', '1309', '1319', '1438', '1458','1488', '1498', '1528']
#intake_ls = ['1289', '1299']

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
df_intake_online_noNAN = df_intake_online.dropna(how='all', subset = df_intake_online.columns[:-1])
df_intake_online_noNAN.reset_index(drop = True, inplace = True)
df_intake_online_noNAN = df_intake_online_noNAN.replace(555, 0.5)
df_intake_online_noNAN = df_intake_online_noNAN.replace(444, 0.25)
df_intake_online_noNAN = df_intake_online_noNAN.replace(200, 3)
df_intake_online_noNAN = df_intake_online_noNAN.replace(300, 4)
df_intake_online_noNAN = df_intake_online_noNAN.replace(400, 5)
df_intake_online_noNAN = df_intake_online_noNAN.replace(500, 6)
df_intake_online_noNAN = df_intake_online_noNAN.replace(600, 7)

for j in range(len(food_online)):
    phe = [col for col in df_intake_online_noNAN.columns if food_online[j] in col]
    df_intake_online_noNAN[food_online[j]] = 0
    print(j)
    for i in range(df_intake_online_noNAN.shape[0]):
        df_intake_online_noNAN[food_online[j]].loc[i] = np.nanmean(df_intake_online_noNAN[phe].loc[i])

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
df_liking_noNeg_noNAN.reset_index(drop = True, inplace = True)

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
df_energy_noNeg.reset_index(drop = True, inplace = True)

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
----------------------------------------------------------------------------------
corr100 = pd.read_csv('corr100.txt', dtype = str, names = ['file'], header = None)
corr100['eid'] = corr100['file'].str[:7].astype(float)

df_liking_all_noNAN['eid'] = df_liking_all_noNAN['eid'].astype(str) 
df_energy_noNeg['eid'] = df_energy_noNeg['eid'].astype(str) 

df_liking_energy = pd.merge(df_liking_all_noNAN, df_energy_noNeg, on = 'eid')
df_liking_mri = pd.merge(df_liking_all_noNAN, corr100, on = 'eid')
df_energy_mri = pd.merge(df_energy_noNeg, corr100, on = 'eid')

df_energy_liking_mri = pd.merge(df_liking_mri, df_energy_mri, on = 'eid')

----------------------------------------------------------------------------------
intake_online = pd.read_csv('intake_online.csv', encoding = 'unicode_escape', index_col=0)
df_intake_online_all = intake_online
df_intake_online_all_10 = df_intake_online_all.dropna(thresh = 11)
df_intake_online_all_10.reset_index(drop = True, inplace = True)
df_liking_noNeg_noNAN_10 = df_liking_noNeg_noNAN.dropna(thresh = 11)
df_liking_noNeg_noNAN_10.reset_index(drop = True, inplace = True)

df_liking_noNeg_noNAN['eid'] = df_liking_noNeg_noNAN['eid'].astype(str) 

df_liking_mri = pd.merge(df_liking_noNeg_noNAN, corr, on = 'eid')
df_liking_mri = df_liking_mri[~df_liking_mri.duplicated(subset='eid')]

df_intake_online_all_10['eid'] = df_intake_online_all_10['eid'].astype(str) 
df_intake_mri_10 = pd.merge(df_intake_online_all_10, corr100, on = 'eid')

df_intake_liking_mri_10 = pd.merge(df_liking_mri_10, df_intake_mri_10, on = 'eid')

intake_overlap = pd.read_csv('intake_overlap.csv', dtype = str, names = ['code', 'name'])
intake_overlap_code = intake_overlap['code'].to_list()

df_intake_overlap = df_intake_online_all_10.filter(regex = convstr(intake_overlap_code, '|'))
df_intake_overlap['eid'] = df_intake_online_all_10['eid']
df_intake_overlap_10 = df_intake_overlap.dropna(thresh = 11)
df_intake_overlap_10 = df_intake_overlap_10.dropna(subset = ['eid'])
df_intake_overlap_10.reset_index(drop = True, inplace = True)


liking_overlap = pd.read_csv('liking_overlap.csv', dtype = str, names = ['code', 'name'])
liking_overlap_code = liking_overlap['code'].to_list()
liking_overlap_code = [s + '-0.0' for s in liking_overlap_code]

def convstr(input_seq, seperator):
    return seperator.join(input_seq)

df_liking_overlap = df_liking_noNeg_noNAN.filter(regex = convstr(liking_overlap_code, '|'))
df_liking_overlap['eid'] = df_liking_noNeg_noNAN['eid']
df_liking_overlap = df_liking_overlap.dropna(thresh = 22)
df_liking_overlap.reset_index(drop = True, inplace = True)
df_liking_overlap['eid'] = df_liking_overlap['eid'].astype(str)

new_col = []
for i in range(df_liking_overlap.shape[1] - 1):
    new_col.append(df_liking_overlap.columns[i][0:5])
new_col.append('eid')

df_liking_overlap.columns = new_col

df_intake_overlap_10['eid'] = df_intake_overlap_10['eid'].astype(int)
df_intake_overlap_10['eid'] = df_intake_overlap_10['eid'].astype(str)

df_intake_over10_TF = df_intake_overlap_10[df_intake_overlap_10.columns[:-1]].notna()
df_intake_over10_TF['eid'] = df_intake_overlap_10['eid']

df_liking_over10_TF = df_liking_overlap_10[df_liking_overlap_10.columns[:-1]].notna()
df_liking_over10_TF['eid'] = df_liking_overlap_10['eid']

df_intake_liking_over10_TF = pd.merge(df_intake_over10_TF, df_liking_over10_TF, on = 'eid')
df_TF = pd.DataFrame(np.zeros([df_intake_liking_over10_TF.shape[0], 41])*np.nan)
f
for j in range(df_intake_liking_over10_TF.shape[0]):
    for i in range(len(intake_overlap_code)):
        if df_intake_liking_over10_TF.loc[j][intake_overlap_code[i]] & df_intake_liking_over10_TF.loc[j][liking_overlap_code[i]]:
            df_TF.loc[j][i] = 0
    print(j)
df_TF['eid'] = df_intake_liking_over10_TF['eid']
df_TF_10 = df_TF.dropna(thresh = 11)       
------------------------------------------------------------------------------------------------------------
intake_online['eid'] = intake_online['eid'].astype(str)

liking = pd.read_csv('liking_0.9.csv', encoding = 'unicode_escape')
liking = liking[liking.columns[1:]]

df_energy = df.filter(regex = '100002|eid')
df_energy_noNeg = df_energy.mask(df_energy < 0).dropna(subset = df_energy.columns[1:],how = 'all')
df_energy_noNeg.reset_index(drop = True, inplace = True)
df_energy_noNeg['100002'] = 0
for i in range(df_energy_noNeg.shape[0]):
    df_energy_noNeg['100002'].loc[i] = np.nanmean(df_energy_noNeg[df_energy_noNeg.columns[1:6]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_energy_noNeg.shape[0])), end="")

df_energy_noNeg.to_csv('energy_mean.csv')

df_energy_noNeg = pd.read_csv('energy_mean.csv', index_col=0)

df_liking_noNeg_noNAN_10.to_csv('liking_90_noNAN.csv')

df_energy_mean = df_energy_noNeg[['eid', '100002']]
df_energy_mean['eid'] = df_energy_mean['eid'].astype(str)
corr = pd.read_csv('corr_files.txt', dtype = str, names = ['file'], header = None)
corr['eid'] = corr['file'].str[:7]

df_liking_noNeg_noNAN_10['eid'] = df_liking_noNeg_noNAN_10['eid'].astype(str)
df_energy_mean['eid'] = df_energy_mean['eid'].astype(str)

df_intake_liking_overlap_10 = pd.merge(df_intake_overlap_10, df_liking_overlap_10, on = 'eid')
df_liking_mri = pd.merge(df_liking_noNeg_noNAN_10, corr, on = 'eid')
df_liking_energy = pd.merge(df_liking_noNeg_noNAN_10, df_energy_mean, on = 'eid')
df_mri_energy = pd.merge(corr, df_energy_mean, on = 'eid')
df_liking_mri_energy = pd.merge(df_liking_mri, df_liking_energy, on = 'eid')

df_intake_liking_overlap_10_energy = pd.merge(df_intake_liking_overlap_10, df_energy_mean, on = 'eid')

df_intake_liking_overlap_10_energy.to_csv('intak_liking_energy.csv', index = False)

------------------------------------------------------------------------------------------------------------
df_intake_liking_overlap_10_energy = pd.read_csv('intak_liking_energy.csv')
df_intake_liking_overlap_10_energy['eid'] = df_intake_liking_overlap_10_energy['eid'].astype(str)
df_intake_liking_overlap_10_energy = df_intake_liking_overlap_10_energy.loc[df_intake_liking_overlap_10_energy['100002']/4.184 < 6000]
df_intake_liking_overlap_10_energy = df_intake_liking_overlap_10_energy.loc[df_intake_liking_overlap_10_energy['100002']/4.184 > 500]

corr = pd.read_csv('corr_files.txt', dtype = str, names = ['file'], header = None)
corr['eid'] = corr['file'].str[:7]

df_intake_liking_energy_noMRI = df_intake_liking_overlap_10_energy.loc[~df_intake_liking_overlap_10_energy['eid'].isin(corr['eid'])]
df_intake_liking_energy_noMRI.reset_index(drop = True, inplace = True)

df_intake_liking_energy_MRI = df_intake_liking_overlap_10_energy.loc[df_intake_liking_overlap_10_energy['eid'].isin(corr['eid'])]
df_intake_liking_energy_MRI = pd.merge(df_intake_liking_overlap_10_energy, corr, on = 'eid')

pd.merge(df_liking_overlap_10, df_energy_mean, on = 'eid').shape
pd.merge(pd.merge(df_liking_overlap_10, df_energy_mean, on = 'eid'), corr, on = 'eid').shape
pd.merge(pd.merge(pd.merge(df_liking_overlap_10, df_energy_mean, on = 'eid'), corr, on = 'eid'), df_intake_overlap_10, on = 'eid').shape

df_demo200 = df_intake_liking_energy_noMRI.sample(n = 200)
df_demo200.to_csv('demo200.csv', index = False)
df_nan = df_intake_liking_energy_noMRI.isnull().sum(axis = 1)
df_sort_nan = df_intake_liking_energy_noMRI.iloc[df_intake_liking_energy_noMRI.isnull().sum(axis=1).mul(-1).argsort()]
df_demo200 = df_sort_nan.tail(200)

df_demo200_full = pd.DataFrame(columns = ['consumption', 'liking', 'items', 'subjects', 'items_calorie', 'energy'])
intake_liking_overlap_item = pd.read_csv('intake_liking_overlap.csv')
intake_liking_overlap_item['intake'] = intake_liking_overlap_item['intake'].astype(str)
intake_liking_overlap_item['liking'] = intake_liking_overlap_item['liking'].astype(str)

df_demo200.reset_index(drop = True, inplace = True)
df_demo200_noNAN = df_demo200.fillna(df_demo200.mean())

df_demo200_full['subjects'] = np.array(df_demo200['eid']).repeat(38)
df_demo200_full['items'] = np.array(list(np.arange(1,39))*200)
df_demo200_full['items_calorie'] = np.array(list(intake_liking_overlap_item['cal_item'])*200)
df_demo200_full['energy'] = np.array(df_demo200['100002']/4.184).repeat(38)

liking_full = []
for i in range(0, df_demo200_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo200_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        liking_full.append(np.float(df_demo200_noNAN[df_demo200_noNAN['eid'] == df_demo200_full['subjects'].loc[i]][intake_liking_overlap_item['liking'].loc[j]]))

df_demo200_full['liking'] = np.array(liking_full)

consumption_full = []
for i in range(0, df_demo200_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo200_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        consumption_full.append(np.float(df_demo200_noNAN[df_demo200_noNAN['eid'] == df_demo200_full['subjects'].loc[i]][intake_liking_overlap_item['intake'].loc[j]]))

df_demo200_full['consumption'] = np.array(consumption_full)

df_demo200_full.reset_index(drop = True, inplace = True)

df_demo200_full = df_demo200_full[df_demo200_full['energy'] < 6000]
df_demo200_full = df_demo200_full[df_demo200_full['energy'] > 500]
df_demo200_full.to_csv('demo200_full.csv', index = False)
df_demo200_full = pd.read_csv('demo200_full.csv')

demo100_enegy = df_demo1000_full.drop_duplicates('subjects')
demo100_enegy = df_gmv_SCE[['BMI', 'energy']]

sns.set(style="ticks", color_codes=True)    
#g = sns.pairplot(df_demo2000_full[['liking', 'consumption']])
g = sns.pairplot(demo100_enegy, vars = ['BMI', 'energy'], plot_kws={'alpha':0.75}, diag_kws={'bins': 9}, aspect= 1.2)
#g.set(xlabel="BMI", ylabel = "Total energy intake")
plt.show()

df_energy_WHR = df_gmv_SCE[['BMI', 'waist_hip_ratio']]

sns.set(style="ticks", color_codes=True)    
#g = sns.pairplot(df_demo2000_full[['liking', 'consumption']])
g = sns.pairplot(df_energy_WHR, vars = ['BMI', 'waist_hip_ratio'], plot_kws={'alpha':0.75}, diag_kws={'bins': 9}, aspect= 1.2)
#g.set(xlabel="BMI", ylabel = "Total energy intake")
plt.show()

np.corrcoef(demo100_enegy['BMI'], demo100_enegy['energy'])

dfh = pd.DataFrame(df_demo200_full['consumption'])

sns.histplot(
    df_demo200_full, x="consumption",
    stat="density", common_norm=False,
)
dfh.hist(density = 1)
dfh.hist(weights=np.ones_like(dfh[dfh.columns[0]]) * 100. / len(dfh))
plt.hist(df_demo200_full['consumption'],stacked=False,  density=1)

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
df_demo2000 = df_intake_liking_energy_noMRI.sample(n = 10000)
df_demo2000.reset_index(drop = True, inplace = True)
#df_demo200.to_csv('demo2000.csv', index = False)
#df_nan = df_intake_liking_energy_noMRI.isnull().sum(axis = 1)
#df_sort_nan = df_intake_liking_energy_noMRI.iloc[df_intake_liking_energy_noMRI.isnull().sum(axis=1).mul(-1).argsort()]
#df_demo200 = df_sort_nan.tail(200)

df_demo2000_full = pd.DataFrame(columns = ['consumption', 'liking', 'items', 'subjects', 'items_calorie', 'energy'])
intake_liking_overlap_item = pd.read_csv('intake_liking_overlap.csv')
intake_liking_overlap_item['intake'] = intake_liking_overlap_item['intake'].astype(str)
intake_liking_overlap_item['liking'] = intake_liking_overlap_item['liking'].astype(str)

df_demo2000_full['subjects'] = np.array(df_demo2000['eid']).repeat(38)
df_demo2000_full['items'] = np.array(list(np.arange(1,39))*10000)
df_demo2000_full['items_calorie'] = np.array(list(intake_liking_overlap_item['cal_item'])*10000)
df_demo2000_full['energy'] = np.array(df_demo2000['100002']/4.184).repeat(38)

liking_full = []
for i in range(0, df_demo2000_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo2000_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        liking_full.append(np.float(df_demo2000[df_demo2000['eid'] == df_demo2000_full['subjects'].loc[i]][intake_liking_overlap_item['liking'].loc[j]]))

df_demo2000_full['liking'] = np.array(liking_full)

consumption_full = []
for i in range(0, df_demo2000_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo2000_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        consumption_full.append(np.float(df_demo2000[df_demo2000['eid'] == df_demo2000_full['subjects'].loc[i]][intake_liking_overlap_item['intake'].loc[j]]))

df_demo2000_full['consumption'] = np.array(consumption_full)

df_demo2000_full = df_demo2000_full[df_demo2000_full['energy'] < 6000]
df_demo2000_full = df_demo2000_full[df_demo2000_full['energy'] > 500]

df_demo2000_full.to_csv('demo10000_full.csv', index = False)

df_demo2000_full = pd.read_csv('demo10000_full.csv')

sns.set(style="ticks", color_codes=True)    
#g = sns.pairplot(df_demo2000_full[['liking', 'consumption']])
#g = sns.pairplot(df_demo2000_full, vars = ['liking', 'consumption'], plot_kws={'alpha':0.01}, diag_kws={'bins': [0,1,2,3,4,5,6,7,8,9,10,11]}, aspect= 1.2)
g = sns.pairplot(df_demo2000_full, vars = ['liking', 'consumption'], plot_kws={'alpha':0.01}, diag_kws={'bins': 9, 'range': (0, 10)}, aspect= 1.2)
plt.show()

plt.hist(df_demo2000_full[['liking', 'consumption']], bins = 9, range = (0, 10))
df_demo2000_full['consumption'].hist(bins = 9, range = (0, 10))

df_demo2000_full[['liking', 'consumption']].corr()
pp = pd.DataFrame(np.unique(df_demo2000_full['consumption'], return_counts = True))

df_demo2000_full_mean = pd.merge(df_demo2000_full.groupby('items', as_index=False)['consumption'].mean(), df_demo2000_full.groupby('items', as_index=False)['liking'].mean(), on = 'items')

sns.set(style="ticks", color_codes=True)    
g = sns.pairplot(df_demo2000_full_mean[['liking', 'consumption']])
plt.show()

plt.bar(df_demo2000_full['liking'], df_demo2000_full['consumption'])
------------------------------------------------------------------------------------------------------------------------------------------

You can make use of pd.cut to partition the values into bins corresponding to each interval and then take each interval's total counts using pd.value_counts. Plot a bar graph later, additionally replace the X-axis tick labels with the category name to which that particular tick belongs.

out = pd.cut(s, bins=[0, 0.35, 0.7, 1], include_lowest=True)
ax = out.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(6,4))
ax.set_xticklabels([c[1:-1].replace(","," to") for c in out.cat.categories])
plt.show()

date = pd.date_range('2017-02-23', periods=10*12, freq='2h')
freq = np.random.poisson(lam=2, size=(len(date)))
df = pd.DataFrame({"freq":freq}, index=date)

df["hours"] = df.index.hour
df["days"] = df.index.map(lambda x: x.strftime('%b-%d'))   

piv = pd.pivot_table(df, values="freq",index=["hours"], columns=["days"], fill_value=0)
#plot pivot table as heatmap using seaborn
ax = sns.heatmap(piv, square=True)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90 )
plt.tight_layout()
plt.show()

df_item = pd.pivot_table(df_demo2000_full, values="items_calorie",index=["liking"], columns=["consumption"])
ax = sns.heatmap(df_item, square=True)
plt.setp( ax.xaxis.get_majorticklabels(), rotation=90 )
plt.tight_layout()
plt.show()

--------------------------------------------------------------------------------------------------
df_BMI = df.filter(regex = '21001|23104|738|6138|845|31|21000|eid')

df_sex = df.filter(regex = '31|eid')[['eid', '31-0.0']]
df_sex_noNAN = df_sex.dropna(subset = df_sex.columns[1:],how = 'all')
df_sex_noNAN = df_sex_noNAN.rename(columns={'eid':'subjects', '31-0.0': 'sex'})
df_sex_noNAN.to_csv('sex.csv', index = False)

df_BMI = df.filter(regex = '21001|eid')
df_BMI_noNeg = df_BMI.mask(df_BMI < 0).dropna(subset = df_BMI.columns[1:],how = 'all')
df_BMI_noNeg.reset_index(drop = True, inplace = True)
df_BMI_noNeg['eid'] = df_BMI_noNeg['eid'].astype(str)
df_BMI_noNeg_noMRI = pd.merge(df_liking_energy_mri, df_BMI_noNeg, on = 'eid')
df_BMI_noNeg_noMRI['21001'] = 0
for i in range(df_BMI_noNeg_noMRI.shape[0]):
    df_BMI_noNeg_noMRI['21001'].loc[i] = np.nanmean(df_BMI_noNeg_noMRI[df_BMI_noNeg.columns[1:5]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_BMI_noNeg_noMRI.shape[0])), end="")

df_BMI_liking_energy_mri = df_BMI_noNeg_noMRI.drop(columns = df_BMI.columns[1:])

''''
df_BMI_online = pd.merge(df_BMI_noNeg, df_online[['eid', 'NAN']], on = 'eid')
df_BMI_online['21001'] = 0
for i in range(df_BMI_online.shape[0]):
    df_BMI_online['21001'].loc[i] = np.nanmean(df_BMI_online[df_BMI_online.columns[1:5]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_BMI_online.shape[0])), end="")

from scipy import stats
stats.pearsonr(df_BMI_online['21001'], df_BMI_online['NAN'])
'''

df_income = df.filter(regex = '738|eid').drop(columns = ['20738-0.0'])
df_income_noNeg = df_income.mask(df_income < 0).dropna(subset = df_income.columns[1:], how = 'all')
df_income_noNeg.reset_index(drop = True, inplace = True)
df_income_noNeg['eid'] = df_income_noNeg['eid'].astype(str)
df_income_noNeg_noMRI = pd.merge(df_BMI_noNeg_noMRI, df_income_noNeg, on = 'eid')
df_income_noNeg_noMRI['738'] = 0
for i in range(df_income_noNeg_noMRI.shape[0]):
    df_income_noNeg_noMRI['738'].loc[i] = np.nanmean(df_income_noNeg_noMRI[df_income_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_income_noNeg_noMRI.shape[0])), end="")
   
df_income_noNeg_noMRI = df_income_noNeg_noMRI.drop(columns = df_income.columns[1:])

df_edu = df.filter(regex = '6138|eid')
df_edu_noNeg = df_edu.mask(df_edu == -3).dropna(subset = df_edu.columns[1:],how = 'all')
df_edu_noNeg.reset_index(drop = True, inplace = True)
df_edu_noNeg['eid'] = df_edu_noNeg['eid'].astype(str)
df_edu_noNeg_noMRI = pd.merge(df_income_noNeg_noMRI, df_edu_noNeg, on = 'eid')
df_edu_noNeg_noMRI['6138'] = 9999
df_edu_noNeg_noMRI[df_edu_noNeg.columns[1:]] = df_edu_noNeg_noMRI[df_edu_noNeg.columns[1:]].replace(-7, 7)
for i in range(df_edu_noNeg_noMRI.shape[0]):      
    df_edu_noNeg_noMRI['6138'].loc[i] = np.nanmin(df_edu_noNeg_noMRI[df_edu_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_edu_noNeg_noMRI.shape[0])), end="")
df_edu_noNeg_noMRI = df_edu_noNeg_noMRI.drop(columns = df_edu.columns[1:])

df_edu_noNeg_noMRI.to_csv('liking_intake_BMI_edu_income_July9.csv', index = False)
'''    
for i in range(df_edu_noNeg_noMRI.shape[0]):
    if (df_edu_noNeg_noMRI['6138'].loc[i] > 3):
        df_edu_noNeg_noMRI['6138'].loc[i] = 4
    print("\r Process{0}%".format(round((i+1)*100/df_edu_noNeg_noMRI.shape[0])), end="")
'''
'''
df_edu1 = df.filter(regex = '845|eid')
df_edu1_noNeg = df_edu1.mask(df_edu1 < 0).dropna(subset = df_edu1.columns[1:], how = 'all')
df_edu1_noNeg.reset_index(drop = True, inplace = True)
df_edu1_noNeg['eid'] = df_edu1_noNeg['eid'].astype(str)
df_edu1_noNeg_noMRI = pd.merge(df_income_noNeg_noMRI, df_edu1_noNeg, on = 'eid')
df_edu1_noNeg_noMRI['845'] = 0
for i in range(df_edu1_noNeg_noMRI.shape[0]):
    df_edu1_noNeg_noMRI['845'].loc[i] = np.nanmax(df_edu1_noNeg_noMRI[df_edu1_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_edu1_noNeg_noMRI.shape[0])), end="")

df_BMI_edu_income_noMRI = df_edu1_noNeg_noMRI.drop(columns = ['21001-0.0', '21001-1.0', '21001-2.0','21001-3.0', '738-0.0', '738-1.0', '738-2.0', '738-3.0','845-0.0', '845-1.0', '845-2.0'])
df_BMI_edu_income_noMRI= df_BMI_edu_income_noMRI[df_BMI_edu_income_noMRI['100002']/4.184 < 6000]
df_BMI_edu_income_noMRI = df_BMI_edu_income_noMRI[df_BMI_edu_income_noMRI['100002']/4.184 > 500]
'''

df_demo1000 = df_edu_noNeg_noMRI.sample(n = 1000, random_state = 9)
df_demo1000.reset_index(drop = True, inplace = True)

for food in df_demo1000.columns[:41]:
    demo1000_min = np.nanmin(df_demo1000[food])
    for i in range(df_demo1000.shape[0]):
        if(df_demo1000[food].loc[i] == demo1000_min):
            df_demo1000[food].loc[i] = 0
        if(df_demo1000[food].loc[i] > demo1000_min):
            df_demo1000[food].loc[i] = 1
    print("\r Process{0}%".format(df_demo1000.columns.get_loc(food)*100/40), end="")
    

#df_demo200.to_csv('demo2000.csv', index = False)
#df_nan = df_intake_liking_energy_noMRI.isnull().sum(axis = 1)
#df_sort_nan = df_intake_liking_energy_noMRI.iloc[df_intake_liking_energy_noMRI.isnull().sum(axis=1).mul(-1).argsort()]
#df_demo200 = df_sort_nan.tail(200)

df_demo1000_full = pd.DataFrame(columns = ['consumption', 'liking', 'items', 'subjects', 'items_calorie', 'energy', 'income', 'edu', 'BMI'])
intake_liking_overlap_item = pd.read_csv('intake_liking_overlap.csv')
intake_liking_overlap_item['intake'] = intake_liking_overlap_item['intake'].astype(str)
intake_liking_overlap_item['liking'] = intake_liking_overlap_item['liking'].astype(str)

df_demo1000_full['subjects'] = np.array(df_demo1000['eid']).repeat(38)
df_demo1000_full['items'] = np.array(list(np.arange(1,39))*1000)
df_demo1000_full['items_calorie'] = np.array(list(intake_liking_overlap_item['cal_item'])*1000)
df_demo1000_full['energy'] = np.array(df_demo1000['100002']/4.184).repeat(38)
df_demo1000_full['BMI'] = np.array(df_demo1000['21001']).repeat(38)
df_demo1000_full['income'] = np.array(df_demo1000['738']).repeat(38)
df_demo1000_full['edu'] = np.array(df_demo1000['6138']).repeat(38)

liking_full = []
for i in range(0, df_demo1000_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo1000_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        liking_full.append(np.float(df_demo1000[df_demo1000['eid'] == df_demo1000_full['subjects'].loc[i]][intake_liking_overlap_item['liking'].loc[j]]))

df_demo1000_full['liking'] = np.array(liking_full)

consumption_full = []
for i in range(0, df_demo1000_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo1000_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        consumption_full.append(np.float(df_demo1000[df_demo1000['eid'] == df_demo1000_full['subjects'].loc[i]][intake_liking_overlap_item['intake'].loc[j]]))

df_demo1000_full['consumption'] = np.array(consumption_full)

df_demo1000_full.to_csv('demo1000_full_July9_binary.csv', index = False)

df_demo1000_full = pd.read_csv('demo1000_full.csv')
sum(df_demo1000_full['consumption'].isnull())/len(df_demo1000_full['consumption'])
df_demo2000_nan = df_demo2000_full['consumption'].isnull().sum(axis = 1)

df_online = pd.read_csv('intake_online.csv')
df_online = df_online.dropna(how='all', subset = ['eid'])
df_online['NAN'] = df_online.isnull().sum(axis = 1)
df_online['eid'] = df_online['eid'].astype(int)
df_online['eid'] = df_online['eid'].astype(str)
df_online = df_online[df_online.columns[1:]]

df_online_BMI = pd.merge(df_BMI_noNeg_noMRI, df_online, on = 'eid')[['eid', 'NAN', '21001']]

stats.pearsonr(df_online_BMI['21001'], df_online_BMI['NAN'])

df_sort = pd.DataFrame(columns=['code'])
df_sort['code'] = df_online.isnull().sum(axis = 0).sort_values(ascending = False).index
df_sort['NAN'] = np.asanyarray(df_online.isnull().sum(axis = 0).sort_values(ascending = False))
df_sort_name = pd.merge(df_sort, intake_online, on = 'code')
df_sort_name['NAN%'] = df_sort_name['NAN']/df_online.shape[0]

df_sort_name.to_csv('sort_NAN_name.csv', index = False)

lst = list(intake_liking_overlap_item['intake'])
lst.append('eid')
df_intake_38 = df_demo1000[lst]
df_sort_38 = pd.DataFrame(columns=['code'])
df_sort_38['code'] = df_intake_38.isnull().sum(axis = 0).sort_values(ascending = False).index
df_sort_38['NAN'] = np.asanyarray(df_intake_38.isnull().sum(axis = 0).sort_values(ascending = False))
df_sort_38_name = pd.merge(df_sort_38, intake_online, on = 'code')
df_sort_38_name['NAN%'] = df_sort_38_name['NAN']/df_intake_38.shape[0]

df_sort_38_name.to_csv('sort_NAN_name_38.csv', index = False)
stats.pearsonr(df_demo1000['NAN'], df_demo1000['21001'])

df_online['100150']

h = sns.boxplot(x="6138", y="21001", data=df_edu_noNeg_noMRI)
h.set(xlabel="Education level (1 is college degree)", ylabel = "BMI")

hh = sns.boxplot(x="6138", y="21001", data=df_demo1000)
hh.set(xlabel="Education level (1 is college degree)", ylabel = "BMI")

incom = sns.pairplot(df_edu_noNeg_noMRI, vars = ['21001', '100002'], plot_kws={'alpha':0.75})
incom.set(xlabel="Energy", ylabel = "BMI")
-----------------------------------------------------------------------------------------------------------
df_liking_mri = pd.merge(df_liking_overlap, corr, on = 'eid')
df_liking_mri = df_liking_mri[~df_liking_mri.duplicated(subset='eid')]
df_liking_mri_energy = pd.merge(df_liking_mri, df_energy_mean, on = 'eid')
df_liking_mri_energy_ecg = pd.merge(df_liking_mri_energy, df_ecg_noNAN, on = 'eid')

df_liking_ecg = pd.merge(df_liking_overlap, df_ecg_noNAN, on = 'eid')
df_mri_energy = pd.merge(corr, df_energy_mean, on = 'eid')
df_liking_energy_ecg = pd.merge(df_liking_ecg, df_energy_mean, on = 'eid')
df_liking_energy_mri = pd.merge(df_liking_mri, df_energy_mean, on = 'eid')
df_liking_energy_mri_ecg = pd.merge(df_liking_energy_mri, df_liking_ecg, on = 'eid')

df_BMI = df.filter(regex = '21001|eid')
df_BMI_noNeg = df_BMI.mask(df_BMI < 0).dropna(subset = df_BMI.columns[1:],how = 'all')
df_BMI_noNeg.reset_index(drop = True, inplace = True)
df_BMI_noNeg['eid'] = df_BMI_noNeg['eid'].astype(str)
df_BMI_noNeg_noMRI = pd.merge(df_liking_mri_energy_ecg, df_BMI_noNeg, on = 'eid')
df_BMI_noNeg_noMRI['21001'] = 0
for i in range(df_BMI_noNeg_noMRI.shape[0]):
    df_BMI_noNeg_noMRI['21001'].loc[i] = np.nanmean(df_BMI_noNeg_noMRI[df_BMI_noNeg.columns[1:5]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_BMI_noNeg_noMRI.shape[0])), end="")

df_BMI_liking_energy_mri = df_BMI_noNeg_noMRI.drop(columns = df_BMI.columns[1:])

df_age_sex = df[['eid', '31-0.0', '21003-0.0']]
df_age_sex_noNAN = df_age_sex.dropna(subset = df_age_sex.columns[1:],how = 'any')
df_age_sex_noNAN = df_age_sex_noNAN.rename(columns={'31-0.0': 'sex', '21003-0.0': 'age'})
df_age_sex_noNAN['eid'] = df_age_sex_noNAN['eid'].astype(str)

df_BMI_liking_energy_mri_age_sex = pd.merge(df_BMI_liking_energy_mri, df_age_sex_noNAN, on = 'eid')

df_ecg = pd.read_csv('./ECGdata/ukb47672.csv', encoding = 'unicode_escape')[['eid', '22333-2.0']]
df_ecg_noNAN = df_ecg.dropna(subset = df_ecg.columns[1:],how = 'all')
df_ecg_noNAN['eid'] = df_ecg_noNAN['eid'].astype(str)

df_ecg_BMI_liking_energy_mri_age_sex = pd.merge(df_BMI_liking_energy_mri_age_sex, df_ecg_noNAN, on = 'eid')

df_waist = df[['eid', '48-0.0', '48-1.0', '48-2.0', '48-3.0']]
df_waist_noNeg = df_waist.mask(df_waist < 0).dropna(subset = df_waist.columns[1:], how = 'all')
df_waist_noNeg.reset_index(drop = True, inplace = True)
df_waist_noNeg['eid'] = df_waist_noNeg['eid'].astype(str)
df_waist_noNeg_noMRI = pd.merge(df_SCE_gmv_2000, df_waist_noNeg, on = 'eid')
#df_waist_noNeg_noMRI = pd.merge(df_BMI_liking_energy_mri_age_sex, df_waist_noNeg, on = 'eid')
df_waist_noNeg_noMRI['48'] = 0
for i in range(df_waist_noNeg_noMRI.shape[0]):
    df_waist_noNeg_noMRI['48'].loc[i] = np.nanmean(df_waist_noNeg_noMRI[df_waist_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_waist_noNeg_noMRI.shape[0])), end="")
   
df_waist_ecg_BMI_liking_energy_mri_age_sex = df_waist_noNeg_noMRI.drop(columns = df_waist.columns[1:])

df_hip = df[['eid', '49-0.0', '49-1.0', '49-2.0', '49-3.0']]
df_hip_noNeg = df_hip.mask(df_hip < 0).dropna(subset = df_hip.columns[1:], how = 'all')
df_hip_noNeg.reset_index(drop = True, inplace = True)
df_hip_noNeg['eid'] = df_hip_noNeg['eid'].astype(str)
df_hip_noNeg_noMRI = pd.merge(df_waist_ecg_BMI_liking_energy_mri_age_sex, df_hip_noNeg, on = 'eid')
df_hip_noNeg_noMRI['49'] = 0
for i in range(df_hip_noNeg_noMRI.shape[0]):
    df_hip_noNeg_noMRI['49'].loc[i] = np.nanmean(df_hip_noNeg_noMRI[df_hip_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_hip_noNeg_noMRI.shape[0])), end="")
   
df_hipwaist_ecg_BMI_liking_energy_mri_age_sex = df_hip_noNeg_noMRI.drop(columns = df_hip.columns[1:])
df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['waist_hip_ratio'] = df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['48']/df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['49']

tttt = df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['waist_hip_ratio'][df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['waist_hip_ratio'] == 0.9]
df_SCE_gmv_2000[df_SCE_gmv_2000['eid'] == '3290038']['waist_hip_ratio']

df_hipwaist_ecg_BMI_liking_energy_mri_age_sex.to_csv('liking_mri_BMI_ECG_control.csv', index = False)

df_sport = df[['eid', '991-0.0', '991-1.0', '991-2.0', '991-3.0']]
df_sport_noNeg = df_sport.mask(df_sport < 0).dropna(subset = df_sport.columns[1:], how = 'all')
df_sport_noNeg.reset_index(drop = True, inplace = True)
df_sport_noNeg['eid'] = df_sport_noNeg['eid'].astype(str)
df_sport_noNeg_noMRI = pd.merge(df_hipwaist_ecg_BMI_liking_energy_mri_age_sex, df_sport_noNeg, on = 'eid')
df_sport_noNeg_noMRI['991'] = 0
for i in range(df_sport_noNeg_noMRI.shape[0]):
    df_sport_noNeg_noMRI['991'].loc[i] = np.nanmean(df_sport_noNeg_noMRI[df_sport_noNeg.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_sport_noNeg_noMRI.shape[0])), end="")
   
df_sport_hipwaist_ecg_BMI_liking_energy_mri_age_sex = df_sport_noNeg_noMRI.drop(columns = df_sport.columns[1:])

df_hipwaist_ecg_BMI_liking_energy_mri_age_sex.to_csv('foods38_liking_mri_BMI_ECG_control.csv', index = False)

df_demo = df_hipwaist_ecg_BMI_liking_energy_mri_age_sex.drop(['49', '48', '21001', 'waist-hip ratio', '22333-2.0', 'file', 'sex', 'age'], axis = 1)
df_demo.to_csv('food38_liking_mri_ECG.csv', index = False)

df_demo_control_energy500_6000 = df_hipwaist_ecg_BMI_liking_energy_mri_age_sex[df_hipwaist_ecg_BMI_liking_energy_mri_age_sex['100002']/4.184 < 6000]
df_demo_control_energy500_6000 = df_demo_control_energy500_6000[df_demo_control_energy500_6000['100002']/4.184 > 500]
df_demo_control = df_demo_control_energy500_6000[['eid', '21001', 'waist-hip ratio', '22333-2.0', 'file', 'sex', 'age']]
df_demo_control = df_demo_control.rename(columns={"21001": "BMI", "22333-2.0": "ecg", "eid": "subjects"})
df_demo_control.to_csv('demo_all_control.csv', index = False)

df_demo_full = pd.DataFrame(columns = ['liking', 'items', 'subjects', 'items_calorie', 'energy'])
intake_liking_overlap_item = pd.read_csv('intake_liking_overlap.csv')
intake_liking_overlap_item['intake'] = intake_liking_overlap_item['intake'].astype(str)
intake_liking_overlap_item['liking'] = intake_liking_overlap_item['liking'].astype(str)

df_demo_full['subjects'] = np.array(df_demo['eid']).repeat(38)
df_demo_full['items'] = np.array(list(np.arange(1,39))*df_demo.shape[0])
df_demo_full['items_calorie'] = np.array(list(intake_liking_overlap_item['cal_item'])*df_demo.shape[0])
df_demo_full['energy'] = np.array(df_demo['100002']/4.184).repeat(38)

liking_full = []
for i in range(0, df_demo_full.shape[0], 38):
    print("\r Process{0}%".format(round((i+1)*100/df_demo_full.shape[0])), end="")
    for j in range(intake_liking_overlap_item.shape[0]):
        liking_full.append(np.float(df_demo[df_demo['eid'] == df_demo_full['subjects'].loc[i]][intake_liking_overlap_item['liking'].loc[j]]))

df_demo_full['liking'] = np.array(liking_full)

df_demo_full = df_demo_full[df_demo_full['energy'] < 6000]
df_demo_full = df_demo_full[df_demo_full['energy'] > 500]
df_demo_full.to_csv('demo_all_H1.csv', index = False)

df_height = df.filter(regex = '12144|eid')
df_height['eid'] = df_height['eid'].astype(str)
#df_height_noNeg = df_height.mask(df_height[df_height.columns[1:]] < 0).dropna(subset = df_height.columns[1:],how = 'all')
df_height_noNAN = df_height.dropna(subset = df_height.columns[1:],how = 'all')
df_height_noNAN.reset_index(drop = True, inplace = True)
df_height_noNAN['height'] = 0

for i in range(df_height_noNAN.shape[0]):
    df_height_noNAN['height'].loc[i] = np.nanmean(df_height_noNAN[df_height_noNAN.columns[1:3]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_height_noNAN.shape[0])), end="")

df_height_noNAN.to_csv('height_mean.csv', index = False)
df_height_noNAN = pd.read_csv('height_mean.csv')
df_height_1 = df_height_noNAN[['eid', 'height']]
df_height_1['height'] = stats.zscore(df_height_1['height'])
df_SCE_update = pd.merge(df_SCE, df_height_1, on = 'eid')

df_hand = df.filter(regex = '1707-0.0|eid')
df_hand_noNAN = df_hand.mask(df_hand < 0).dropna(subset = df_hand.columns[1:], how = 'all')
df_hand_noNAN['eid'] = df_hand_noNAN['eid'].astype(str)
df_hand_noNAN = df_hand_noNAN.rename(columns={'1707-0.0': 'hand'})
df_hand_noNAN.reset_index(drop = True, inplace = True)
df_SCE_update = pd.merge(df_SCE_update, df_hand_noNAN, on = 'eid')

df_education = df.filter(regex = '6138-2.|eid')
df_education_noNAN = df_education.mask(df_education < 0).dropna(subset = df_education.columns[1:], how = 'all')
df_education_noNAN['eid'] = df_education_noNAN['eid'].astype(str)
df_education_noNAN.reset_index(drop = True, inplace = True)
df_education_noNAN['edu'] = 7

for i in range(df_education_noNAN.shape[0]):
    df_education_noNAN['edu'].loc[i] = np.nanmin(df_education_noNAN[df_education_noNAN.columns[1:]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_education_noNAN.shape[0])), end="")

df_education_noNAN['edu'] = stats.zscore(df_education_noNAN['edu'])
df_SCE_update = pd.merge(df_SCE_update, df_education_noNAN[['eid', 'edu']], on = 'eid')

df_IQ = df.filter(regex = '20016-2|eid')
df_IQ_noNAN = df_IQ.mask(df_IQ < 0).dropna(subset = df_IQ.columns[1:], how = 'all')
df_IQ_noNAN['eid'] = df_IQ_noNAN['eid'].astype(str)
df_IQ_noNAN = df_IQ_noNAN.rename(columns={'20016-2.0': 'IQ'})
df_IQ_noNAN.reset_index(drop = True, inplace = True)
df_IQ_noNAN['IQ'] = stats.zscore(df_IQ_noNAN['IQ'])
df_SCE_update = pd.merge(df_SCE_update, df_IQ_noNAN[['eid', 'IQ']], on = 'eid')

df_income = df.filter(regex = '738|eid').drop(columns = ['20738-0.0'])
df_income_noNAN = df_income.mask(df_income < 0).dropna(subset = df_income.columns[1:], how = 'all')
df_income_noNAN.reset_index(drop = True, inplace = True)
df_income_noNAN['eid'] = df_income_noNAN['eid'].astype(str)
df_income_noNAN['income'] = 0

for i in range(df_income_noNAN.shape[0]):
    df_income_noNAN['income'].loc[i] = np.nanmean(df_income_noNAN[df_income_noNAN.columns[1:5]].loc[i])
    print("\r Process{0}%".format(round((i+1)*100/df_income_noNAN.shape[0])), end="")
   
df_income_noNAN['income'] = stats.zscore(df_income_noNAN['income'])
df_SCE_update = pd.merge(df_SCE_update, df_income_noNAN[['eid', 'income']], on = 'eid')
   
df_house = df.filter(regex = '709-0.0|eid').drop(columns = ['20709-0.0'])
df_house_noNAN = df_house.mask(df_house < 0).dropna(subset = df_house.columns[1:], how = 'all')
df_house_noNAN['eid'] = df_house_noNAN['eid'].astype(str)
df_house_noNAN.reset_index(drop = True, inplace = True)
df_house_noNAN = df_house_noNAN.rename(columns={'709-0.0': 'household'})
df_house_noNAN['household'] = stats.zscore(df_house_noNAN['household'])  
df_SCE_update = pd.merge(df_SCE_update, df_house_noNAN[['eid', 'household']], on = 'eid')

df_genetic = df.filter(regex = '22009|eid')
df_genetic_noNAN = df_genetic.dropna(subset = df_genetic.columns[1:], how = 'any')
df_genetic_noNAN['eid'] = df_genetic_noNAN['eid'].astype(str)
df_genetic_noNAN[list(df_genetic_noNAN.columns[1:])] = df_genetic_noNAN[list(df_genetic_noNAN.columns[1:])].apply(stats.zscore)
df_genetic_noNAN.to_csv('df_genetic.csv', index = False)
df_SCE_update1 = pd.merge(df_SCE_update, df_genetic_noNAN, on = 'eid')
df_SCE_update1['sexage'] = df_SCE_update1['sex']*df_SCE_update1['age']

df_SCE_update1.to_csv('df_SCE_update1.csv', index = False)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

df_SCE = pd.read_csv('X_value_control.csv', index_col=0)
df_SCE = df_SCE.rename(columns={"subjects": "eid"})
df_SCE['eid'] = df_SCE['eid'].astype(str)

IC1 = np.loadtxt('1000048_25751_2_0.txt')
size = 55
corr_matrix = np.zeros((size,size))
corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC1
corr_matrix = corr_matrix + corr_matrix.T

df_SCE_2000 = df_SCE.sample(n = 2000, random_state = 99)
df_SCE_2000.reset_index(drop = True, inplace = True)
IC55 = []
for i in range(0, df_SCE_2000.shape[0]):
    tem = stats.zscore(np.loadtxt(df_SCE_2000['file'].loc[i]))
    IC55.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_2000.shape[0])), end="")

IC55_sex_slope = []
for i in range(0, df_SCE_2000.shape[0]):
    #tem = np.append(np.append(stats.zscore(np.loadtxt(df_SCE_2000['file'].loc[i])), df_SCE_2000['slope'].loc[i]), df_SCE_2000['sex'].loc[i])
    tem = np.append(stats.zscore(np.loadtxt(df_SCE_2000['file'].loc[i])), df_SCE_2000['sex'].loc[i])
    IC55_sex_slope.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_SCE_2000.shape[0])), end="")

from scipy import stats

#X = np.array(IC55)
X = np.array(IC55_sex_slope)
#y = df_SCE_2000['SCE']
y = stats.zscore(df_SCE_2000['energy'])
#y = stats.zscore(df_SCE_2000['slope'])

r2e = []
r2svr = []

outer_cv = KFold(n_splits = 5, shuffle = True, random_state = 20)
inner_cv = KFold(n_splits = 5, shuffle = True, random_state = 20)
p_grid = {'alpha': [1e-2, 1e-1, 1e-8, 1e-4, 1e-6]}
p_gridsvr = {'C': [0.01, 0.1, 1.0, 10.0, 100.0]}

for j, (train, test) in enumerate(outer_cv.split(X, y)):
    #split dataset to decoding set and test set
    x_train, x_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    #find optim paramater setting in the inner cv
    clf =  GridSearchCV(estimator = ElasticNet(tol = 1e-1), param_grid = p_grid, cv = inner_cv, scoring = "r2")
    clf.fit(x_train, y_train)
    print(j)

    #predict labels on the test set
    y_pred = clf.predict(x_train)
    print(r2_score(y_train, y_pred))
    #print(y_pred)
    #calculate metrics
    r2e.append(r2_score(y_train, y_pred))
    
    #predict labels on the test set
    y_pred = clf.predict(x_test)
    print(r2_score(y_test, y_pred))
    #print(y_pred)
    #calculate metrics
    r2e.append(r2_score(y_test, y_pred))

print(r2e)

for j, (train, test) in enumerate(outer_cv.split(X, y)):
    #split dataset to decoding set and test set
    x_train, x_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    #find optim paramater setting in the inner cv
    clf =  GridSearchCV(estimator = SVR(), param_grid = p_gridsvr, cv = inner_cv, scoring = "r2")
    clf.fit(x_train, y_train)
    print(j)
    
    #predict labels on the test set
    y_pred = clf.predict(x_test)
    #calculate metrics
    #print(r2_score(y_test, y_pred))
    #print(y_pred)
    r2svr.append(r2_score(y_test, y_pred))

print(r2svr)


gmv = np.genfromtxt('GMV.txt', dtype='str')
gmv_l = np.char.add(gmv, '-2.0')
df_gmv = df.filter(regex = convstr(gmv_l, '|'))
df_gmv['eid'] = df['eid']
df_gmv_noNAN = df_gmv.dropna(how='all', subset = df_gmv.columns[:-1])
df_gmv_noNAN.reset_index(drop = True, inplace = True)
df_gmv_noNAN['eid'] = df_gmv_noNAN['eid'].astype(str)

df_gmv_noNAN_scaled = preprocessing.scale(df_gmv_noNAN[gmv_l])
pca = PCA(n_components = 'mle')
pca = PCA(n_components = 0.63)
gmv_pca = pca.fit_transform(df_gmv_noNAN_scaled)
print(pca.explained_variance_ratio_)
len(pca.explained_variance_ratio_)

def plotCumSumVariance(var=None):
    #PLOT FIGURE
    #You can use plot_color[] to obtain different colors for your plots
    #Save file
    cumvar = var.cumsum()

    plt.figure()
    plt.bar(np.arange(len(var)), cumvar, width = 1.0)
    plt.axhline(y=0.64, color='r', linestyle='-')

plotCumSumVariance(pca.explained_variance_ratio_)

for j in range(len(gmv)):
    phe = [col for col in df_gmv.columns if gmv[j] in col]
    df_gmv_noNAN[gmv[j]] = 0
    print(j)
    for i in range(df_gmv_noNAN.shape[0]):
        df_gmv_noNAN[gmv[j]].loc[i] = np.nanmean(df_gmv_noNAN[phe].loc[i])

df_gmv_noNAN_mean = pd.DataFrame(columns = gmv)

for j in range(len(gmv)):
    phe = [col for col in df.columns if gmv[j] in col]
    df_gmv_noNAN_mean[gmv[j]] = 0
    print(j)
    for i in range(df_gmv_noNAN.shape[0]):
        df_gmv_noNAN_mean[gmv[j]].loc[i] = np.nanmean(df_gmv_noNAN[phe].loc[i])
 
-------------------------------------------------------------------------------------------------
from ECGXMLReader import ECGXMLReader

ecg = ECGXMLReader('./ECGtime/1000351_20205_2_0.xml', augmentLeads=True)

print( ecg.getLeadVoltages('I') )

from GEMuseXMLReader import GEMuseXMLReader

GEMuseData = GEMuseXMLReader('./ECGtime/1000351_20205_2_0.xml')

GEMuseData.header ## Header containing the patient, device and acquisition session parameters

GEMuseData.dataObject ## Dictionary containing the data separated by lead

GEMuseData.dataFrame ## Panda's data frame containg the acquisition data

GEMuseData.dataArray ## Numpy matrix containing the acquisition data