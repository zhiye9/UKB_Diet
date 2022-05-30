# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

df = pd.read_csv('ukb44644.csv', encoding = 'unicode_escape')

carb = [col for col in df.columns if '100005' in col]
df_carb = df[carb]
df_carb['eid'] = df['eid']

df_carb_noNAN = df_carb.dropna(how='all', subset=carb)
df_carb_noNAN.reset_index(drop = True, inplace = True)
df_carb_noNAN['kilocalories_carb'] = 0
for i in range(df_carb_noNAN.shape[0]):
    df_carb_noNAN['kilocalories_carb'].loc[i] = np.nanmean(df_carb_noNAN[carb].loc[i])*4


protein = [col for col in df.columns if '100003' in col]
df_protein = df[protein]
df_protein['eid'] = df['eid']

df_protein_noNAN = df_protein.dropna(how='all', subset=protein)
df_protein_noNAN.reset_index(drop = True, inplace = True)
df_protein_noNAN['kilocalories_protein'] = 0
for i in range(df_protein_noNAN.shape[0]):
    df_protein_noNAN['kilocalories_protein'].loc[i] = np.nanmean(df_protein_noNAN[protein].loc[i])*4


fat = [col for col in df.columns if '100004' in col]
df_fat = df[fat]
df_fat['eid'] = df['eid']

df_fat_noNAN = df_fat.dropna(how='all', subset=fat)
df_fat_noNAN.reset_index(drop = True, inplace = True)
df_fat_noNAN['kilocalories_fat'] = 0
for i in range(df_fat_noNAN.shape[0]):
    df_fat_noNAN['kilocalories_fat'].loc[i] = np.nanmean(df_fat_noNAN[fat].loc[i])*9
    
    
sugar = [col for col in df.columns if '100008' in col]
df_sugar = df[sugar]
df_sugar['eid'] = df['eid']

df_sugar_noNAN = df_sugar.dropna(how='all', subset=sugar)
df_sugar_noNAN.reset_index(drop = True, inplace = True)
df_sugar_noNAN['kilocalories_sugar'] = 0
for i in range(df_sugar_noNAN.shape[0]):
    df_sugar_noNAN['kilocalories_sugar'].loc[i] = np.nanmean(df_sugar_noNAN[sugar].loc[i])*4


alcohol = [col for col in df.columns if '100022' in col]
df_alcohol = df[alcohol]
df_alcohol['eid'] = df['eid']

df_alcohol_noNAN = df_alcohol.dropna(how='all', subset=alcohol)
df_alcohol_noNAN.reset_index(drop = True, inplace = True)
df_alcohol_noNAN['kilocalories_alcohol'] = 0
for i in range(df_alcohol_noNAN.shape[0]):
    df_alcohol_noNAN['kilocalories_alcohol'].loc[i] = np.nanmean(df_alcohol_noNAN[alcohol].loc[i])*7
    

energy = [col for col in df.columns if '100002' in col]
df_energy = df[energy]
df_energy['eid'] = df['eid']

df_energy_noNAN = df_energy.dropna(how='all', subset=energy)
df_energy_noNAN.reset_index(drop = True, inplace = True)
df_energy_noNAN['kilocalories_energy'] = 0
for i in range(df_energy_noNAN.shape[0]):
    df_energy_noNAN['kilocalories_energy'].loc[i] = np.nanmean(df_energy_noNAN[energy].loc[i])/4.184

df_energy_noNAN.loc[df_energy_noNAN['kilocalories_energy'] < 500)] 
df_energy_noNAN.loc[df_energy_noNAN['kilocalories_energy'] < 6000)] 

RR_interval = [col for col in df.columns if '22333' in col]
