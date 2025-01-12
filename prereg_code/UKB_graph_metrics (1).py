#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 03:43:59 2021

@author: zhye
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import centrality as cen
from networkx.algorithms import efficiency_measures as eff
import copy

#Only keep edges with positive correlation
def pos_net(correlation_matrix):
  correlation_matrix[correlation_matrix < 0] = 0
  return correlation_matrix

def neg_net(correlation_matrix):
  correlation_matrix[correlation_matrix > 0] = 0
  return abs(correlation_matrix)

def reciprocal_net(correlation_matrix):
  correlation_matrix_rep = copy.copy(correlation_matrix)
  mask = correlation_matrix_rep != 0
  correlation_matrix_rep[mask] = np.reciprocal(correlation_matrix_rep[mask])
  return correlation_matrix_rep

#Convert correlation matrix to adjacency matrix in networkx format
def adj_netx(correlation_matrix):
  return nx.from_numpy_array(correlation_matrix)

def weight_central(G1):
    return np.array(G1.degree(weight = 'weight'))[:,1]

def efficiency_mean(G1):
    eff = []
    for i in range(55):
        short_path_length = nx.single_source_dijkstra_path_length(G1, source=i, weight = 'weight')
        eff.append(sum(short_path_length.values())/(len(short_path_length) - 1))
    return np.array(eff)

def Graph_metrics(filename, n_ICs, eff = False):
    IC = np.loadtxt(filename)
    size = n_ICs
    corr_matrix = np.zeros((size,size))
    corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC
    corr_matrix = corr_matrix + corr_matrix.T
    if eff == True:
        return np.concatenate((weight_central(adj_netx(pos_net(copy.copy(corr_matrix)))), weight_central(adj_netx(neg_net(copy.copy(corr_matrix)))), efficiency_mean(adj_netx(reciprocal_net(pos_net(copy.copy(corr_matrix))))), efficiency_mean(adj_netx(neg_net(reciprocal_net(copy.copy(corr_matrix)))))), axis = None)
    elif eff == False:
        return np.concatenate((weight_central(adj_netx(pos_net(copy.copy(corr_matrix)))), weight_central(adj_netx(neg_net(copy.copy(corr_matrix))))), axis = None)
