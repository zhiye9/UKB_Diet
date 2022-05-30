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

#IC1 = np.loadtxt('1000048_25751_2_0.txt')

def weight_central(G1, p):
    if p == True:
        pos_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']>0]
        G = nx.Graph()
        G.add_edges_from(pos_edges)
    elif (p == False):
        neg_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']<0]
#       for i in range(len(neg_edges)):
#          neg_edges[i][2]['weight'] = abs(neg_edges[i][2]['weight'])
        G = nx.Graph()
        G.add_edges_from(neg_edges)    
    return np.array(G.degree(weight = 'weight'))[:,1]
    
def global_eff(G1, p):
    if p == True:
        pos_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']>0]
        G = nx.Graph()
        G.add_edges_from(pos_edges)
    elif (p == False):
        neg_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']<0]
        for i in range(len(neg_edges)):
            neg_edges[i][2]['weight'] = abs(neg_edges[i][2]['weight'])
        G = nx.Graph()
        G.add_edges_from(neg_edges)
        
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_bellman_ford_path_length(G)
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
        # g_eff = sum(1 / d for s, tgts in lengths
        #                   for t, d in tgts.items() if d > 0) / denom
    else:
        g_eff = 0
        # TODO This can be made more efficient by computing all pairs shortest
    # path lengths in parallel.
    return g_eff

def local_eff(G1, p):
    efficiency_list = (global_eff(G1.subgraph(G1[v]), p) for v in G1)
    return sum(efficiency_list) / len(G1)

def binary_edges(G1, p):
    Gsorted_edges = sorted(G1.edges(data=True), key=lambda x: x[2].get('weight', 1), reverse = True)
    G = nx.Graph()
    if (p == True):     
        G.add_edges_from(Gsorted_edges[:350])
    elif (p == False):
        G.add_edges_from(Gsorted_edges[-350:])
    return G

def Graph_metrics(filename, n_ICs):
    IC = np.loadtxt(filename)
    size = n_ICs
    corr_matrix = np.zeros((size,size))
    corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC
    corr_matrix = corr_matrix + corr_matrix.T
    return np.concatenate((weight_central(nx.from_numpy_array(corr_matrix), p = True), weight_central(nx.from_numpy_array(corr_matrix), p = False), global_eff(nx.from_numpy_array(corr_matrix), p = True), 
                global_eff(nx.from_numpy_array(corr_matrix), p = False), local_eff(nx.from_numpy_array(corr_matrix), p = True),
                local_eff(nx.from_numpy_array(corr_matrix), p = False)), axis = None)

def Graph_binary_metrics(filename, n_ICs):
    IC = np.loadtxt(filename)
    size = n_ICs
    corr_matrix = np.zeros((size,size))
    corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC
    corr_matrix = corr_matrix + corr_matrix.T
    return np.concatenate((np.array(list(cen.degree_centrality(binary_edges(nx.from_numpy_array(corr_matrix), p = True)).values())), np.array(list(cen.degree_centrality(binary_edges(nx.from_numpy_array(corr_matrix), p = False)).values())),  
                np.array(eff.local_efficiency(binary_edges(nx.from_numpy_array(corr_matrix), p = True))), np.array(eff.local_efficiency(binary_edges(nx.from_numpy_array(corr_matrix), p = False))), 
                np.array(eff.global_efficiency(binary_edges(nx.from_numpy_array(corr_matrix), p = True))), np.array(eff.global_efficiency(binary_edges(nx.from_numpy_array(corr_matrix), p = False)))), axis = None)

