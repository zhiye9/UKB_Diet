#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:25:24 2021

@author: zhye
"""

'''np.array(list(cen.degree_centrality(binary_edges_centrality(nx.from_numpy_array(corr_matrix), p = True)).values())).shape
GTTTTT = binary_edges_centrality(nx.from_numpy_array(corr_matrix), p = True)
GTTTTT.number_of_nodes()
G = nx.Graph()
nx.add_path(G, [0, 1, 2])
nx.add_path(G, [0, 10, 2])
print([p for p in nx.all_shortest_paths(G, source=0, target=2)])

nx.to_numpy_array(FG)
nx.draw(GTTTTT)
nx.draw(nx.from_numpy_array(corr_matrix))

G = nx.path_graph(5)

length = dict(nx.all_pairs_dijkstra_path_length(G))
for node in [0, 1, 2, 3, 4]:
    print(f"1 - {node}: {length[1][node]}")

eff.global_efficiency

FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (1, 4, 0.5), (2, 4, 1.2), (3, 4, 0.375)])

IC = np.loadtxt(filename)
size = n_ICs
corr_matrix = np.zeros((size,size))
corr_matrix[np.triu_indices(corr_matrix.shape[0], k = 1)] = IC1
corr_matrix = corr_matrix + corr_matrix.T

G9 = nx.from_numpy_array(corr_matrix)

nx.all_pairs_dijkstra_path_length(G)

filename = x_train_all['file'].loc[i]

ap = global_eff(nx.from_numpy_array(corr_matrix), p = True)
bp = local_eff(nx.from_numpy_array(corr_matrix), p = True)
an = global_eff(nx.from_numpy_array(corr_matrix), p = False)
bn = local_eff(nx.from_numpy_array(corr_matrix), p = False)

pos=nx.spring_layout(G)

pos_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']>0]
neg_edges = [(u,v,w) for (u,v,w) in G1.edges(data=True) if w['weight']<0]
neg_edges[0:9]

for i in range(len(neg_edges)):
    neg_edges[i][2]['weight'] = abs(neg_edges[i][2]['weight'])
neg_edges[0:9]
    
neg_pos_edges = list(map(edge_abs, neg_edges))

Hpos = nx.Graph()
Hneg = nx.Graph()

Hpos.add_edges_from(pos_edges)
Hneg.add_edges_from(neg_edges)'''
