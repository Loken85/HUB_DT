#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 09:24:41 2022

@author: proxy_loken

Clustering functions and utilities for unsupervised clustering of behaviour/neural data using HDBSCAN



"""

import scipy.io
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import cluster_metrics
import b_utils

import hdbscan

import pandas as pd






# helper for plotting hdb labelling in t-sne space
# takes in labels and probabilities, points are desaturated based on probs
# if noise is True, plots noise points in grey, if False noise is cut(plots in white)
# if compare to is True, plots the provided labels over top for comparison
def plot_hdb_over_tsne(t_out, hdb_labels, hdb_probs, noise=False, compare_to=False, comp_label=0):
    
    if noise:
        noise_color = (0.5,0.5,0.5)
    else:
        noise_color = (1,1,1)
    
    hdb_labels = hdb_labels.astype(int)
    
    n_clusts = max(hdb_labels)+1
    color_palette = sns.color_palette('dark', n_colors=n_clusts)
    cluster_colors = [color_palette[x] if x>=0
                      else noise_color
                      for x in hdb_labels]
    cluster_member_colors = [sns.desaturate(x,p) for x, p in
                             zip(cluster_colors, hdb_probs)]
    fig, ax = plt.subplots()
    ax.scatter(t_out[:,0],t_out[:,1], s=10, linewidth=0, c=cluster_member_colors, alpha=0.25)
    if compare_to:
        comp_points = t_out[comp_label,:]
        ax.scatter(comp_points[:,0], comp_points[:,1], s=50, marker='x', c='black', alpha=0.25)
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    
    return fig


def plot_condensed_tree(clusterer, select_clusts=False, label_clusts=False, draw_epsilon=False, epsilon=0, n_clusts=0):
    
    fig, ax = plt.subplots()
    
    clusterer.condensed_tree_.plot(select_clusters=select_clusts, label_clusters=label_clusts, selection_palette = sns.color_palette('dark', n_clusts))
    if draw_epsilon:
        if epsilon > 0:
            lam = 1/epsilon
        else:
            lam = 0
        
        ax.axhline(y=lam, color='red', linestyle = '-')
    
    return fig

    

# function for searching parameter ranges for best fits based on purity
# TODO: refactor for variable metric of comparison
# INPUTS: data - array to cluster(samples x dims), true_labels - labels for purity comparison
# min_sizes, min_samps - parameters to search over, mode - '3_contexts' for default 3 context setup, 'single' otherwise
# NOTE: only computes split_join distance if '3_contexts' is selected(sj_dists have no meaning when compared to a single category/context) 
def hdb_parameter_search(data, true_labels, min_sizes, min_samps, ignore_noise=False, mode='3_contexts', selection='leaf'):
    
    results = []
    
    purity_list = []
    
    distances = []
    
    for i in range(len(min_sizes)):
        
        for j in range(len(min_samps)):
            
            
            hdb_clust = hdbscan.HDBSCAN(min_cluster_size=min_sizes[i], min_samples=min_samps[j], cluster_selection_method=selection)
            
            hdb_clust.fit(data)
            
            labels_hdb = hdb_clust.labels_
            
            if mode == '3_contexts':
                purity = b_utils.compute_overall_purity(b_utils.compute_context_purity(labels_hdb,true_labels,ignore_noise,mode))
                dists, max_a, max_b = cluster_metrics.split_join_distance(true_labels, labels_hdb,ignore_noise)
                sj_dist = cluster_metrics.compute_relative_distance(dists, true_labels)
                num_clusts = max(labels_hdb)+1
                clf = {'type': 'HDBscan', 'classifier': hdb_clust, 'min_cluster_size': min_sizes[i], 'min_samples': min_samps[j], 'purity': purity, 's-j distance': sj_dist, 'num_clusts': num_clusts}
                purity_list.append(purity)
                distances.append(sj_dist)
            elif mode == 'single':
                purities = b_utils.compute_context_purity(labels_hdb,true_labels,ignore_noise,mode)
                purity_clust = b_utils.compute_max_purity(purities,comp='clust')
                purity_context = b_utils.compute_max_purity(purities,comp='context')
                num_clusts = max(labels_hdb)+1
                clf = {'type': 'HDBscan', 'classifier': hdb_clust, 'min_cluster_size': min_sizes[i], 'min_samples': min_samps[j], 'purity_clust': purity_clust, 'purity_context': purity_context, 'num_clusts': num_clusts}
                purity_comb = np.stack((purity_clust, purity_context))
                purity_list.append(purity_comb)
            
            results.append(clf)
            
    
    return results, purity_list, distances


# function to build and train a HDB cluster model
# INPUTS: data - numpy array of data to fit, min_cluster_size - int of minimum size for any cluster, 
# min_samples - int of minimum number of samples to consider, selection - method for cluster selection
# OUTPUTS: hdb_clust - fit hdb model
def hdb_scan(data, min_cluster_size, min_samples, selection='leaf', cluster_selection_epsilon=0.0):
    
    hdb_clust = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=selection)
    
    hdb_clust.fit(data)
    
    return hdb_clust


