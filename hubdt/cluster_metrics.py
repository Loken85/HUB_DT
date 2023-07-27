#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 17:26:22 2020

comparative metrics/stats for clustering

@author: proxy_loken
"""


import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score



def onehot_to_categorical(olabels):
    
    clabels = np.zeros((np.size(olabels,0)))
    for i in range(0,np.size(olabels,1)):
        idx = olabels[:,i] ==1
        clabels[idx] = i
    
    return clabels




# Calculate the adjusted mutual information score between two sets of labels
# adjusted to account for chance, independent of cluster label values/permutations and symetric
# Homogenous labellings will return an AMI value of 1 (completely disparate labellings return 0 )
# INPUTS: true_labels, pred_labels - arrays of labelled frames (frames x 1) (do not have to actually be true/pred labels)
def calc_AMI(true_labels, pred_labels):
    
    return adjusted_mutual_info_score(true_labels,pred_labels)



# Calculate the adjusted Rand index score between two sets of labels
# adjusted to account for chance, independent of cluster label values/permutations and symetric
# Homogenous labellings will return a RAND value of 1 (completely disparate labellings return 0 )
# INPUTS: true_labels, pred_labels - arrays of labelled frames (frames x 1) (do not have to actually be true/pred labels)
def calc_RAND(true_labels, pred_labels):
    
    return adjusted_rand_score(true_labels, pred_labels)



# calculate the KL divergence between two gaussian mixture models
# INPUTS: gmm_p, gmm_q - sklearn gmm models, n_samples - number of samples to
# sample each distribution
def gmm_kl(gmm_p, gmm_q,n_samples=10**5):
    # sample the model p's distribution
    X, _ = gmm_p.sample(n_samples)
    # log of the probabilites of X belonging to the distributions
    # KL divergence is defined with ln, but log(2) is much easier computationally
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    # Another computational simplification: mean(log(p(X)) / q(X)) = mean(log(p(X)))-mean(log(q(X)))
    kl = log_p_X.mean()-log_q_X.mean()
    
    return kl
    


# Calculate the Jensen-Shannon divergence between two gaussian mixture models
# the JS devergence is a symetric and smoothed adaption fo the KL divergence
# equivalent to the MI of a rv X associated to the mixture distribution between P and Q
# and a binary indicator Z (used to switch between P and Q)
# Note: the JSd between a joint distribution and the product of the marginals can be used
# as a reliability measure for determining if a given response is from one or the other
# INPUTS: gmm_p, gmm_q - sklearn gmm models, n_samples - number of samples to
# sample each distribution
def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    # sample the model p's distribution
    X, _ = gmm_p.sample(n_samples)
    # log of the probabilities of X belonging to the distributions
    log_p_X = gmm_p.score_samples(X)
    log_q_X = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X,log_q_X)
    # sample the model q's distribution
    Y, _ = gmm_q.sample(n_samples)
    # log of the probabilites of Y belonging to the distributions
    log_p_Y = gmm_p.score_samples(Y)
    log_q_Y = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y,log_q_Y)
    # JS = KL(p||(p+q)/2) + KL(q||(p+q)/2)
    js = (log_p_X.mean() - (log_mix_X.mean() - np.log(2)) + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2
    
    return js

    
# Calculate the Fowlkes-Mallows score between two labellings
# the FMI is the geometric mean of pairwise recall and precision between two 
# sets of labels. ( FMI = TP/root((TP+FP)(TP+FN)))
# generally used to compare a clustering with "true" labels
# Ranges from 0-1. A score of zero indicates an independent labelling, 1 is a
# perfect match
# Usefull for comapring heirarchichal clusterings of different structure
def calc_fmi(labels_a, labels_b):
    
    score = fowlkes_mallows_score(labels_a, labels_b)
    
    return score


# Calculate the split/join distance, also known as the van dongen metric
# This distance is defined on the space of partitions(clustertings) of a given set
# it is the sum of the projection distance from one partition to another, and vice versa
# this measure is asymmetric, and so must be calculated for each direction
# INPUTS: labels_a,labels_b: vectors of labels
# ignore_noise: wether to include the noise category as a cluster (noise is by default labelled -1)
# OUTPUTS: a list of the two distances, a projected to b, and b to a
# lists of the maximal overlap clusters for a to b, and b to a
def split_join_distance(labels_a, labels_b, ignore_noise=False):
    
    labels_a = labels_a.astype(int)
    labels_b= labels_b.astype(int)
    
    if ignore_noise:
        first_clust = 0        
        size_a = np.count_nonzero((labels_a >=0))
        size_b = np.count_nonzero((labels_b >=0))
    else:
        first_clust = -1
        size_a = np.size(labels_a,0)
        size_b = np.size(labels_b,0)
    
    # Get Distance for A vs B
    # sum of the max overlaps for each cluster in a
    sum_overlap = 0
    # clusters with max overlap
    max_clusters_a = []
    for i in range(first_clust, max(labels_a)+1):
        # holds current maximum overlap between clusters
        max_overlap = 0
        # holds current max overlap cluster
        max_cluster = -2
        # cast labels to boolean for the given current cluster
        a_bool = labels_a == i
        for j in range(first_clust, max(labels_b)+1):
            # cast label b to boolean for given current cluster
            b_bool = labels_b == j
            # compare boolean arrays to find matches between two current clusters
            comp = a_bool & b_bool
            # sum matches to find size of overlap
            overlap = np.count_nonzero(comp)
            if overlap > max_overlap:
                max_overlap = overlap
                max_cluster = j
                
            
        
        sum_overlap = sum_overlap + max_overlap
        max_clusters_a.append(max_cluster)
    
    a_dist = size_a-sum_overlap
    
    # Get Distance for B vs A
    # sum of the max overlaps for each cluster in b
    sum_overlap = 0
    #clusters with max overlap
    max_clusters_b = []
    for i in range(first_clust, max(labels_b)+1):
        # holds current maximum overlap between clusters
        max_overlap = 0
        # holds current max overlap cluster
        max_cluster = -2
        # cast labels to boolean for the given current cluster
        b_bool = labels_b == i
        for j in range(first_clust, max(labels_a)+1):
            # cast label a to boolean for given current cluster
            a_bool = labels_a == j
            # compare boolean arrays to find matches between two current clusters
            comp = b_bool & a_bool
            # sum matches to find size of overlap
            overlap = np.count_nonzero(comp)
            if overlap > max_overlap:
                max_overlap = overlap
                max_cluster = j
        
        sum_overlap = sum_overlap + max_overlap
        max_clusters_b.append(max_cluster)
    
    b_dist = size_b-sum_overlap
    
    return [a_dist, b_dist], max_clusters_a, max_clusters_b



# computes % distance from split_join distance for simple comparison of two labellings
# NOTE: comparing clustertings this way makes a lot of assumptions, so this is best
# reserved for cases where a large number of quick comparisons are required
# INPUTS: dists - list of split-join dists between two label sets, labels - one set of labels
def compute_relative_distance(dists,labels):
    
    samples = len(labels)
    
    rel_d1 = dists[0] / samples
    rel_d2 = dists[1] / samples
    
    avg_rel_d = np.mean((rel_d1,rel_d2))
    
    return avg_rel_d



