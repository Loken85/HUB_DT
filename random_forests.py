#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:45:14 2021

@author: proxy_loken

Functions and helpers for Random Forests: for Behavioural/Neural Classification 



"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import cross_val_predict

from collections import OrderedDict

from sklearn.inspection import permutation_importance

#from itertools import cycle

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


"""
Example Random Forest setup:

rf_estimator = RandomForestClassifier(n_estimators=1000, max_features='sqrt', oob_score='True', class_weight='balanced')

scores = cross_val_score(rf_estimator, X_train,Y_train, cv=5)
model = rf_estimator.fit(X_train, Y_train)

"""





# PLOTTING FUNCTIONS


# function for plotting scores(cross-validated) from a random forest
# mean_scores is meant for averaging across categories 
def plot_scores(scores, mean_scores=False, multi_colour=False):
    
    if mean_scores:
        bar_locs = np.arange(np.size(scores,0))
    else:
        bar_locs = np.arange(np.size(scores))
    
    if multi_colour:
        bar_cols = ['blue', 'orange', 'purple', 'yellow','green']
    else:
        bar_cols = 'cornflowerblue'
    
    fig, ax = plt.subplots()
    
    if mean_scores:
        
        means = np.mean(scores,1)
        stds = np.std(scores,1)
        ax.bar(bar_locs, means, yerr=stds, align='center', color=bar_cols)
    else:
    
        ax.bar(bar_locs, scores, align='center', color=bar_cols)
    
    ax.set_ylabel('Accuracy %')
    
    ax.set_xticks(bar_locs)
    
    ax.set_xlabel('Folds')
    
    ax.set_yticklabels([0,20,40,60,80,100])
    
    plt.tight_layout()
    
    return fig
    
    



# function for plotting feature importances(MDI) from a random forest
def plot_mdi(forest, num_feats=0):
    
    importances = forest.feature_importances_
    sorted_idx = importances.argsort()
    # take top num_feat
    if ((num_feats > 0) and (num_feats <= len(sorted_idx))):
        sorted_idx = sorted_idx[-num_feats:]    
    labels = np.arange(0,len(importances))
    labels = labels[sorted_idx]
    
    
    
    y_ticks = np.arange(0,len(sorted_idx))
    fig, ax = plt.subplots()
    ax.barh(y_ticks, importances[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(labels)    
    ax.set_title("RF Feature Importances (MDI)")
    fig.tight_layout
    
    return fig



# fucntion for plotting permutation importances from a random forest
def plot_perm_importance(perm_results, num_feats=0, sort=True):
    
    if sort:
        sorted_idx = perm_results.importances_mean.argsort()
    else:
        sorted_idx = perm_results.importances_mean
    
    if (num_feats > 0) and (num_feats <= len(sorted_idx)):
        sorted_idx = sorted_idx[-num_feats:]
    
    labels = np.arange(0,len(perm_results.importances_mean))
    labels = labels[sorted_idx]
    
    fig, ax = plt.subplots()
    ax.boxplot(perm_results.importances[sorted_idx].T, vert=False, labels = labels)
    ax.set_title("RF Permutation Importances")
    fig.tight_layout
    
    return fig



# function for plotting scores(cross-validated) from single category random forest
def plot_single_scores(scores, multi_colour=False):
    
    bar_locs = np.arange(np.size(scores,1))
    
    if multi_colour:
        bar_cols = ['blue', 'orange', 'purple', 'yellow','green']
    else:
        bar_cols = 'cornflowerblue'
    
    figs_per_row = 2
    n_rows = np.size(scores,0) // figs_per_row
    if np.size(scores,0) % figs_per_row:
        n_rows = n_rows+1
    
    
    fig, axs = plt.subplots(n_rows,figs_per_row,sharex=True,sharey=True)
    
    ax = axs.ravel()
    
    for i in range(np.size(scores,0)):
        
        score = scores[i]
    
        ax[i].bar(bar_locs, score, align='center', color=bar_cols)
    
        ax[i].set_ylabel('Accuracy %')
    
        ax[i].set_xticks(bar_locs)
    
        ax[i].set_xlabel('Folds')
    
        ax[i].set_yticklabels([0,20,40,60,80,100])
    
    plt.tight_layout()
    
    return fig


# function for generating grouped bar plot for comparing scores (requires matched length score vectors)
def plot_score_comp(score1_means, score1_stds, score2_means,score2_stds, labels=['Raw','Behaviour Residuals'], colours=['orange','blue'], prod=False):
    
    score1_mean_stds = np.mean(score1_stds,axis=1)
    
    score2_mean_stds = np.mean(score2_stds,axis=1)
    
    bar_locs = np.arange(len(score1_means))
    
    bar_width = 0.35
    
    fig, ax = plt.subplots()
    
    bars1 = ax.bar(bar_locs-bar_width/2, score1_means, bar_width, yerr=score1_mean_stds, label=labels[0], color=colours[0])
    
    bars2 = ax.bar(bar_locs+bar_width/2, score2_means, bar_width, yerr=score2_mean_stds, label=labels[1], color=colours[1])
    
    ax.set_xticks([])
    
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.set_yticklabels([0,20,40,60,80,100])
    
    if prod:
        
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    return fig    
    
    


# function for plotting feature importances(MDI) from single category random forests
def plot_single_mdis(forests, num_feats=0, sort=True):
    
    figs_per_row = 2
    n_rows = len(forests) // figs_per_row
    if len(forests) % figs_per_row:
        n_rows = n_rows+1
    
    #fig, axs = plt.subplots(n_rows,figs_per_row,gridspec_kw={'hspace': 0,'wspace' : 0})
    fig, axs = plt.subplots(n_rows,figs_per_row, sharex=True)
    fig.suptitle("RF Feature Importances (MDI) for Single Catgegory RFs")    
    
            
    ax = axs.ravel()
    
    for i in range(len(forests)):
    
        importances = forests[i].feature_importances_
        if sort:
            sorted_idx = importances.argsort()
        else:
            sorted_idx = np.arange(np.size(importances))
        # take top num_feat
        if ((num_feats > 0) and (num_feats <= len(sorted_idx))):
            sorted_idx = sorted_idx[-num_feats:]    
        labels = np.arange(0,len(importances))
        labels = labels[sorted_idx]
        
        y_ticks = np.arange(0,len(sorted_idx))
        ax[i].barh(y_ticks, importances[sorted_idx])
        ax[i].set_yticks(y_ticks)
        ax[i].set_yticklabels(labels)
    
        
    fig.tight_layout
    
    return fig


# fucntion for plotting permutation importances from single category random forests
def plot_perm_importances(perm_results, num_feats=0, sort=True):
    
    figs_per_row = 2
    n_rows = len(perm_results) // figs_per_row
    if len(perm_results) % figs_per_row:
        n_rows = n_rows+1
    
    fig, axs = plt.subplots(n_rows,figs_per_row, sharex=True)
    fig.suptitle("RF Permutation Importances for Single Catgegory RFs")
    
    ax = axs.ravel()
    
    for i in range(len(perm_results)):
        
        if sort:
            sorted_idx = perm_results[i].importances_mean.argsort()
        else:
            sorted_idx = np.arange(np.size(perm_results[i].importances_mean))
    
        if (num_feats > 0) and (num_feats <= len(sorted_idx)):
            sorted_idx = sorted_idx[-num_feats:]
    
        labels = np.arange(0,len(perm_results[i].importances_mean))
        labels = labels[sorted_idx]

        ax[i].boxplot(perm_results[i].importances[sorted_idx].T, vert=False, labels = labels)
    
    fig.tight_layout
    
    return fig


    

# TRAINING/EVALUTATION FUNCTIONS


# function for training a RF estimator on multiple behaviours
# INPUTS: X (features), Ys (labels for all categories)
# Default params are given, pass if necessary
def rf_multi_category(X_train, X_val, y_train, y_val, n_estimators=100, max_features='sqrt', oob_score='True', class_weight='balanced', min_samples_leaf=10):
    
    rf_estimator = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, oob_score=oob_score, class_weight=class_weight, min_samples_leaf=min_samples_leaf)
    
    #X_train, X_val, Y_train, Y_val = train_test_split(X, Ys, test_size=0.5)
    
    scores = cross_val_score(rf_estimator, X_train, y_train, cv=5)
    
    rf_estimator.fit(X_train, y_train)
    
    perm_result = permutation_importance(rf_estimator, X_val, y_val, n_repeats=10)
    
    predict_probas = rf_estimator.predict_proba(X_val)
    
    return rf_estimator, perm_result, predict_probas, scores




# function for training a RF estimator on a single behaviour
# INPUTS: X (features), Ys (labels for all categories), behav (category)
def rf_single_category(X, Ys, behav):
    
    Y = Ys[:,behav]
    
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.5)
    
    rf_estimator = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score='True', class_weight='balanced', min_samples_leaf=10)
    
    scores = cross_val_score(rf_estimator, X_train,Y_train, cv=5)
    rf_estimator.fit(X_train, Y_train)
    
    plot_mdi(rf_estimator, num_feat=20)
    
    perm_result = permutation_importance(rf_estimator, X_val, Y_val, n_repeats=10)
    
    plot_perm_importance(perm_result, num_feats=20)
    
    predict_probas = rf_estimator.predict_proba(X_val)
    
    return rf_estimator, perm_result, predict_probas, scores


# function for training RF estimators on all single behaviours
# INPUTS: X(features), Ys (labels for all categories), plot (generate default plots)
def rf_single_categories(X_train, X_val, Y_train, Y_val, n_estimators=100, max_features='sqrt', min_samples_leaf=10, plot=True):
    
    #Y = Ys
    #num_beh = np.size(Y,1)
    num_beh = np.size(Y_train,1)
    
    #X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.5)
    
    models = []
    scores = []
    perm_results = []
    predict_probs = []
    
    for i in range(num_beh):
        
        rf_estimator = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, oob_score='True', class_weight='balanced', min_samples_leaf=min_samples_leaf)
        
        model = rf_estimator.fit(X_train, Y_train[:,i])
        score = cross_val_score(rf_estimator, X_train,Y_train[:,i], cv=10)
        perm_result = permutation_importance(rf_estimator, X_val, Y_val[:,i], n_repeats=10)
        predict_probas = rf_estimator.predict_proba(X_val)
        
        models.append(model)
        scores.append(score)
        perm_results.append(perm_result)
        predict_probs.append(predict_probas)
        
    
    if plot:
        plot_single_mdis(models, 20)
        plot_perm_importances(perm_results, 20)
    
    return models, perm_results, predict_probs, scores


# Function for generating permutation importance from a given RF estimator
def gen_perm_importances(rf_estimator, X_val, Y_val, n_repeats=10):
    
    perm_results = permutation_importance(rf_estimator, X_val, Y_val, n_repeats=n_repeats)
    
    return perm_results




# TESTING FUNCTIONS

# OOB error testing for parameter selection
# This should not need to be run more than once: choose a broad range of parameters
# and fine tune with individual runs on reasonable intervals
def prog_OOB_test(X_train, Y_train):
    
    RANDOM_STATE = 123
    
    ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
    ]
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 100
    max_estimators = 1000

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X_train, Y_train)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))
    
    # Generate the "OOB error rate" vs. "n_estimators" plot.
    plt.figure()
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    #plt.show()
    
    return ensemble_clfs