#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 12:18:40 2021

@author: proxy_loken

Shapley and Owen Value Functions


"""

import numpy as np
import pandas as pd

import shap

import seaborn as sns
import matplotlib.pyplot as plt


shap.initjs()



# function to generate a shap test for a given model and dataset
# INPUTS: model - ml model to be evaluated, X - dataset to generate shap values for (usually the test/val set)
# Note: this uses the default explainer without an explicit mask
# This can be optimised via partioning and probabilistic explainers for a particular model
def gen_shap_test(model, X):
    
    explainer = shap.Explainer(model)
    
    shap_test = explainer(X)
    
    return shap_test


# function to make a dataframe containing just positive shap test values from binary classification
def get_positive_shapdf(shap_test):
    
    shap_df = pd.DataFrame(shap_test.values[:,:,1])
    
    return shap_df



# Gutcheck for shapley values for classification
# So I don't forget how to check this
def binary_shap_check(model, explainer, shap_df, X):
    
    check = np.isclose(model.predict_proba(X)[:,1], explainer.expected_value[1]+shap_df.sum(axis=1))

    return check




# SHAP Visualisations
# Note: mostly wrappers of built in plotting functions with optional modifications for reference


# Global Bar plot for shap_test 
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_gshap_bar(shap_test):
    
    fig, ax = plt.subplots()
    shap.plots.bar(shap_test)
        
# Global beeswarm plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_gshap_beeswarm(shap_test):
    
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_test)
    # alternate colourmap
    # shap.plots.beeswarm(shap_test, cmap=plt.get_cmap("winter_r))
    
# Global violin plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_gshap_violin(shap_test):
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_test, plot_type='violin')
    

# Global heatmap plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_gshap_heatmap(shap_test):
    
    fig, ax = plt.subplots()
    shap.plots.heatmap(shap_test)
    # alternate colourmap
    # shap.plots.heatmap(shap_test, cmap=plt.get_cmap("winter_r))
    
# Global forceplot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
# Expected values can be passed from explainer(explainer.expecetd_value)
# X is the dataset for the explainer (usually test/val)
def plot_gshap_force(expected_values, shap_test, X):
    
    fig, ax = plt.subplots()
    shap.force_plot(expected_values, shap_test.values, X)


    
# Global mean shap value and distribution bar plots
def plot_gshap_mean_dis(shap_df):
    
    columns = shap_df.apply(np.abs).mean().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(1,2)
    
    sns.barplot(data=shap_df[columns].apply(np.abs), orient='h', ax=ax[0])
    ax[0].set_title("Mean Absolute Shap Value")
    
    sns.boxplot(data=shap_df[columns], orient='h', ax=ax[1])
    ax[1].set_title("Distribution of Shap Values")


# Local bar plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_lshap_bar(shap_test, example):
    
    fig, ax = plt.subplots()
    shap.plots.bar(shap_test[:,:][example])

    
# Local force plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_lshap_force(shap_test, example):
    
    fig, ax = plt.subplots()
    shap.plots.force(shap_test[:,:][example])


# Local waterfall plot for shap_test
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_lshap_water(shap_test, example):
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_test[:,:][example])


# alternate Local waterfall plot for shap_test
# Redundant, but shows how the waterfall plot is constructed
# If passing shap values for classification, pre-select the output to plot (shap_test[:,:,1])
def plot_lshap_altwater(shap_test, example):
    
    class WaterfallData():
        def __init__ (self, shap_test, index):
            self.values = shap_test[index].values
            self.base_values = shap_test[index].base_values[0]
            self.data = shap_test[index].data
            #self.feature_names = shap_test.feature_names
    
    fig, ax = plt.subplots()
    shap.plots.waterfall(WaterfallData(shap_test, example))




# # Example for tracking down possibly systemic errors


# # make data into dataframes to facilitate indexing and search

# Xval_df = pd.DataFrame(X_val)

# Yval_df = pd.DataFrame(Y_val, columns=['locomotion','rearing','grooming','nose','feeding','none'])

# Ytrain_df = pd.DataFrame(Y_train, columns=['locomotion','rearing','grooming','nose','feeding','none'])

# Xtrain_df = pd.DataFrame(X_train)

# # make one frame for both features and outputs
# test = pd.concat([Xval_df, Yval_df], axis=1)
# # add model output probs (of positive class) column
# test['probability'] = rf.predict_proba(X_val)[:,1]
# # add order column to track pre-sort order
# test['order'] = np.arange(len(test))
# # Query for biggest error
# errors = test.query('locomotion==1').nsmallest(5,'probability')
# # get index of first entry in error
# ind0 = errors['order'].values[0]
# # waterfall plot for that entry, lets us see which features had the biggest impact
# shap.plots.waterfall(shap_test[:,:,1][ind0])
# # search for similar example in training set (this only works if the features are binary,categorical, or discrete bins)
# similars = pd.concat([Xtrain_df, Ytrain_df], 
#                      axis=1) [(Xtrain_df[3].between(1,3)) &
#                               (Xtrain_df[30]==0) &
#                               (Xtrain_df[54]==0)]











    
    
    