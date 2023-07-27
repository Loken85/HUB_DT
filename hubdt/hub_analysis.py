#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Analysis Functions for the HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np
from openTSNE import TSNE

import data_loading
import b_utils
import wavelets
import hdb_clustering
import glms
import random_forests
import video_utils
import t_sne

import hub_utils

# DICT/DATA GENERATORS

#@st.cache
def generate_hdb_dict():
    
    hdb_dict = {}
    if 'hdb_labels' in st.session_state and 'hdb_probs' in st.session_state:
        hdb_dict['hdb_labels'] = st.session_state.hdb_labels
        hdb_dict['hdb_probs'] = st.session_state.hdb_probs
    
    
    return hdb_dict


def generate_context_dict():
    
    context_dict = {}
    if 'context_labels' in st.session_state:
        context_dict['context_labels'] = st.session_state.context_labels
        
    return context_dict



#@st.cache
def generate_embedding_dict():
    
    embed_dict = {}
    if 'b_tout' in st.session_state:
        embed_dict['projections tsne'] = np.array(st.session_state.b_tout)
    if 's_out' in st.session_state:
        embed_dict['st tsne'] = np.array(st.session_state.s_tout)
    
    
    return embed_dict


def generate_glm_dict(glm_type='multi'):
    
    glm_dict = {}
    if 'multiglm' in st.session_state and glm_type=='multi':
        glm_dict['multimse'] = st.session_state.mse
        glm_dict['multimae'] = st.session_state.mae
        glm_dict['multimpd'] = st.session_state.mpd
        glm_dict['multi_mean_r2s'] = st.session_state.multi_mean_r2s
        glm_dict['multi_sum_r2s'] = st.session_state.multi_sum_r2s
        glm_dict['multi_max_r2s'] = st.session_state.multi_max_r2s
        glm_dict['multi_raw_r2s'] = st.session_state.multi_raw_r2s
        glm_dict['multi_resids'] = st.session_state.multi_resids
        
    if 'singleglm' in st.session_state and glm_type=='single':
        glm_dict['single_r2s'] = st.session_state.single_r2s
        glm_dict['single_resids'] = st.session_state.single_resids
        glm_dict['num_sigrs'] = st.session_state.num_sigrs
        glm_dict['sig_prop'] = st.session_state.sig_prop
        
    return glm_dict


def generate_rf_dict(rf_type='singles'):
    
    rf_dict = {}
    
    if st.session_state.rfs_single and rf_type=='singles':
        
        rf_dict['rf_scores'] = st.session_state.rfs_scores
        rf_dict['rf_perms'] = st.session_state.rfs_perms
        rf_dict['rf_probs'] = st.session_state.rfs_probs
        
    if st.session_state.rf_multi and rf_type=='multi':
        
        rf_dict['rf_scores'] = st.session_state.rf_scores
        rf_dict['rf_perms'] = st.session_state.rf_perms
        rf_dict['rf_probs'] = st.session_state.rf_probs
        
    return rf_dict


def generate_proj_dict(proj_type='behaviour'):
    
    proj_dict = {}
    
    if proj_type=='behaviour':
        if 'b_projections' in st.session_state:
            proj_dict['behaviour_projections'] = st.session_state.b_projections
    elif proj_type=='neural':
        if 's_projections' in st.session_state:
            proj_dict['neural_projections'] = st.session_state.s_projections
    elif proj_type=='both':
        if 'b_projections' in st.session_state:
            proj_dict['behaviour_projections'] = st.session_state.b_projections
        if 's_projections' in st.session_state:
            proj_dict['neural_projections'] = st.session_state.s_projections
    
    return proj_dict




@st.cache
def generate_matlab_export_dict():
    
    exp_dict = {}
    if 'hdb_labels' in st.session_state:
        exp_dict['hdb_tracking_labels'] = st.session_state.hdb_labels
    if 'behav_labels' in st.session_state:
        exp_dict['manual_behav_labels'] = st.session_state.behav_labels
    if 'b_projections' in st.session_state:
        exp_dict['tracking_projections'] = st.session_state.b_projections
    if 's_projections' in st.session_state:
        exp_dict['neural_projections'] = st.session_state.s_projections
    if 'stbin' in st.session_state:
        exp_dict['stbin'] = st.session_state.stbin
    # TODO: Fix GLM exporting to fit new format
    if 'multiglm' in st.session_state:
        exp_dict['multimse'] = st.session_state.mse
        exp_dict['multimae'] = st.session_state.mae
        exp_dict['multimpd'] = st.session_state.mpd        
    if 'singleglm' in st.session_state:
        exp_dict['single_r2s'] = st.session_state.single_r2s
        exp_dict['single_resids'] = st.session_state.single_resids
    
    return exp_dict



def generate_stbin_dict():
    
    stbin_dict = {}
    
    if 'stbin' in st.session_state:
        stbin_dict['stbin'] = st.session_state.stbin
        
    return stbin_dict
        

def generate_video_streams():
    # TODO: check if streams already exist before overwriting
    
    stack = False
    stack_path = st.session_state.curr_sess.stack_path
    
    
    cam0 = False
    camera0_path = st.session_state.curr_sess.camera0_path
    
    cam1 = False
    camera1_path = st.session_state.curr_sess.camera1_path
        
    if stack_path:
        st.session_state.vidstream_stack = video_utils.open_stream(stack_path)
        stack = True
    
    
    if camera0_path:
        st.session_state.vidstream0 = video_utils.open_stream(camera0_path)
        cam0 = True
        
    if camera1_path:
        st.session_state.vidstream1 = video_utils.open_stream(camera1_path)
        cam1 = True
    
    return stack, cam0, cam1

# ANALYSIS GENERATORS


#@st.cache()
def generate_projections(proj_type,minf,maxf,nump):
    if proj_type=='Tracking':
        data = st.session_state.tracking
    elif proj_type=='Neural':
        data = st.session_state.stbin
    else:
        return
    scales, freqs = wavelets.calculate_scales(minf,maxf,st.session_state.curr_sess.sfreq,nump)
    st.session_state.freqs = freqs
    scalonp = wavelets.wavelet_transform_np(data, scales, freqs, st.session_state.curr_sess.sfreq)
    if proj_type=='Tracking':
        if st.session_state.dlc:
            sub_scalonp = data_loading.convert_fr(np.transpose(scalonp), 15, 5)
            stbin, sub_scalonp = data_loading.trim_to_match(st.session_state.stbin, sub_scalonp)
            st.session_state.b_projections = sub_scalonp
        else:
            st.session_state.b_projections = np.transpose(scalonp)
    elif proj_type=='Neural':
        st.session_state.s_projections = np.transpose(scalonp)
    return

@st.cache()
def generate_tracking_embedding(perp):
    tsne = TSNE(perplexity=perp, initialization='pca', metric='cosine', n_jobs=8)
    t_outp = tsne.fit(st.session_state.b_projections)
    st.session_state.b_tout = t_outp
    # clear the stored density (will be recalculated if the embedding is plotted)
    if 'b_density' in st.session_state:
        del st.session_state['b_density']
    
    return

@st.cache()
def generate_neural_embedding(perp):
    tsne = TSNE(perplexity=perp, initialization='pca', metric='cosine', n_jobs=8)
    t_outp = tsne.fit(st.session_state.s_projections)
    st.session_state.s_tout = t_outp
    if 's_density' in st.session_state:
        del st.session_state['s_density']
    
    return

#@st.cache()
def generate_clustering(min_clust_size, min_samples, selection_method, selection_epsilon):
    hdb_clust = hdb_clustering.hdb_scan(st.session_state.b_tout, min_cluster_size=min_clust_size, min_samples=min_samples, selection=selection_method, cluster_selection_epsilon=selection_epsilon)
    st.session_state.hdbclust = hdb_clust
    st.session_state.hdb_labels = hdb_clust.labels_
    st.session_state.hdb_probs = hdb_clust.probabilities_
    st.session_state.hdb_eps = selection_epsilon
    
    return

# Function for experimental trajectory based smoothing (applies jitter removal and gap filling)
def generate_cluster_smoothing(labels, gap, jitter, gap_size, jitter_size):
    # TODO: add option for doing neural cluster smoothing    
    if jitter:
        labels = b_utils.remove_label_jitter(labels, jitter_size=jitter_size)
        
    if gap:
        labels = b_utils.gap_fill_labels(labels, gap_size)
        
    st.session_state.smoothed_labels = labels    
    return


# Wrapper/selector for GLM regression generation functions
def generate_regression(reg_type, tar_type, target, pred_type, predictor, tar_noise=True, pred_noise=True, tar_pre=True, pred_pre=True, m_iter=300, cut_pre=False):
    
    # Fetch data
    tar_set = hub_utils.get_data(tar_type, target)
    pred_set = hub_utils.get_data(pred_type, predictor)     
    
    if tar_set is None:
        st.error('Ensure the selected Target Data are Generated/Loaded before Regression')
        return
        
    if pred_set is None:
        st.error('Ensure the selected Predictor Data are Generated/Loaded before Regression')
        return
    
    
    if reg_type =='Multifactor GLM':
        
        generate_multiglm(tar_type, target, tar_set, tar_pre, tar_noise, pred_type, predictor, pred_set, pred_pre, pred_noise, m_iter, cut_pre)
                                
        return
    
    elif reg_type=='Single-factor GLM':
        
        generate_singleglm(tar_type, target, tar_set, tar_pre, tar_noise, pred_type, predictor, pred_set, pred_pre, pred_noise, m_iter)
        
        return


        
def generate_multiglm(tar_type, target, tar_set, tar_pre, tar_noise, pred_type, predictor, pred_set, pred_pre, pred_noise, m_iter, cut_pre=False):
    
    if tar_type=='Categorical' and cut_pre:
        tar_set, pred_set = b_utils.precut_noise(tar_set, pred_set)
        # not keeping noise would erroneosly cut the first category if already precut, so set noise keep vars to true
        tar_noise = True
        pred_noise = True
    elif pred_type=='Categorical' and cut_pre:
        pred_set, tar_set = b_utils.precut_noise(pred_set, tar_set)
        tar_noise = True
        pred_noise = True
    
    
    
    if tar_type=='Categorical' and tar_pre:
        ys = glms.preprocess_targets(tar_set, keep_noise=tar_noise)
        # This branch should be avoided, will likely cause a dim error downstream if the targets are of the wrong shape
    elif tar_type=='Categororical' and not tar_pre:
        ys = tar_set
    elif tar_type=='Numerical' and tar_pre:
        ys = glms.standardize(tar_set)
    elif tar_type=='Numerical' and not tar_pre:
        ys = tar_set
            
    if pred_type=='Categorical' and pred_pre:
        Xs = glms.preprocess_data(pred_set,input_cats=True, keep_noise=pred_noise)
    elif pred_type=='Categorical' and not pred_pre:
        Xs = pred_set
    elif pred_type=='Numerical' and pred_pre:
        Xs = glms.preprocess_data(pred_set)
    elif pred_type=='Numerical' and not pred_pre:
            Xs = pred_set
        
    X_train, X_test, y_train, y_test = glms.split_data(Xs, ys)
        
    estimator = glms.fit_glm(X_train, y_train,m_iter=m_iter)
    mse, mae, mpd = glms.score_estimator(estimator, X_test, y_test)
    multi_mean_r2s = glms.calc_r2s(estimator, X_test, y_test, out_type='mean')
    multi_sum_r2s = glms.calc_r2s(estimator, X_test, y_test, out_type='sum')
    multi_max_r2s = glms.calc_r2s(estimator, X_test, y_test, out_type='max')
    multi_raw_r2s = glms.calc_r2s(estimator, X_test, y_test, out_type='raw')
    multi_resids = glms.generate_residuals(estimator, Xs, ys)
    target_deltas = b_utils.calc_resid_diffs(ys, multi_resids)
        
    st.session_state.multiglm = True
    st.session_state.glm_est = estimator
    st.session_state.mse = mse
    st.session_state.mae = mae
    st.session_state.mpd = mpd
    st.session_state.multi_mean_r2s = multi_mean_r2s
    st.session_state.multi_sum_r2s = multi_sum_r2s
    st.session_state.multi_max_r2s = multi_max_r2s
    st.session_state.multi_raw_r2s = multi_raw_r2s
    st.session_state.multi_resids = multi_resids
    st.session_state.multi_tar_dels = target_deltas
    st.session_state.multiglm_preds = predictor
    st.session_state.multiglm_pred_type = pred_type
    st.session_state.multiglm_targs = target
    st.session_state.multiglm_tar_type = tar_type
    
    # Temp location for specific GLM caching
    # This will put GLM data for specific variable sets in session state for use in Supervised ML
    # TODO: relocate into it's own helper(maybe add selectability)
    if target == 'stbin' and predictor == 'HDB Labels':
        
        st.session_state.hdb_fr_glm = True
        st.session_state.hdb_fr_resids = multi_resids
        st.session_state.hdb_fr_r2s = multi_raw_r2s
        
    elif target == 'stbin' and predictor == 'Context Labels':
        
        st.session_state.context_fr_glm = True
        st.session_state.context_fr_resids = multi_resids
        st.session_state.context_fr_r2s = multi_raw_r2s
    
    
    return



def generate_singleglm(tar_type, target, tar_set, tar_pre, tar_noise, pred_type, predictor, pred_set, pred_pre, pred_noise, m_iter):
    
    if tar_type=='Categorical' and tar_pre:
        ys = glms.preprocess_targets(tar_set, keep_noise=tar_noise)
    # This branch should be avoided, will likely cause a dim error downstream if the targets are of the wrong shape
    elif tar_type=='Categororical' and not tar_pre:
        ys = tar_set
    elif tar_type=='Numerical' and tar_pre:
        ys = glms.standardize(tar_set)
    elif tar_type=='Numerical' and not tar_pre:
        ys = tar_set
            
    if pred_type=='Categorical' and pred_pre:
        Xs = glms.preprocess_data(pred_set,input_cats=True, keep_noise=pred_noise)
    elif pred_type=='Categorical' and not pred_pre:
        Xs = pred_set
    elif pred_type=='Numerical' and pred_pre:
        Xs = glms.preprocess_data(pred_set)
    elif pred_type=='Numerical' and not pred_pre:
        Xs = pred_set
        
    X_train, X_test, y_train, y_test = glms.split_data(Xs, ys)
        
    r2s, resids = glms.fit_single_glms(X_train, y_train, Xs, ys, m_iter=m_iter)
        
    st.session_state.num_sigrs, st.session_state.sig_prop = glms.calc_proportions(r2s)        
        
    st.session_state.singleglm = True
    st.session_state.single_r2s = r2s
    st.session_state.single_resids = resids
    
    return


def generate_random_forest(forest_type, target, predictor, n_estimators, n_leaf, features):
    
    ys = hub_utils.get_data('Categorical', target)
    X = hub_utils.get_data('Numerical', predictor)
    
    if ys is None or X is None:
        st.sidebar.error('Ensure Selected Data is generated/loaded before fitting')
        return
    ys = glms.preprocess_targets(ys)
    
    X_train, X_test, y_train, y_test = glms.split_data(X, ys)
    
    if forest_type=='Single Class Forests':
        
        rf_models, rf_perms, rf_probs, rf_scores = random_forests.rf_single_categories(X_train, X_test, y_train, y_test, n_estimators=n_estimators, max_features=features, min_samples_leaf=n_leaf, plot=False)
        
        st.session_state.rfs_single = True
        st.session_state.rfs_models = rf_models
        st.session_state.rfs_perms = rf_perms
        st.session_state.rfs_probs = rf_probs
        st.session_state.rfs_scores = rf_scores
    
    elif forest_type=='Multiclass Forest':
        
        rf_model, rf_perms, rf_probs, rf_scores = random_forests.rf_multi_category(X_train, X_test, y_train, y_test, n_estimators=n_estimators, max_features=features, min_samples_leaf=n_leaf)
        
        st.session_state.rf_multi = True
        st.session_state.rf_model = rf_model
        st.session_state.rf_perms = rf_perms
        st.session_state.rf_probs = rf_probs
        st.session_state.rf_scores = rf_scores
        
    else:
        
        return
    
    return
    


# helper to generate embedding densities. This is computationally expensive, so this saves on time when plotting
def generate_embedding_density(embed_type):
    embedding = hub_utils.get_embedding(embed_type)
    if embedding is None:
        st.error('Generate Embedding before Plotting')
        return
    density, xi,yi = t_sne.calc_density(embedding)
    if embed_type == 'Tracking':
        st.session_state.b_density = density
        st.session_state.b_xi = xi
        st.session_state.b_yi = yi
    elif embed_type == 'Neural':
        st.session_state.s_density = density
        st.session_state.s_xi = xi
        st.session_state.s_yi = yi
        

# helper for generating list of list of purities between two labelsets
def generate_labelset_purities(labelset_1,labelset_2,ignore_noise=False):
    
    purities = b_utils.compute_labelset_purity(labelset_1,labelset_2,ignore_noise=ignore_noise)
    
    return purities






