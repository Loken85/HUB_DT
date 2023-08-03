#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering page for HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np

from hubdt import hub_utils, hub_constants, hub_analysis

from hubdt import hdb_clustering






def display_clustering_vis():
    
    draw_top_clust_display(st.session_state.top_clustering_display)
    
    draw_bottom_clust_display(st.session_state.bottom_clustering_display)
    
    return
           
    
    

def draw_top_clust_display(disp_type):
    
    if disp_type=='Density Tree':
        st.markdown('Density Tree for HDBScan Clustering')
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting the Density Tree')
        else:
            hdb_tree_select_clusts = st.checkbox(label='Select Clusters', key='hdb_tree_select_clusts_top')
            hdb_tree_label_clusts = st.checkbox(label='Label Clusters', key='hdb_tree_label_clusts_top')
            hdb_tree_draw_eps = st.checkbox(label='Draw Cut-off', key='hdb_tree_draw_eps_top')
            fig = hub_utils.draw_hdb_tree(hdb_tree_select_clusts, hdb_tree_label_clusts, hdb_tree_draw_eps)
            st.pyplot(fig)
    elif disp_type=='Embedding Density':
        st.markdown('Density Map of the 2d Embedding')
        if 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting the Density')
        else:
            fig = hub_utils.draw_embedding_density()
            st.pyplot(fig)
    elif disp_type=='Clusters over Embedding':
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting')
        elif 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_top')
            fig = hdb_clustering.plot_hdb_over_tsne(st.session_state.b_tout, st.session_state.hdb_labels, st.session_state.hdb_probs,noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    elif disp_type=='Smoothed Clusters':
        if 'smoothed_labels' not in st.session_state:
            st.error('Perform Smoothing before Plotting')
        elif 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_top')
            fig = hdb_clustering.plot_hdb_over_tsne(st.session_state.b_tout, st.session_state.smoothed_labels, st.session_state.hdb_probs, noise=hdb_clust_plot_noise)
            st.pyplot(fig)
    else:
        st.error('Not Yet Implemented')
    

def draw_bottom_clust_display(disp_type):
    
    if disp_type=='Density Tree':
        st.markdown('Density Tree for HDBScan Clustering')
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting the Density Tree')
        else:
            hdb_tree_select_clusts = st.checkbox(label='Select Clusters', key='hdb_tree_select_clusts_bottom')
            hdb_tree_label_clusts = st.checkbox(label='Label Clusters', key='hdb_tree_label_clusts_bottom')
            hdb_tree_draw_eps = st.checkbox(label='Draw Cut-off', key='hdb_tree_draw_eps_bottom')
            fig = hub_utils.draw_hdb_tree(hdb_tree_select_clusts, hdb_tree_label_clusts, hdb_tree_draw_eps)
            st.pyplot(fig)
    elif disp_type=='Embedding Density':
        st.markdown('Density Map of the 2d Embedding')
        if 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting the Density')
        else:
            fig = hub_utils.draw_embedding_density()
            st.pyplot(fig)
    elif disp_type=='Clusters over Embedding':
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting')
        elif 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_bottom')
            fig = hdb_clustering.plot_hdb_over_tsne(st.session_state.b_tout, st.session_state.hdb_labels, st.session_state.hdb_probs,noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    elif disp_type=='Smoothed Clusters':
        if 'smoothed_labels' not in st.session_state:
            st.error('Perform Smoothing before Plotting')
        elif 'b_tout' not in st.session_state:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_bottom')
            fig = hdb_clustering.plot_hdb_over_tsne(st.session_state.b_tout, st.session_state.smoothed_labels, st.session_state.hdb_probs, noise=hdb_clust_plot_noise)
            st.pyplot(fig)
    else:
        st.error('Not Yet Implemented')



def clustering():
        
    #sidebar section
    st.sidebar.markdown("# Clustering")
    # TODO cluster parameter setters and cluster run button
    cluster_form = st.sidebar.form(key='cluster_form')
    cluster_form_select_type = cluster_form.selectbox(label='Cluster Type',options=('Tracking','Neural'))
    cluster_form_select_method = cluster_form.selectbox(label='Selection Method',options=('leaf','eom'))
    cluster_form_minsize = cluster_form.number_input(label='Minimum Cluster Size', min_value=50, max_value=2000, value= 200)
    cluster_form_samplesize = cluster_form.number_input(label='Minimum Sample Size', min_value=1, max_value=200, value= 5)
    cluster_form_selectepsilon = cluster_form.slider(label='Clustering Cutoff Epsilon', min_value=0.00, max_value=0.99, value=0.25)
    cluster_form_submit = cluster_form.form_submit_button(label='Perform Clustering')
    
    if cluster_form_submit:
        if cluster_form_select_type=='Tracking':
            hub_analysis.generate_clustering(cluster_form_minsize, cluster_form_samplesize, cluster_form_select_method, cluster_form_selectepsilon)
        elif cluster_form_select_type=='Neural':
            st.error('Not Yet Implemented')
    
    
    st.sidebar.markdown('# Experimental Smoothing')
    smoothing_form = st.sidebar.form(key='smoothing_form')
    smoothing_form_select_type = smoothing_form.selectbox(label='Cluster Type', options=('Tracking', 'Neural'))
    smoothing_form_gap_check = smoothing_form.checkbox(label='Gap Smoothing', key='smoothing_form_gap_check')
    smoothing_form_gap_size = smoothing_form.number_input(label='Gap Size', min_value=1, max_value=10, value=1, key='smoothing_form_gap_size')
    smoothing_form_jitter_check = smoothing_form.checkbox(label='Jitter Smoothing', key='smoothing_form_jitter_check')
    smoothing_form_jitter_size = smoothing_form.number_input(label='Minimum Instance Size', min_value=1, max_value=10, value=3, key='smoothing_form_jitter_size')
    
    smoothing_form_submit = smoothing_form.form_submit_button(label='Perform Smoothing')
    
    if smoothing_form_submit:
        if smoothing_form_select_type=='Tracking':
            labelset = hub_utils.get_labelset('HDB Labels')
            if labelset is not None:
                hub_analysis.generate_cluster_smoothing(labelset, gap=smoothing_form_gap_check, jitter=smoothing_form_jitter_check, gap_size=smoothing_form_gap_size, jitter_size=smoothing_form_jitter_size)
            else:
                st.sidebar.error('Ensure Clustering Labelset exists before smoothing')
        elif smoothing_form_select_type=='Neural':
            st.sidebar.error('Not Yet Implemented')
    
    st.sidebar.markdown("# Clustering Visualisations")
    
    display_options = ['Density Tree','Embedding Density','Clusters over Embedding','Smoothed Clusters']
    top_clustering_display = st.sidebar.selectbox(label='Top Display', key='top_clustering_display', options=display_options)
    st.session_state.top_clustering_display_options = st.empty()
    bottom_clustering_display = st.sidebar.selectbox(label='Bottom Display', key='bottom_clustering_display', options=display_options, index=2)
    st.session_state.bottom_clustering_display_options = st.empty()
    
    # TODO: top panel display based on sidebar selector. default to placeholder image
    # TODO: bottom panel display based on sidebar selector. default to placeholder image
    # main section
    st.markdown('# HDBClust Results')
    
    #TODO: Put in a container to control spacing
    display_clustering_vis()



clustering()
