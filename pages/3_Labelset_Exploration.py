#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Labelset Exploration/Details page for HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np

from hubdt import hub_utils, hub_constants, hub_analysis

from hubdt import hdb_clustering



def display_details_vis():
    
    draw_top_details_display(st.session_state.top_details_display)
    
    draw_bottom_details_display(st.session_state.bottom_details_display)
    
    return


def draw_top_details_display(disp_type):
    
    # TODO Single Labelset stats: Bar of avg continuous time in each cluster, bar of cluster size
    # for selected cluster: hist of membership confidence, hist of continuous time, eventually gif generated from longest example
    if disp_type=='Clusters over Embedding':
        embedding = hub_utils.get_embedding(st.session_state.details_top_embed_radio)
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting')
        elif embedding is None:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_top')
            fig = hdb_clustering.plot_hdb_over_tsne(embedding, st.session_state.hdb_labels, st.session_state.hdb_probs,noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    
    elif disp_type=='Clusters Overlay':
        embedding = hub_utils.get_embedding(st.session_state.details_top_embed_radio)
        baselabel_set = hub_utils.get_labelset(st.session_state.details_top_label_select)
        overlabel_set = hub_utils.get_labelset(st.session_state.details_top_over_label_select)
        if baselabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        elif embedding is None:
            st.error('Perform Embedding before Plotting')
        elif overlabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        else:
            if st.session_state.details_top_label_select=='HDB Labels':
                probs = st.session_state.hdb_probs
            else:
                probs = np.ones(len(baselabel_set))
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_top')
            fig = hdb_clustering.plot_hdb_over_tsne(embedding, baselabel_set, probs, compare_to=True, comp_label=(overlabel_set==st.session_state.details_top_over_label_slider).astype(bool),noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    
    elif disp_type=='Embedding Overlay':
        embedding = hub_utils.get_embedding(st.session_state.details_top_embed_radio)
        overlabel_set = hub_utils.get_labelset(st.session_state.details_top_over_label_select)
        if embedding is None:
            st.error('Perform Embedding before Plotting')
        elif overlabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        else:
            fig = hub_utils.draw_embedding_density(embed_type=st.session_state.details_top_embed_radio,compare_to=True, comp_label=(overlabel_set==st.session_state.details_top_over_label_slider).astype(bool))
            st.pyplot(fig)
    
    elif disp_type=='Labelset Statistics':
        label_set = hub_utils.get_labelset(st.session_state.details_top_label_select)
        if label_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        # Top Row: Set wide Plots
        top_row_cont = st.container()
        top_row_col1, top_row_col2 = top_row_cont.columns(2)
        top_row_col1.markdown('Mean Continuous Counts for Each Label')
        top_row_col1.pyplot(hub_utils.draw_cons_labelset(label_set))
        top_row_col2.markdown('Total Membership for Each Label')
        top_row_col2.pyplot(hub_utils.draw_labelset_sizes(label_set))
        # Bottom Row: Specific Label Plots
        bottom_row_cont = st.container()
        bottom_row_col1, bottom_row_col2 = bottom_row_cont.columns(2)        
        title1,fig1 = hub_utils.draw_selectable_single_plots(st.session_state.details_top_label_plot1_radio,st.session_state.details_top_label_select,label_set,st.session_state.details_top_label_slider)
        bottom_row_col1.markdown('**' + title1 + '**')
        bottom_row_col1.pyplot(fig1)
        title2,fig2 = hub_utils.draw_selectable_single_plots(st.session_state.details_top_label_plot2_radio,st.session_state.details_top_label_select,label_set,st.session_state.details_top_label_slider)
        bottom_row_col2.markdown('**' + title2 + '**')
        bottom_row_col2.pyplot(fig2)
        
    elif disp_type=='Comparative Statistics':
        label_set = hub_utils.get_labelset(st.session_state.details_top_label_select)
        if label_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        complabel_set = hub_utils.get_labelset(st.session_state.details_top_comp_label_select)
        if complabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        # Top Row Comparative Plots
        top_row_cont = st.container()
        top_row_col1, top_row_col2 = top_row_cont.columns(2)
        top_row_col1.markdown('**Comparative Purities**')
        top_row_col1.pyplot(hub_utils.draw_comparative_cluster_purities(hub_utils.generate_labelset_purities(label_set,complabel_set)))
        top_row_col2.markdown('**Purities Between Labelsets**')
        direction_radio = top_row_col2.radio(label="Direction",options=('Base -> Comp','Comp -> Base'))
        base_df, comp_df = hub_utils.draw_comparative_purities_df(hub_utils.generate_labelset_purities(label_set,complabel_set))
        if direction_radio=='Base -> Comp':
            top_row_col2.dataframe(base_df.style.pipe(hub_utils.df_styler_purities), height=150)
        elif direction_radio=='Comp -> Base':
            top_row_col2.dataframe(comp_df, height=150)
        
        # Bottom Row Selectable Comparative Plots
        bottom_row_cont = st.container()
        bottom_row_col1,bottom_row_col2 = bottom_row_cont.columns(2)        
        hub_utils.draw_selectable_comparative_plots(bottom_row_col1, st.session_state.details_top_comp_plot1_radio, label_set,complabel_set, st.session_state.details_top_label_slider)
        hub_utils.draw_selectable_comparative_plots(bottom_row_col2, st.session_state.details_top_comp_plot2_radio, label_set,complabel_set, st.session_state.details_top_label_slider)
        
        
    
    return






def draw_bottom_details_display(disp_type):
    # TODO Single Labelset stats: Bar of avg continuous time in each cluster, bar of cluster size
    # for selected cluster: hist of membership confidence, hist of continuous time, eventually gif generated from longest example
    if disp_type=='Clusters over Embedding':
        embedding = hub_utils.get_embedding(st.session_state.details_bottom_embed_radio)
        if 'hdbclust' not in st.session_state:
            st.error('Perform Clustering before Plotting')
        elif embedding is None:
            st.error('Perform Embedding before Plotting')
        else:
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_bottom')
            fig = hdb_clustering.plot_hdb_over_tsne(embedding, st.session_state.hdb_labels, st.session_state.hdb_probs,noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    
    elif disp_type=='Clusters Overlay':
        embedding = hub_utils.get_embedding(st.session_state.details_bottom_embed_radio)
        baselabel_set = hub_utils.get_labelset(st.session_state.details_bottom_label_select)
        overlabel_set = hub_utils.get_labelset(st.session_state.details_bottom_over_label_select)
        if baselabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        elif embedding is None:
            st.error('Perform Embedding before Plotting')
        elif overlabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        else:
            if st.session_state.details_bottom_label_select=='HDB Labels':
                probs = st.session_state.hdb_probs
            else:
                probs = np.ones(len(st.session_state.hdb_probs))
            hdb_clust_plot_noise = st.checkbox(label='Plot Noise', key='hdb_clust_plot_noise_bottom')
            fig = hdb_clustering.plot_hdb_over_tsne(embedding, baselabel_set, probs, compare_to=True, comp_label=(overlabel_set==st.session_state.details_bottom_over_label_slider).astype(bool),noise= hdb_clust_plot_noise)
            st.pyplot(fig)
    
    elif disp_type=='Embedding Overlay':
        embedding = hub_utils.get_embedding(st.session_state.details_bottom_embed_radio)
        overlabel_set = hub_utils.get_labelset(st.session_state.details_bottom_over_label_select)
        if embedding is None:
            st.error('Perform Embedding before Plotting')
        elif overlabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
        else:
            fig = hub_utils.draw_embedding_density(embed_type=st.session_state.details_bottom_embed_radio,compare_to=True, comp_label=(overlabel_set==st.session_state.details_bottom_over_label_slider).astype(bool))
            st.pyplot(fig)
    
    elif disp_type=='Labelset Statistics':
        label_set = hub_utils.get_labelset(st.session_state.details_bottom_label_select)
        if label_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        # Top Row: Set wide Plots
        top_row_cont = st.container()
        top_row_col1, top_row_col2 = top_row_cont.columns(2)
        top_row_col1.markdown('**Mean Continuous Counts for Each Label**')
        top_row_col1.pyplot(hub_utils.draw_cons_labelset(label_set))
        top_row_col2.markdown('**Total Membership for Each Label**')
        top_row_col2.pyplot(hub_utils.draw_labelset_sizes(label_set))
        # Bottom Row: Specific Label Plots
        bottom_row_cont = st.container()
        bottom_row_col1, bottom_row_col2 = bottom_row_cont.columns(2)        
        title1,fig1 = hub_utils.draw_selectable_single_plots(st.session_state.details_bottom_label_plot1_radio,st.session_state.details_bottom_label_select,label_set,st.session_state.details_bottom_label_slider)
        bottom_row_col1.markdown('**' + title1 + '**')
        bottom_row_col1.pyplot(fig1)
        title2,fig2 = hub_utils.draw_selectable_single_plots(st.session_state.details_bottom_label_plot2_radio,st.session_state.details_bottom_label_select,label_set,st.session_state.details_bottom_label_slider)
        bottom_row_col2.markdown('**' + title2 + '**')
        bottom_row_col2.pyplot(fig2)
        
    elif disp_type=='Comparative Statistics':
        label_set = hub_utils.get_labelset(st.session_state.details_bottom_label_select)
        if label_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        complabel_set = hub_utils.get_labelset(st.session_state.details_bottom_comp_label_select)
        if complabel_set is None:
            st.error('Ensure Selected Labelset is Generated/Loaded')
            return
        # Top Row Comparative Plots
        top_row_cont = st.container()
        top_row_col1, top_row_col2 = top_row_cont.columns(2)
        top_row_col1.markdown('**Comparative Purities**')
        top_row_col1.pyplot(hub_utils.draw_comparative_cluster_purities(hub_utils.generate_labelset_purities(label_set,complabel_set)))
        top_row_col2.markdown('**Purities Between Labelsets**')
        direction_radio = top_row_col2.radio(label="Direction",options=('Base -> Comp','Comp -> Base'))
        base_df, comp_df = hub_utils.draw_comparative_purities_df(hub_utils.generate_labelset_purities(label_set,complabel_set))
        if direction_radio=='Base -> Comp':
            top_row_col2.dataframe(base_df.style.pipe(hub_utils.df_styler_purities), height=150)
        elif direction_radio=='Comp -> Base':
            top_row_col2.dataframe(comp_df, height=150)
        
        # Bottom Row Selectable Comparative Plots
        bottom_row_cont = st.container()
        bottom_row_col1,bottom_row_col2 = bottom_row_cont.columns(2)        
        hub_utils.draw_selectable_comparative_plots(bottom_row_col1, st.session_state.details_bottom_comp_plot1_radio, label_set,complabel_set, st.session_state.details_bottom_label_slider)
        hub_utils.draw_selectable_comparative_plots(bottom_row_col2, st.session_state.details_bottom_comp_plot2_radio, label_set,complabel_set, st.session_state.details_bottom_label_slider)
        
            
    return



def details_sidebar_top(disp_type):
    
    # TODO: add block/time period labelset options
    #label_options = ['HDB Labels', 'Manual Labels', 'Full-depth Labels', 'Mid Labels', 'Rough Labels']
    
    if disp_type=='Clusters over Embedding':
        st.sidebar.radio(label='Embedding', key='details_top_embed_radio', options=('Tracking','Neural'))
        st.sidebar.selectbox(label='Labelset', key='details_top_label_select', options=st.session_state.label_options)
    elif disp_type=='Clusters Overlay':
        st.sidebar.radio(label='Embedding', key='details_top_embed_radio', options=('Tracking','Neural'))
        st.sidebar.selectbox(label='Base Labelset', key='details_top_label_select', options=st.session_state.label_options)
        overlabel_select = st.sidebar.selectbox(label='Overlay Labelset', key='details_top_over_label_select', options=st.session_state.label_options)
        overlabel_set = hub_utils.get_labelset(overlabel_select)
        if overlabel_set is not None:
            st.sidebar.slider(label='Select Label to Overlay', key='details_top_over_label_slider', min_value=0, max_value=int(np.max(overlabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
    elif disp_type=='Embedding Overlay':
        st.sidebar.radio(label='Embedding', key='details_top_embed_radio', options=('Tracking','Neural'))
        overlabel_select = st.sidebar.selectbox(label='Overlay Labelset', key='details_top_over_label_select', options=st.session_state.label_options)
        overlabel_set = hub_utils.get_labelset(overlabel_select)
        if overlabel_set is not None:
            st.sidebar.slider(label='Select Label to Overlay', key='details_top_over_label_slider', min_value=0, max_value=int(np.max(overlabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')        
    elif disp_type=='Labelset Statistics':
        label_plot_options = ['Mean Wavelets', 'Continuous Counts', 'Membership Confidence', 'Mean Firing Rate']
        labelset_select = st.sidebar.selectbox(label='Labelset', key='details_top_label_select', options=st.session_state.label_options)
        labelset = hub_utils.get_labelset(labelset_select)
        if labelset is not None:
            st.sidebar.slider(label='Select Label', key='details_top_label_slider', min_value=0, max_value=int(np.max(labelset)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
        st.sidebar.radio(label='Label Plot 1', key='details_top_label_plot1_radio',options=label_plot_options, index=1)
        st.sidebar.radio(label='Label Plot 2', key='details_top_label_plot2_radio',options=label_plot_options, index=2)
        
    elif disp_type=='Comparative Statistics':
        comp_plot_options = ['Scores','Overlap','Single Category Purity']
        st.sidebar.selectbox(label='Labelset', key='details_top_label_select', options=st.session_state.label_options)
        complabel_select = st.sidebar.selectbox(label='Comparison Labelset', key='details_top_comp_label_select', options=st.session_state.label_options)
        complabel_set = hub_utils.get_labelset(complabel_select)
        if complabel_set is not None:
            st.sidebar.slider(label='Select Label for single Category Comparisons', key='details_top_label_slider', min_value=0, max_value=int(np.max(complabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
        st.sidebar.radio(label='Label Comparison 1', key='details_top_comp_plot1_radio',options=comp_plot_options)
        st.sidebar.radio(label='Label Comparison 2', key='details_top_comp_plot2_radio',options=comp_plot_options)
    else:
        st.sidebar.error('Not Yet Implemented')
    return



def details_sidebar_bottom(disp_type):
    # TODO: add block/time period labelset options
    #label_options = ['HDB Labels', 'Manual Labels', 'Full-depth Labels', 'Mid Labels', 'Rough Labels']
    
    if disp_type=='Clusters over Embedding':
        st.sidebar.radio(label='Embedding', key='details_bottom_embed_radio', options=('Tracking','Neural'))
        st.sidebar.selectbox(label='Labelset', key='details_bottom_label_select', options=st.session_state.label_options)
    elif disp_type=='Clusters Overlay':
        st.sidebar.radio(label='Embedding', key='details_bottom_embed_radio', options=('Tracking','Neural'))
        st.sidebar.selectbox(label='Base Labelset', key='details_bottom_label_select', options=st.session_state.label_options)
        overlabel_select = st.sidebar.selectbox(label='Overlay Labelset', key='details_bottom_over_label_select', options=st.session_state.label_options)
        overlabel_set = hub_utils.get_labelset(overlabel_select)
        if overlabel_set is not None:
            st.sidebar.slider(label='Select Label to Overlay', key='details_bottom_over_label_slider', min_value=0, max_value=int(np.max(overlabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
    elif disp_type=='Embedding Overlay':
        st.sidebar.radio(label='Embedding', key='details_bottom_embed_radio', options=('Tracking','Neural'))
        overlabel_select = st.sidebar.selectbox(label='Overlay Labelset', key='details_bottom_over_label_select', options=st.session_state.label_options)
        overlabel_set = hub_utils.get_labelset(overlabel_select)
        if overlabel_set is not None:
            st.sidebar.slider(label='Select Label to Overlay', key='details_bottom_over_label_slider', min_value=0, max_value=int(np.max(overlabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')        
    elif disp_type=='Labelset Statistics':
        label_plot_options = ['Mean Wavelets', 'Continuous Counts', 'Membership Confidence', 'Mean Firing Rate']
        labelset_select = st.sidebar.selectbox(label='Labelset', key='details_bottom_label_select', options=st.session_state.label_options)
        labelset = hub_utils.get_labelset(labelset_select)
        if labelset is not None:
            st.sidebar.slider(label='Select Label', key='details_bottom_label_slider', min_value=0, max_value=int(np.max(labelset)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
        st.sidebar.radio(label='Label Plot 1', key='details_bottom_label_plot1_radio',options=label_plot_options, index=1)
        st.sidebar.radio(label='Label Plot 2', key='details_bottom_label_plot2_radio',options=label_plot_options, index=2)
        
    elif disp_type=='Comparative Statistics':
        comp_plot_options = ['Scores','Overlap','Single Category Purity']
        st.sidebar.selectbox(label='Labelset', key='details_bottom_label_select', options=st.session_state.label_options)
        complabel_select = st.sidebar.selectbox(label='Comparison Labelset', key='details_bottom_comp_label_select', options=st.session_state.label_options)
        complabel_set = hub_utils.get_labelset(complabel_select)
        if complabel_set is not None:
            st.sidebar.slider(label='Select Label for single Category Comparisons', key='details_bottom_label_slider', min_value=0, max_value=int(np.max(complabel_set)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
        st.sidebar.radio(label='Label Comparison 1', key='details_bottom_comp_plot1_radio',options=comp_plot_options)
        st.sidebar.radio(label='Label Comparison 2', key='details_bottom_comp_plot2_radio',options=comp_plot_options)
    else:
        st.sidebar.error('Not Yet Implemented')
    return




def labelset_details():
    #sidebar section
    st.sidebar.markdown('# Labelset Exploration')
    # TODO: selectors for two displays (with subselectors if needed) for:
    # cluster stats for each label set, cluster over tsne, highlight a cluster over tsne, label overlay on top of other labelling
    display_options = ['Clusters over Embedding', 'Clusters Overlay', 'Embedding Overlay', 'Labelset Statistics', 'Comparative Statistics']
    st.sidebar.markdown('## Top Display')
    top_details_display = st.sidebar.selectbox(label='Select Type', options=display_options, key='top_details_display')
    details_sidebar_top(top_details_display)
    st.sidebar.markdown('## Bottom Display')
    bottom_details_display = st.sidebar.selectbox(label='Select Type', options=display_options, key='bottom_details_display')
    details_sidebar_bottom(bottom_details_display)
    
    
    # main section
    st.markdown('# Label Visualisations')
    
    display_details_vis()
    

labelset_details()


