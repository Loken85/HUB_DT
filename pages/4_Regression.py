#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression page for HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np
import pickle

from hubdt import hub_utils, hub_constants, hub_analysis

from hubdt import b_utils





def component_regression():
    # sidebar section
    st.sidebar.markdown('# Component Regression')
    
    # GLM Selector and Form
    regression_type_radio = st.sidebar.radio(label='Regression Type', options=('Multifactor GLM', 'Single-factor GLM'), key='regression_type_radio')
    reg_tar_radio = st.sidebar.radio(label='Target Type', options=('Categorical','Numerical'), key='reg_tar_radio')
    reg_pred_radio = st.sidebar.radio(label='Predictor Type', options=('Categorical', 'Numerical'), key='reg_pred_radio')
    
    # TODO: move this options list to session state and write a wrapper to do name translation
    num_options = ['tracking','stbin','b_projections','s_projections']
    
    regression_form = st.sidebar.form(key='regression_form')
    
    if reg_tar_radio=='Categorical':
        reg_form_target_select = regression_form.selectbox(label='Select Target Labelset', options=st.session_state.label_options, key='reg_form_target_select')
        reg_form_tar_noise_check = regression_form.checkbox(label='Include Noise Target?', key='reg_form_tar_noise_check')
    elif reg_tar_radio=='Numerical':
        reg_form_target_select = regression_form.selectbox(label='Select Target Data', options=num_options, key='reg_form_target_select')
        reg_form_tar_pre_check = regression_form.checkbox(label='Preprocess Target?', key='reg_form_tar_pre_check')
    if reg_pred_radio == 'Categorical':
        reg_form_pred_select = regression_form.selectbox(label='Select Predictor Labelset', options=st.session_state.label_options, key='reg_form_pred_select')
        reg_form_pred_noise_check = regression_form.checkbox(label='Include Noise Predictor?', key='reg_form_pred_noise_check')
    elif reg_pred_radio == 'Numerical':
        reg_form_pred_select = regression_form.selectbox(label='Select Predictor Data',options=num_options, key='reg_form_pred_select')
    reg_form_pre_check = regression_form.checkbox(label='Preprocess Predictors?', key='reg_form_pre_check')
    
    reg_form_iter_num = regression_form.number_input(label='# Interations', min_value=100, max_value=2000, value=300)
    
    reg_form_precut_check = regression_form.checkbox(label='Precut Noise Bins?', key='reg_form_precut_check')
            
    reg_form_submit = regression_form.form_submit_button(label='Fit GLM')
    
    if reg_form_submit:
        
        if reg_tar_radio=='Categorical':
            tar_noise = reg_form_tar_noise_check
            tar_pre = True
        elif reg_tar_radio=='Numerical':
            tar_noise = True
            tar_pre = reg_form_tar_pre_check
            
        if reg_pred_radio=='Categorical':
            pred_noise = reg_form_pred_noise_check
            pred_pre = True
        elif reg_pred_radio=='Numerical':
            pred_noise = True
            pred_pre = reg_form_pre_check
            
        hub_analysis.generate_regression(regression_type_radio, reg_tar_radio, reg_form_target_select, reg_pred_radio, reg_form_pred_select, tar_noise=tar_noise, pred_noise=pred_noise, tar_pre=tar_pre, pred_pre=pred_pre, m_iter=reg_form_iter_num, cut_pre=reg_form_precut_check)
    
    # Display Regression Results if any
    if 'multiglm' in st.session_state:
        
        st.markdown('## Multifactor GLM Results')
        st.markdown('---')
        # Display Scores
        multiglm_scores1, multiglm_scores2, multiglm_scores3 = st.columns(3)
        multiglm_scores1.markdown('**Mean Squared Error:** {:0.3f}'.format(st.session_state.mse))
        multiglm_scores2.markdown('**Mean Absolute Error:** {:0.3f}'.format(st.session_state.mae))
        multiglm_scores3.markdown('**Mean Poisson Divergence:** {:0.3f}'.format(st.session_state.mpd))
        # Display r2s
        multiglm_scores1.markdown('**Mean r2 Value:** {:0.3f}'.format(st.session_state.multi_mean_r2s))
        multiglm_scores2.markdown('**Max r2 Value:** {:0.3f}'.format(st.session_state.multi_max_r2s))
        multiglm_scores3.markdown('**Summed r2 Value:** {:0.3f}'.format(st.session_state.multi_sum_r2s))
               
        #multiglm_r2select = multiglm_scores1.selectbox(label='R2 Values', options=('Raw', 'Mean', 'Sum', 'Max'))
        #multiglm_scores2.markdown('Display Values')
        #multiglm_plot_button = multiglm_scores2.button(label='Plot')
        st.pyplot(b_utils.plot_r2s(st.session_state.multi_raw_r2s))
        
        multiglm_plot_radio = st.radio(label='Plot Difference in Residuals', options=('True', 'False'))
        
        if multiglm_plot_radio == 'True':            
            
            st.pyplot(b_utils.plot_fr_props(b_utils.calc_fr_props(hub_utils.get_data(st.session_state.multiglm_tar_type,st.session_state.multiglm_targs), st.session_state.multi_raw_r2s)))
            
                
        
        
        
    else:
        st.markdown('Generate a Multi Component Regression to See Results')
        
    st.markdown('---')
    
    if 'singleglm' in st.session_state:
        
        st.markdown('## Single-Factor GLM Results')
        st.markdown('---')
        single_glm_stats1, single_glm_stats2, = st.columns(2)
        single_glm_stats1.markdown('**# of Significant Factors:** {:0.1f}'.format(st.session_state.num_sigrs))
        single_glm_stats2.markdown('**Proportion of Significant Factors:** {:0.2f}'.format(st.session_state.sig_prop))
        # TODO: Add plotting options
        st.pyplot(b_utils.plot_r2s(st.session_state.single_r2s))
        
    else:
        
        st.markdown('Generate a Single Component Regression to See Results')
    
    
    # Download section
    st.markdown('---')
    st.markdown('Export Regression Data')
    st.download_button(label='Download Multi-GLM',data=pickle.dumps(hub_analysis.generate_glm_dict(glm_type='multi')),file_name='current_multiglm_data.p')
    st.download_button(label='Download Single-GLM',data=pickle.dumps(hub_analysis.generate_glm_dict(glm_type='single')),file_name='current_singleglm_data.p')
    


component_regression()
