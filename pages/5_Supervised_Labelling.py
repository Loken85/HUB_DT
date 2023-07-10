#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supervised Labelling/Learning page for HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np
import pickle

import hub_utils
import hub_constants
import hub_analysis

import random_forests


def random_forest_display(rf_type, display_type):
    
    
    if rf_type=='Single Class Forests':
        
        rf_models = st.session_state.rfs_models
        rf_perms = st.session_state.rfs_perms
        rf_probs = st.session_state.rfs_probs
        rf_scores = st.session_state.rfs_scores
        
        # TODO: Add Number of features selector for plot customisation
        
        if display_type=='Statistics':
            #TODO: Add further stats to this display
            st.pyplot(random_forests.plot_single_scores(rf_scores))
        elif display_type=='Feature Importances':
            st.pyplot(random_forests.plot_single_mdis(rf_models, num_feats=10))
        elif display_type=='Permutation Importances':
            st.pyplot(random_forests.plot_perm_importances(rf_perms, sort=True, num_feats=10))
        elif display_type=='Shapley Display':
            # TODO: Write a wrapper function for the shapley utils/options
            st.error('Not Yet Implemented')
        else:
            st.error('Not Yet Implemented')
        
    
    elif rf_type=='Multiclass Forest':
        
        rf_models = st.session_state.rf_model
        rf_perms = st.session_state.rf_perms
        rf_probs = st.session_state.rf_probs
        rf_scores = st.session_state.rf_scores
        
        if display_type=='Statistics':
            #TODO: add further stats to this display
            st.pyplot(random_forests.plot_scores(rf_scores))
        elif display_type=='Feature Importances':
            st.pyplot(random_forests.plot_mdi(rf_models, num_feats=10))
        elif display_type=='Permutation Importances':
            st.pyplot(random_forests.plot_perm_importance(rf_perms, num_feats=10, sort=True))
        elif display_type=='Shapley Display':
            # TODO: Write a wrapper function for the shapley utils/options
            st.error('Not Yet Implemented')
        else:
            st.error('Not Yet Implemented')





def supervised_labelling():
    
    st.sidebar.markdown('# Supervised Labelling')
    
    # Model Type selection
    model_type_radio = st.sidebar.radio(label='Model Type', options=('Random Forest', 'Neural Network'), key='superl_model_type_radio')
    
    # options for numerical data (this list is specific to the supervised labelling page)
    num_options = ['tracking','stbin','b_projections','s_projections', 'hdb_fr_resids', 'contextg_fr_resids']
    
    forest_types = ['Single Class Forests', 'Multiclass Forest']
    
    if model_type_radio == 'Random Forest':
        # Random Forest setup
        rf_type_radio = st.sidebar.radio(label='Forest Type', options=forest_types)
        
        # Random forest parameter form
        rf_form = st.sidebar.form(key='superl_rf_form')
        rf_target_select = rf_form.selectbox(label='Select Target Labelset', options=st.session_state.label_options, key='rf_form_target_select')
        rf_predictor_select = rf_form.selectbox(label='Select Predictor Data', options=num_options, key='rf_form_pred_select')
                
        rf_estims_number = rf_form.number_input(label='Number of Trees', min_value=10, max_value=1000, value=100, key='rf_form_estims_number')
        rf_leaf_number = rf_form.number_input(label='Samples per Leaf', min_value=1, max_value=100, value=10, key='rf_form_leaf_number')
        rf_features_select = rf_form.selectbox(label='Max Features per Split', options=('sqrt','log2','None'), key='rf_form_features_select')        
        
        rf_form_submit = rf_form.form_submit_button(label='Fit RF')
        
        
        if rf_form_submit:
            
            hub_analysis.generate_random_forest(rf_type_radio, rf_target_select, rf_predictor_select, rf_estims_number, rf_leaf_number, rf_features_select)
        
        
        
    elif model_type_radio == 'Neural Network':
        
        st.sidebar.error('Not Yet Implemented')
        
    # Display Selection
    
    # defined as two vars to allow for future customisation
    single_forest_displays = ['Statistics','Feature Importances', 'Permutation Importances', 'Shapley Display']
    multiclass_forest_displays = ['Statistics', 'Feature Importances', 'Permutation Importances', 'Shapley Display']
    
    rf_top_disp_radio = st.sidebar.radio(label='Top RF Display Type', options=forest_types, key='rf_top_disp_radio')
    if rf_top_disp_radio=='Single Class Forests':
        rf_top_disp_select = st.sidebar.selectbox(label='Display Type', options=single_forest_displays, key='rf_top_disp_select')
    elif rf_top_disp_radio=='Multiclass Forest':
        rf_top_disp_select = st.sidebar.selectbox(label='Display Type', options=multiclass_forest_displays, key='rf_top_disp_select')
    
      
    
    rf_bot_disp_radio = st.sidebar.radio(label='Bottom RF Display Type', options=forest_types, key='rf_bot_disp_radio')
    if rf_bot_disp_radio=='Single Class Forests':
        rf_bot_disp_select = st.sidebar.selectbox(label='Display Type', options=single_forest_displays, key='rf_bot_disp_select')
    elif rf_bot_disp_radio=='Multiclass Forest':
        rf_bot_disp_select = st.sidebar.selectbox(label='Display Type', options=multiclass_forest_displays, key='rf_bot_disp_select')
    
    
    # Main page displays
    
    if rf_top_disp_radio=='Single Class Forests':
        
        if st.session_state.rfs_single == True:
            
            random_forest_display(rf_top_disp_radio, rf_top_disp_select)
        
        else:
            
            st.error('Generate Single Class Random Forests before Plotting')
            
    elif rf_top_disp_radio=='Multiclass Forest':
        
        if st.session_state.rf_multi==True:
            
            random_forest_display(rf_top_disp_radio, rf_top_disp_select)
            
        else:
            
            st.error('Generate a Multiclass Random Forest before Plotting')
    
    
    if rf_bot_disp_radio=='Single Class Forests':
        
        if st.session_state.rfs_single == True:
            
            random_forest_display(rf_bot_disp_radio, rf_bot_disp_select)
        
        else:
            
            st.error('Generate Single Class Random Forests before Plotting')
            
    elif rf_bot_disp_radio=='Multiclass Forest':
        
        if st.session_state.rf_multi==True:
            
            random_forest_display(rf_bot_disp_radio, rf_bot_disp_select)
            
        else:
            
            st.error('Generate a Multiclass Random Forest before Plotting')
    
    
    # Download Section
    st.markdown('---')
    rfexport_col1, rfexport_col2 = st.columns(2)
    
    rfexport_col1.markdown('Export Random Forest Data')
    rfexport_col1.download_button(label='Download Single Class Forests Data',data=pickle.dumps(hub_analysis.generate_rf_dict(rf_type='singles')),file_name='current_singlesrf_data.p')
    rfexport_col1.download_button(label='Download Multiclass Forest Data',data=pickle.dumps(hub_analysis.generate_rf_dict(rf_type='multi')),file_name='current_multirf_data.p')
    
    rfexport_col2.markdown('Export Random Forest Models')
    if st.session_state.rfs_single:
        rfexport_col2.download_button(label='Download Single Class Forests Models', data=pickle.dumps(st.session_state.rfs_models),file_name='current_singlesrf_models.p')
    if st.session_state.rf_multi:
        rfexport_col2.download_button(label='Download Multiclass Forest Model', data=pickle.dumps(st.session_state.rf_model),file_name='current_multirf_model.p')



supervised_labelling()
