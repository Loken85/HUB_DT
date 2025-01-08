#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session Loading/Details Page for HUB-D streamlit application


@author: proxy_loken
"""

import streamlit as st
import numpy as np
import pickle

from hubdt import hub_utils, hub_constants, hub_analysis, data_loading





def change_session():
    st.session_state.sess_name = st.session_state.session_select
    hub_utils.load_session()



def session_details_load():
    
    # sidebar section
    sess_load_form = st.sidebar.form(key='session_load_form')
    session_select = sess_load_form.selectbox("Load Session", key='session_select', options=st.session_state.session_names)
    
    sess_load_button = sess_load_form.form_submit_button(label='Load')
    if sess_load_button:
        change_session()
    
    # main section
    st.write('# Current Session: ', st.session_state.sess_name)
    st.write("---")
    st.markdown("## Session Details")
    sd_load_animal, sd_load_date, sd_load_name = st.columns(3)
    sd_load_animal = st.write('Animal: ', st.session_state.curr_sess.animal_id)
    sd_load_date = st.write('Session Date: ', st.session_state.curr_sess.session_date)
    sd_load_name = st.write('Full Session Name: ', st.session_state.curr_sess.session_name)
    
    # tabs processing
    tracking_tab, neural_tab, trimming_tab, labels_tab, projection_tab, embedding_tab = st.tabs(['Tracking','Neural','Trimming','Labels','Projections','Embedding'])
    
    with tracking_tab:
        st.markdown("### Tracking")
        if not st.session_state.is_tracking:
            st.warning('No Tracking Data Specified for Session')
        if st.session_state.tracking.any():
            st.write('Tracking Loaded: True --- Length: ' + str(np.size(st.session_state.tracking,0)))
        else:
            st.write('Tracking Loaded: False')
            
        if st.session_state.dlc and st.session_state.tracking_type=='dlc':
            feature_selection = True
            sess_feats = hub_utils.load_feats(st.session_state.curr_sess)
        else:
            feature_selection = False
            
        tracking_load_form = st.form(key='tracking_load_form')
        tracking_camera_radio = tracking_load_form.radio(label='Tracking type', options=['2 Camera 3D', 'Single Camera'], horizontal=True)
        if feature_selection:
            tracking_feature_select = tracking_load_form.multiselect(label='Select Features', options=sess_feats, default=sess_feats)
            if tracking_camera_radio == '2 Camera 3D':
                tracking_load_form.warning('Ensure Features for both Cameras are Selected')
        tracking_smooth_check = tracking_load_form.checkbox(label='Smooth Tracking', key='tracking_load_smooth_check')
        tracking_load_button = tracking_load_form.form_submit_button(label='Load Tracking')
        if tracking_load_button:
            if st.session_state.is_tracking:
                if feature_selection:
                    hub_utils.load_tracking(st.session_state.curr_sess, tracking_smooth_check, tracking_camera_radio, features=tracking_feature_select)
                else:
                    hub_utils.load_tracking(st.session_state.curr_sess, tracking_smooth_check, tracking_camera_radio, features=[])
        
        
    with neural_tab:
        st.markdown("### Neural Data")
        if not st.session_state.is_neural:
            st.warning('No Neural Data Specified for Session')
        if st.session_state.stbin.any():
            st.write('Neural Data: True --- Length: ' + str(np.size(st.session_state.stbin,0)))
        else:
            st.write('Neural Data: False')
            
        neural_load_form = st.form(key='neural_load_form')
        
        neural_bin_selector = neural_load_form.slider(label='Select Bin Size (ms)',min_value=10,max_value=2000,value=200)
        neural_load_button = neural_load_form.form_submit_button(label='Load Neural Data')
        if neural_load_button:
            if st.session_state.is_neural:
                hub_utils.load_neural(st.session_state.curr_sess, neural_bin_selector)
            else:
                st.error('No neural data to load')
        
    with trimming_tab:
        st.markdown('### Trimming')
        preprocess_form = st.form(key='preprocess_form')
        preprocess_threshold_tracking = preprocess_form.checkbox(label='Threshold Tracking', key='preprocess_track_thresh_check')
        preprocess_threshold_slider = preprocess_form.slider(label='Threshold (Pixels)', min_value=0, max_value=50, value=5)
        # TEMPORARY trimming tool
        # for debugging of mismatch length in labels 
        preprocess_trim_projs = preprocess_form.checkbox(label='Trim Projections (TEMP)', key='preprocess_trim_projs_check')
        preprocess_trim_length = preprocess_form.number_input(label='Trim Length', min_value=0, max_value=np.size(st.session_state.tracking,0), value=np.size(st.session_state.stbin,0))
        preprocess_submit = preprocess_form.form_submit_button(label='Run Pre-processing')
    
        if preprocess_submit:
            if preprocess_threshold_tracking:
                st.session_state.tracking = data_loading.threshold_tracking(st.session_state.tracking, threshold=preprocess_threshold_slider)
            if preprocess_trim_projs:
                st.session_state.stbin = st.session_state.stbin[0:preprocess_trim_length]
                st.session_state.b_projections = st.session_state.b_projections[0:preprocess_trim_length]
                st.session_state.b_tout = st.session_state.b_tout[0:preprocess_trim_length]
    
    with labels_tab:
        st.markdown("### Labels")
        if st.session_state.behav_labels.any():
            st.write('Behavioural Labels: True')
        else:
            st.write('Behavioural Labels: False')    
        if st.session_state.hdb_labels.any():
            st.write('Clustering Labels: True')
        else:
            st.write('Clustering Labels: False')    
        if st.session_state.context_labels.any():
            st.write('Context Labels: True')
        else:
            st.write('Context Labels: False')
            
        sd_labels_form = st.form(key='sd_labels_form')
        sd_labels_load_button = sd_labels_form.form_submit_button(label='Load Session Labels')
        
        if sd_labels_load_button:
            if st.session_state.is_labels:
                hub_utils.load_encodings(st.session_state.curr_sess)
            else:
                st.error('No labels to load')
        
    
    with projection_tab:
        st.markdown("### Projections")
        if st.session_state.b_projections.any():
            st.write('Tracking Projections: True '+str(np.shape(st.session_state.b_projections)))
        else:
            st.write('Tracking Projections: False')
    
        if st.session_state.s_projections.any():
            st.write('Neural Projections: True')
        else:
            st.write('Neural Projections: False')
            
        projections_form = st.form(key='projections_form')
        projections_form_select_type = projections_form.selectbox(label='Projection Type',options=('Tracking','Neural'))
        projections_form_minf = projections_form.slider(label='Minimum Frequency', min_value=0.1, max_value=1.0, value=0.5)
        projections_form_maxf = projections_form.slider(label='Maximum Frequency', min_value=0.5, max_value=30.0, value=5.0, step=0.5)
        projections_form_nump = projections_form.slider(label='Number of Periods', min_value=1, max_value=15, value=5)
        projections_form_submit = projections_form.form_submit_button(label='Generate Projections')
    
        if projections_form_submit:
            hub_analysis.generate_projections(projections_form_select_type,projections_form_minf,projections_form_maxf,projections_form_nump)
        
    with embedding_tab:
        st.markdown("### Embedding")
        if st.session_state.b_tout.any():
            st.write('Tracking Embedding: True '+str(np.shape(st.session_state.b_tout)))
        else:
            st.write('Tracking Embedding: False')
    
        if st.session_state.s_tout.any():
            st.write('Neural Embedding: True')
        else:
            st.write('Neural Embedding: False')
        
        embedding_select_input = st.radio(label='Embedding Input', options=['Tracking', 'Neural'])
        
        tsne_tab, umap_tab, topo_tab = st.tabs(['t-SNE', 'UMAP', 'Topology'])
        
        with tsne_tab:
            embedding_form = st.form(key='embedding_form_tsne')        
            embedding_form_perp = embedding_form.number_input(label='Perplexity', min_value=10, max_value=10000, value=50)
            embedding_form_submit = embedding_form.form_submit_button(label='Generate Embedding')
            embedding_form.text('Warning: May take considerable time')
            
            if embedding_form_submit:
                if embedding_select_input=='Tracking':
                    hub_analysis.generate_tracking_embedding_tsne(embedding_form_perp)
                elif embedding_select_input=='Neural':
                    hub_analysis.generate_neural_embedding_tsne(embedding_form_perp)
        
        with umap_tab:
            embedding_form = st.form(key='embedding_form_umap')
            embedding_form_nn = embedding_form.number_input(label='Neighbours', min_value=10, max_value=1000, value=30)
            embedding_form_mdist = embedding_form.slider(label='Min Distance', min_value=0.0, max_value=1.0, value=0.0)
            embedding_form_submit = embedding_form.form_submit_button(label='Generate Embedding')
            embedding_form.text('Warning: May take considerable time')
            
            if embedding_form_submit:
                if embedding_select_input=='Tracking':
                    hub_analysis.generate_tracking_embedding_umap(embedding_form_nn, embedding_form_mdist)
                elif embedding_select_input=='Neural':
                    hub_analysis.generate_neural_embedding_umap(embedding_form_nn, embedding_form_mdist)
        
        with topo_tab:
            embedding_form = st.form(key='embedding_form_topo')
            embedding_form_submit = embedding_form.form_submit_button(label='Generate Embedding')
            embedding_form.text('NOT YET IMPLEMENTED')
        
        
        
    
               
    



def session_details_import():
    
    # sidebar section
    st.sidebar.markdown("# Import Data to Session")
    st.sidebar.markdown('Note: Currently only supports Pickle format')
    
    # Main section
    st.write('# Current Session: ', st.session_state.sess_name)
    st.write("---")
    st.markdown('### Load Data')
    # Label Load Form
    label_load_options = ['HDB Tracking Labels', 'HDB Neural Labels', 'Comparative HDB Labels', 'External Comparative Labels', 'External Full Labels', 'Extneral Neural Labels']
    label_load = st.form(key='label_load_form')
    label_path = label_load.text_input(label='Enter Filepath of Labels')
    label_load_col1,label_load_col2 = label_load.columns(2)
    label_load_name = label_load_col1.text_input(label='Enter Name of Labelset')
    label_load_submit = label_load_col1.form_submit_button(label='Load Labels')
    label_load_select = label_load_col2.selectbox(label='Label Type', options=label_load_options)
    
    if label_load_submit:
        hub_utils.load_labels(label_path,label_load_select,label_load_name)
    
    # Embedding Load Form
    tsne_load = st.form(key='tsne_load_form')
    tsne_path = tsne_load.text_input(label='Enter Filepath of TSNE Embeddings')
    tsne_load_submit = tsne_load.form_submit_button(label='Load Embeddings')
    
    if tsne_load_submit:
        hub_utils.load_embeddings(tsne_path)
    


def session_details_export():
    
    #sidebar section
    st.sidebar.markdown("# Export Data from Session")
    
    # main section
    st.write('# Current Session: ', st.session_state.sess_name)
    st.write("---")
    # Downloads Section
    # TODO: add options for selecting specific data and changing filenames(auto use the session name)
    st.markdown('### Export Data')
    export_col1,export_col2,export_col3 = st.columns(3)
    export_col1.markdown('HDB Labels')
    export_col1.download_button(label='Download',data=pickle.dumps(hub_analysis.generate_hdb_dict()),file_name='current_hdb_labels.p')
    export_col2.markdown('Context Labels')
    export_col2.download_button(label='Download',data=pickle.dumps(hub_analysis.generate_context_dict()),file_name='current_context_labels.p')
    export_col2.markdown('Embeddings')
    export_col2.download_button(label='Download',data=pickle.dumps(hub_analysis.generate_embedding_dict()),file_name='current_embedding.p')
    export_col3.markdown('Session Data for MATLAB')
    export_col3.button(label='Download',on_click=hub_utils.save_matlab_data, args=(hub_analysis.generate_matlab_export_dict(),'working_data/current_sess_dict.mat'),key='download_sess_mat_button')
    export_col1.markdown('Aligned Neural Data')
    export_col1.download_button(label='Download', data=pickle.dumps(hub_analysis.generate_stbin_dict()),file_name='current_stbin.p')
    export_col3.markdown('Behaviour Projs for MATLAB')
    export_col3.button(label='Download', on_click=hub_utils.save_matlab_data, args=(hub_analysis.generate_proj_dict(proj_type='behaviour'),'working_data/current_bproj.mat'), key='download_bproj_mat_button')



def session_details():
    # sidebar section
    st.sidebar.markdown("# Session")
    st.sidebar.markdown("Current Session: "+ st.session_state.sess_name)
    sd_mode_select = st.sidebar.radio(label='Select Mode', options=['Load','Import','Export'], horizontal=True)
    
    if sd_mode_select == 'Load':
        session_details_load()
    elif sd_mode_select == 'Import':
        session_details_import()
    elif sd_mode_select == 'Export':
        session_details_export()
        
    
    # TODO: Image from the associated video
    
    
   




session_details()

