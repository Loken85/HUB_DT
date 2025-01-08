#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Exploration/Visualisation page for HUB-D streamlit application

@author: proxy_loken
"""

import streamlit as st
import numpy as np


from hubdt import hub_utils, hub_constants, hub_analysis

from hubdt import b_utils



def frame_vis_gif(position):
    # Gif generator section (use one of the displays to show the gif makin interface instead)
    st.markdown('GIF Generator')
    
    
    # get the frame index and image for display
    if position == 'top':
        # get frame index
        ind = st.session_state.fv_top_frame_slider
        # get labelset
        labelset = hub_utils.get_labelset(st.session_state.fv_top_labelset_select)
        camera = st.session_state.fv_top_camera_radio
    
        
    elif position == 'bottom':
        # get frame index
        ind = st.session_state.fv_bottom_frame_slider
        # get labelset
        labelset = hub_utils.get_labelset(st.session_state.fv_bottom_labelset_select)
        camera = st.session_state.fv_bottom_camera_radio
    
    frame = hub_utils.get_frame(ind, camera)   
    
    gif_gen_cont = st.container()
    gif_gen_left, gif_gen_right = gif_gen_cont.columns(2)
    
    gif_form = gif_gen_left.form(key='gif_gen_form')
    gif_type_radio = gif_form.radio(label='GIF Type', options=['Set Frames', 'Set Cluster'], key='gif_form_type_radio')
    gif_length_select = gif_form.number_input(label='Num Frames', min_value=1, max_value=100, value=10, key='gif_form_length_number')
    gif_framerate_select = gif_form.number_input(label='GIF Framerate', min_value=1, max_value=60, value=5, key='gif_form_framerate_number')
    gif_form.markdown('Note: Only Set Frames is currently functioning')
    #TODO: add filepath select, add functionality for set cluster(indexing needs to be refactored)
    gif_form_submit = gif_form.form_submit_button(label='Make GIF')
    
    if gif_form_submit:
        # do the thing
        frames = np.arange(frame, ind+gif_length_select)
        hub_utils.generate_gif(frames, camera, 'current_gif.gif', gif_framerate_select)   
    
   
    disp_frame = hub_utils.prep_frame(frame)    
    
    gif_gen_frame_cont = gif_gen_right.container()
    gif_gen_frame_cont.image(disp_frame, channels='BGR')
    
    return




def frame_vis_top():
    
    # get camera
    camera = st.session_state.fv_top_camera_radio      
    # get frame index
    ind = st.session_state.fv_top_frame_slider
    # get labelset
    labelset = hub_utils.get_labelset(st.session_state.fv_top_labelset_select)
    
    frame = hub_utils.get_frame(ind, camera)
        
    # Draw the selected frame
    # TODO: add option for drawing annotated frame (with DLC tracking)
    hub_utils.draw_frame(frame)
    # display frame data
    if labelset is not None:
        label = labelset[ind]
    else:
        label = -1
        
    data_string = 'Frame Index: ' + str(ind) + '  Label: ' + str(label)
    st.markdown(data_string)
    st.checkbox(label='Plot Cluster Overlay?', key= 'fv_top_overlay_check')
    # plot overlay if checked
    # TODO
    return


def frame_vis_bottom():
    
    # get camera
    camera = st.session_state.fv_bottom_camera_radio
    
    # get frame index
    ind = st.session_state.fv_bottom_frame_slider
    # get labelset
    labelset = hub_utils.get_labelset(st.session_state.fv_bottom_labelset_select)
            
    frame = hub_utils.get_frame(ind, camera)
        
    # Draw the selected frame
    # TODO: add option for drawing annotated frame (with DLC tracking)
    hub_utils.draw_frame(frame)
    # display frame data
    if labelset is not None:
        label = labelset[ind]
    else:
        label = -1
        
    data_string = 'Frame Index: ' + str(ind) + '  Label: ' + str(label)
    st.markdown(data_string)
    st.checkbox(label='Plot Cluster Overlay?', key= 'fv_bottom_overlay_check')
    # plot overlay if checked
    # TODO
    return



def frame_vis_sidebar_top(disp_type):
    # select labelset
    fv_top_labelset_select = st.sidebar.selectbox(label= 'Select Labelset', options= st.session_state.label_options, key= 'fv_top_labelset_select')
    labelset = hub_utils.get_labelset(fv_top_labelset_select)
    counts_list, max_inds = b_utils.count_consecutive_labels(labelset)
    if disp_type == 'All Frames':
        bin_tracking = st.session_state.tracking
        # Frame Selector
        if labelset is not None:
            max_frame = np.size(labelset,0)
        else:
            max_frame = (np.size(bin_tracking,0)-1)
        fv_top_frame_slider = st.sidebar.slider(label= 'Select Frame', key='fv_top_frame_slider', min_value=0, max_value=max_frame, value=0)
        
    elif disp_type == 'Frames by Cluster':    
        if labelset is not None:
            fv_top_label_slider = st.sidebar.slider(label= 'Select Label', key='fv_top_label_slider', min_value=0, max_value=int(np.max(labelset)), value=0)
            max_cons_string = 'Max Consecutive Index: ' + str(max_inds[fv_top_label_slider])
            st.sidebar.markdown(max_cons_string)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
            return
    
        # Frame Selector
        frame_indices = np.argwhere(labelset == fv_top_label_slider)
        fv_top_frame_slider = st.sidebar.select_slider(label= 'Select Frame', key='fv_top_frame_slider', options=frame_indices)
    
    return


def frame_vis_sidebar_bottom(disp_type):
    # select labelset
    fv_bottom_labelset_select = st.sidebar.selectbox(label= 'Select Labelset', options= st.session_state.label_options, key= 'fv_bottom_labelset_select')
    labelset = hub_utils.get_labelset(fv_bottom_labelset_select)
    if disp_type == 'All Frames':
        bin_tracking = st.session_state.bin_tracking
        # Frame Selector
        if labelset is not None:
            max_frame = np.size(labelset,0)
        else:
            max_frame = (np.size(bin_tracking,0)-1)
        fv_bottom_frame_slider = st.sidebar.slider(label= 'Select Frame', key='fv_bottom_frame_slider', min_value=0, max_value=max_frame, value=0)
        
    elif disp_type == 'Frames by Cluster':    
        if labelset is not None:
            fv_bottom_label_slider = st.sidebar.slider(label= 'Select Label', key='fv_bottom_label_slider', min_value=0, max_value=int(np.max(labelset)), value=0)
        else:
            st.sidebar.error('Ensure Selected Labelset is Generated/Loaded')
            return
    
        # Frame Selector
        frame_indices = np.argwhere(labelset == fv_bottom_label_slider)
        fv_bottom_frame_slider = st.sidebar.select_slider(label= 'Select Frame', key='fv_bottom_frame_slider', options=frame_indices)
        
    return




def frame_visualisations():
    #sidebar section
    st.sidebar.markdown('# Frame Exploration')
    # TODO: modify frame selector ui to fit here (add tracking overlay option)
    
    fv_disp_options = ['Full Image', 'GIF Generator']
    fv_frame_options = ['All Frames', 'Frames by Cluster']
    fv_camera_options = ['Stack', 'Camera 1', 'Camera 2']
    
    # Display Selection
    
    # top display selector
    fv_top_disp_radio = st.sidebar.radio(label= 'Top Display Type', options= fv_disp_options, key= 'fv_top_disp_radio')
    # frame selector
    fv_top_frame_radio = st.sidebar.radio(label='Top Frame Selection', options=fv_frame_options, key='fv_top_frame_radio')
    # camera selector (designed for two camera setup)
    fv_top_camera_radio = st.sidebar.radio(label= 'Top Display Camera', options= fv_camera_options, key= 'fv_top_camera_radio')
    
    
    frame_vis_sidebar_top(fv_top_frame_radio)
    
    # bottom display selector
    fv_bottom_disp_radio = st.sidebar.radio(label= 'Bottom Display Type', options= fv_disp_options, key= 'fv_bottom_disp_radio')
    # frame selector
    fv_bottom_frame_radio = st.sidebar.radio(label='Bottom Frame Selection', options=fv_frame_options, key='fv_bottom_frame_radio')
    # camera selector (designed for two camera setup)
    fv_bottom_camera_radio = st.sidebar.radio(label= 'Bottom Display Camera', options= fv_camera_options, key= 'fv_bottom_camera_radio')
    
    frame_vis_sidebar_bottom(fv_bottom_frame_radio)
    
    
    # main section
    
    # open video streams for frame grabbing
    stack, cam0, cam1 = hub_analysis.generate_video_streams()
    
    st.markdown('# Frame Display')
    # Call display functions
    if fv_top_disp_radio == 'Full Image':
        if fv_top_camera_radio == 'Camera 1' and cam0 == True:
            frame_vis_top()
        elif fv_top_camera_radio == 'Camera 2' and cam1 == True:
            frame_vis_top()
        elif fv_top_camera_radio == 'Stack' and stack == True:
            frame_vis_top()
        else:
            st.error('No Video Stream for Selected Camera. Make sure the filepath is correct')
    elif fv_top_disp_radio == 'GIF Generator':
        if fv_top_camera_radio == 'Camera 1' and cam0 == True:
            frame_vis_gif('top')
        elif fv_top_camera_radio == 'Camera 2' and cam1 == True:
            frame_vis_gif('top')
        elif fv_top_camera_radio == 'Stack' and stack == True:
            frame_vis_gif('top')
        else:
            st.error('No Video Stream for Selected Camera. Make sure the filepath is correct')
        
    if fv_bottom_disp_radio == 'Full Image':
        if fv_bottom_camera_radio == 'Camera 1' and cam0 == True:
            frame_vis_bottom()
        elif fv_bottom_camera_radio == 'Camera 2' and cam1 == True:
            frame_vis_bottom()
        elif fv_bottom_camera_radio == 'Stack' and stack == True:
            frame_vis_bottom()
        else:
            st.error('No Video Stream for Selected Camera. Make sure the filepath is correct')
    elif fv_bottom_disp_radio == 'GIF Generator':
        if fv_bottom_camera_radio == 'Camera 1' and cam0 == True:
            frame_vis_gif('bottom')
        elif fv_bottom_camera_radio == 'Camera 2' and cam1 == True:
            frame_vis_gif('bottom')
        elif fv_bottom_camera_radio == 'Stack' and stack == True:
            frame_vis_gif('bottom')
        else:
            st.error('No Video Stream for Selected Camera. Make sure the filepath is correct')


frame_visualisations()

