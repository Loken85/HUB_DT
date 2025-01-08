#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Utilities for HUB-D Streamlit Application



@author: proxy_loken
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import skimage.io as skio
from PIL import Image

from hubdt import b_utils, data_loading, cluster_metrics, hdb_clustering, video_utils, behav_session_params

from hubdt import hub_constants, hub_analysis


#INITIALISATION
def initialise_session():
    if 'session_names' not in st.session_state:
        st.session_state.session_names = behav_session_params.load_session_names()
    
    if 'sess_name' not in st.session_state:
        st.session_state.sess_name = hub_constants.sess_name
    
    if 'sess_loaded' not in st.session_state:
        st.session_state.sess_loaded = load_session()
        
    if 'b_projections' not in st.session_state:
        st.session_state.b_projections = np.zeros((2,2))
        
    if 's_projections' not in st.session_state:
        st.session_state.s_projections = np.zeros((2,2))
        
    if 'b_tout' not in st.session_state:
        st.session_state.b_tout = np.zeros((2,2))
        
    if 's_tout' not in st.session_state:
        st.session_state.s_tout = np.zeros((2,2))
    
    if 'hdb_labels' not in st.session_state:
        st.session_state.hdb_labels = np.zeros((2,2))
    
    if 'context_labels' not in st.session_state:
        st.session_state.context_labels = np.zeros((2,2))       
    if 'feats' not in st.session_state:
        #st.session_state.feats = ['Nose', 'Shoulder', 'Left-forepaw', 'Right-forepaw', 'Pelvis', 'Left-hindpaw', 'Right-hindpaw', 'Tail-base', 'Tail-mid']
        st.session_state.feats = []
        
    if 'label_options' not in st.session_state:
        st.session_state.label_options = ['HDB Labels', 'Manual Labels', 'Context Labels', 'Full-depth Labels', 'Mid Labels', 'Rough Labels', 'Smoothed Labels']
    
    if 'rfs_single' not in st.session_state:
        st.session_state.rfs_single=False
        
    if 'rf_multi' not in st.session_state:
        st.session_state.rf_multi = False
        
    if 'is_tracking' not in st.session_state:
        st.session_state.is_tracking = False
    
    if 'tracking_type' not in st.session_state:
        st.session_state.tracking_type= False
    
    if 'is_neural' not in st.session_state:
        st.session_state.is_neural = False
    
    if 'is_labels' not in st.session_state:
        st.session_state.is_labels = False
    
    return



# SAVING/LOADING


# load features from the DLC file for the current session
def load_feats(curr_sess):
    
    if st.session_state.dlc:
        all_feats = data_loading.dlc_get_feats(curr_sess)
    else:
        st.error('Feature Selection Requires DLC H5 File')
        return
       
    return all_feats



def load_tracking(curr_sess, smooth_tracking, tracking_type, features):
    
    
    if tracking_type == '2 Camera 3D':
        st.session_state.three_d =True
    else:
        st.session_state.three_d = False
    
    if curr_sess.tracking_mat:
        tracking = data_loading.load_tracking(curr_sess, smooth=smooth_tracking)
        st.session_state.dlc = False
        st.session_state.tracking_type = 'manual'
    elif curr_sess.tracking_h5:
        tracking = data_loading.load_tracking(curr_sess, st.session_state.three_d, dlc=True, feats=features)
        st.session_state.dlc = True
        st.session_state.tracking_type = 'dlc'
        st.session_state.feats = features
    elif curr_sess.tracking_pickle:
        # if you load directly from pickle, feature selection and tracking type don't work
        tracking = data_loading.load_tracking(curr_sess,from_pickle=True)
        st.session_state.dlc = True
        st.session_state.tracking_type = 'pickle'
    
    st.session_state.tracking = tracking


def load_encodings(curr_sess):
    
    # Not all sessions have supervised(Manual or Automatic) behavioural encodings
    if curr_sess.behav_mat:
        encodings = data_loading.load_encodings(curr_sess)
        encodings = data_loading.add_unlabelled_cat(encodings)
        encodings = b_utils.one_hot_to_vector(encodings)
        #stbin, encodings = data_loading.trim_to_match(stbin, encodings)
        st.session_state.behav_labels = encodings
        
        
        
    # Not all sessions have block start/end markers
    if curr_sess.neutral_start and 'bin_tracking' in st.session_state:
        context_labels, context_arr = data_loading.create_context_labels(curr_sess, st.session_state.bin_tracking)
        st.session_state.context_labels = context_labels
        
    

def load_neural(curr_sess, bin_size):
    
    
    if curr_sess.stmtx_mat:
        stbin = data_loading.load_stmtx(curr_sess)
    else:
        st.error('Missing Binned Neural Data File')
        return
        
    #convert bin size to a freq/fr
    bin_fr = 1/(bin_size/1000)
    # bin_tracking is downsampled to match the binned neural data
    if curr_sess.stmtx_mat and curr_sess.vid_fr and 'tracking' in st.session_state:
        bin_tracking = data_loading.convert_fr(st.session_state.tracking, curr_sess.vid_fr, bin_fr)
        stbin, bin_tracking = data_loading.trim_to_match(stbin, st.session_state.tracking)
        st.session_state.bin_tracking = bin_tracking
    st.session_state.stbin = stbin

    

def load_session():
    
    # TODO: add block and tone/outcome period label loading
    curr_sess = behav_session_params.load_session_params(st.session_state.sess_name)
    st.session_state.curr_sess = curr_sess
        
    clear_session()
    
    # TODO: Add functionality for loading multiple tracking sets at once
    if curr_sess.tracking_mat:    
        st.session_state.dlc = False
        st.session_state.is_tracking = True
        st.session_state.tracking_type = 'manual'
    elif curr_sess.tracking_h5:
        st.session_state.dlc = True
        st.session_state.is_tracking = True
        st.session_state.tracking_type = 'dlc'
    elif curr_sess.tracking_pickle:
        st.session_state.dlc = True
        st.session_state.is_tracking = True
        st.session_state.tracking_type = 'pickle'
    
    if curr_sess.stmtx_mat:
        st.session_state.is_neural = True
        
    if curr_sess.behav_mat or curr_sess.neutral_start:
        st.session_state.is_labels = True
    
    return True


# Helper to clear session variables before loading new ones. Should be called from the load function
def clear_session():    
    st.session_state.tracking = np.zeros((2,1))
    st.session_state.bin_tracking = np.zeros((2,1))
    st.session_state.stbin = np.zeros((2,1))
    st.session_state.behav_labels = np.zeros((2,1))
    st.session_state.hdb_labels = np.zeros((2,1))
    st.session_state.b_projections = np.zeros((2,1))
    st.session_state.s_projections = np.zeros((2,1))
    st.session_state.b_tout = np.zeros((2,1))
    st.session_state.s_tout = np.zeros((2,1)) 
    st.session_state.context_labels = np.zeros((2,1))
    st.session_state.feats = []
    st.session_state.tracking_type = False
    
    # existential vars
    st.session_state.is_tracking = False
    st.session_state.is_neural = False
    st.session_state.is_labels = False
    
    # GLM vars
    st.session_state.hdb_fr_glm = False
    st.session_state.context_fr_glm = False
    
    # Video Streams (technically, this should be a call to release, but on a file this should be done automatically)
    st.session_state.vidstream0 = False
    st.session_state.vidstream1 = False
    st.session_state.vidstream_stack = False
    
    return


@st.cache()
def load_labels(file,labels_type,name=None):
    # options = ['HDB Tracking Labels', 'HDB Neural Labels', 'Comparative HDB Labels', 'External Comparative Labels', 'External Full Labels', 'Extneral Neural Labels']
    labels_dict = data_loading.load_pickle(file)
    
    if labels_type=='External Full Labels':
        st.session_state.behav_labels = labels_dict['behav vector']
        st.session_state.full_clusters = labels_dict['full clusters']
        st.session_state.mid_clusters = labels_dict['mid clusters']
        st.session_state.rough_clusters = labels_dict['rough clusters']
    elif labels_type=='HDB Tracking Labels':
        st.session_state.hdb_labels = labels_dict['hdb_labels']
        st.session_state.hdb_probs = labels_dict['hdb_probs']
        
    elif labels_type=='HDB Neural Labels':
        st.session_state.hdbs_labels = labels_dict['hdb_labels']
        st.session_state.hdbs_probs = labels_dict['hdb_probs']
        if 'hdbs_labels' not in st.session_state.label_options:
            st.session_state.label_options.append('hdbs_labels')
    elif labels_type=='Comparative HDB Labels':
        name_l = name + '_labels'
        name_p = name + '_probs'
        st.session_state[name_l] = labels_dict['hdb_labels']
        st.session_state[name_p] = labels_dict['hdb_probs']
        if name_l not in st.session_state.label_options:
            st.session_state.label_options.append(name_l)
    elif labels_type=='External Comparative Labels':
        name_l = name + '_labels'
        st.session_state[name_l] = labels_dict[list(labels_dict.keys())[0]]
        if name_l not in st.session_state.label_options:
            st.session_state.label_options.append(name_l)
    
    return

@st.cache()
def load_embeddings(file):
    
    tsne_dict = data_loading.load_pickle(file)
    
    if 'projections tsne' in tsne_dict:
        st.session_state.b_tout = tsne_dict['projections tsne']
    if 'rfshap tsne' in tsne_dict:
        st.session_state.rfshap_tsne = tsne_dict['rfshap tsne']
    if 'st tsne' in tsne_dict:
        st.session_state.s_tout = tsne_dict['st tsne']


# NOTE: Only works if the app is running on localhost, otherwise this will require a built-in download button
def save_matlab_data(export_data,path):
    
    scipy.io.savemat(path,export_data)
    
    return

@st.cache()
def load_label_data(filename):
    l_data = data_loading.load_pickle(filename)
    behav_labels = l_data["behav labels"]
    behav_vector = l_data["behav vector"]
    full_labels = l_data["full clsuters"]
    mid_labels = l_data["mid clusters"]
    rough_labels = l_data["rough clusters"]
    
    return behav_labels, behav_vector, full_labels, mid_labels, rough_labels


#@st.cache()
def load_image(img_path):
    # TODO: grab frame from video using vid utils
    return skio.imread(img_path)






# FETCHING/CHECKING

# Helper to check for labelsets being generated/loaded
def label_exists(labelset):
    
    if labelset=='HDB Labels':
        return st.session_state.hdb_labels.any()
    elif labelset=='Manual Labels':
        if 'behav_labels' in st.session_state:
            return True
        else:
            return False
    elif labelset=='Full-depth Labels':
        if 'full_clusters' in st.session_state:
            return True
        else:
            return False
    elif labelset=='Mid Labels':
        if 'mid_clusters' in st.session_state:
            return True
        else:
            return False
    elif labelset=='Rough Labels':
        if 'rough_clusters' in st.session_state:
            return True
        else:
            return False
    elif labelset=='Smoothed Labels':
        if 'smoothed_labels' in st.session_state:
            return True
        else:
            return False    
    elif labelset in st.session_state:
        return True
    else:
        return False


# Helper for fectching the specified labelset
# Renders the label exist function redundant, simply check the return of this function for "None"
def get_labelset(labelset):
   
    if labelset=='HDB Labels':
        if st.session_state.hdb_labels.any():
            return st.session_state.hdb_labels
        else:
            return None
    elif labelset=='Manual Labels':
        if 'behav_labels' in st.session_state:
            return st.session_state.behav_labels
        else:
            return None
    elif labelset=='Full-depth Labels':
        if 'full_clusters' in st.session_state:
            return st.session_state.full_clusters
        else:
            return None
    elif labelset=='Mid Labels':
        if 'mid_clusters' in st.session_state:
            return st.session_state.mid_clusters
        else:
            return None
    elif labelset=='Rough Labels':
        if 'rough_clusters' in st.session_state:
            return st.session_state.rough_clusters
        else:
            return None
    elif labelset=='Context Labels':
        if st.session_state.context_labels.any():
            return st.session_state.context_labels
        else:
            return None
    elif labelset=='Smoothed Labels':
        if 'smoothed_labels' in st.session_state:
            return st.session_state.smoothed_labels
        else:
            return None
    elif labelset in st.session_state:        
        return st.session_state[labelset]
    else:
        return None


# Helper for fetchinga given embedding
# Returns "None" if embedding is not generated/loaded
def get_embedding(embedding):
    if embedding=='Tracking':
        if st.session_state.b_tout.any():
            return st.session_state.b_tout
        else:
            return None
    elif embedding=='Neural':
        if st.session_state.s_tout.any():
            return st.session_state.s_tout
        else:
            return None
    else:
        return None
    


# helper for fetching a given numerical set from the session state
# Returns "None" if set is not generated/loaded
def get_numset(numset):
    
    # options = ['tracking','stbin','b_projections','s_projections']
    
    if numset=='tracking':
        if 'tracking' in st.session_state:
            return st.session_state.tracking
        else:
            return None
    elif numset=='stbin':
        if 'stbin' in st.session_state:
            return st.session_state.stbin
        else:
            return None
    elif numset=='b_projections':
        if st.session_state.b_projections.any():
            return st.session_state.b_projections
        else:
            return None
    elif numset=='s_projections':
        if st.session_state.s_projections.any():
            return st.session_state.s_projections
        else:
            return None
    elif numset in st.session_state:
        if st.session_state[numset].any():
            return st.session_state[numset]
        else:
            return None
    else:
        return None


# Helper to call the correct data fetch function for a given type
def get_data(fetch_type, data_name):
    
    if fetch_type =='Categorical':
        fetch_data = get_labelset(data_name)
    elif fetch_type =='Numerical':
        fetch_data = get_numset(data_name)
    elif fetch_type == 'Embedding':
        fetch_data = get_embedding(data_name)
    else:
        fetch_data = None
        
    return fetch_data


def get_frame(ind, camera):
    
    # get the appropriate frame number, convert fr if necessary
    if st.session_state.curr_sess.sfreq != st.session_state.curr_sess.vid_fr:
        conversion = video_utils.calc_fr_conversion(st.session_state.curr_sess.sfreq, st.session_state.curr_sess.vid_fr)
        frame_ind = video_utils.index_to_frame(ind,conversion,st.session_state.curr_sess.frame_start)
    else:
        frame_ind = video_utils.index_to_frame(ind, offset=st.session_state.curr_sess.frame_start)
        
    
    if camera == 'Camera 1':
        vidstream = st.session_state.vidstream0
    elif camera == 'Camera 2':
        vidstream = st.session_state.vidstream1
    elif camera == 'Stack':
        vidstream = st.session_state.vidstream_stack
        
    frame = video_utils.extract_frame_st(vidstream, frame_ind)
    
    return frame

# Helper for generating gifs. Calls video utils frames_to_gif on a given camera and set of frames
def generate_gif(inds, camera, gif_name, gif_fr):
    
    if camera == 'Camera 1':
        vidstream = st.session_state.vidstream0
    elif camera == 'Camera 2':
        vidstream = st.session_state.vidstream1
    elif camera == 'Stack':
        vidstream = st.session_state.vidstream_stack
    
    # add offset if needed (this might be redundant)
    frames = inds+st.session_state.curr_sess.frame_start
    # create gif
    video_utils.frames_to_gif2(vidstream, frames, gif_name, gif_fr)
    


# DRAWING

# styler generater for styling df tables for display
def df_styler_purities(styler):
    styler.highlight_max(props='color:white;background-color:darkblue',axis=0)
    styler.format(precision=2)
    
    return styler


#@st.cache
def draw_hdb_tree(select_clusts=False, label_clusts=False, draw_eps=False):
    n_clusts = np.max(st.session_state.hdb_labels)+1
    return hdb_clustering.plot_condensed_tree(st.session_state.hdbclust, select_clusts=select_clusts,label_clusts=label_clusts, draw_epsilon=draw_eps, epsilon=st.session_state.hdb_eps, n_clusts=n_clusts)
    
    

#@st.cache()    
def draw_embedding_density(embed_type='Tracking',compare_to=False,comp_label=0):
    embedding = get_embedding(embed_type)
    if embedding is None:
        st.error('Generate Embedding before Plotting')
        return
    #calculate the density
    if embed_type == 'Tracking':
        if 'b_density' not in st.session_state:
            hub_analysis.generate_embedding_density(embed_type)
        density = st.session_state.b_density
        xi = st.session_state.b_xi
        yi = st.session_state.b_yi
    elif embed_type == 'Neural':
        if 's_density' not in st.session_state:
            hub_analysis.generate_embedding_density(embed_type)
        density = st.session_state.s_density
        xi = st.session_state.s_xi
        yi = st.session_state.s_yi
    
    #plot the resulting map
    fig, ax = plt.subplots()
    plt.pcolormesh(xi, yi, density)
    
    if compare_to:
        comp_points = embedding[comp_label,:]
        ax.scatter(comp_points[:,0], comp_points[:,1], s=50, marker='x', c='black', alpha=0.25)
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    return fig


def draw_cons_labelset(labelset):
    
    counts_list,max_inds = b_utils.count_consecutive_labels(labelset)
    means_list = b_utils.mean_consecutive_labels(counts_list)
    
    fig,ax = plt.subplots()
    
    ax.bar(x=range(0,len(means_list)),height=means_list,color='blue')
    
    #ax.xaxis.set_ticks([])
    
    return fig



def draw_labelset_sizes(labelset):
    
    label_sizes = []
    labelset = labelset.astype(int)
    for i in range(0,int(np.max(labelset)+1)):
        size = np.sum(labelset==i)
        label_sizes.append(size)
    
    fig, ax = plt.subplots()
    
    ax.bar(x=range(0,len(label_sizes)),height=label_sizes,color='purple')
    
    #ax.xaxis.set_ticks([])    
    
    return fig


def draw_selectable_single_plots(plot_type, label_type, labelset, label):
    
    # TODO: add avg FR plot
    if plot_type=='Mean Wavelets':
        #TODO: add capability for using neural projections
        title = ('Mean Wavelet Amplitudes for Label: ' + str(label))        
        fig = draw_mean_wavelets(labelset,label)
         
    elif plot_type=='Continuous Counts':
        title = ('Continuous Counts for Label: ' + str(label))
        fig = draw_continuous_counts(labelset,label)
        
    elif plot_type=='Membership Confidence':
        if label_type=='HDB Labels':
            probs = st.session_state.hdb_probs
        else:
            probs = np.ones(len(labelset))
        title = ('Membership Confidence for Label: ' + str(label))
        fig = draw_membership_confidence(labelset,label,probs)
        
    elif plot_type=='Mean Firing Rate':
        title = ('Mean Firing Rate for Label: ' + str(label))
        fig = draw_mean_fr_label(labelset, label)
    else:
        title=''
        fig=0
        st.error('Not Yet Implemented')
    
    return title,fig
    


def draw_mean_wavelets(labelset,label,proj_type='Tracking', wave_correct=True, response_correct=True, mean_response=True, color='lightblue'):
    
    if proj_type=='Tracking':
        projections = st.session_state.b_projections
    elif proj_type=='Neural':
        projections = st.session_state.s_projections
    else:
        st.error('Generate Projections before Plotting')
        return
    #correct feats list to account for camera input
    if st.session_state.three_d == True:
        feats = st.session_state.feats[0:(len(st.session_state.feats)//2)]
        cam_dims = 3
    else:
        feats = st.session_state.feats
        cam_dims = 2
    fig = b_utils.plot_cluster_wav_mags(projections, labelset, label, feats, st.session_state.freqs, dims=cam_dims, wave_correct = wave_correct, response_correct = response_correct, mean_response = mean_response, colour=color)
    
    return fig


def draw_continuous_counts(labelset,label,bins=20,color='green'):
    
    counts_list,max_inds = b_utils.count_consecutive_labels(labelset)
    fig, ax = plt.subplots()
            
    ax.hist(counts_list[label],bins=bins,color=color)
    
    return fig


def draw_membership_confidence(labelset,label,probs,bins=20,color='orange'):
    
    inds = labelset==label
    fig, ax = plt.subplots()
    
    ax.hist(probs[inds],bins=bins,color=color)
    
    return fig


def draw_mean_fr_label(labelset,label,color='brown'):
    
    stbin = st.session_state.stbin
    #print(np.shape(stbin))
    labelset, stbin = data_loading.trim_to_match(labelset, stbin)
    inds = labelset==label
    st_cat = stbin[inds,:]
    
    mean_frs = np.mean(st_cat,axis=0)
    
    fig, ax = plt.subplots()
    
    ax.bar(x=(range(0,np.size(st_cat,1))),height=mean_frs,color=color)
    #print(np.shape(st_cat))
    return fig


def draw_selectable_comparative_plots(container, plot_type, labelset,complabel_set, label):
    # make sure labelsets match in length
    labelset, complabel_set = data_loading.trim_to_match(labelset, complabel_set)    
    # options = ['Scores','Overlap','Single Category Purity']
    if plot_type=='Scores':
        draw_comparative_scores(container, labelset, complabel_set)
    elif plot_type=='Overlap':
        draw_comparative_overlap(container, labelset, complabel_set)
    elif plot_type=='Single Category Purity':
        container.markdown('**Single Category Purities**')
        fig = draw_comparative_single_purity(hub_analysis.generate_labelset_purities(labelset, complabel_set),label)
        container.pyplot(fig)
    else:
        return    
    
    return


def draw_image(image, header, description):
    
    # Draw header and image
    st.subheader(header)
    st.markdown(description)
    st.image(image.astype(np.uint8), use_column_width=True)    
    return


def draw_image_annotated(image, pose, header, description):
    # draw header and image
    st.subheader(header)
    st.markdown(description)
    frame = image.astype(np.float64)
    
    colours = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[125,50,50]]
    
    for i in range(0,5):
        frame[pose[i,1]-3:pose[i,1]+3,pose[i,0]-3:pose[i,0]+3,:] +=colours[i]
        frame[pose[i,1]-3:pose[i,1]+3,pose[i,0]-3:pose[i,0]+3,:] /=2
    
    st.image(frame.astype(np.uint8), use_column_width=True)
    return

# Note: this "draw" function operates on the container object itself and therefor does not return a figure
def draw_comparative_overlap(container,labelset,complabel_set):
    
    container.markdown('**Overlap Between Labelsets: Split-Join Distance**')
    noise_check = container.checkbox(label='Include Noise?')
    if noise_check:
        dists, max_base, max_comp = cluster_metrics.split_join_distance(labelset, complabel_set)
    else:
        dists, max_base, max_comp = cluster_metrics.split_join_distance(labelset, complabel_set, ignore_noise=True)
    
    container.markdown('Set Distances')
    container.markdown(' - - *Base -> Comp*: ' + str(dists[0]))
    container.markdown(' - - *Comp -> Base*: ' + str(dists[1]))
    
    base_df = pd.DataFrame(max_base)
    comp_df = pd.DataFrame(max_comp)
    container.markdown('---')
    container.markdown('Closest Matches for Base Labelset')
    container.dataframe(base_df.T)
    container.markdown('Closest Matches for Comparison Labelset')
    container.dataframe(comp_df.T)
    
    
    return


def draw_comparative_single_purity(purities,label,color='brown'):
    
    pur_arr = np.array(purities)
    
    cat_purs = pur_arr[:,label,1]
    
    comps = range(0,np.size(cat_purs,0))
    fig,ax = plt.subplots()
    
    ax.bar(x=comps,height=cat_purs,color=color)
    
    
    return fig


def draw_comparative_cluster_purities(purities,colors=['blue','purple']):
    
    max_purities = b_utils.compute_max_labelset_purities(purities, comp='clust')
    max_purities = np.array(max_purities)
    
    comps = range(0,np.size(max_purities,0))
    fig, axs = plt.subplots(2,1,sharex=True,sharey=True)
    
    axs[0].bar(x=comps,height=max_purities[:,1],color=colors[0])
    axs[1].bar(x=comps,height=max_purities[:,2],color=colors[1])
    
    for ax in axs.flat:
        ax.label_outer()
        
    return fig


def draw_comparative_purities_df(purities):
    
    # TODO: Add Styling and column/row labelling to the dataframe
    # Purity array is complabels x baselabels x 3
    pur_arr = np.array(purities)
    # each column sums to 1, shows where the mass of that base label is in the comp labels
    base_purs = pur_arr[:,:,1]
    # each row sums to 1, shows where the mass of the comp label is in the base labels
    comp_purs = pur_arr[:,:,2]
    
    base_df = pd.DataFrame(base_purs)
    comp_df = pd.DataFrame(comp_purs)
        
    return base_df, comp_df


# Note: this "draw" function operates on the container object itself and therefor does not return a figure
def draw_comparative_scores(container,labelset,complabel_set):
    
    container.markdown('**Comparative Metric Scores**')
    container.markdown('---')
    ami_score = cluster_metrics.calc_AMI(labelset,complabel_set)
    container.markdown('Adjust Mutual Information: ' + str(ami_score))
    container.markdown('---')
    rand_score = cluster_metrics.calc_RAND(labelset,complabel_set)
    container.markdown('Adjusted RAND Score: ' + str(rand_score))
    container.markdown('---')
    fmi_score = cluster_metrics.calc_fmi(labelset,complabel_set)
    container.markdown('Fowlkes-Mallow Index: ' + str(fmi_score))
        
    return

def draw_frame(frame):
    
    # convert to pil image
    pil_image = Image.fromarray(frame)
    
    st.image(pil_image, channels='BGR')

# helper to preprocess a frame to a streamlit displayable image
def prep_frame(frame):
    
    pil_image = Image.fromarray(frame)
    
    return pil_image



# MISC

@st.cache()
def get_category_frames(labels, frame_inds, category=0):
    # retrieves indices for frames from a given cluster/category
    inds = np.argwhere(labels[:,category])
    cluster_inds = frame_inds[inds]
    return cluster_inds



@st.cache()
# count consecutive occurances of labels in label array
def count_consecutive_labels(labels):
    counts_list = []
    
    for i in range(0,np.size(labels,1)):
        
        bool_arr = labels[:,i] == 1
        count = np.diff(np.where(np.concatenate(([bool_arr[0]],bool_arr[:-1] != bool_arr[1:], [True])))[0])[::2]
        counts_list.append(count)
        
    return counts_list


