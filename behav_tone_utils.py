#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 10:09:13 2022

@author: proxy_loken

Utilities (data loading, formatting, etc) for the three context boxes task
Data sources are DLC annotations and frame ranges for tone periods

"""

import numpy as np
import scipy.io
import pandas as pd


# Current default parameters for these functions

feats = ["head","rightshoulder","leftshoulder", "tailbase"]

framerate = 30

sfreq = 30

minf = 0.5

maxf = 30

periods = 10

# End Parameters






# loader for DLC dataframe
def load_dlc_hdf(filename):
    
    dlc_df = pd.read_hdf(filename)
    
    return dlc_df


# loader for tones
def load_tone_ts(filename):
    
    mat = scipy.io.loadmat(filename)
    
    neutral_tones = mat['neutral_tone_frames']
    food_tones = mat['food_tone_frames']
    shock_tones = mat['shock_tone_frames']
    
    return neutral_tones, food_tones, shock_tones



# Function to generate frame range for tones
# Returns range as side by side start and end of range vectors
def gen_tone_frames(tones, frame_rate):
    
    frames = np.zeros((len(tones),2))
    
    for i in range(len(tones)):
        # start at start of first tone
        frames[i,0] = int(tones[i,0])
        # end at last tone + frame_rate (tones last one second)
        frames[i,1] = int(tones[i,2]+frame_rate)
    
    return frames.astype(int)


# Function for slicing DLC hdf dataframes for specific features
# ONLY WORKS ON DLC DATAFRAMES (the slicing is dependent on the multindex arrangement)
# Strips the "scorer" index (irrelevent in this analysis) 
def select_dlc_feats(data, feats):
    
    idx = pd.IndexSlice
    out = data.loc[:,idx[:,feats,:]].copy()    
    out = out.droplevel('scorer',axis=1)
    return out


# Function to pull x,y points from dataframe and turn into numpy array
def df_to_arr(data):
    
    darray = []
    
    for feature in list(data.T.index.get_level_values(0).unique()):
        darray.append(data.loc(axis=1)[feature,['x','y']].to_numpy())
    
    out = np.concatenate(darray, axis=1)
    
    return out


# Function to align to generated centre of mass
# TEMP: requires a specific set of features
def align_to_centre(df):
    
    data = df.copy()
    
    cen_x = (data.loc(axis=1)['rightshoulder','x'].to_numpy() + data.loc(axis=1)['leftshoulder','x'].to_numpy()) / 2
    
    cen_y = (data.loc(axis=1)['rightshoulder','y'].to_numpy() + data.loc(axis=1)['leftshoulder','y'].to_numpy()) / 2
    
    for feature in list(data.T.index.get_level_values(0).unique()):
        
        data.loc(axis=1)[feature,'x'] = data.loc(axis=1)[feature,'x']-cen_x
        data.loc(axis=1)[feature,'y'] = data.loc(axis=1)[feature,'y']-cen_y
        
    data.loc(axis=1)['com','x'] = cen_x
    data.loc(axis=1)['com','y'] = cen_y
    
    return data


# function for extracting the tone period from DLC data/projections/embeddings
# data should be a numpy array, frames is from the gen_tone_frames function
def extract_tone_frames(frames, data):
    
    extracted = []
    indices = []
    
    for i in range(len(frames)):
        
        extracted.append(data[frames[i,0]:frames[i,1],:])
        indices.append(np.arange(frames[i,0],frames[i,1]))
        
    return np.concatenate(extracted, axis=0), np.concatenate(indices, axis=0)

