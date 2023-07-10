#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 12:26:24 2021


Video Utility functions for behavioural classification / clustering

@author: proxy_loken
"""
import numpy as np
import imageio

import cv2


# Helper to open video stream for streamlit app
def open_stream(filepath):
    
    vidstream = cv2.VideoCapture(filepath)
    
    return vidstream



# Function to extract a set of specific frames from a video and save them as .pngs 
def extract_frames(filepath, frames):
    
    vid = cv2.VideoCapture(filepath)
    
    for i in frames:
        vid.set(1,i)
        
        success, frame = vid.read()
        
        if not success:
            continue
        
        cv2.imwrite(f'{filepath}_frame{i}.png', frame)
        

# Function to extract and return a particular frame for a video
def extract_frame(filepath, frame):
    
    vid = cv2.VideoCapture(filepath)
    
    vid.set(1,frame)
    
    success, im = vid.read()
    
    if not success:
        return 0
    
    return im
    


def extract_frame_st(vid, ind):
    
    vid.set(1, ind)
    
    success, frame = vid.read()
    
    if not success:
        return 0
    
    return frame


# Function to take a set of frames and write to a gif
def frames_to_gif(filepath, frames, gif_name, fps=5):
    
    vid = cv2.VideoCapture(filepath)
    
    image_list = []
    
    for i in frames:
        vid.set(1,i)
        
        success, frame = vid.read()
        
        if not success:
            continue
        
        
        image_list.append(frame)
        
    imageio.mimsave(gif_name, image_list, fps=fps)


# Same as above, takes in an already open videostream 
def frames_to_gif2(vidstream, frames, gif_name, fps=5):
    
    vid = vidstream
    
    image_list = []
    
    for i in frames:
        vid.set(1,i)
        
        success, frame = vid.read()
        
        if not success:
            continue
        
        
        image_list.append(frame)
        
    imageio.mimsave(gif_name, image_list, fps=fps)

