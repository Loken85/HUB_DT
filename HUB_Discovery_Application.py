#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:11:33 2023


NEW MULTI-PAGE VERSION OF HUB-D: Heirarchical Unsupervised Behavioural Discovery

This is the landing/front page of the Streamlit App



@author: proxy_loken
"""

import streamlit as st

import hub_utils
import hub_constants
import hub_analysis



def main():
    
    hub_utils.initialise_session()
    
    st.sidebar.success('Navigate to Session Management to Begin')
    
    # Front Page Image
    title_image = hub_utils.load_image('imgs/hub_dt_front.png')
    hub_utils.draw_image(title_image,'','')
    
    st.markdown('---')
    st.markdown("### An Application for Hierarchical Unsupervised Behavioural Discovery and Comparative Analysis with Electrophysiology Data")
    st.markdown('---')
    st.markdown("By")
    st.markdown("Adrian Lindsay (2023)")    
    st.markdown('Seamans Lab at the University of British Columbia')








if __name__ == "__main__":
    main()
