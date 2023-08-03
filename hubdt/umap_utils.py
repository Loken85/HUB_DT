#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:06:00 2023


UMAP functions

@author: proxy_loken
"""

import numpy as np

import umap


def umap_embedding(data, dims=2, neighbours=30, m_dist=0.0):
    
    umap_out = umap.UMAP(n_neighbors=neighbours, n_components=dims, min_dist=m_dist).fit_transform(data)
    
    return umap_out


def umap_visualisation(data, dims=2, neighbours=30):
    
    umap_out = umap.UMAP(n_neighbors=neighbours, n_components=dims).fit_transform(data)
    
    return umap_out
