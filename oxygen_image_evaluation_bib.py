# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:11:36 2019

@author: Philipp
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path


def load_ox_data(fp):
    ''' Loads all text files in the folder'''
    files = [np.loadtxt(file) for file in glob.glob(fp)]
    
    return files
    

def ox_zone_size(ox_zone_data, ox_thresh, pixel_area):   
    '''Takes an array of oxygen values and calculates the surface and mean oxygen saturation of the oxygenated zone. 
    ox_thresh = minimum saturation what is considered as oxic 
    pixel_area = surface of one pixel'''
    
    #creates condition and extraction to keep only oxygen values larger than threshold sat
    condition = (ox_zone_data>ox_thresh) == True 
    ox_pixel = np.extract(condition, ox_zone_data)
    ox_area = ox_pixel.shape[0]*pixel_area 
    
    return ox_area



def ox_zone_mean(ox_zone_data, ox_thresh):   
    '''Takes an array of oxygen values and calculates the mean oxygen saturation of the oxygenated zone. 
    ox_thresh = minimum saturation what is considered as oxic'''
    
    #creates condition and extraction to keep only oxygen values larger than threshold sat
    condition = (ox_zone_data>ox_thresh) == True 
    ox_pixel = np.extract(condition, ox_zone_data)
    # calculates mean of all oxygenated pixels
    ox_mean = np.mean(ox_pixel)
    
    return ox_mean


def ox_zone_split(ox_zone_data, thresholds, pixel_area):   
    '''Takes an array of oxygen values and calculates the surface and mean oxygen saturation of the oxygenated zone. 
    thresholds = list of desired categories of oxygen saturation 
    pixel_area = surface of one pixel'''
    
    #creates condition and extraction to keep only oxygen values larger than threshold sat
    ox_zone_splitted = []
    for thresh in thresholds:
        condition = (ox_zone_data>thresh) == True 
        ox_pixel = np.extract(condition, ox_zone_data)
        ox_area = ox_pixel.shape[0]*pixel_area
        ox_zone_splitted = ox_zone_splitted.append(ox_area)
        
    return ox_zone_splitted 

def swb_ox_an_change(ox_data_cropb):
    ''' creates figure of swb and ox/an change over time'''
    condition = ox_data_crop<10
    ox_data_crop[condition] = 0
    ox_data_crop[:,425:430]
    #still working on this
    np.concatenate((img1, img2), axis=1)