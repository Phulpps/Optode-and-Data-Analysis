# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:11:36 2019

@author: Philipp
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import natsort

pixel_size =  0.00015625 # in cm²
flume_width = 29 #in cm
flume_length = 240

def load_ox_data(fp):
    '''
    Loads all text files in the folder
    '''
    files = [np.loadtxt(file) for file in natsort.natsorted(glob.glob(fp))]
    
    return files
    

def ox_unit_conv(sat_sw, do_sw, sat_hz):
    ''' 
    takes the oxygen saturation of the surface water, the saturation from the hyporheic zone, 
    the calculated DO from the presens oxygen converter and calculates the DO of the hz
    ''' # testes linear relationship
    do_hz = (do_sw * sat_hz) / sat_sw
    return do_hz

def ox_zone_size(ox_zone_data, ox_thresh, pixel_size):    
    '''
    Takes an array of oxygen values and calculates the surface and mean oxygen saturation of the oxygenated zone. 
    ox_thresh = minimum saturation what is considered as oxic 
    pixel_size = surface of one pixel in cm²
    '''
    
    #creates condition and extraction to keep only oxygen values larger than threshold sat
    condition = (ox_zone_data>ox_thresh) == True 
    ox_pixel = np.extract(condition, ox_zone_data)
    ox_area = ox_pixel.shape[0]*pixel_size 
    
    return ox_area



def ox_zone_mean(ox_zone_data, ox_thresh):   
    '''
    Takes an array of oxygen values and calculates the mean oxygen saturation of the oxygenated zone. 
    ox_thresh = minimum saturation what is considered as oxic
    '''
    
    #creates condition and extraction to keep only oxygen values larger than threshold sat
    condition = (ox_zone_data>ox_thresh) == True 
    ox_pixel = np.extract(condition, ox_zone_data)
    # calculates mean of all oxygenated pixels
    ox_mean = np.mean(ox_pixel)
    
    return ox_mean


def ox_zone_split(ox_zone_data, thresholds, pixel_size):   
    '''
    Takes an array of oxygen values and calculates the surface and 
    mean oxygen saturation of the oxygenated zone. 
    thresholds = list of desired categories of oxygen saturation 
    pixel_size = surface of one pixel
    '''    
    ox_zone_splitted = []
    
    for thresh in thresholds:
        condition = (ox_zone_data>thresh) == True  #creates condition and extraction to keep only oxygen values larger than threshold sat
        ox_pixel = np.extract(condition, ox_zone_data)
        ox_area = ox_pixel.shape[0]*pixel_size
        ox_zone_splitted.append(ox_area)
        
    return ox_zone_splitted 



def volumetric_hef_flume(hef, flume_length, flume_width):
    '''
    If the specific flux is in [cm/d] and area is in [cm^2],
    the result will be in units of [l/d] 
    '''
    exchange_volume = (hef * flume_length*flume_width) / 1000
    
    return exchange_volume


def volumetric_hef_bedform(hef, bedform_length, flume_width):
    '''
    If the specific flux is in [cm/d] and area is in [cm^2],
    the result will be in units of [l/d] for the area of one bedform
    Hint: Its the same function like above, but it just calculates the influx
    in the area of one bedform instead of the whole flume.
    '''
    exchange_volume_bedform = (hef * bedform_length*flume_width) / 1000
    
    return exchange_volume_bedform



def vol_oxic_zone(oxic_zone_area, flume_width):
    '''
    Calculates volume of the HZ in liter
    '''
    return oxic_zone_area*flume_width  / 1000


def O2_mass_oxic_zone(oxic_zone_area, flume_width, O2_conc_oxic_zone):
    '''
    takes surface of oxic zone, flume width and O2 concentration of oxic zone and
    calculates the O2 mass in the oxic zone in mg
    '''
    return oxic_zone_area * flume_width * O2_conc_oxic_zone / 1000


def get_O2_mass_flux(exchange_volume, O2_concentration_sw):
    '''
    Takes hyporheic exchange volume in l/d and O2 conc and give o2 mass flux rate in mg/d
    '''
    o2_mass_flux = exchange_volume * O2_concentration_sw # mg/d
    
    return o2_mass_flux


def get_respiration_rate_hr(oxic_zone_area, O2_conc_oxic_zone, O2_mass_flux, flume_width):
    '''
    Takes the size of oxic zone, the O2 concentration in the zone, 
    the mass of O2 that goes into the zone and the flume width, 
    to get respiration rate as mg O2 per liter sediment per hour [mg/L/hr].
    '''
    ox_zone_vol = vol_oxic_zone(oxic_zone_area, flume_width)
    ox_zone_mass = O2_mass_oxic_zone(oxic_zone_area,flume_width, O2_conc_oxic_zone)
    ox_consumed_day = O2_mass_flux - ox_zone_mass
    respiration_rate_hr = ox_consumed_day / ox_zone_vol / 24  #in mg/hr*l
    
    return respiration_rate_hr 

