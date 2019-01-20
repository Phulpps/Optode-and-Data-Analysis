# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:01:57 2019

@author: Philipp
"""
import os
os.chdir('/Users/Philipp/Dropbox/Masterarbeit/img_processing_scripts')

import matplotlib.pyplot as plt
import oxygen_image_evaluation_bib as ox
import pandas as pd

fp = 'D:/Philipp_israel_data/Processed Optode Images/600 lmin s2/*.txt'
#fp = '/Users/Philipp/Downloads/600 lmin/*.txt' # for mac fp

################################################
#### defining constants of each experimental set
###############################################

o2_molar_sw = 233.03 # in mili mol/l
o2_conc_sw =  7.457 # in mg/l
o2_sat_sw = 94.6 #in % sat
hef = 13.45 # hyporheic exchange flux in cm/day
flume_width = 29 #in cm
flume_length = 240 #in cm
pixel_size =  0.00015625 # in cm²

###############################################
###############################################


oxygen_data = ox.load_ox_data(fp)

result_list = []
for index, ox_dat in enumerate(oxygen_data):
    
    do_conv = ox.ox_unit_conv(o2_sat_sw, o2_conc_sw, ox_dat)
    hz_size = ox.ox_zone_size(do_conv,2, pixel_size)
    hz_ox_mean = ox.ox_zone_mean(do_conv,2)
    hef_vol_bf = ox.volumetric_hef_bedform(hef, 15, flume_width)
    hz_vol_bf = ox.vol_oxic_zone(hz_size, flume_width)
    o2_mass_flux_bf = ox.get_O2_mass_flux(hef_vol_bf,o2_conc_sw)
    o2_mass_hz = ox.O2_mass_oxic_zone(hz_size, flume_width, hz_ox_mean)
    resp_rate_bf = ox.get_respiration_rate_hr(hz_size,hz_ox_mean,o2_mass_flux_bf,flume_width)
    
    name = '600 s2 Img-{}'.format(index+1)
    result_list.append([name, hz_size, hz_vol_bf,
                              hz_ox_mean,
                              o2_mass_hz,
                              resp_rate_bf])
    #respiration rate is mg of 02 respired per hour per litre of sediment
    df = pd.DataFrame(result_list, columns=['Name','Area of HZ in cm²', 'Volume of HZ in l',
                                                      'Mean DO conc in HZ in mg/l',
                                                      'O2 mass in hz in mg',
                                                      'Respiration [mg/L/hr]',])

os.chdir('D:/Philipp_israel_data//Processed Optode Images/') # set directory where to save csv file
    
df.to_csv('Results_600s2.csv')