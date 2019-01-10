# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:04:14 2018

@author: Philipp Wolke
"""

# -*- coding: utf-8 -*-
'''script for the processing of optode images'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.optimize import curve_fit
import cv2 as cv
import imutils as im
import os.path
import glob
import seaborn as sns
import uncertainties.unumpy as unp
import uncertainties as unc
# Constants

# water stage = 7 cm [Measured from the crest of the bedform to the left of the optode]
# flume width = 29 cm
# flume length = 260 cm [Measured from where the losing and gaining system is i.e. exclude ramp]

FLUME_LENGTH = 260  # cm -(40cm ramp, 10cm end) = -50 cm So
FLUME_WIDTH = 29 # cm

#Unisense Conversion
MOLECULAR_WEIGHT_O2 = 31.998
O2_µMOL_L = 257.9 # Original value provided by Unisense
O2_µG_L = O2_µMOL_L * MOLECULAR_WEIGHT_O2 # Multiply by the molecular weight of oxygen (31.998) to get microgram/l
O2_MG_L = O2_µG_L / 1000

# y_exp array of exchange fluxes from the relationship between velocity and hyporheic flux
#EEF = calc_volumetric_hyporheic_exchange_flux(y_exp, (29*260))

# Calculate the size represented by 1 pixel. Area covered by optodes is 10.5 cm deep by 14.1 width. 
# Size of image is 2281 px by 3469 px.

fp_anchor_120 = '' # PATH TO TYPICAL IMAGE (same dimensions)

# Optode foil is 10.5 cm in height
# Optode foil is 14.1 cm in length
IMAGE = cv.imread(fp_anchor_120)
OPTODE_H = 10.5 # Optode h in cm
OPTODE_L = 14.1 # Optode l in cm
OPTODE_AREA = OPTODE_H * OPTODE_L # Optode area in cm2
TOTAL_PX_IN_IMAGE = IMAGE.shape[0] * IMAGE.shape[1]
SQ_CM_PER_PX = OPTODE_AREA/TOTAL_PX_IN_IMAGE # cm2 / px

# Functions for display/plotting
def showimage(image, name="No name given"):
    cv.imshow(name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

from IPython.display import display_html

def display_side_by_side(*args):
    """To display df previews side by side."""
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
# Functions for conversions
def lmin_to_cm_s(lmin, depth, width=29):
    """Coverts L/min to cm/s given the water stage [cm] amd width [cm]."""
    cm3_min = lmin * 1000
    cm_s = cm3_min / 60
    area = depth * width
    cm_sec = cm_s / area
    return cm_sec

# Functions for calculations/fitting
def fit_exp_func(df):
    """This function fits data from salt tracer tests to get the relationship between velocity and hyporheic exchange flux. Returns paramter A and B which
    needed to calculate the HEF for a given velocity."""
    # Fit the function, y = A ^ Bx
    df = df.dropna()
    x, y = df['Velocity [cm/s]']. values, df['Hyporheic flux [cm/d]'].values
    result = curve_fit(lambda t, a, b: a * np.exp(b*t), x, y, p0=(4, 0.1))
    return result

def get_hyporheic_flux(lmin, depth, A, B):
    """Takes discharge [L/min], water stage [cm] along with A & B which are derived from fitting
    an exponential curve to the salt injection HEF data. Returns hyporheic exchange [cm/day]."""
    flux_cm_d = A * np.exp(B*lmin_to_cm_s(lmin, depth)) # cm / day
    return flux_cm_d

def get_o2_mass_flux_rate(hyporheic_flux_cmd, o2_concentration=O2_MG_L, flume_length=FLUME_LENGTH,
                          flume_width=FLUME_WIDTH):
    """Takes hyporheic flux in cm/d and give o2 mass flux rate in mg/d"""
    hyporheic_flux_vol = hyporheic_flux_cmd * flume_length * flume_width # cm3/day
    hyporheic_flux_vol_litres = hyporheic_flux_vol / 1000 # litres/day
    o2_mass_flux_rate = hyporheic_flux_vol_litres * o2_concentration # mg/d
    return o2_mass_flux_rate

def calc_volumetric_hyporheic_exchange_flux(specific_flux, area):
    """If the specific flux is in [cm/d] and area is in [cm^2],
    the result will be in units of [cm^3/d]"""
    effective_exchange_flux = specific_flux * area
    return effective_exchange_flux

def get_greyscale_im(fp):
    """Take a file path to an image and imports image as greyscale"""
    image = cv.imread(fp, 0) # Import image as greyscale
    return image

def threshold_im(image, threshold=47):
    """Returns a thresholded image"""
    ret, thresh_im = cv.threshold(image, threshold, 255, cv.THRESH_BINARY) # Keep px with value 100 and higher
    return thresh_im

def get_size_of_oxic_zone(thresh_im):
    oxic_zone_in_px = np.count_nonzero(thresh_im)
    return oxic_zone_in_px

def open_close_im(binary_im, kernel=np.ones((32,32), np.uint8)):
    closed_im = cv.morphologyEx(binary_im, cv.MORPH_CLOSE, kernel)
    opened_im = cv.morphologyEx(closed_im, cv.MORPH_OPEN, kernel)
    return opened_im

def get_centre_of_mass(im_fp, debug='n'):
    """Returns a tuple of x, y coordinates and area of oxic zone.
    Area from get_size_of_oxic_zone() more accurate."""
    thresholded_im = threshold_im(get_greyscale_im(im_fp), threshold=100)
    open_closed_im = open_close_im(thresholded_im, kernel=np.ones((32,32), np.uint8) )
    ret_im, cnts, hierarchy = cv.findContours(open_closed_im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt_1 = cnts[0] # This should be the largest contour
    moments = cv.moments(ret_im) # This should be the first contour in list of all contours
    centre_of_mass_x = int(moments['m10']/moments['m00'])
    centre_of_mass_y = int(moments['m01']/moments['m00'])
    num_of_pixels = np.count_nonzero(open_closed_im) # Check 2nd bedform removed from images
    if debug == 'y':
        print('Centre of mass is at x: {}, y: {}. Area of oxic zone is: {}.'.format(centre_of_mass_x, 
                                                                                centre_of_mass_y, num_of_pixels))
    return centre_of_mass_x, centre_of_mass_y, num_of_pixels

def convert_px_to_cm(size_in_px, sq_cm_per_px=SQ_CM_PER_PX):
    area_cm_2 = size_in_px * sq_cm_per_px
    return area_cm_2

def extract_oxic_area(fp, sq_cm_per_px=SQ_CM_PER_PX):
    """Takes a file path to an image, imports it as greyscale, thresholds this image and returns
    the area of the oxic zone in cm2."""
    image = get_greyscale_im(fp)
    thresh_im = threshold_im(image)
    oxic_zone_in_px = get_size_of_oxic_zone(thresh_im)
    area_cm_2 = convert_px_to_cm(oxic_zone_in_px, sq_cm_per_px=SQ_CM_PER_PX)
    return area_cm_2

def get_respiration_rate_hr(fp, number_of_bedforms=19, o2_flux_mg_d=None, width=FLUME_WIDTH,
                            sq_cm_per_px=SQ_CM_PER_PX):
    """Get respiration rate as mg O2 per litre sediment per hour [mg/L/hr]."""
    total_volume_of_oxic_zone_litres = (extract_oxic_area(fp, sq_cm_per_px=SQ_CM_PER_PX) *
                                        number_of_bedforms * width) / 1000
    o2_mg_L_hr = (o2_flux_mg_d / total_volume_of_oxic_zone_litres) / 24 # mg / L / hr
    return o2_mg_L_hr, total_volume_of_oxic_zone_litres

def return_velocity_deltas(df):
    """Takes a df, looks for column called 'Velocity [cm/s]', finds the delta between rows and 
    'wraps' to find the delta between the last row and the first row."""
    deltas = df['Velocity [cm/s]'].diff()
    deltas[0] = df.loc[0,'Velocity [cm/s]'] - df.loc[10,'Velocity [cm/s]']
    return deltas

def get_vector_magnitude(x, y):
    mag = np.sqrt(x**2 + y**2)
    return mag


def result_df(root_folder, paramsAB, water_stage_csv = 'water_stage_discharge_ideal.csv', full_O2_sat=O2_MG_L,
             debug='n', losing=0, sq_cm_per_px=SQ_CM_PER_PX):
    root_fp = '/PATHTOSAVE/{}/*.tif'.format(root_folder)
    list_of_fps = glob.glob(root_fp)
    water_stage_discharge = pd.read_csv(water_stage_csv)
    A, B = paramsAB

    respiration_rates = []
    for index, fp in enumerate(sorted(list_of_fps)):
        lmin = water_stage_discharge.iloc[index].values[1]
        water_stage = water_stage_discharge.iloc[index].values[0]
        velocity = lmin_to_cm_s(lmin, water_stage)
        CoM_x, CoM_y, _ = get_centre_of_mass(fp)
        
        hypo_flux = get_hyporheic_flux(lmin, water_stage, A, B)
        downward_flux = get_hyporheic_flux(lmin, water_stage, A, B) + losing
        o2_flux_rate_day = get_o2_mass_flux_rate(downward_flux, full_O2_sat) # rate in mg/day
        o2_flux_hr = o2_flux_rate_day / 24
        respiration_rate_hr, total_volume_of_oxic_zone_litres = get_respiration_rate_hr(fp, 
                                                                    o2_flux_mg_d=o2_flux_rate_day) # mg/L/hr
        name = 'Unsteady T-{}'.format(index)
        respiration_rates.append([name, respiration_rate_hr,
                                  velocity, o2_flux_hr,
                                  hypo_flux, downward_flux,
                                  total_volume_of_oxic_zone_litres,
                                  CoM_x, CoM_y])
        #respiration rate is mg of 02 respired per hour per litre of sediment
    df = pd.DataFrame(respiration_rates, columns=['Name', 'Respiration [mg/hr]',
                                                  'Velocity [cm/s]', 'O2 flux rate [mg/hr]',
                                                  'HEF [cm/d]', 'Downward Water Flux [cm/d]',
                                                  'Volume of oxic zone (L)',
                                                  'Centre of Mass x', 'Centre of Mass y'])
    if debug == 'y':
            print('Volume is: ', volume,'\n', 'File path was: ', 
              fp,'\n', 'Downward flux was: ', hypo_flux,'\n', 'O2 flux was:', o2_flux)
    df['Mag'] = get_vector_magnitude(df['Centre of Mass x'].diff(), df['Centre of Mass y'].diff())
    return df

# Example of extracting the oxic zone for methods/supporting information
'''test_losing_im = cv.imread('PATH_TO_IMAGE')

width = 800
threshold_value = 47
image = test_losing_im
image_fp = test_losing_fp
smaller_test_im = im.resize(image, width=width)
greyscale_im = get_greyscale_im(image_fp)
greyscale_im = im.resize(greyscale_im, width=width)
thresh_im = threshold_im(greyscale_im, threshold_value)
fig, axes = plt.subplots(1,2)
axes[0].imshow(cv.cvtColor(smaller_test_im, cv.COLOR_BGR2RGB))
axes[1].imshow(cv.cvtColor(thresh_im, cv.COLOR_GRAY2RGB))'''
#savepath = os.path.join(os.pardir, 'manuscript_figures/example_extracting_oxic.eps')
#fig.savefig(savepath)
