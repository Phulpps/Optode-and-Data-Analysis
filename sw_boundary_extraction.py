# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:43:45 2018

@author: philipp
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path

def get_trans_coords(img):
    """Function for getting the four x,y coordinates for the geometric transformation.
        Click rule: Upper left, upper right, lower left, lower right"""
        
    def onclick(event):
        # function to handle mouse events and get coordinates
        toolbar = plt.get_current_fig_manager().toolbar
        if toolbar.mode!='':
            print("clicked, but toolbar is in mode {:s}.".format(toolbar.mode))
        elif event.xdata and event.ydata is not None:
            ix, iy = event.xdata, event.ydata
            print( 'x = %d, y = %d ' %(ix, iy))
            coords.append((ix, iy))
            if len(coords) == 4:
                fig.canvas.mpl_disconnect(cid)
                plt.close()
        else:
            print("outside axes")  
            
    # plots the image for extracting coordinates and stores 4 pairs       
    fig = plt.figure()
    plt.imshow(img)
    plt.show()
    coords = [] 
    cid = fig.canvas.mpl_connect('button_press_event',onclick)
    
    return coords


def geo_trans(img,coords):
    ''' takes the coordinates of onclick functions and homogenize
    image size and perspective'''
    
    #pts1 takes coordinates from get_trans_coords function
    pts1 = np.float32(coords)
    #pts2 give shape of final img
    pts2 = np.float32([[0,0],[1200,0],[0,800],[1200,800]])
    # creates transformation matrix 
    M = cv.getPerspectiveTransform(pts1,pts2)
    # creates desired corrected img
    corrected_img = cv.warpPerspective(img,M,(1200,800))
    
    return corrected_img


def get_threshold_img(corrected_img,thresh_value):
    '''takes the geometrically corrected img and and a threshold value and gives
    threshold img threshold needs to be adjusted after every img set'''
    
    # split img and taking red channel, best contrast with red
    b, g, r = cv.split(corrected_img) 
    # add guassian blur to smooth the image
    greyscale = cv.GaussianBlur(r, (9, 9), 0)
    # creates binary image with given threshold
    ret_val, thresh_im = cv.threshold(greyscale, thresh_value, 255, cv.THRESH_BINARY)
    #smo0thes the binary image
    thresh_img = cv.medianBlur(thresh_im, 5)   
    
    return thresh_img


def open_close_im(binary_im, kernel=np.ones((16,16), np.uint8)):
    closed_im = cv.morphologyEx(binary_im, cv.MORPH_CLOSE, kernel)
    opened_im = cv.morphologyEx(closed_im, cv.MORPH_OPEN, kernel)
    return opened_im


def compare_images(img1, img2):
    '''plots two images to compare them'''
    
    plt.subplot(121),plt.imshow(cv.GaussianBlur(img1, (9, 9), 0), cmap = "gray"),plt.title('Digital Image')
    plt.subplot(122),plt.imshow(img2, cmap = "gray"),plt.title('Oxygen Distribution in the Sediment')
    plt.show()


def compare_hist(img1, img2):
    ''' compares the histograms of two images'''
    
    plt.subplot(121),plt.hist(img1.ravel(),256,[0,256]),plt.title('hist 1')
    plt.subplot(122),plt.hist(img2.ravel(),256,[0,256]),plt.title('hist 2')
    plt.show()


def sw_boundary_ox(ox_img, thresh_img):
    '''gives a raw optode image, remove marks with remove_pen_marks function'''
    
    # finds contours in binary image and sorts them after length
    im3, contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
    ox_img_swb = ox_img.copy()
    sorted_conts = sorted(contours, key=cv.contourArea, reverse=True)
    cont = sorted_conts[0]  # takes the largest contour   
    # Draw contour on optode img and use as black mask
    cv.drawContours(ox_img_swb, [cont], 0, (0, 0, 0), -1)
    
    return ox_img_swb


def get_ox_img(ox_img_swb):
    '''Takes output image from sw_boundary_ox function and removes the pen marks
    from the optode image.'''
    
    swb_img_copy = ox_img_swb.copy()
    #convert into gray value for thresholding
    gray = cv.cvtColor(swb_img_copy, cv.COLOR_RGB2GRAY)
    ret_val, thresh_im = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    #contours of pen marks get easily detected in binary image 
    im3, contours, hierarchy = cv.findContours(thresh_im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #sort contours and mask all contours except the largest
    sorted_conts = sorted(contours, key=cv.contourArea, reverse=True)
    cont = sorted_conts[1:] 
    swb_img = cv.drawContours(swb_img_copy, cont, -1, (0, 0, 0), -1)  
   
    return swb_img

    
def get_ox_dat(ox_img_swb,ox_data_trans):
    '''Takes output image from sw_boundary_ox function and removes the pen marks
    from the optode image. Also takes oxygen data from text file and masks surface water'''
    
    swb_img_copy = ox_img_swb.copy()
    #convert into gray value for thresholding
    gray = cv.cvtColor(swb_img_copy, cv.COLOR_RGB2GRAY)
    ret_val, thresh_im = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    #contours of pen marks get easily detected in binary image 
    im3, contours, hierarchy = cv.findContours(thresh_im, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #sort contours and mask all contours except the largest
    sorted_conts = sorted(contours, key=cv.contourArea, reverse=True)
    cont = sorted_conts[1:] 
    swb_img = cv.drawContours(swb_img_copy, cont, -1, (0, 0, 0), -1) 
    # take the processed image and creates another mask to keep only sediment oxygen values
    #for the data text file  
    mask_prep = cv.cvtColor(swb_img, cv.COLOR_RGB2GRAY)
    ret, sw_mask = cv.threshold(mask_prep, 1, 255, cv.THRESH_BINARY)
    ox_zone_data = cv.bitwise_and(ox_data_trans, ox_data_trans, mask = sw_mask)
    
    return ox_zone_data



    
if  __name__ == '__main__':
    
    fp =  'D:/Philipp_israel_data/digital_img/400 lmin/DSC_0008.jpg'
    fp_ox = 'D:/Philipp_israel_data/optode_img/preprocessed images/400 lmin/400 lmin_0004.jpg'

    ox_data = np.loadtxt('D:/Philipp_israel_data/optode_img/preprocessed images/400 lmin_oxdata/400 lmin_0004.txt')
    img = cv.imread(fp, -1)
    img_ox = cv.cvtColor(cv.imread(fp_ox, -1), cv.COLOR_BGR2RGB)
 
    trans_coord = get_trans_coords(img)    #get the 4 coordinate points first before 
    trans_coord_ox = get_trans_coords(img_ox)  #  executing other operations
  
    trans_img = geo_trans(img, trans_coord)
    threshh = 154
    thresh_img = get_threshold_img(trans_img,threshh )
    compare_images(trans_img, get_threshold_img(trans_img,threshh ))
    
    ox_data_trans = geo_trans(ox_data, trans_coord_ox) 
    ox_img_trans = geo_trans(img_ox, trans_coord_ox)
    ox_zone_raw = sw_boundary_ox(ox_img_trans, thresh_img)      
    ox_zone=get_ox_img(ox_zone_raw)
    ox_zone_data= get_ox_dat(ox_zone,ox_data_trans)
  
    # manually saving images
    os.chdir('D:\\Philipp_israel_data\\Results\\400 lmin')
    index = 4
    oname = 'oxic_zone_400_'+str(index)+'.jpg' 
    cv.imwrite(oname,cv.cvtColor(ox_zone,cv.COLOR_BGR2RGB)) 
    
    dname = 'oxic_zone_dat_400_'+str(index)+'.txt' 
    np.savetxt(dname,ox_zone_data)
    
    
###########
###########
########### batch processing tests 
       
    dig_img = [cv.imread(file, -1) for file 
               in glob.glob('D:/Philipp_israel_data/digital_img/900 lmin/*.jpg')]
    
    ox_img = [cv.cvtColor(cv.imread(file, -1), cv.COLOR_BGR2RGB) for file 
              in glob.glob('D:/Philipp_israel_data/optode_img/preprocessed images/900 lmin/*.jpg')]
    
    ox_data = [np.loadtxt(file) for file 
               in glob.glob('D:/Philipp_israel_data/optode_img/preprocessed images/900 lmin_oxdata/*.txt')]
 
    trans_coordb = get_trans_coords(dig_img[0])    #get the 4 coordinate points first before 
    trans_coord_oxb = get_trans_coords(ox_img[0])  #  executing other operations
    
    
    # script for batch processing images
    trans_imgb = [geo_trans(im, trans_coordb) for im in dig_img]
    #look for good threshold value manually here 
    threshold = 36
    b,g,r = cv.split(trans_imgb[0])
    compare_images(r, open_close_im(get_threshold_img(trans_imgb[0], threshold)))
    ########### Best classification results only with manuel threshold ######
    
    thresh_imgb = [get_threshold_img(dgeo, threshold) for dgeo in trans_imgb]
        
   # thresh_imgb = [open_close_im(mor) for mor in thresh_imgb]
        
    ox_img_transb = [geo_trans(img_ox, trans_coord_oxb) for img_ox in ox_img]
    ox_data_transb = [geo_trans(data_ox, trans_coord_oxb) for data_ox in ox_data]
      
    ox_zone_rawb = [sw_boundary_ox(ox, thr) for ox, thr in zip(ox_img_transb, thresh_imgb)]
    
    ox_zoneb = [get_ox_img(oi) for oi in ox_zone_rawb] 
    ox_zone_datab = [get_ox_dat(od, ot) for od, ot in zip(ox_zoneb, ox_data_transb)] 
  
    os.chdir('D:\\Philipp_israel_data\\Results\\900 lmin') # path to your results folder
    Path.cwd()
    
    digimg_col = [cv.cvtColor(digimg, cv.COLOR_BGR2RGB) for digimg in trans_imgb] # convert into rgb before creating digital image
    
    # looops for writing files
    
    for index, digim in enumerate(digimg_col):
        name = 'trans_dig_img_900_'+ str(index+1)+'.jpg'
        cv.imwrite(name, cv.cvtColor(digim,cv.COLOR_BGR2RGB))
    
    for index, oxim in enumerate(ox_zoneb):
        name = 'oxic_zone_900_'+str(index+1)+'.jpg' 
        cv.imwrite(name,cv.cvtColor(oxim,cv.COLOR_BGR2RGB)) 
        
    for index, oxdat in enumerate(ox_zone_datab):
        name = 'oxic_zone_dat_900_'+str(index+1)+'.txt' 
        np.savetxt(name,oxdat)
    
    ## loop for testing the ouput images
    for d,o in zip(digimg_col,ox_zoneb):
        plt.figure()
        compare_images(d,o)
        
    
    
    #ox_meanb, ox_areab = [get_pixel_ox(ox_data_result) for ox_data_result in ox_data_cropb]
        