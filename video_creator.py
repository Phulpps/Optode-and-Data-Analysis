# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 22:34:02 2019

@author: Philipp
"""

import cv2 as cv
import glob
import os

images = []

for image in glob.glob('D:/Philipp_israel_data/Processed Optode Images/800 lmin/*.jpg'):
    img = cv.imread(image)
    height, width, layers = img.shape
    size = (width,height)
    images.append(img)
    
os.chdir('D:/Philipp_israel_data/Processed Optode Images/') 

out = cv.VideoWriter('800 lmin video.mp4',cv.VideoWriter_fourcc(*'mp4v'), 1, size)



for i in range(len(images)):
    out.write(images[i])
    
out.release()
