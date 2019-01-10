# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:43:45 2018

@author: philipp
"""
import cv2
import imutils as im
import glob
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Import test image and create versions of images need for later steps

test_fp = 'C:\\Users\\guest.IR1171\\Dropbox\\Masterarbeit\\Experiment images\\400 lmin\\DSC_0007.jpg'
test_im = cv2.imread(test_fp, -1)
test_im = im.resize(test_im, width=600)
test_im_alpha = test_im[:,:,2]
test_im_greyscale = cv2.cvtColor(test_im, cv2.COLOR_BGR2GRAY)
test_im_lab = cv2.cvtColor(test_im, cv2.COLOR_BGR2LAB)
lum, green_red, test_im_blue_yellow = cv2.split(test_im_lab)
ret_val, thresh_im = cv2.threshold(test_im_blue_yellow, 80, 255, cv2.THRESH_BINARY_INV)

#Convert the from BGR colourspace to LAB colour space, split channels and use 
#the blue/yellow channel for dye extract as this is where the great contrast between 
#blue dye and yellow sand can be seen.

rgba = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGBA)
images = [rgba, test_im_alpha, test_im_greyscale, test_im_blue_yellow]
titles = ["original", "alpha", "greyscale", "blue/yellow greyscale"]
fig, axes = plt.subplots(1,len(images))
for ax, im, title in zip(axes, images, titles):
    ax.imshow(im, cmap='gray')
    ax.set_title(title)
    
fig.set_size_inches(25, 16)

#To find the sediment-water inferface use morphological gradient (the different between 
#image erosion and image dilation) on the image with alpha channel.

kernel = np.ones((3,3), np.uint8)
sw_interface = cv2.morphologyEx(test_im_alpha, cv2.MORPH_GRADIENT, kernel)
ret, sw_interface = cv2.threshold(sw_interface, 1, 255,cv2.THRESH_BINARY)

fig, ax = plt.subplots()
ax.imshow(sw_interface, cmap='gray')
fig.set_size_inches(10, 10)

#Find all contours and then sort contours according to their areas. We're only interested 
# in the largest contour, this a quick and easy method to avoid including noise 
#(the dots). NB. No compression is used to store the coordinates of the shape as 
#I thought this would be easiest/most portable. If you'd like me to use 
#a compression method let me know!

im3, contours, hierarchy = cv2.findContours(sw_interface.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # No compress of contours
copy_orig_im = test_im.copy()
sorted_conts = sorted(contours, key=cv2.contourArea, reverse=True )

### Draw contour on original image
cv2.drawContours(copy_orig_im, sorted_conts, 0, (0,255,0), 3)
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(copy_orig_im, cv2.COLOR_BGR2RGB))
fig.set_size_inches(15, 10)

#Noise removal, dye extraction and finding contours of dye area¶
#Use gaussian blur to remove most of the noise, threshold image 
#(best on blue/yellow channel).
blur_im = cv2.GaussianBlur(test_im_blue_yellow, (5,5), 0)
ret, thresh_2 = cv2.threshold(blur_im,80, 255,cv2.THRESH_BINARY_INV)
#fig, ax = plt.subplots()
#ax.imshow(thresh_2, cmap='gray')
#Use bitwise and operation to subtract the 'water area' and
# be left with only the dye.
dye_im = cv2.bitwise_and(thresh_2, test_im_alpha, mask=test_im_alpha)
fig, ax = plt.subplots()
ax.imshow(dye_im, cmap='gray')
#Use morphological opening to move of the noise. NB. A compromise has to be 
#made between noise/accuracy, so far I've done what I thought to be appropriate. 
#Let me know if I should change something.

kernel = np.ones((3,3), np.uint8)
open_im = cv2.morphologyEx(dye_im, cv2.MORPH_OPEN, kernel, iterations=2)
fig, ax = plt.subplots()
ax.imshow(open_im, cmap='gray')

#Draw find all contours and draw them on the original image to check sufficiency

im3, contours_dye, hierarchy = cv2.findContours(open_im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
copy_orig_im_dye = test_im.copy()
sorted_conts_dye = sorted(contours_dye, key=cv2.contourArea, reverse=True )

### Draw contour on original image
cv2.drawContours(copy_orig_im_dye, sorted_conts_dye, -1, (0,255,0), 3)
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(copy_orig_im_dye, cv2.COLOR_BGR2RGB))
fig.set_size_inches(15, 10)
#fig.savefig('dye_extraction.png')

sw_interface_contours_df = pd.DataFrame(sorted_conts, columns=['Contours'])
dye_contours_df = pd.DataFrame(sorted_conts_dye, columns=['Contours'])

#xlist = [x[0][0] for x in sorted_conts[0]]
#ylist = [x[0][1] for x in sorted_conts[0]]

#fig, ax = plt.subplots()
#ax.plot(xlist, ylist)
#ax.set_ylim(ax.get_ylim()[::-1])
#fig.set_size_inches(10,4)

def cartesian_coords_extraction(sorted_cnts, no_to_extract):
    list_of_coords = []
    count = 0
    while count < no_to_extract:
        #for idx, each_cnt in enumerate(sorted_conts):
        x_coords = [coord_pair[0][0] for coord_pair in sorted_cnts[count]]
        y_coords = [coord_pair[0][1] for coord_pair in sorted_cnts[count]]
        list_of_coords.append([x_coords, y_coords])
        count += 1
    return list_of_coords
        
def create_csv(list_of_coords, name=""):    
    df = pd.DataFrame.from_dict(*list_of_coords)
    df = df.transpose() # x,y data in columns instead of rows
    no_of_col = len(df.columns)
    return df
        
sw_xy_coords = cartesian_coords_extraction(sorted_conts, 1)
sw_xy_coords_df = create_csv(sw_xy_coords)

fig, ax = plt.subplots()
ax.plot(sw_xy_coords_df[0], sw_xy_coords_df[1])
ax.set_ylim(ax.get_ylim()[::-1])
fig.set_size_inches(10,4)
ax.set_title('Sediment water interface')
fig.tight_layout()
fig.savefig('sediment_water_interface.png')
#sw_xy_coords_df.to_csv('sw_interface_contours.csv', index=False)

def get_xy_list_of_coordinates(sorted_contours):
    list_of_xy_coords = []
    list_of_col_names = []
    for num, contour in enumerate(sorted_contours):
        name_x, name_y = "X Contour {}".format(num + 1), "Y Contour {}".format(num + 1)
        x_coords = [x[0][0] for x in sorted_conts_dye[num]]
        y_coords = [y[0][1] for y in sorted_conts_dye[num]]
        list_of_xy_coords.extend([x_coords, y_coords])
        list_of_col_names.extend([name_x, name_y])
    return list_of_xy_coords, list_of_col_names


coords, names = get_xy_list_of_coordinates(sorted_conts_dye)
dye_df = pd.DataFrame(coords, index=names)
dye_df = dye_df.transpose().fillna('')

fig, ax = plt.subplots()

ax.plot(pd.DataFrame(coords, index=names).iloc[0], pd.DataFrame(coords, index=names).iloc[1])
ax.plot(pd.DataFrame(coords, index=names).iloc[2], pd.DataFrame(coords, index=names).iloc[3])
ax.plot(pd.DataFrame(coords, index=names).iloc[4], pd.DataFrame(coords, index=names).iloc[5])
ax.plot(pd.DataFrame(coords, index=names).iloc[6], pd.DataFrame(coords, index=names).iloc[7])
ax.plot(pd.DataFrame(coords, index=names).iloc[8], pd.DataFrame(coords, index=names).iloc[9])
ax.plot(pd.DataFrame(coords, index=names).iloc[10], pd.DataFrame(coords, index=names).iloc[11])

ax.set_ylim(ax.get_ylim()[::-1])
fig.set_size_inches(10,4)
ax.set_title('All dye contours')
fig.tight_layout()
#fig.savefig('all_dye_contours.png')

#dye_df.to_csv('dye_contours.csv', index=False)

fig, ax = plt.subplots()
ax.plot(sw_xy_coords_df[0], sw_xy_coords_df[1])

ax.plot(pd.DataFrame(coords, index=names).iloc[0], pd.DataFrame(coords, index=names).iloc[1])
ax.plot(pd.DataFrame(coords, index=names).iloc[2], pd.DataFrame(coords, index=names).iloc[3])
ax.plot(pd.DataFrame(coords, index=names).iloc[4], pd.DataFrame(coords, index=names).iloc[5])
ax.plot(pd.DataFrame(coords, index=names).iloc[6], pd.DataFrame(coords, index=names).iloc[7])
ax.plot(pd.DataFrame(coords, index=names).iloc[8], pd.DataFrame(coords, index=names).iloc[9])
ax.plot(pd.DataFrame(coords, index=names).iloc[10], pd.DataFrame(coords, index=names).iloc[11])
fig.set_size_inches(10,4)
ax.set_title('Sediment water interface and dye contours')
fig.tight_layout()
ax.set_ylim(ax.get_ylim()[::-1])
fig.savefig('swi_and_contours.png')