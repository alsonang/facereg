# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:29:10 2018

@author: NTU user
"""

#%%
from PIL import Image
import pandas as pd
import os
import numpy as np
from fr_utils import *
from matplotlib.pyplot import imshow, figure
#os.chdir(r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition')
os.chdir(r'/Users/alson/Downloads/MH4510 Face Recognition-2/')





#%%
'''
image_dir_raw = r'images_raw\\test'
images = os.listdir(image_dir_raw)
image_dir_rotated = r'images_rotated\\test'
image_dir_bounded= r'images_bounded\\test'
'''
#%%
def rotate_images(image_dir_raw, image_dir_rotated):
    images = os.listdir(image_dir_raw)
    for i in images:
        imageFile = i
        #imageFile = image_dir_raw + '\\' +i
        imageFile = image_dir_raw + '/' + i
        print(imageFile)
        im = Image.open(imageFile)
        im = im.rotate(90)
        figure()
        imshow(im)
        
        ext = ".jpg"
        #new_name = image_dir_rotated + '\\' +'test'+str(i) + '_rotated'
        new_name = image_dir_rotated + '/' +'test'+str(i) + '_rotated'
        im.save(new_name + ext)

#%%
def auto_crop(image_name, bounding_boxes):
    im = Image.open(image_name)
    
    add_w = bounding_boxes[2]
    add_h = bounding_boxes[3]

    start_w = bounding_boxes[0]
    start_h = bounding_boxes[1]
    
    im_cropped = im.crop((start_w, start_h, add_w, add_h))
    imshow(im_cropped)
    
    rescale_width = 96
    rescale_height = 96
    im_scaled = im_cropped.resize((rescale_width, rescale_height), Image.ANTIALIAS)
    imshow(im_scaled)
    
    return im_scaled

#%%
'''
image_dir_rotated = r'images_rotated\\test'
images = os.listdir(image_dir_rotated)
'''
#%%
def bounding(image_dir_rotated, image_dir_bounded, bounding_boxes_compiled, name):
    images = os.listdir(image_dir_rotated)
    for i in range(len(images)):
        im_cropped = auto_crop(image_dir_rotated +'\\'+ images[i], bounding_boxes_compiled[i])
        ext = ".jpg"
        im_cropped.save(image_dir_bounded +'//' +name+'_bounded'+str(i) + ext, 'JPEG')

#%%
'''
bounding(images, image_dir_rotated, bounding_boxes_compiled)
'''
#%%
'''
image_dir_bounded = r'images_bounded\\test'
images = os.listdir(image_dir_bounded)
'''
#%%

def encode_img(image_dir_bounded, FRmodel, label):
    images = os.listdir(image_dir_bounded)
    N = len(images)
    encoded_array = np.empty([N,128])
    for i in range(len(images)):    
    #    encoded = img_to_encoding(image_dir_bounded+ '\\' + images[i], FRmodel)
        encoded = img_to_encoding(image_dir_bounded + '/' + images[i], FRmodel)
        encoded_array[i,:] = encoded
        
    labels = np.ones([N,1])*label
    encoded_array = np.hstack([encoded_array, labels])
    return encoded_array
    
 


#%%
'''
 make X
X = np.vstack([hy_array, th_array])
X = np.vstack([X, alson_array])
#%% make Y

Y_hy = np.ones([100,1])
Y_th = np.ones([96,1])*2
Y_alson = np.ones([100,1])*3

Y = np.vstack([Y_hy, Y_th])
Y = np.vstack([Y, Y_alson])
'''