# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 20:29:10 2018

@author: NTU user
"""

#%%
from PIL import Image
import pandas as pd
import os
from matplotlib.pyplot import imshow, figure
os.chdir(r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition')

#%%
image_dir = r'images_raw\\test'
images = os.listdir(image_dir)

#%% rotaion
for i in images:
    imageFile = i
    imageFile = 'images_raw\\test\\' + i
    im = Image.open(imageFile)
    im = im.rotate(90)
    figure()
    imshow(im)
    
    ext = ".jpg"
    new_name = 'images_raw\\test\\' + 'test'+str(i) + '_rotated'
    im.save(new_name + ext)
    
#%%
image_dir = r'images_raw\\th'
images = os.listdir(image_dir)

#%%
for i in images:
    imageFile = i
    imageFile = 'images_raw\\th\\' + i
    im = Image.open(imageFile)
    im = im.rotate(90)
    figure()
    imshow(im)
    
    ext = ".jpg"
    new_name = 'images_raw\\th\\' + 'th' + str(i) + '_rotated'
    im.save(new_name + ext)
    
#%%
image_dir = r'images_raw\\alson'
images = os.listdir(image_dir)

#%%
for i in images:
    imageFile = i
    imageFile = 'images_raw\\alson\\' + i
    im = Image.open(imageFile)
    im = im.rotate(90)
    figure()
    imshow(im)
    
    ext = ".jpg"
    new_name = 'images_raw\\alson\\' + 'alson' + str(i) + '_rotated'
    im.save(new_name + ext)
    
#%%
image_dir = r'images_raw\\hy'
images = os.listdir(image_dir)

#%%
def auto_crop(image_name, bounding_boxes):
    im = Image.open('images_raw\\'+image_name)
    
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
    
image_dir = r'images_raw\\test'
images = os.listdir(image_dir)

for i in range(len(images)):
    im_cropped = auto_crop('test\\'+images[i], bounding_boxes_compiled[i])
    ext = ".jpg"
    new_name = images[i][:-4] + '_bounded'
    im_cropped.save('images_bounded\\' + 'testbounded'+str(i) + ext, 'JPEG')
    
#%%
hy_array = np.empty([100,128])
image_dir = r'images_bounded\\hy'
images = os.listdir(image_dir)
for i in range(len(images)):    
    encoded = img_to_encoding('images_bounded\\hy\\' + images[i], FRmodel)
    hy_array[i,:] = encoded
    
#%%
th_array = np.empty([96,128])
image_dir = r'images_bounded\\th'
images = os.listdir(image_dir)
for i in range(len(images)):    
    encoded = img_to_encoding('images_bounded\\th\\' + images[i], FRmodel)
    th_array[i,:] = encoded    
   

#%%
alson_array = np.empty([100,128])
image_dir = r'images_bounded\\alson'
images = os.listdir(image_dir)
for i in range(len(images)):    
    encoded = img_to_encoding('images_bounded\\alson\\' + images[i], FRmodel)
    alson_array[i,:] = encoded    
    
#%% make X
X = np.vstack([hy_array, th_array])
X = np.vstack([X, alson_array])
#%% make Y

Y_hy = np.ones([100,1])
Y_th = np.ones([96,1])*2
Y_alson = np.ones([100,1])*3

Y = np.vstack([Y_hy, Y_th])
Y = np.vstack([Y, Y_alson])
