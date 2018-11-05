# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 15:22:02 2018

@author: NTU user
"""
#%%
from PIL import Image
import os
from matplotlib.pyplot import imshow, figure
os.chdir(r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition')

#%%
image_dir = r'images_raw\\hy'
images = os.listdir(image_dir)


#%% ROTATE

for i in images:
    imageFile = i
    im = Image.open(imageFile)
    im = im.rotate(90)
    figure()
    imshow(im)
    
    ext = ".jpg"
    new_name = imageFile[:-4]+'_rotated'
    im.save(new_name + ext)

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

for i in range(len(images)):
    im_cropped = auto_crop(images[i], bounding_boxes_compiled[i])
    ext = ".jpg"
    new_name = images[i][:-4] + '_bounded'
    im_cropped.save('images_bounded\\' + new_name + ext, 'JPEG')
