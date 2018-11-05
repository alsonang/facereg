# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:27:51 2018

@author: NTU user
"""
import os
os.chdir(r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition')

from full import *
from extract_face_v2 import *

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
#import cv2
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from matplotlib.pyplot import imshow
from PIL import Image
import collections
from keras.models import load_model

#%% Predefined things

gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel = load_model('face-rec_Google.h5')
#%%
name = 'th'
image_dir_raw = r'images_raw'+'//'+name
image_dir_rotated = r'images_rotated' +'\\' + name
image_dir_bounded= r'images_bounded' +'\\' + name


#%%
name_dict=dict()

name_dict[1] = 'alson'
name_dict[2] = 'th'
name_dict[3] = 'hy'
name_dict[4] = 'wf'
name_dict[5] = 'jace'
name_dict[6] = 'kelly'

#%%

#rotate_images(image_dir_raw, image_dir_rotated)

#%%
bounding_boxes_compiled = get_bounding_box_coord(image_dir_rotated)

#%%
bounding(image_dir_rotated, image_dir_bounded, bounding_boxes_compiled, name)

#%%
encoded_array = encode_img(image_dir_bounded, FRmodel, 0)

#%%
def full_stack(name, label):
    image_dir_raw = r'images_raw'+'//'+name
    image_dir_rotated = r'images_rotated' +'\\' + name
    image_dir_bounded= r'images_bounded' +'\\' + name
    
    #rotate_images(image_dir_raw, image_dir_rotated)
    #bounding_boxes_compiled = get_bounding_box_coord(image_dir_rotated)
    #bounding(image_dir_rotated, image_dir_bounded, bounding_boxes_compiled, name)
    encoded_array = encode_img(image_dir_bounded, FRmodel, label)
    
    return encoded_array


#%%
encoded_array_alson = full_stack('alson',1)
encoded_array_th = full_stack('th', 2)
encoded_array_hy = full_stack('hy', 3)
encoded_array_wf = full_stack('wf',4)
encoded_array_jace = full_stack('jace', 5)
encoded_array_kelly = full_stack('kelly', 6)
#%%
encoded_array_test = full_stack('test',99)
test2 = encoded_array_test[:,:-1]

#%%
test_answer = np.array([os.listdir('images_bounded\\test')]).T

#%%
train = encoded_array_alson
train = np.vstack([train, encoded_array_th])
train = np.vstack([train, encoded_array_hy])
train = np.vstack([train, encoded_array_wf])
train = np.vstack([train, encoded_array_jace])
train = np.vstack([train, encoded_array_kelly])

#%%
np.save('train_set_v2', train)
np.save('test_set_v2', test)

np.load('train_set.npy')



