# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:25:49 2018

@author: NTU user
"""

#%%
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
import os
os.chdir(r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition')
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
#%%
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

print("Total Params:", FRmodel.count_params())
#FRmodel = load_model('face-rec_Google.h5')
#%%
def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)
    ### END CODE HERE ###
    
    return loss

#%%
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

#%%
model.fit(X, Y, epochs=1, batch_size=1)
#%%
database = {}
database["Hwee Young_1"] = img_to_encoding(r"hy4.jpg", FRmodel)
database["Hwee Young_2"] = img_to_encoding(r"hy5.jpg", FRmodel)
database["Hwee Young_3"] = img_to_encoding(r"hy6.jpg", FRmodel)


database["Tze Hong_1"] = img_to_encoding(r"th5.jpg", FRmodel)
database["Tze Hong_2"] = img_to_encoding(r"th6.jpg", FRmodel)
database["Tze Hong_3"] = img_to_encoding(r"th7.jpg", FRmodel)

database["Alson_1"] = img_to_encoding(r"alson1.jpg", FRmodel)
database["Alson_2"] = img_to_encoding(r"alson2.jpg", FRmodel)
database["Alson_3"] = img_to_encoding(r"alson3.jpg", FRmodel)

database["Jace_1"] = img_to_encoding(r"j1.jpg", FRmodel)
database["Jace_2"] = img_to_encoding(r"j2.jpg", FRmodel)
database["Jace_3"] = img_to_encoding(r"j3.jpg", FRmodel)




#%%

database = {}
database["jackneo_1"] = img_to_encoding(r"jackneo_1_bounded.jpg", FRmodel)
database["jackneo_2"] = img_to_encoding(r"jackneo_2_bounded.jpg", FRmodel)

database["joshua_1"] = img_to_encoding(r"joshua_1_bounded.jpg", FRmodel)
database["joshua_2"] = img_to_encoding(r"joshua_4_bounded.jpg", FRmodel)


database["marklee_1"] = img_to_encoding(r"marklee_1_bounded.jpg", FRmodel)
database["marklee_2"] = img_to_encoding(r"marklee_2_bounded.jpg", FRmodel)


database["tosh_1"] = img_to_encoding(r"tosh_1_bounded.jpg", FRmodel)
database["tosh_2"] = img_to_encoding(r"tosh_2_bounded.jpg", FRmodel)

database["Apple_1"] = img_to_encoding(r"apple_2_bounded.jpg", FRmodel)
database["Apple_2"] = img_to_encoding(r"apple_3_bounded.jpg", FRmodel)

#%%
def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".
    
    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras
    
    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    
    ### START CODE HERE ###
    
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
        
    ### END CODE HERE ###
        
    return dist, door_open

#%%
def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###
    
    if min_dist > 0.5:
        print("Not in the database." +'...'+image_path)
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist) +'...'+image_path)
        
    return min_dist, identity

#%%
def who_is_it_voting(image_path, database, model, n):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    
    ### START CODE HERE ### 
    
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)
    
    ## Step 2: Find the closest encoding ##
    
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    dist = np.array([])
    # Loop over the database to find the distance of each database image to our test image
    for (_, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.append(dist,np.linalg.norm(encoding-db_enc))
    
    min_n_idx = dist.argsort()[:n] # Get closest 5 images from db
    min_n_array = np.array(list(database.keys()))[min_n_idx] # Get name of 5 closest images from db

    for i in range(len(min_n_array)):
        # clean up names
        min_n_array[i] = min_n_array[i][:-2]

    counter = collections.Counter(min_n_array) # Count number of matches
    #print(counter)
    msk =  np.array(list(counter.values()))>= np.ceil(n/2) # Bool Mask for matches more than or equal 3
    
    
    
    if np.all(np.array(list(counter.values()))<np.ceil(n/2)):
        # If no faces in db with 3 or more matches
        print("Not in the database." +'...'+image_path  +'...'+image_path)
        return
    
    identity = np.array(list(counter.keys()))[msk][0]
    
    print ("it's " + str(identity) +'...'+image_path  +'...'+image_path)
    ### END CODE HERE ###
    
    return identity

#%%
def who_is_it_avg(image_path, database, model, sizefaceset=3):
    
    #distancevec stores the L2 distance between test face and database faces
    distancevec = np.empty(0)
    #Encode images
    encoding = img_to_encoding(image_path, model)
    
    for (name, db_enc) in database.items():  
        
            # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)
            #store into distance vec
        distancevec = np.append(distancevec,[dist],axis = 0)
        
    nfaces = len(list(database.keys())) #Number of faces in database
    #sizefaceset = 3 #fixed?? the number of each faces
    
    avgvec = np.array([]) #This vector stores the average distance
    for i in range(0,nfaces-1,sizefaceset):
        avgvec = np.append(avgvec, (1/sizefaceset)*np.sum(distancevec[i:i+sizefaceset]))
        
    #find largest 
    large = avgvec.argmin()
    
    namelist = []  #Stores the names eg.[Tzehong,Hweeyoung,Alston]
    bignamelist = (np.array(list(database.keys())))#Stores all keys [Tzehong_1,Tzehong_2,Hweeyoung_1...]
    
    for i in range(0,len(bignamelist)-1,sizefaceset):
        namelist.append(bignamelist[i][:-2])
    
    #print(avgvec)
    if avgvec[large] > 0.05:
        print("guess its weifeng" + ', alpha: ' +str(avgvec[large])) 
        
    else:
        print("The face is {}".format(namelist[large])+ ', alpha: ' +str(avgvec[large]))
    
    return
#%%    
    
who_is_it("images/camera_0.jpg", database, FRmodel)

#%%
image_dir = r'C:\Users\NTU user\Desktop\MAEC\Year 3\MH4510 Data Mining\MH4510 Face Recognition\images_straight\far\rotated\bounded'
images = os.listdir(image_dir)

for i in images:
    
    imageFile = i
    im = Image.open(imageFile)
    imshow(im)
    #who_is_it_voting(imageFile, database, FRmodel, 3)
    who_is_it_avg(imageFile, database, FRmodel, 3)
    #who_is_it(imageFile, database, FRmodel)

#%%#%%
who_is_it_voting(imageFile, database, FRmodel)
who_is_it_avg(imageFile, database, FRmodel)
