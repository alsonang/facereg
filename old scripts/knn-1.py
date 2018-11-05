# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:29:22 2018

@author: NTU user
"""

#%%
database = {}
img_dir = "images_bounded"+"\\" +"database"+ "\\"
database["Hwee Young_1"] = img_to_encoding(img_dir + "hy_bounded0.jpg", FRmodel)
database["Hwee Young_2"] = img_to_encoding(img_dir + "hy_bounded1.jpg", FRmodel)
database["Hwee Young_3"] = img_to_encoding(img_dir + "hy_bounded2.jpg", FRmodel)
database["Hwee Young_4"] = img_to_encoding(img_dir + "hy_bounded3.jpg", FRmodel)
database["Hwee Young_5"] = img_to_encoding(img_dir + "hy_bounded4.jpg", FRmodel)

database["Tze Hong_1"] = img_to_encoding(img_dir + "th_bounded0.jpg", FRmodel)
database["Tze Hong_2"] = img_to_encoding(img_dir + "th_bounded1.jpg", FRmodel)
database["Tze Hong_3"] = img_to_encoding(img_dir + "th_bounded2.jpg", FRmodel)
database["Tze Hong_4"] = img_to_encoding(img_dir + "th_bounded3.jpg", FRmodel)
database["Tze Hong_5"] = img_to_encoding(img_dir + "th_bounded4.jpg", FRmodel)

database["Alson_1"] = img_to_encoding(img_dir + "alson_bounded0.jpg", FRmodel)
database["Alson_2"] = img_to_encoding(img_dir + "alson_bounded1.jpg", FRmodel)
database["Alson_3"] = img_to_encoding(img_dir + "alson_bounded2.jpg", FRmodel)
database["Alson_4"] = img_to_encoding(img_dir + "alson_bounded3.jpg", FRmodel)
database["Alson_5"] = img_to_encoding(img_dir + "alson_bounded4.jpg", FRmodel)

database["Jace_1"] = img_to_encoding(img_dir + "jace_bounded0.jpg", FRmodel)
database["Jace_2"] = img_to_encoding(img_dir + "jace_bounded1.jpg", FRmodel)
database["Jace_3"] = img_to_encoding(img_dir + "jace_bounded2.jpg", FRmodel)
database["Jace_4"] = img_to_encoding(img_dir + "jace_bounded3.jpg", FRmodel)
database["Jace_5"] = img_to_encoding(img_dir + "jace_bounded4.jpg", FRmodel)


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
    
    if min_dist > 0.07:
        print("Not in the database." +'...'+'aplha: '+str(dist) +'...'+image_path)
        return None
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
    if avgvec[large] > 0.06:
        print("Not in database" + ', alpha: ' +str(avgvec[large])) 
        
    else:
        print("The face is {}".format(namelist[large])+ ', alpha: ' +str(avgvec[large]))
    
    return

#%%
img_dir = "images_bounded"+"\\" +"test"+ "\\"
who_is_it_avg(img_dir + 'wf26.jpg', database, FRmodel, sizefaceset=5)

image_path = img_dir + 'wf1.jpg'

model= FRmodel

sizefaceset=5

#%%
img_dir = "images_bounded"+"\\" +"test"+ "\\"
images = os.listdir(img_dir)
for i in images:
    who_is_it(img_dir + i, database, FRmodel)