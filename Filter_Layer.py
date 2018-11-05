# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:29:22 2018

@author: NTU user
"""

#%%
database = {}
img_dir = "images_bounded"+"\\" +"database"+ "\\"
database["3_Hwee Young_1"] = img_to_encoding(img_dir + "hy_bounded0.jpg", FRmodel)
database["3_Hwee Young_2"] = img_to_encoding(img_dir + "hy_bounded1.jpg", FRmodel)
database["3_Hwee Young_3"] = img_to_encoding(img_dir + "hy_bounded2.jpg", FRmodel)
database["3_Hwee Young_4"] = img_to_encoding(img_dir + "hy_bounded3.jpg", FRmodel)
database["3_Hwee Young_5"] = img_to_encoding(img_dir + "hy_bounded4.jpg", FRmodel)

database["2_Tze Hong_1"] = img_to_encoding(img_dir + "th_bounded0.jpg", FRmodel)
database["2_Tze Hong_2"] = img_to_encoding(img_dir + "th_bounded1.jpg", FRmodel)
database["2_Tze Hong_3"] = img_to_encoding(img_dir + "th_bounded2.jpg", FRmodel)
database["2_Tze Hong_4"] = img_to_encoding(img_dir + "th_bounded3.jpg", FRmodel)
database["2_Tze Hong_5"] = img_to_encoding(img_dir + "th_bounded4.jpg", FRmodel)

database["1_Alson_1"] = img_to_encoding(img_dir + "alson_bounded0.jpg", FRmodel)
database["1_Alson_2"] = img_to_encoding(img_dir + "alson_bounded1.jpg", FRmodel)
database["1_Alson_3"] = img_to_encoding(img_dir + "alson_bounded2.jpg", FRmodel)
database["1_Alson_4"] = img_to_encoding(img_dir + "alson_bounded3.jpg", FRmodel)
database["1_Alson_5"] = img_to_encoding(img_dir + "alson_bounded4.jpg", FRmodel)

'''
database["5_Jace_1"] = img_to_encoding(img_dir + "jace_bounded0.jpg", FRmodel)
database["5_Jace_2"] = img_to_encoding(img_dir + "jace_bounded1.jpg", FRmodel)
database["5_Jace_3"] = img_to_encoding(img_dir + "jace_bounded2.jpg", FRmodel)
database["5_Jace_4"] = img_to_encoding(img_dir + "jace_bounded3.jpg", FRmodel)
database["5_Jace_5"] = img_to_encoding(img_dir + "jace_bounded4.jpg", FRmodel)
'''

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
    
    if min_dist > 0.045:
        print("Not in the database." +'...'+', the distance is '+str(min_dist) +'...'+image_path)
        return False, identity, min_dist, encoding
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist) +'...'+image_path)
        return True, identity, min_dist, encoding

#%%
img_dir = "images_bounded"+"\\" +"test"+ "\\"
images = os.listdir(img_dir)0
for i in images:
    who_is_it(img_dir + i, database, FRmodel)