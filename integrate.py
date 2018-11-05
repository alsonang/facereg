# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 14:32:47 2018

@author: NTU user
"""
import numpy as np
#%%
img_dir = "images_bounded"+"\\" +"test"+ "\\"
images = os.listdir(img_dir)

final = np.empty([0,5])

N = len(images)
for i in range(N):
    final_row = np.array([])
    check, identity_filterlayer, min_dist, encoding = who_is_it(img_dir + images[i], database, FRmodel)
    identity_filterlayer = int(identity_filterlayer[0])
    #identity_filterlayer = name_dict[identity_filterlayer]
    final_row = np.append(final_row, check)
    final_row = np.append(final_row, identity_filterlayer)
    
    if check == True:
        final_row = np.append(final_row, model_lr.predict(encoding))
        final_row = np.append(final_row, model_rf.predict(encoding))
        final_row = np.append(final_row, 0)
    
    else:
        final_row = np.append(final_row, 99)
        final_row = np.append(final_row, 99)
        final_row = np.append(final_row, 0)
        
    final = np.vstack([final, final_row])

num_results = final.shape[0]
#%%
for i in range(num_results):
    voting_check = np.unique(final[i][1:4], return_counts= True)
    voting_idx = np.argmax(voting_check[1])
    final[i][4] = voting_check[0][voting_idx]
    
#%%
np.array([name_dict[i] for i in list(final[:,4])])