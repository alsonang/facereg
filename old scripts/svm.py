# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:38:28 2018

@author: NTU user
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
#%%

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)

#%%
clf = SVC(kernel= 'sigmoid', degree=2 ,gamma='auto')
clf.fit(X_train, y_train) 



#%%
prediction = clf.predict(X_test).reshape([-1,1])

test = np.hstack([prediction, y_test.reshape([-1,1])])
np.mean(prediction == y_test.reshape([-1,1]))

