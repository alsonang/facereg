# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 21:38:28 2018

@author: NTU user
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
#%%
train = np.load('train_set_v4.npy')
test = np.load('test_set_v4.npy')
X = train[:,:-1]
Y = train[:,128]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=10, stratify=Y)

#%%

model_lr = LogisticRegressionCV(cv = 10, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
model_rf = RandomForestClassifier(n_estimators=1000, criterion ='gini', max_features = 'sqrt', bootstrap  = True, oob_score  = True)

model_rf.fit(X_train, y_train)
