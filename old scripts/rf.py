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

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)

#%%

model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)

#%%
prediction = model.predict(X_test).reshape([-1,1])

test = np.hstack([prediction, y_test.reshape([-1,1])])
np.mean(prediction == y_test.reshape([-1,1]))

