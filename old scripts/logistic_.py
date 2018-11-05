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
train = np.load('train_set_v2.npy')
test = np.load('test_set_v2.npy')
X = train[:,:-1]
Y = train[:,128]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=10, stratify=Y)

#%%

model = LogisticRegressionCV(cv = 10, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)

prediction = model.predict(X_test).reshape([-1,1])
prediction_prob = np.max(model.predict_proba(X_test), axis=1)

testing = np.hstack([prediction, y_test.reshape([-1,1])])
testing = np.hstack([testing, prediction_prob.reshape([-1,1])])

np.mean(prediction == y_test.reshape([-1,1]))


#%%
prediction = model.predict(test)
prediction_translated = np.array([name_dict[i] for i in prediction])
test_answer_translated = np.array([name_dict[i] for i in list(test_answer)])
prediction_prob = np.max(model.predict_proba(test), axis=1)

testing = np.hstack([prediction_translated.reshape([-1,1]),test_answer_translated.reshape([-1,1])])
testing = np.hstack([testing, prediction_prob.reshape([-1,1])])

#%%
accepted_idx = np.max(model.predict_proba(test), axis=1) > 0.5
prediction_accepted = prediction[accepted_idx]
test_answer_accepted = test_answer[accepted_idx]
prediction_prob_accepted = prediction_prob[accepted_idx]
prediction_accepted_translated = np.array([name_dict[i] for i in prediction_accepted])

testing = np.hstack([prediction_accepted_translated.reshape([-1,1]),test_answer_accepted.reshape([-1,1])])
testing = np.hstack([testing, prediction_prob_accepted.reshape([-1,1])])

