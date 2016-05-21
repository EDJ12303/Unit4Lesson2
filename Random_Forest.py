# -*- coding: utf-8 -*-
"""
Created on Wed May 18 21:58:08 2016

@author: Erin
"""

import csv
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pandas as pd
import numpy as np
import pylab as pl

#read csv that Roy sent to me
df = pd.read_csv('C:/Users/Erin/thinkful/Unit4Lesson2/samsungdata_raw.csv', index_col=0)

#change the "activity" column to a categorical variable
df['activity'] = pd.Categorical(df['activity']).codes

# partition data 
fortrain = df.query('subject >= 27')
fortest = df.query('subject <= 6')
forval = df.query("(subject >= 21) & (subject < 27)")

# fit random forest model
train_target = fortrain['activity']
train_data = fortrain.ix[:,1:-2]
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc.fit(train_data, train_target)

# show oob score
rfc.oob_score_
print "oob_score="
print (rfc.oob_score_) 

#determine important features
important = rfc.feature_importances_
indices = np.argsort(important)[::-1]
for i in range(10):
    print("%d.feature %d (%f)" % (i + 1, indices[i], important[indices[i]]))

#print the 10th feature's importance score 
print "The importance score for the 10th most important feature is:"
print (important[indices[i]])

# define validation set 
val_target = forval['activity']
val_data = forval.ix[:,1:-2]
val_pred = rfc.predict(val_data)

# define test set 
test_target = fortest['activity']
test_data = fortest.ix[:,1:-2]
test_pred = rfc.predict(test_data)

# mean accuracy scores
print("mean accuracy score for validation set = %f" %(rfc.score(val_data, val_target)))
print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_target)))

# visualize confusion matrix
test_cm = skm.confusion_matrix(test_target, test_pred)
pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

print("Accuracy = %f" %(skm.accuracy_score(test_target, test_pred)))
print("Precision = %f" %(skm.precision_score(test_target, test_pred)))
print("Recall = %f" %(skm.recall_score(test_target, test_pred)))
print("F1 score = %f" %(skm.f1_score(test_target, test_pred)))

#getting a deprication warning that I don't understand
#how to map back to original column names?

    


        
        