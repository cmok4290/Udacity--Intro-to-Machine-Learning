#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
from sklearn.svm import SVC
import numpy as np

# create model
clf = SVC(C=10000, kernel="rbf", gamma="auto")

# fit training data to model
clf.fit(features_train, labels_train)

# predict with test data
pred = clf.predict(features_test)
print "email 10:", pred[10]
print "email 26:", pred[26]
print "email 50:", pred[50]
# print "email 100:", pred[100]

# print number of predictions for Chris(1)
pred_arr = np.array(pred)
print "no. of predictions for Chris: ", np.sum(pred_arr)

# print accuracy score
print clf.score(features_test, labels_test)
#########################################################


