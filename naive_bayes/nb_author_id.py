#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

from sklearn.naive_bayes import GaussianNB

# create model
clf = GaussianNB()

# start train time 
t0 = time()

# fit training data to model
clf.fit(features_train, labels_train)

# end train time
print "training time:", round(time()-t0, 3), "s"

# start predict time
t0 = time()

# predict with test data
clf.predict(features_test)

# end predict time
print "prediction time:", round(time()-t0, 3), "s"

# print accuracy score
print "accuracy score:", clf.score(features_test, labels_test)

#########################################################


