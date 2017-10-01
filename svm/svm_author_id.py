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
#from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import confusion_matrix


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
t0 = time()
#classifier = linearSVC(random_state= 0) 99%
#classifier = svm.OneClassSVM( kernel="rbf", gamma=0.1)
from sklearn.svm import SVC
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

classifier = SVC(kernel = 'rbf', random_state = 0, C = 10000)

y_pred = classifier.fit(features_train, labels_train).predict(features_test)


cm = confusion_matrix(labels_test, y_pred)

print cm
print y_pred[10]
print y_pred[26]
print y_pred
print "training time:", round(time()-t0, 3), "s"
#########################################################


