#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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



t0  = time()
#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import confusion_matrix

classifier = tree.DecisionTreeClassifier(min_samples_split = 40)


from sklearn.metrics import accuracy_score
y_pred = classifier.fit(features_train, labels_train).predict(features_test)



print len(features_train[0])

cm = confusion_matrix(labels_test, y_pred)


#print cm

print accuracy_score(labels_test, y_pred)

print "training time:", round(time()-t0, 3), "s"
#########################################################


