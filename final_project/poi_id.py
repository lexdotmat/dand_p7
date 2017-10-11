#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm



### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary',  'exercised_stock_options', 'bonus', 'total_payments','total_stock_value','long_term_incentive','shared_receipt_with_poi'] # You will need to use more features
#' financial features: ['salary', 'deferral_payments', 'total_payments',
# 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income',
# 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
# 'restricted_stock', 'director_fees'] (all units are in US dollars)
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 0)

### Task 1: Select what features you'll use.
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
treeclf = tree.DecisionTreeClassifier()
SVMclf = svm.SVC( C = 0.1, gamma = 1000)
NBclf = GaussianNB()

treeclf.fit(features_train, labels_train)
SVMclf.fit(features_train, labels_train)
NBclf.fit(features_train, labels_train)

y_pred_tree = treeclf.predict(features_test)
y_pred_SVM  = SVMclf.predict(features_test)
y_pred_NB   = NBclf.predict(features_test)

print "Number of training points", len(labels_train)
print "Number of test points", len(labels_test)

print "Confusion Matrix Prediction"
# print confusion_matrix(labels_test, y_pred)
print "Classification Tree Scores"
print "Accuracy Tree: ", accuracy_score(labels_test, y_pred_tree)
print "precision Score: ", precision_score(labels_test, y_pred_tree)
print "Recall Score: ", recall_score(labels_test, y_pred_tree)

parameters = {'min_samples_split': [2, 4, 10, 100, 1000], 'min_samples_leaf': [1000,100,10,2,1, 0.01, 0.001, 0.0001],}
svr = tree.DecisionTreeClassifier()
clf = GridSearchCV(svr, parameters)
clf.fit(features_train, labels_train)
y_pred = clf.predict(features_test)
print "Best", clf.best_params_
print "Accuracy SVM clf: ", accuracy_score(labels_test, y_pred)
print "precision Score SVM clf: ", precision_score(labels_test, y_pred)
print "Recall Score SVM clf: ", recall_score(labels_test, y_pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "NB Scores"
print "Accuracy NB: ", accuracy_score(labels_test, y_pred_NB)
print "precision Score: ", precision_score(labels_test, y_pred_NB)
print "Recall Score: ", recall_score(labels_test, y_pred_NB)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)