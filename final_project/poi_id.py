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
SVMclf = svm.SVC()
NBclf = GaussianNB()

treeclf.fit(features_train, labels_train)
SVMclf.fit(features_train, labels_train)
NBclf.fit(features_train, labels_train)

y_pred_tree = treeclf.predict(features_test)
y_pred_SVM = SVMclf.predict(features_test)
y_pred_NB = NBclf.predict(features_test)

print "Number of training points", len(labels_train)
print "Number of test points", len(labels_test)


print "Confusion Matrix Prediction"
# print confusion_matrix(labels_test, y_pred)
print "Classification Tree Scores"
print "Accuracy Tree: ", accuracy_score(labels_test, y_pred_tree)
print "precision Score: ", precision_score(labels_test,y_pred_tree)
print "Recall Score: ", recall_score(labels_test,y_pred_tree)

print "SVM Scores"
print "Accuracy SVM: ", accuracy_score(labels_test, y_pred_SVM)
print "precision Score: ", precision_score(labels_test,y_pred_SVM)
print "Recall Score: ", recall_score(labels_test,y_pred_SVM)

print "NB Scores"
print "Accuracy NB: ", accuracy_score(labels_test, y_pred_NB)
print "precision Score: ", precision_score(labels_test,y_pred_NB)
print "Recall Score: ", recall_score(labels_test,y_pred_NB)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
for score in scores:
    print "# Tuning hyper-parameters for %s" % score
    print

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    clf.fit(features_train, labels_train)

    print "Best parameters set found on development set:"

    print clf.best_params_

    print "Grid scores on development set:"

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print "%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)

    print "Detailed classification report:"
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."

    y_true, y_pred = labels_test, clf.predict(features_test)

    print classification_report(y_true, y_pred)


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)