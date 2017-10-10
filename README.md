# Intro to Machine Learning (DAND)

==============
## Projects : Enron Dataset

This document is linked to the Intro to machine learning project. 
This project is part of Udacity Data Analyst NanoDegree and aims to use and apply machine learning techniques in a real-world use case.

This document's follows this structure:
- Chapter 1, Enron Dataset / Questions, describes the main stakes of the project.
- Chapter 2, Feature selection, describes the feature selection process and the results
- Chapter 3, Algorithm selection and tuning, describes the algorithm selection and which parameters have been chosen.
- Chapter 4, Evaluation, discusses validation and validation strategy as well as metrics.


### 1 Dataset and Questions

#### Dataset introduction
The dataset is downloaded from https://www.cs.cmu.edu/~./enron/.
Here is an introduction about the dataset provided by the website, please for more information visit the page Here is an introduction about the dataset provided by the website, please for more information visit the page https://www.cs.cmu.edu/~./enron/ . 

"This dataset was collected and prepared by the CALO Project (A Cognitive Assistant that Learns and Organizes). It contains data from about 150 users, mostly senior management of Enron, organized into folders. The corpus contains a total of about 0.5M messages. This data was originally made public, and posted to the web, by the Federal Energy Regulatory Commission during its investigation.
Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]"

#### Project Questions 

The project aims to identify POI (Person of Interest) based on e-mail content. In order to do so, analyses of text and 
### 2 Feature selection
What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

### 3 Algorithm selection 
What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]
What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

Interesting algorithms : 
SVM - http://scikit-learn.org/stable/modules/svm.html 
Decision Tree :  http://scikit-learn.org/stable/modules/tree.html#classification 
Stochastic Gradient Descent  - http://scikit-learn.org/stable/modules/sgd.html#classification

test 2 out of the 3 

### 4 Evaluation
What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]
Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics
 that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]
 
 
### Resources

Scikit Learn Supervised learning 
http://scikit-learn.org/stable/supervised_learning.html 
