#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0 )
for i in data_dict.keys():
    if (data_dict[i]['bonus'] > 5000000) & (data_dict[i]['bonus'] != 'NaN'):
        print i
        print data_dict[i]['bonus']
        print data_dict[ i ][ 'salary' ]
data = featureFormat(data_dict, features)


### your code below


for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
