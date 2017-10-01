#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data[ "SKILLING JEFFREY K" ][ "bonus" ] = 5600000

print enron_data.keys()
print len(enron_data["SKILLING JEFFREY K"].keys())
POI_count = 0
for i in enron_data.keys():

    if enron_data[i]["poi"] == 1:
        POI_count +=1

print POI_count
print enron_data[ 'PRENTICE JAMES']

print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K'][ 'exercised_stock_options']
print enron_data[ 'LAY KENNETH L'][ 'total_payments']
print enron_data['SKILLING JEFFREY K'][ 'total_payments']
print enron_data[ 'FASTOW ANDREW S'][ 'total_payments']

#data[person_name]["poi"]==1

# check NaNs
import pandas as pd
import numpy as np
df = pd.DataFrame(enron_data)
df = df.T
print (df['salary'] != 'NaN').sum()
print (df['email_address'] != 'NaN').sum()
print (df['total_payments'] == 'NaN').sum()
print (df['total_payments'] == 'NaN').sum()/(df['total_payments'].count())
print (df['poi'] == True) .sum()
print df[(df['poi'] == True) & (df['total_payments'] == 'NaN')]
