#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from scipy.stats import trim_mean
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from feature_selection import new_features,test_kbest

from algorithms import *
from explore_data import *

feature_list = ['poi','retention_incentives/key_payments','bonus/salary','poi_emailratio',
    'exercised_stock_options/salary','key_payments',
    'retention_incentives','bonus', 'deferral_payments', 'deferred_income', 
    'director_fees', 'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person',
       'from_this_person_to_poi', 'loan_advances', 'long_term_incentive',
       'other', 'restricted_stock', 'restricted_stock_deferred',
       'salary', 'shared_receipt_with_poi', 'to_messages',
       'total_payments', 'total_stock_value']

old_feature_list = ['poi','bonus', 'deferral_payments', 'deferred_income', 'director_fees',
       'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person',
       'from_this_person_to_poi', 'loan_advances', 'long_term_incentive',
       'other', 'restricted_stock', 'restricted_stock_deferred',
       'salary', 'shared_receipt_with_poi', 'to_messages',
       'total_payments', 'total_stock_value']



data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

for items in data_dict:
    for i in data_dict[items]:
        if data_dict[items][i] == "NaN":
            data_dict[items][i] = 0

for individual in data_dict:
    del data_dict[individual]['email_address']


for indiv in data_dict:
    for i in data_dict[indiv]:
        if i == 'poi':
            if data_dict[indiv][i] == True:
                data_dict[indiv][i] = 1
            else:
                data_dict[indiv][i] = 0

del data_dict['TOTAL']
del data_dict['LOCKHART EUGENE E']
del data_dict['THE TRAVEL AGENCY IN THE PARK']



for items in data_dict:
        data_dict[items]['key_payments'] = data_dict[items]['salary'] + data_dict[items]['bonus'] + data_dict[items]['other']
        data_dict[items]['retention_incentives'] = data_dict[items]['long_term_incentive'] + data_dict[items]['total_stock_value']
        if data_dict[items]['salary'] != 0:
            data_dict[items]['bonus/salary'] = data_dict[items]['bonus']/data_dict[items]['salary']
            data_dict[items]['exercised_stock_options/salary'] = data_dict[items]['exercised_stock_options']/data_dict[items]['salary']
        else:
            data_dict[items]['bonus/salary'] = 0
            data_dict[items]['exercised_stock_options/salary'] = 0
        if data_dict[items]['key_payments'] != 0:
            data_dict[items]['retention_incentives/key_payments'] = data_dict[items]['retention_incentives']/data_dict[items]['key_payments']
        else:
            data_dict[items]['retention_incentives/key_payments'] = 0
        messagetotal = data_dict[items]['from_messages'] + data_dict[items]['to_messages']
        if messagetotal > 0:
            poitotal = data_dict[items]['from_this_person_to_poi'] + data_dict[items]['from_poi_to_this_person']
            data_dict[items]['poi_emailratio'] = poitotal/messagetotal
        else:
            data_dict[items]['poi_emailratio'] = 0

my_dataset = data_dict

data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

featureimportance(data,feature_list)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
'''clf_dt0 = DT(scaler, old_features, old_labels)
clf_rf0 = RForest(scaler, old_features, old_labels)
clf_gb0 = GradientBoost(scaler, old_features, old_labels)
clf_ad0 = Ada(scaler, old_features, old_labels)'''

clf_dt = DT(scaler, features, labels)
clf_rf = RForest(scaler, features, labels)
clf_gb = GradientBoost(scaler, features, labels)
#clf_lg = Logit(scaler, features, labels)
### Task : Algorithm Choice and Parameter Tuning (functions in algorithms.py)
#uncomment the function calls below to test on each algorithm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
#eclf0 = VotingClassifier(estimators=[('rf', clf_r0), ('dt',clf_d0),('gnb', clf_g0)], voting='soft')
clf_v = VotingClassifier(estimators=[('rf', clf_rf), ('dt',clf_dt),('gnb', clf_gb)], voting='soft')

Xtrain, Xtest, ytrain, ytest = train_test_split( features, labels, test_size = 0.2, random_state = None)
clf_v.fit(Xtrain, ytrain)

y_pred = clf_v.predict(Xtest)
### Task : Cross Validation and Testing - using StratifiedShuffleSplit and report in tester.py (Udacity function)
print '↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Voting Classifier Performance ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
#test_classifier(model  ,my_dataset,feature_list)
print '\n',classification_report(ytest, y_pred)
print '↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Voting Classifier Performance ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑','\n\n' 

dump_classifier_and_data(clf_v, my_dataset, feature_list)
