#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from feature_selection import new_features,test_kbest

from algorithms import *
from explore_data import *

### feature_list and old_feature_list are lists of strings, each of which is a feature name.
### The first feature must always be "poi".

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

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

    #Task : Initial exploration and Null value and Outlier analysis (functions in explore_data.py)

    printresults(data_dict)
    updatenulls(data_dict)

    checktotals(data_dict)


    #Task : Outlier removal (functions in explore_data.py)
    data_dict.pop('TOTAL')
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
    data_dict.pop('LOCKHART EUGENE E') #all null values.  Cheating as the reviewer told me about this one.
    update_data_errors(data_dict)

    ### Task : New features (functions in explore_data.py)

    #look at data graphically
    finstats,poi_finstats = graphstats(data_dict,finlistall)
    #drawboxes(finstats,poi_finstats)

    data_dict = new_features(data_dict)
    #draw graphs for new features.
    finstats,poi_finstats = graphstats(data_dict,newfeaturelist)
    #drawboxes(finstats,poi_finstats)

    ### Store to my_dataset for easy export below.
    my_dataset = data_dict

### Task
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
old_data = featureFormat(my_dataset, old_feature_list, sort_keys = True)
old_labels, old_features = targetFeatureSplit(old_data)

### Task : Feature Importance
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k=9)
featureimportance(data,feature_list)

### Task : Feature scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

### Task : Model Selection
clf_dt0 = DT(scaler, old_features, old_labels)
clf_rf0 = RForest(scaler, old_features, old_labels)
clf_gb0 = GradientBoost(scaler, old_features, old_labels)
clf_ad0 = Ada(scaler, old_features, old_labels)

clf_dt = DT(scaler, features, labels)
clf_rf = RForest(scaler, features, labels)
clf_gb = GradientBoost(scaler, features, labels)
clf_ad = Ada(scaler, features, labels)
### Task : Algorithm Choice and Parameter Tuning (functions in algorithms.py)
#uncomment the function calls below to test on each algorithm
clf_d0 = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Decision Trees', clf_dt0 )])
clf_r0 = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Random Forest', clf_rf0)])
clf_g0 = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Gradient Boosting', clf_gb0 )])
clf_a0 = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Ada Boosting', clf_ad0 )])

clf_d = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Decision Trees', clf_dt )])
clf_r = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Random Forest', clf_rf)])
clf_g = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Gradient Boosting', clf_gb )])
clf_a = Pipeline(steps=[('Scaler',scaler), ('SKB', skb),('Ada Boosting', clf_ad )])
from sklearn.ensemble import VotingClassifier
eclf0 = VotingClassifier(estimators=[('rf', clf_r0), ('dt',clf_d0)], voting='soft')
eclf = VotingClassifier(estimators=[('rf', clf_r), ('dt',clf_d),('gnb', clf_g),('ada',clf_a)], voting='soft')

#Task 7: Cross Validation and Testing - using StratifiedShuffleSplit and report in tester.py (Udacity function)
print '↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Original Features Performance ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
test_classifier(eclf0,my_dataset,old_feature_list)
print '↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Original Features Performance ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ '
print'\n\n\n', '↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓New Features Performance ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓','\n\n'
test_classifier(eclf,my_dataset,feature_list)
print'\n\n\n', '↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ New Features Performance ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ '
### Task 6: Dump your classifier, dataset, and features_list
dump_classifier_and_data(eclf, my_dataset, feature_list)