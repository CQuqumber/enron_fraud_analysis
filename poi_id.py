#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, mean_squared_error
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


plt.style.use('ggplot')

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


###############  轉換型態   ############
df = pd.DataFrame(data_dict).T

flt_col = ['bonus', 'deferral_payments', 'deferred_income', 'director_fees',
       'exercised_stock_options', 'expenses',
       'from_messages', 'from_poi_to_this_person',
       'from_this_person_to_poi', 'loan_advances', 'long_term_incentive',
       'other', 'restricted_stock', 'restricted_stock_deferred',
       'salary', 'shared_receipt_with_poi', 'to_messages',
       'total_payments', 'total_stock_value']

df[flt_col] = df[flt_col].apply(pd.to_numeric, errors='coerce')

################## 移除遺失過多的feature ###################
df = df.fillna(0)

################# 刪除特徵 poi 重新編碼 ######################
df = df.drop('email_address', axis = 1)
df = df.drop('TOTAL', axis = 0)
df = df.drop('THE TRAVEL AGENCY IN THE PARK')

label = LabelEncoder()
df['poi'] = label.fit_transform(df['poi'])
df['poi'].tail()

################## 创建新特征 ###################
df['extra_payment_std'] = np.log10((df['total_payments'] -df['bonus'] - df['salary']+0.0001)**2)

n_comp = 4
# ICA
ica = FastICA(n_components=n_comp, random_state=None)
ica_results = ica.fit_transform(df)
# kPCA
kpca = KernelPCA(n_components=n_comp,kernel='rbf', random_state=None)
kca_results = ica.fit_transform(df)
# PCA
pca = PCA(n_components=n_comp, random_state=None)
pca_results = pca.fit_transform(df)

for i in range(1, n_comp + 1):
    df['pca_' + str(i)] = pca_results[:,i-1]    
    df['ica_' + str(i)] = ica_results[:,i-1]
    df['kca_' + str(i)] = kca_results[:,i-1]

################## 选择特征 / 特徵縮放 ###################
features_list = df.drop('poi', axis = 1).columns.values
X_train = df[features_list]
y_train = df.poi.values
MinMax = MinMaxScaler()
X_train = MinMax.fit_transform(X_train)
feat_labels = df[features_list].columns

forest = RandomForestClassifier(n_estimators=100,
                                criterion = 'entropy',
                                max_features = None,
                                max_depth = 5,
                                random_state=None,
                                n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

Features = list(feat_labels[indices])[:-15]
X_train = df[Features]
print('Total Importance Features : %d'%(len(Features)))

################# 选择和调整算法 / 验证和评估 #####################
Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train, test_size = 0.2, random_state = None)
Xtrain = MinMax.fit_transform(Xtrain)
Xtest = MinMax.fit_transform(Xtest)

param_test={'learning_rate':[1/(10**i) for i in range(1,4)],
           'max_depth':[2,3,4,5],
           'min_samples_split':[2,3,4,5],
           'subsample':[i/10 for i in range(7,10)]}

gb_model = GradientBoostingClassifier()

kfold = KFold(n_splits=5, shuffle = True,random_state=9487)
Gridsearch = GridSearchCV(estimator = gb_model, param_grid = param_test, scoring='accuracy', cv=kfold)
Gridsearch.fit(Xtrain, ytrain)

print('Best score: {}'.format(Gridsearch.best_score_))
print('Best parameters: {}'.format(Gridsearch.best_params_))
print('Best estimator: {}'.format(Gridsearch.best_estimator_ ))
model = Gridsearch.best_estimator_
model.fit(Xtrain,ytrain)
y_pred = model.predict(Xtest)
print classification_report(ytest, y_pred)


############## 寫入檔案 ##################
enron_data = {}

for i in df.index:
    enron_attr = {}
    for j in features_list:
        enron_attr[j] = df.loc[i][j]
    enron_data[i] = enron_attr

features_list = ['poi'] + list(feat_labels[indices])[:-15]
my_dataset = enron_data
clf = Gridsearch.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### Extract features and labels from dataset for local testing

#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

