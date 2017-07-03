#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import division
import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from numpy import mean
from sklearn.cross_validation import train_test_split
from explore_data import featureimportance
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']
features_list = poi_label + email_features + financial_features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# Total number of data points
print("Total number of data points: %i" %len(data_dict))
# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))
       
# Number of features used
all_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features \
are used" %(len(all_features), len(features_list)))
# Are there features with many missing values? etc.
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1
print("The number of missing values for each feature: ")
for feature in missing_values:
    print("%s: %i" %(feature, missing_values[feature]))
    
### Task 2: Remove outliers
def plotOutliers(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
# Visualize data to identify outliers
print(plotOutliers(data_dict, 'total_payments', 'total_stock_value'))
print(plotOutliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(plotOutliers(data_dict, 'salary', 'bonus'))
print(plotOutliers(data_dict, 'total_payments', 'other'))
identity = []
for person in data_dict:
    if data_dict[person]['total_payments'] != "NaN":
        identity.append((person, data_dict[person]['total_payments']))
print("Outlier:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])

# Find persons whose financial features are all "NaN"
fi_nan_dict = {}
for person in data_dict:
    fi_nan_dict[person] = 0
    for feature in financial_features:
        if data_dict[person][feature] == "NaN":
            fi_nan_dict[person] += 1
sorted(fi_nan_dict.items(), key=lambda x: x[1])

# Find persons whose email features are all "NaN"
email_nan_dict = {}
for person in data_dict:
    email_nan_dict[person] = 0
    for feature in email_features:
        if data_dict[person][feature] == "NaN":
            email_nan_dict[person] += 1
sorted(email_nan_dict.items(), key=lambda x: x[1])

# Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0
new_features_list = features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
featureimportance(data,new_features_list )
#Select the best features: 
#Removes all features whose variance is below 80% 
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
features = sel.fit_transform(features)

#Removes all but the k highest scoring features
from sklearn.feature_selection import f_classif
k = 12
selector = SelectKBest(f_classif, k=k)
selector.fit_transform(features, labels)
print("Best features:")
scores = zip(new_features_list[1:],selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
print sorted_scores
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:k]
print(optimized_features_list)

# Extract from dataset without new features
data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# Extract from dataset with new features
data = featureFormat(my_dataset, optimized_features_list + \
                     ['msg_to_poi_ratio', 'msg_from_poi_ratio'], \
                     sort_keys = True)
new_f_labels, new_f_features = targetFeatureSplit(data)
new_f_features = scaler.fit_transform(new_f_features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def evaluate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)] 
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
    #return grid_search.best_estimator_

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
###############################
features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

clf_dt = DecisionTreeClassifier()
dt_param = {'criterion' : ['gini', 'entropy'], 'max_depth':[ 2,3,4] }
dt_param = {}
dt_grid_search = GridSearchCV(clf_dt, dt_param)
dt_grid_search.fit(features_train, labels_train)
clf_dt = dt_grid_search.best_estimator_
print "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Evaluate DecisionTree model↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ "
evaluate_clf(dt_grid_search, features, labels, dt_param)

clf_nb = GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(clf_nb, nb_param)
nb_grid_search.fit(features_train, labels_train)
clf_nb = nb_grid_search.best_estimator_
print '\n\n',"↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Evaluate naive bayes model↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ "
evaluate_clf(nb_grid_search, features, labels, nb_param)

clf_gb= GradientBoostingClassifier()
gb_param = {'max_depth':[3,4,5,6],'min_samples_split':[3,4,5],'subsample':[.8,.9]}
gb_param = {}
gb_grid_search = GridSearchCV(clf_gb, gb_param)
gb_grid_search.fit(features_train, labels_train)
clf_gb = gb_grid_search.best_estimator_
print'\n\n',"↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Evaluate GradientBoosting model↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ "
evaluate_clf(gb_grid_search, features, labels, gb_param)

clf_rf = RandomForestClassifier()
rf_param = {'n_estimators' : [10,20,30], 'max_features':[None],'criterion':['entropy'],'max_depth':[2,3,4,5] }
rf_param = {}
rf_grid_search = GridSearchCV(clf_rf , rf_param)
rf_grid_search.fit(features_train, labels_train)
clf_rf = rf_grid_search.best_estimator_
print'\n\n',"↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Evaluate RandomForest model↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ "
evaluate_clf(rf_grid_search, features, labels, rf_param)
#############################
#clf_v = VotingClassifier(estimators=[('rf', clf_rf), ('dt',clf_dt), ('gb', clf_gb), ('nb', clf_nb)], voting='soft')
print '\n\n', '↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  模型調參表現結果，選擇Naive Bays ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ '



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

final_features_list = optimized_features_list #+ ['msg_to_poi_ratio', 'msg_from_poi_ratio']

print '↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ Naive Bays test_classifier 的表現 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓'
test_classifier(clf_nb , my_dataset, final_features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf_nb, my_dataset, final_features_list)
