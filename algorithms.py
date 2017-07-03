from sklearn.pipeline import Pipeline
#Five different algorithms all using piplines
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
kfold = KFold(n_splits=3, shuffle = True,random_state=42)

def DT(scaler, features, labels):
    model = DecisionTreeClassifier(random_state = 42)
    param_test={'criterion' : ['gini', 'entropy'], 'max_depth':[ 2,3,4] }
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='recall', cv=kfold )
    #scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    return clf


def RForest(scaler, features, labels):

    model = RandomForestClassifier(random_state = 42)
    param_test={'n_estimators' : [10,20,30], 'max_features':[None],'criterion':['entropy'],'max_depth':[2,3,4,5] }
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='recall', cv=kfold )
    scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    return clf

def Logit(scaler, features, labels):

    model = LogisticRegression(random_state = 42)
    param_test={'penalty':['l1','l2'], 'C':[0.1 ,0.3 ,1 , 3]}
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='recall', cv=kfold )
    scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    return clf


def GradientBoost(scaler, features, labels):

    model = GradientBoostingClassifier(random_state = 42)
    param_test={'max_depth':[3,4,5,6],'min_samples_split':[3,4,5],'subsample':[.8,.9]}
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='recall', cv=kfold )
    scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    return clf