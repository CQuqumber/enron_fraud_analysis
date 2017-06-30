from sklearn.pipeline import Pipeline
#Five different algorithms all using piplines
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
kfold = KFold(n_splits=5, shuffle = True,random_state=42)

def DT(scaler, features, labels):
    model = DecisionTreeClassifier()
    param_test={'criterion' : ['gini', 'entropy'], 'max_depth':[ 2,3,4] }
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='accuracy', cv=5)
    scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    return clf


def RForest(scaler, features, labels):

    model = RandomForestClassifier()
    param_test={'n_estimators' : [10,20,30],'criterion':['gini','entropy'],'max_depth':[2,3,4] }
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='accuracy', cv=5)
    scaler.fit_transform(features)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    #clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Random Forest', model)])
    return clf



def GradientBoost(scaler, features, labels):

    model = GradientBoostingClassifier()
    param_test={'max_depth':[3,4,5],'min_samples_split':[3,4,5],'subsample':[.7,.8,.9]}
    Gridsearch = GridSearchCV(estimator = model, param_grid = param_test, scoring='accuracy', cv=5)
    Gridsearch.fit(features, labels)
    clf = Gridsearch.best_estimator_
    #clf = Pipeline(steps=[('Scaler',scaler), ('SKB', skb), ('Gradient Boosting', model)])
    return clf