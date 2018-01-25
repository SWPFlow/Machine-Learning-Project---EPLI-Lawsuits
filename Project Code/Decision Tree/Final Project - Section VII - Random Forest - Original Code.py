# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:52:48 2017

@author: Chris.Cirelli
"""



import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz    
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

####    DATA FILES

File_1_original_encoded = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section V - ABT - Features Encoded.xlsx')
DataFrame_1_original_encoded = pd.DataFrame(File_1_original_encoded)


####    DEFINE FEATURES & TARGET

Features = DataFrame_1_original_encoded.drop('Claims Count', axis = 1)
Targets = DataFrame_1_original_encoded['Claims Count']

####    DECISION TREE I - GridSearchCV



#   Original Tree

#   Tree Using GridSearchCV
def tree_I_GridSearchCV(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 100, stratify = Targets)
    param_grid = {'max_depth': [2,4,6,8,10], 
                  'min_samples_split':[25,30,35], 
                  'min_samples_leaf': [25,30,35], 
                  'min_weight_fraction_leaf': [.01, .025, .05], 
                  'max_features':[2,3,4], 
                  'max_leaf_nodes':[15,20,25], 
                 }
    tree = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree, param_grid, scoring = 'f1')
    grid_search.fit(X_train, y_train)
    prediction = grid_search.predict(X_train)
    best_params = grid_search.best_params_
    print(best_params)
    class_report = sklearn.metrics.classification_report(y_train, prediction)
    print(class_report)
#    prediction = tree.predict(X_train)
#    class_report = sklearn.metrics.classification_report(y_train, prediction)
#    print(class_report)



#    Test GridSearchCV Tree on test subjects
def tree_I_best_fit(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 100, stratify = Targets)
    tree = DecisionTreeClassifier(max_depth= 4, 
                                  max_features= 4, 
                                  max_leaf_nodes = 20, 
                                  min_samples_leaf= 25, 
                                  min_samples_split = 35, 
                                  min_weight_fraction_leaf = 0.01)
    tree.fit(X_train, y_train)   
    
    #prediction_train = tree.predict(X_train)
    #class_report_train = sklearn.metrics.classification_report(y_train, prediction_train)
    #print(class_report_train)
    prediction_test = tree.predict(X_test)
    class_report_test = sklearn.metrics.classification_report(y_test, prediction_test)
    print(class_report_test)

'''Best Fit to Recall - Issue.  it is fitting recall very high for the non-claims, but still very low for teh actual claims.  See if you can run this for precision to get a better score. '''


#   Original Random Tree
def random_tree_I(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size = 0.3, random_state = 50)
    forest = RandomForestClassifier(n_estimators = 100, max_depth = 55, min_samples_split = 2, max_features = 4, min_samples_leaf = 1, min_impurity_split = .03)
    forest.fit(X_train, y_train)
    prediction_train = forest.predict(X_train)
    recall_score_train = sklearn.metrics.recall_score(y_train, prediction_train)
    prediction_test = forest.predict(X_test)
    recall_score_test = sklearn.metrics.recall_score(y_test, prediction_test)
    print('Recall score train => ', recall_score_train)
    print('')
    print('Recall score test => ', recall_score_test)
        
#   Best fit Random Tree





#   Example of Bagging
    
from sklearn.ensemble import BaggingClassifier
#from sklear.tree import DecisionTreeClassifier

def decision_tree_baggin(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size = 0.3)
    bag_clf = BaggingClassifier(
            n_estimators = 5,                      #train 500 different Decision Tree Classifiers. 
            max_samples = 200, 
            bootstrap = True, 
            n_jobs = -1)                            #number of classifiers to run at the same time.  If -1, then num jobs = num cores. 

    bag_clf.fit(X_train, y_train)
    y_pred = bag_clf.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(score)


def Bagger(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets)
    bg = BaggingClassifier(
            RandomForestClassifier(),
            max_samples = 2000, 
            max_features = 3, 
            n_estimators = 100)
    bg.fit(X_train, y_train)
    prediction = bg.predict(X_test)
    score = sklearn.metrics.classification_report(y_test, prediction)
    print(score)

from sklearn.ensemble import AdaBoostClassifier

def Booster(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets)    
    ada_clf = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth= 8, 
                                  max_features= 4, 
                                  max_leaf_nodes = 20, 
                                  min_samples_leaf= 25, 
                                  min_samples_split = 35, 
                                  min_weight_fraction_leaf = 0.01),
            n_estimators = 200)
    ada_clf.fit(X_train, y_train)
    #prediction = ada_clf.predict((X_train))
    #score = sklearn.metrics.classification_report(y_train, prediction)
    prediction = ada_clf.predict((X_test))
    score = sklearn.metrics.classification_report(y_test, prediction)
    print(score)





def DecisionTreeRegressor_Boost(Features, Targets):
     X_train, X_test, y_train, y_test = train_test_split(Features, Targets)  
     tree_reg1 = DecisionTreeRegressor()
     tree_reg1.fit(X_train, y_train)   
     prediction = tree_reg1.predict(X_train)
     report = sklearn.metrics.classification_report(y_train, prediction)
     return report
     
report = DecisionTreeRegressor(Features, Targets)

print(report)






















