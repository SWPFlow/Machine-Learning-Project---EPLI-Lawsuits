# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:03:47 2017

@author: Chris.Cirelli
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
import graphviz
from sklearn.tree import export_graphviz    
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

#   Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier




#   File Import

File_1_Final_ABT_Encoded = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section V - ABT - Features Encoded.xlsx')
df1_ABT_Encoded = pd.DataFrame(File_1_Final_ABT_Encoded)

df1_Features = df1_ABT_Encoded.drop('Claims Count', axis = 1)
df1_Targets = df1_ABT_Encoded['Claims Count']

#   New ABT

File_New_ABT = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section VI - tree_II - Binary + Encoded Features.xlsx')
df2_New_ABT = pd.DataFrame(File_New_ABT)

df2_Features = df2_New_ABT.drop(['Claim Count'], axis = 1)
df2_Target = df2_New_ABT['Claim Count']


def encode_datset(Dataframe):
    encoder = LabelEncoder()
    df_Change_Revenues = Dataframe['Change Revenues']
    df_Change_Employees = Dataframe['Change Employees']
    Dataframe['Change Revenues Encoded'] = encoder.fit_transform(df_Change_Revenues)
    Dataframe['Change Employees Encoded'] = encoder.fit_transform(df_Change_Employees)
    df_final = Dataframe.drop(['Change Revenues', 'Change Employees'], axis = 1)
    return df_final

df1_features_encoded = encode_datset(df1_Features)


#   Simple Decision Tree

def simple_decision_tree(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 50)
    clf = DecisionTreeClassifier(max_depth = 200)
    clf.fit(X_train, y_train)
    
    #   Train
    clf_pred = clf.predict(X_train)
    report_train = sklearn.metrics.classification_report(y_train, clf_pred)
    matrix_train = sklearn.metrics.confusion_matrix(y_train, clf_pred)
    
    #   Test
    clf_pred = clf.predict(X_test)
    report_test = sklearn.metrics.classification_report(y_test, clf_pred)
    matrix_test = sklearn.metrics.confusion_matrix(y_test, clf_pred)
    
    #   Create Confusion Matrix DataFrame
    df1 = pd.DataFrame(matrix_test)
    df1['Predicted NO'] = df1[0]
    df1['Predicted Yes'] = df1[1]
    List = ['Actual No', 'Actual Yes']
    df1['Index'] = List
    df1_set_index = df1.set_index('Index')
    df_Final = df1_set_index.drop([0,1], axis = 1) 

    #   Print Report & Matrix
    print('Training Results')
    print(report_train)
    print('')
    print('Test Results')
    print(report_test)
    print('')
    print(df_Final)
    
    return None    
    


#   Bagging Classifier
    
def Bagging_Decision_Tree(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Targets, 
                                                        test_size = 0.3, 
                                                        random_state = 50, 
                                                        stratify = Targets)
    bag_clf =   BaggingClassifier(
                    DecisionTreeClassifier(),
                    n_estimators = 500,             #train 500 different Decision Tree Classifiers. 
                    max_samples = 2200,
                    bootstrap = True)
    bag_clf.fit(X_train, y_train)
    
    #   Train
    y_pred_train = bag_clf.predict(X_train)
    report_train = sklearn.metrics.classification_report(y_train, y_pred_train)
    
    #   Test
    y_pred_test = bag_clf.predict(X_test)
    report_test = sklearn.metrics.classification_report(y_test, y_pred_test)
 
    #   Print Results
    print('Training Results')
    print(report_train)
    print('')
    print('Test Results')
    print(report_test)

    return None




#   Random Forest

def random_tree_no_mods(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Targets, 
                                                        test_size = 0.3, 
                                                        random_state = 50, 
                                                        stratify = Targets)
    forest = RandomForestClassifier(
                                    n_estimators = 500) 
    forest.fit(X_train, y_train)
    
    # Train
    forest_pred_train = forest.predict(X_train)
    report_train = sklearn.metrics.classification_report(y_train, forest_pred_train)
    
    # Test
    forest_pred_test = forest.predict(X_test)
    report_test = sklearn.metrics.classification_report(y_test, forest_pred_test)
    
    # Print Report
    print('Training Results')
    print(report_train)
    print('')
    print('Test Results')
    print(report_test)

    return None



#    Best Fit - Random Forest - GridSearchCV
    

def random_tree_bestFit_GridSearchCV(Features, Targets, TestTrain, ScoringSelection):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Targets, 
                                                        test_size=0.3, 
                                                        random_state = 50, 
                                                        stratify = Targets)
    param_grid = {'max_depth': [20, 25, 30],                      
                  'min_samples_split':[2,3,4],                
                  'max_features':[4,5,6],                    
                  'max_leaf_nodes':[25, 50, 75]               
                 }
    forest = RandomForestClassifier(
                    n_estimators = 400) 
    
    grid_search = GridSearchCV(forest, param_grid, scoring = ScoringSelection)
    grid_search.fit(X_train, y_train)

    if TestTrain == Train:
         prediction = grid_search.predict(X_train)
         best_params_train = grid_search.best_params_
         class_report_train = sklearn.metrics.classification_report(y_train, prediction)
         print('Training Results')
         print(best_params_train)
         print(class_report_train)
         print('')
    elif TestTrain == Test:
        prediction = grid_search.predict(X_test)
        best_params_test = grid_search.best_params_
        class_report_test = sklearn.metrics.classification_report(y_test, prediction)
        print('Test Results')
        print(best_params_test)
        print(class_report_test)
    else:
        print('Error in input')
        
    return None



#   AbaBoostClassifier
     
def clf_AbaBoostClassifier(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Targets, 
                                                        test_size = 0.3, 
                                                        random_state = 50, 
                                                        stratify = Targets)
    clf = AdaBoostClassifier(
            RandomForestClassifier(
                                    n_estimators = 10,                
                                    max_depth = 20,
                                    min_samples_split = 2, 
                                    max_features = 6
                                    ), 
            n_estimators = 100, 
                            )  
    clf.fit(X_train, y_train)
    
    # Train
    clf_train = clf.predict(X_train)
    clf_train_report = sklearn.metrics.classification_report(y_train, clf_train)
    
    # Test
    clf_test = clf.predict(X_test)
    clf_test_report = sklearn.metrics.classification_report(y_test, clf_test)
    
    # Print Report
    print('Training Results')
    print(clf_train_report)
    print('')
    print('Test Results')
    print(clf_test_report)
    
    return None



#   Second Attempt Best Fit

def random_tree_bestFit_GridSearchCV_2(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Targets, 
                                                        test_size=0.3, 
                                                        random_state = 50, 
                                                        stratify = Targets)
    param_grid = {
                  'max_depth': [20],                      
                  'min_samples_split':[6],                   
                  'min_samples_leaf': [1],        
                  #'min_weight_fraction_leaf': [.1, 0.001],      
                  'max_features':[7],                      
                  'max_leaf_nodes':[300]              
                 }
    
    forest = RandomForestClassifier(
                                      criterion = 'entropy', 
                                      n_estimators = 50) 
    
    grid_search = GridSearchCV(forest, param_grid, scoring = 'neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    #   Train Algorithm
    
#    prediction = grid_search.predict(X_train)
#    best_params_train = grid_search.best_params_
#    class_report_train = sklearn.metrics.classification_report(y_train, prediction)

    #   Test Algorithm
    prediction = grid_search.predict(X_test)
    best_params_test = grid_search.best_params_
    class_report_test = sklearn.metrics.classification_report(y_test, prediction)
       
    #   Print
#    print('Training Results')
#    print(best_params_train)
#    print(class_report_train)
#    print('')

    print('Test Results')
    print(best_params_test)
    print(class_report_test)

    return None









