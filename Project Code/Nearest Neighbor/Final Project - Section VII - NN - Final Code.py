# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:38:03 2017

@author: Chris.Cirelli
"""

####   LOAD LIBRARIES 
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import decimal
import pandas as pd

#####   LOAD DATA

File_1_Final_ABT_Encoded = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section V - ABT - Features Encoded.xlsx')
df1_ABT_Encoded = pd.DataFrame(File_1_Final_ABT_Encoded)

df1_Features = df1_ABT_Encoded.drop('Claims Count', axis = 1)
df1_Targets = df1_ABT_Encoded['Claims Count']


#####   ONE HOT ENCODING

df2_Features = pd.get_dummies(df1_Features, columns=['SIC Code', 
                                                     'Change Revenues', 
                                                     'Change Employees',
                                                     'Replace E-Count(0) w-Median', 
                                                     'Replace Rev(0) w-Median',
                                                     'State_Encoded', 
                                                     'Broker_Encoded', 
                                                     'Industry Encoded'])
df2_Targets = df1_Targets

#####   TERMS

Train = 'Train'
Test = 'Test'
ClassificationReport = 'ClassificationReport'
Accuracy = 'Accuracy'

####   ALGORITHM

def NN_no_Mods(Features, Target, Score):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Target, 
                                                        stratify = Target, 
                                                        random_state = 50, 
                                                        test_size = 0.3)
    clf_knn = KNeighborsClassifier()                        
    clf_knn.fit(X_train, y_train)                           

    # Accuracy 
    if Score == 'Accuracy':
        print('Accuracy score on training set => {:.2f}'.format(clf_knn.score(X_train, y_train)))
        print('Accuracy score on test set => {:.2f}'.format(clf_knn.score(X_test, y_test)))
    
    elif Score == 'ClassificationReport':
        clf_predict_train = clf_knn.predict(X_train)
        clf_predict_test = clf_knn.predict(X_test)
        
        clf_class_report_train = sklearn.metrics.classification_report(y_train, clf_predict_train)
        clf_class_report_test = sklearn.metrics.classification_report(y_test, clf_predict_test)
        print('Classification Report Results - Train')
        print(clf_class_report_train)
        print('')
        print('Classiciation Report Results - Test')
        print(clf_class_report_test)
    
    else:
        print('Input Error')
        
           
    #    END    
    



####   FINE TUNING

def NN_Accuracy_graph_TrainTest(Features, Target):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Target, 
                                                        stratify = Target,
                                                        test_size = 0.3, 
                                                        random_state = 100)
    
    Training_accuracy = []
    Test_accuracy = []
    Num_neighbors = range(1,11)
    
    for x in Num_neighbors:
        clf_knn = KNeighborsClassifier(n_neighbors = x)
        clf_knn.fit(X_train, y_train)
        Training_accuracy.append(clf_knn.score(X_train, y_train))              
        Test_accuracy.append(clf_knn.score(X_test, y_test))
    
    plt.plot(Num_neighbors, Training_accuracy, label = 'Accuracy Training Set')
    plt.plot(Num_neighbors, Test_accuracy, label = 'Accuracy Test Set')
    plt.ylabel('Accuracy')
    plt.xlabel('Num Neighbors')
    plt.legend()
    print('')
        
    #    END


def NN_Recall_graph_TrainTest(Features, Target):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Target, 
                                                        stratify = Target,
                                                        test_size = 0.3, 
                                                        random_state = 100)
    
    Train_recall_score = []
    Test_recall_score = []
    
    Num_neighbors = range(1,11)
    
    for x in Num_neighbors:
        clf_knn = KNeighborsClassifier(n_neighbors = x)
        clf_knn.fit(X_train, y_train)
        
        #   Train 
        clf_pred_train = clf_knn.predict(X_train)
        clf_recall_train = sklearn.metrics.recall_score(y_train, clf_pred_train)
        Train_recall_score.append(clf_recall_train)
        
        #   Test              
        clf_pred_test = clf_knn.predict(X_test)
        clf_recall_test = sklearn.metrics.recall_score(y_test, clf_pred_test)
        Test_recall_score.append(clf_recall_test)

    
    plt.plot(Num_neighbors, Train_recall_score, label = 'Recall Score For Training Set')
    plt.plot(Num_neighbors, Test_recall_score, label = 'Recall Score For Test Set')
    plt.ylabel('Recall Score')
    plt.xlabel('Num Neighbors')
    plt.legend()
    print('')

    #   END




####   PREDICT

def NN_Class_Report(Features, Target, TrainTest):
    X_train, X_test, y_train, y_test = train_test_split(Features, 
                                                        Target, 
                                                        stratify = Target, 
                                                        test_size = 0.3, 
                                                        random_state = 50)
    clf_knn = KNeighborsClassifier(
                                    n_neighbors = 3, 
                                    weights = 'distance', 
                                    p = 1, 
                                    leaf_size = 40,
                                    algorithm = 'kd_tree')                      
    clf_knn.fit(X_train, y_train)                           
    
    if TrainTest == Train:
        clf_pred = clf_knn.predict(X_train)
        report_train = sklearn.metrics.classification_report(y_train, clf_pred)
        print('Classiffication Report - Train Results')
        print(report_train)    
    
    elif TrainTest == Test: 
        clf_pred = clf_knn.predict(X_test)
        report_test = sklearn.metrics.classification_report(y_test, clf_pred)
        print('Classiffication Report - Test Results')
        print(report_test)

    else:
        print('Error with input')
    
    #   END










