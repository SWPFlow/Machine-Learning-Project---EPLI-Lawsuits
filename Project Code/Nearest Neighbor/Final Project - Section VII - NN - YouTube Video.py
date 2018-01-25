# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:46:52 2017

@author: Chris.Cirelli
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import decimal
import sklearn 

#   Datasets

cancer = load_breast_cancer()
Features = cancer['data']
Target = cancer['target']

#   Defined Variables

Train = 'Train'
Test =  'Test'

#   Algorithms


def NN_test_no_Mods(Features, Target):
    X_train, X_test, y_train, y_test = train_test_split(Features, Target, stratify = Target, random_state = 42)
    clf_knn = KNeighborsClassifier()                        #select classifier
    clf_knn.fit(X_train, y_train)                           #fit classifer to dataset. 
    
    print('Accuracy score on training set{:.2f}'.format(clf_knn.score(X_train, y_train)))
    print('Accuracy score on test set{:.2f}'.format(clf_knn.score(X_test, y_test)))
    
     
    ####    END    
    
    

#   Fine Tuning - Accuracy

def NN_test_w_Mods_print_score(Features, Target, TrainTest):
    X_train, X_test, y_train, y_test = train_test_split(Features, Target, stratify = Target, random_state = 66)
    
    Training_accuracy = []
    Test_accuracy = []
    
    Num_neighbors = range(1,11)
    
    if TrainTest == Train:
        for x in Num_neighbors:
            clf_knn = KNeighborsClassifier(n_neighbors = x)
            clf_knn.fit(X_train, y_train)
            clf_knn.predict(X_test)
            Recall_score = sklearn.metrics
            Training_accuracy.append(round(float(clf_knn.score(X_train, y_train)),3))
        print('Train Accuracy', Training_accuracy)
        
    elif TrainTest == Test:
        for x in range(1,11,1):
            clf_knn = KNeighborsClassifier(n_neighbors = x)
            clf_knn.fit(X_train, y_train)                   
            Training_accuracy[x] = round(float(clf_knn.score(X_test, y_test)),3)
        print('Test Accuracy', Test_accuracy)
        
    else:
        print('Error with input')
    
    
    ####    END


def NN_test_w_Mods_graph_TrainTest(Features, Target):
    X_train, X_test, y_train, y_test = train_test_split(Features, Target, stratify = Target, random_state = 66)
    
    Training_accuracy = []
    Test_accuracy = []
    Num_neighbors = range(1,11)
    
    for x in Num_neighbors:
        clf_knn = KNeighborsClassifier(n_neighbors = x)
        clf_knn.fit(X_train, y_train)
        clf_knn.predict(X_test)
        Recall = sklearn.metrics.recall_score()
        Test_accuracy.append()
        Training_accuracy.append(clf_knn.score(X_train, y_train))              
        Test_accuracy.append(clf_knn.score(X_test, y_test))
        
    plt.plot(Num_neighbors, Training_accuracy, label = 'Accuracy Training Set')
    plt.plot(Num_neighbors, Test_accuracy, label = 'Accuracy Test Set')
    plt.ylabel('Accuracy')
    plt.xlabel('Num Neighbors')
    plt.legend()
    ####    END




import sklearn
sklearn.metrics.classifier
























