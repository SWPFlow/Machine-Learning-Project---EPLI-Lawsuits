# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:49:29 2017

@author: Chris.Cirelli
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


####    SESSION I 

from sklearn.datasets import load_breast_cancer         #import our dataset
from sklearn.tree import DecisionTreeClassifier         #import the type of tree we want to use
from sklearn.model_selection import train_test_split    #Split arrays or matrices into random train and test subsets


def train_tree_I_cancer_dataset():
    Data = cancer['data']                               #define feature data
    Target = cancer['target']                           #define target data
    X_train, X_test, y_train, y_test = train_test_split(Data, Target, stratify = Target, random_state = 42)                                                                   
    #split the training and test sets. In our dataset you are going to have to define the target and test dataframes beforehand 
    tree = DecisionTreeClassifier(random_state = 0)     #create the classifier
    tree.fit(X_train, y_train)                          #train model on train dataset. 
    score_train = tree.score(X_train, y_train)          #get score on training set
    score_test = tree.score(X_test, y_test)             #get score on test set
    print('Accuracy on training subset  {:.3f}'.format(score_train))
    print('Accuracy on test subset  {:.3f} '.format(score_test))


'''
Accuracy on the training set is 100% because it is overfitting.  This happens because there is no restriction on its depth. 
Note:  run this test on your dataset in order to determine in the beginning if it is overfit. 
'''


####    SESSION II - PRUNE TREE


def train_tree_II_maxdepth_4():
    Data = cancer['data']                               #define feature data
    Target = cancer['target']                           #define target data
    X_train, X_test, y_train, y_test = train_test_split(Data, Target, stratify = Target, random_state = 42)                                                                   
    #split the training and test sets. In our dataset you are going to have to define the target and test dataframes beforehand 
    tree = DecisionTreeClassifier(max_depth= 4, random_state = 0)     #create the classifier
    tree.fit(X_train, y_train)                          #train model on train dataset. 
    score_train = tree.score(X_train, y_train)          #get score on training set
    score_test = tree.score(X_test, y_test)             #get score on test set
    print('Accuracy on training subset  {:.3f}'.format(score_train))
    print('Accuracy on test subset  {:.3f} '.format(score_test))
    #training set accuracy reduced in order to reduce overfitting and improve prediction. 
    

####    SESSION III - VIZUALIZE TREE

import graphviz 
from sklearn.tree import export_graphviz    

def train_tree_III_vizualize():
    Data = cancer['data']
    Target = cancer['target']                           
    X_train, X_test, y_train, y_test = train_test_split(Data, Target, stratify = Target, random_state = 42)                                                                   
    tree = DecisionTreeClassifier(max_depth= 4, random_state = 0)
    tree.fit(X_train, y_train)                          
    export_graphviz(tree, out_file = 'cancertree.dot', class_names = ['malignant', 'benign'], feature_names =    
    cancer.feature_names, impurity = False, filled =True) #export tree structure to a dot file.  Then cut and past into     
    #graphviz web page. to vizualize. 
    

####    SESSION IV - FEATURE IMPORTANCES
    
'''Each feature is attributed a value between 0 and 1, 1 being a perfect prediction'''

def train_tree_IV_feature_importances():
    Data = cancer['data']
    Target = cancer['target']                           
    X_train, X_test, y_train, y_test = train_test_split(Data, Target, stratify = Target, random_state = 42)                                                                   
    tree = DecisionTreeClassifier(max_depth= 4, random_state = 0)
    tree.fit(X_train, y_train)                          
    feature_importance = tree.feature_importances_      #returns a numpy array without the features.
    n_features = Data.shape[1]
    plt.barh(range(n_features), feature_importance, align = 'center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature', fontsize = 12)
    plt.figure(figsize = (12,12))
    plt.show()
    

train_tree_IV_feature_importances()






    
    
    
    
    
    
    
    
    
    
    
    

    





