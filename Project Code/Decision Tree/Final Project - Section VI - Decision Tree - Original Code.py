# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 07:13:51 2017

@author: Chris.Cirelli
"""

import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import export_graphviz    
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 


####    TREE BUILDING STEPS
'''
1.)     Call dataframe
2.)     Define Features and Targets
3.)     Define training and testing datasets using 'train_test_split'
4.)     Define DecisionTreeClassifier
5.)     Fit tree to either the training or test datasets
6.)     Define the measurements you want to use 'or' vizualize your tree. 
'''

####    APPROACH
'''
1.)     Train a decision tree using the encoded ABT
2.)     Take a look at the accuracy using confusion matrix and metrics.classifier report. 
3.)     Modify depth and check accuracy
4.)     Inspect the features using feature_importance
5.)     Graph decision tree
6.)     Group features Revenues / Employee Count / State / Broker / ^Rev & Employees, etc.
7.)     Re-run 1-5 to see if you can get a better accuracy score with a simpler model. 
'''

####    DATA FILES

File_1_original_encoded = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section V - ABT - Features Encoded.xlsx')
DataFrame_1_original_encoded = pd.DataFrame(File_1_original_encoded)

File_2_ABT_pre_encoding = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section IV - Final ABT_pre_encoding.xlsx')

Dataframe_2_pre_encoding = pd.DataFrame(File_2_ABT_pre_encoding)
df2_dropped_rev_empl_count = Dataframe_2_pre_encoding.drop(['Revenues', 'Employees'], axis = 1)


####    DEFINE FEATURES & TARGET

df1_tree_I_Features = DataFrame_1_original_encoded.drop('Claims Count', axis = 1)
df1_tree_I_Targets = DataFrame_1_original_encoded['Claims Count']



####    DECISION TREE I - KEY MEASUREMENTS & VIZUALIZATION

def tree_I_get_score(Features, Targets):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 0)
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    score_train = tree.score(X_train, y_train)
    score_test = tree.score(X_test, y_test)
    print('Training score => {:.3f}'.format(score_train))
    print('Testing score => {:.3f}'.format(score_test))

def tree_I_test_depth(Features, Targets):
    Dictionary = {}
    Training_score = []
    Testing_score = []
    Index_depth = []
    for x in range(2,30,2):
        X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 0)
        tree = DecisionTreeClassifier(max_depth = x)
        tree.fit(X_train, y_train)
        score_train = tree.score(X_train, y_train)
        score_test = tree.score(X_test, y_test)
        Index_depth.append(x)
        Training_score.append(score_train)
        Testing_score.append(score_test)
    Dictionary['Training_score'] = Training_score
    Dictionary['Testing_score'] = Testing_score
    Dictionary['Index_depth'] = Index_depth
    df_results = pd.DataFrame(Dictionary)
    df_final = df_results.set_index('Index_depth')
    df_final.plot.bar(figsize = (12,12), title = 'Tree Accuracy Scores by Depth', fontsize = 15)


def tree_I_get_feature_importance_train(Features, Targets, Max_depth):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 100)
    tree = DecisionTreeClassifier(max_depth = Max_depth)
    tree.fit(X_train, y_train)
    feature_importance = tree.feature_importances_
    n_features = Features.shape[1]
    plt.barh(range(n_features), feature_importance, align = 'center')
    plt.yticks(np.arange(n_features), Features.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature', fontsize = 12)
    plt.show()

def tree_I_get_feature_importance_test(Features, Targets, Max_depth):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 100)
    tree = DecisionTreeClassifier(max_depth = Max_depth)
    tree.fit(X_test, y_test)
    feature_importance = tree.feature_importances_
    n_features = Features.shape[1]
    plt.barh(range(n_features), feature_importance, align = 'center')
    plt.yticks(np.arange(n_features), Features.columns)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature', fontsize = 12)
    plt.show()
    
def tree_I_vizualization(Features, Targets, Max_depth):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 100)
    tree = DecisionTreeClassifier(max_depth = Max_depth)
    tree.fit(X_train, y_train)
    export_graphviz(tree, out_file = 'Decision_Tree_I.dot', feature_names = Features.columns)

def tree_I_confusion_matrix(Features, Targets, Max_depth):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 50)
    tree = DecisionTreeClassifier(max_depth = Max_depth)
    tree.fit(X_train, y_train)
    predict_X_test = tree.predict(X_test)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predict_X_test)
    df1 = pd.DataFrame(confusion_matrix)
    df1['Predicted NO'] = df1[0]
    df1['Predicted Yes'] = df1[1]
    List = ['Actual No', 'Actual Yes']
    df1['Index'] = List
    df1_set_index = df1.set_index('Index')
    df_Final = df1_set_index.drop([0,1], axis = 1)   
    print(df_Final)


def tree_I_classification_report(Features, Targets, Max_depth, Max_leaf_nodes):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 50)
    tree = DecisionTreeClassifier(max_depth = Max_depth, max_leaf_nodes = Max_leaf_nodes)
    tree.fit(X_train, y_train)
    predict_X_test = tree.predict(X_test)
    classification_report = sklearn.metrics.classification_report(y_test, predict_X_test)
    print(classification_report)

for x in range(2,30,2):
    tree_I_classification_report(df1_tree_I_Features, df1_tree_I_Targets, 6, x)

def get_recall_score(Features, Targets, Max_depth, Max_nodes):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 50)
    tree = DecisionTreeClassifier(max_leaf_nodes = Max_nodes, max_depth = Max_depth)
    tree.fit(X_train, y_train)
    predict_X_test = tree.predict(X_test)
    recall_score = sklearn.metrics.recall_score(y_test, predict_X_test)
    return recall_score

def tree_I_try_maxNodes_maxDepth():
    for x in range(5, 25,5):
        for y in range(5, 25,5):
            print('Max_depth =>' +str(x))
            print('Max_nodes => '+ str(y))
            print('')
            report = get_recall_score(df1_tree_I_Features, df1_tree_I_Targets, x, y)
            print(report)
            print('')

'''
Accuracy: Overall, how often is the classifier correct?
    (TP+TN)/total

Misclassification Rate: Overall, how often is it wrong?
    (FP+FN)/total
    equivalent to 1 minus Accuracy also known as "Error Rate"

True Positive Rate: When it's actually yes, how often does it predict yes?
    TP/actual yes = 100/105 = 0.95
    also known as "Sensitivity" or "Recall"

False Positive Rate: When it's actually no, how often does it predict yes?
    FP/actual no = 10/60 = 0.17

Specificity: When it's actually no, how often does it predict no?
    TN/actual no = 50/60 = 0.83
    equivalent to 1 minus False Positive Rate

Precision: When it predicts yes, how often is it correct?
    TP/predicted yes = 100/110 = 0.91

Prevalence: How often does the yes condition actually occur in our sample?
    actual yes/total = 105/165 = 0.64
'''

####    DECISION TREE II - CREATE BINARY FEATURES - REV, EE, REV^, EE^

def write_to_excel(dataframe, filename):
    writer = pd.ExcelWriter(filename+'.xlsx')
    dataframe.to_excel(writer, 'Data')
    writer.save()


def simplify_dataframe_by_Median(dataframe):
    #Calculate Median Values
    Revenue_median = dataframe['Replace Rev(0) w-Median'].median()
    Employee_count_median = dataframe['Replace E-Count(0) w-Median'].median()
    #Create new columns to capture binary values
    New_col_revenues = []
    New_col_employees = []
    New_col_change_rev = []
    New_col_change_empl = []
    # Convert values to 1/0
    for x in dataframe['Replace Rev(0) w-Median']:
        if x > Revenue_median:
            x = 1
        else: x = 0
        New_col_revenues.append(x)
    
    for x in dataframe['Replace E-Count(0) w-Median']:
        if x > Employee_count_median:
            x = 1
        else: x = 0
        New_col_employees.append(x)
    
    for x in dataframe['Change Revenues']:
        if x > 0:
            x = 1
        else:
            x = 0
        New_col_change_rev.append(x)
    
    for x in dataframe['Change Employees']:
        if x > 0:
            x = 1
        else: x = 0
        New_col_change_empl.append(x)
    #Reconstruct ABT
    dataframe['Revenues_(1/0)'] = New_col_revenues
    dataframe['Employees_(1/0)'] = New_col_employees
    dataframe['Change Revenues_(1/0)'] = New_col_change_rev
    dataframe['Change Employees_(1/0)'] = New_col_change_empl
    dataframe_final = dataframe.drop(['Change Revenues', 'Change Employees','Replace E-Count(0) w-Median','Replace Rev(0) w-Median'], axis = 1)
    #Encode New Dataframe
    encoder = LabelEncoder()
    df_State = dataframe_final['State']
    df_Broker = dataframe_final['Broker']
    df_Industry = dataframe_final['Industry']
    dataframe_final['State Encoded'] = encoder.fit_transform(df_State)
    dataframe_final['Broker Encoded'] = encoder.fit_transform(df_Broker)
    dataframe_final['Industry Encoded'] = encoder.fit_transform(df_Industry)
    df_final = dataframe_final.drop(labels = ['State', 'Broker', 'Industry'], axis = 1)
    write_to_excel(df_final, 'Final Project - Section VI - tree_II - Binary + Encoded Features')    
    return df_final


####    DEISION TREE II - TRY NEW TREE DATAFRAME (BINARY + ENCODED)

File_3 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section VI - tree_II - Binary + Encoded Features.xlsx')
df_ABT_Binary_Encoded = pd.DataFrame(File_3)

df2_tree_II_Features = df_ABT_Binary_Encoded.drop(['Claim Count'], axis = 1)
df2_tree_II_Target = df_ABT_Binary_Encoded['Claim Count']


def classification_report_try_depths():
    for x in range(2,30,2):
        print('Max_depth => ', str(x))
        print('')
        tree_I_classification_report(df_ABT_Binary_Encoded_Features, df_ABT_Binary_Encoded_Target,x)

def confusion_matrix_try_depths():
    for x in range(2,30,2):
        print('Max_depth => ', str(x))
        print('')
        tree_I_confusion_matrix(df_ABT_Binary_Encoded_Features, df_ABT_Binary_Encoded_Target, x)
        print('')

def get_feature_importance_try_depth():
    for x in range(2,30,2):
        print('Max_depth => ', str(x))
        tree_I_get_feature_importance_train(df_ABT_Binary_Encoded_Features,     
        df_ABT_Binary_Encoded_Target,x)
        tree_I_get_feature_importance_test(df_ABT_Binary_Encoded_Features,df_ABT_Binary_Encoded_Target,x)
        print('')

def get_recall_score(Features, Targets, Max_depth):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 50)
    tree = DecisionTreeClassifier(max_depth = Max_depth)
    tree.fit(X_train, y_train)
    predict_X_test = tree.predict(X_test)
    recall_score = sklearn.metrics.recall_score(y_test, predict_X_test)
    return recall_score


def get_recall_score_depth_nodes(Features, Targets, Max_depth, Max_nodes):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3,       
    random_state = 50)
    tree = DecisionTreeClassifier(max_depth = Max_depth, max_leaf_nodes = Max_nodes)
    tree.fit(X_train, y_train)
    predict_X_test = tree.predict(X_test)
    recall_score = sklearn.metrics.recall_score(y_test, predict_X_test)
    return recall_score

def try_max_depth_nodes_recall(Features, Targets):
    for x in range(2,40,2):
        for y in range(2,40,2):
            score = get_recall_score_depth_nodes(Features, Targets, x, y)
            print('Training score => {:.3f}'.format(score))

def try_max_depth_nodes_class_report(Features, Targets):
    for x in range(2,40,2):
        for y in range(2,40,2):
            report = tree_I_classification_report(Features, Targets, x, y)
            print('')
            print('Max_depth => ' + str(x))
            print('Max_nodes => ' + str(y))
            print(report)
            print('')



'''Package Tree Functions
tree_I_get_score()
tree_I_test_depth()
tree_I_get_feature_importance_train()
tree_I_get_feature_importance_test()
tree_I_vizualization()
tree_I_confusion_matrix()
tree_I_classification_report()
'''


####    THIRD ATTEMPT - RECONSTRUCT TREE BY FEATURE IMPORTANCE 'RECALL'

'''
Dataframe:  DataFrame_1_original_encoded
Approach:   Iterate the tree over each feature to find the most predictive, then the next. 

'''
#   FILES

df3_tree_III = DataFrame_1_original_encoded
df3_tree_III_Features = df3_tree_III.drop('Claims Count', axis = 1)
df3_tree_III_Target = df3_tree_III['Claims Count']
df3_tree_III_Feature_names_list = list(df3_tree_III_Features.columns)

def get_recall_score_indv_features(Dataframe, Targets, Max_depth):   
    # define lists 
    Feature_names = list(Dataframe.columns)
    Recall_scores_list = []
    Recall_scores_names = []
    # Run algorithm
    for x in Feature_names:
        Feature_indv = pd.DataFrame(df3_tree_III[x])
        X_train, X_test, y_train, y_test = train_test_split(Feature_indv, Targets, test_size=0.3, random_state = 0)
        tree = DecisionTreeClassifier(max_depth = Max_depth)
        tree.fit(X_train, y_train)
        predict_X_test = tree.predict(X_test)
        recall_score = sklearn.metrics.recall_score(y_test, predict_X_test)
        # Append vlaues
        Recall_scores_list.append(recall_score)
        Recall_scores_names.append(str(x))
    # Create Dataframe
    Recall_scores_dataframe = pd.DataFrame(Recall_scores_list, Recall_scores_names)
    return Recall_scores_dataframe

def run_max_depth_test_INDV_features(Features, Target):
    for x in range(5,30,5):
        depth_score = get_recall_score_indv_features(Features, Target, x)
        print('Max_depth => ' + str(x))
        print('')
        print(depth_score)
        print('')

# Try Different Feature Combinations

Feature_grouping_I = df3_tree_III[['Replace Rev(0) w-Median', 'Change Revenues']]
Feature_grouping_II = df3_tree_III[['Replace Rev(0) w-Median', 'Change Revenues', 'Replace E-Count(0) w-Median']]
Feature_grouping_III = df3_tree_III[['SIC Code', 'State_Encoded']]
Feature_grouping_IV = df3_tree_III[['Industry Encoded', 'State_Encoded']]
Feature_grouping_V = df3_tree_III[['SIC Code', 'State_Encoded', 'Broker_Encoded']]
Feature_grouping_VI = df3_tree_III[['Replace Rev(0) w-Median', 'Change Revenues', 'Replace E-Count(0) w-Median', 'SIC Code']]

 
List_groupings = [Feature_grouping_I, Feature_grouping_II, Feature_grouping_III, Feature_grouping_IV, Feature_grouping_V, Feature_grouping_VI]


def run_max_depth_feature_combinations():
    for x in List_groupings:
        print('')
        run_max_depth_test_ALL_feature(x, df3_tree_III_Target)
        print('')

def run_max_depth_test_ALL_feature(Features, Target):
    for x in range(5,50,5):
        depth_score = get_recall_score(Features, Target, x)
        print('Max_depth => ' + str(x))
        print('')
        print(depth_score)
        print('')



####    DECISION TREE I - TRY MODIFYING LEAVES

def tree_I_get_report_try_nodes_depth(Features, Targets, Max_depth, Max_leaf_nodes):
    X_train, X_test, y_train, y_test = train_test_split(Features, Targets, test_size=0.3, random_state = 0)
    tree = DecisionTreeClassifier(max_depth = Max_depth, max_leaf_nodes = Max_leaf_nodes)
    tree.fit(X_train, y_train)
    prediction = tree.predict(X_test)
    report = sklearn.metrics.classification_report(y_test, prediction)
    return report















