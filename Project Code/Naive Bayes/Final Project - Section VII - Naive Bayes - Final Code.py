# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:59:31 2017

@author: Chris.Cirelli
"""

####   LOAD LIBRARIES 

import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def write_to_excel(dataframe, filename):
    writer = pd.ExcelWriter(filename+'.xlsx')
    dataframe.to_excel(writer, 'Data')
    writer.save()


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


####    TRAIN ALGORITHMS


#   Gaussian_NB

def Gaussian_NB(Feature, Target, Score, TestTrain):
    
    X_train, X_test, y_train, y_test = train_test_split(Feature,
                                                        Target, 
                                                        stratify = Target, 
                                                        random_state = 50)
    clf_NB = GaussianNB()
    clf_NB.fit(X_train, y_train)
    
    clf_NB_pred_train = clf_NB.predict(X_train)
    clf_NB_pred_test = clf_NB.predict(X_test)
        
    if Score == 'Classification_Report':
        if TestTrain == 'Train':
            report_train = sklearn.metrics.classification_report(y_train, clf_NB_pred_train)
            print('Training Results')
            print(report_train)
        elif TestTrain == 'Test':
            report_test = sklearn.metrics.classification_report(y_test, clf_NB_pred_test)
            print('Test Results')
            print(report_test)
        
    elif Score == 'Recall_Score':
        if TestTrain == 'Train':
            recall_train = sklearn.metrics.recall_score(y_train, clf_NB_pred_train)
            return '{:.3f}'.format(recall_train)
        elif TestTrain == 'Test':
            recall_test = sklearn.metrics.recall_score(y_test, clf_NB_pred_test)
            return '{:.3f}'.format(recall_test)

    elif Score == 'Accuracy':
        if TestTrain == 'Train':
            recall_train = sklearn.metrics.accuracy_score(y_train, clf_NB_pred_train)
            return '{:.3f}'.format(recall_train)
        elif TestTrain == 'Test':
            recall_test = sklearn.metrics.accuracy_score(y_test, clf_NB_pred_test)
            return '{:.3f}'.format(recall_test)
    #   END



#   Gaussian_NB - Test Feature Exclusion

def GaussianNB_drop_feature(Score, TrainTest):
    Features = df1_Features.columns
    List_score = []
    List_feature_dropped = []
    
    for x in Features:
        drop_feature = df1_Features.drop(x, axis = 1)
        df_Features_encode = pd.get_dummies(drop_feature, columns = drop_feature.columns)
        Score_NB = Gaussian_NB(df_Features_encode, df1_Targets, Score, TrainTest)
        List_score.append(float(Score_NB))
        List_feature_dropped.append(x)
    zipped = zip(List_feature_dropped, List_score)
    List1 = list(zipped)
    df = pd.DataFrame(List1)
    df_final = df.set_index(0)
    df_sorted = df_final.sort_values(1, ascending = False)
    return df_sorted

Test = GaussianNB_drop_feature('Recall_Score', 'Test')

print(Test.plot.barh( use_index = True),title = 'Recall Score - Feature Drop')




def Gaussian_NB_drop_multi_feature(Score, TrainTest):
    Features = df1_Features.columns
    List_score = []
    List_feature_dropped = []
    
    for x in Features:
        for y in Features:
            if x == y:
                df_drop_features = df1_Features.drop(y, axis = 1)
                df_Features_encode = pd.get_dummies(df_drop_features, columns = df_drop_features.columns)
                List_score.append(Gaussian_NB(df_Features_encode, df1_Targets, Score, TrainTest))
                List_feature_dropped.append(y)
            else:
                df_drop_features = df1_Features.drop([x,y], axis = 1)
                df_Features_encode = pd.get_dummies(df_drop_features, columns = df_drop_features.columns)
                List_score.append(Gaussian_NB(df_Features_encode, df1_Targets, Score, TrainTest))
                List_feature_dropped.append(str(x)+','+str(y))
    zipped = zip(List_feature_dropped, List_score)
    List1 = list(zipped)
    df = pd.DataFrame(List1)
    df_final = df.set_index(0)
    return df_final

#   Graph Feature Exclusion
    
def Graph_Feature_Exclusion():
    File_Exclusion_Results = pd.read_excel(r'Merged Results Multi Feature Exclusion.xlsx')
    df3_Exclusion_Results = pd.DataFrame(File_Exclusion_Results)
    print(df3_Exclusion_Results.columns)
    df3_set_index = df3_Exclusion_Results.set_index('Features Excluded ')
    print(df3_set_index.plot.barh(figsize = (15,15), fontsize = 15))
    return None



#   Multinomial_NB

def Multinomial_NB(Feature, Target):
    X_train, X_test, y_train, y_test = train_test_split(Feature,
                                                        Target, 
                                                        stratify = Target, 
                                                        random_state = 50)
    clf_NB = MultinomialNB()
    clf_NB.fit(X_train, y_train)
    
    clf_NB_pred_train = clf_NB.predict(X_train)
    clf_NB_pred_test = clf_NB.predict(X_test)
    
    report_train = sklearn.metrics.classification_report(y_train, clf_NB_pred_train)
    report_test = sklearn.metrics.classification_report(y_test, clf_NB_pred_test)

    print('Training Results')
    print(report_train)
    print('Test Results')
    print(report_test)
    return None


#   Bernoulli_BN

def Bernoulli_NB(Feature, Target):
    X_train, X_test, y_train, y_test = train_test_split(Feature,
                                                        Target, 
                                                        stratify = Target, 
                                                        random_state = 50)
    clf_NB = BernoulliNB()
    clf_NB.fit(X_train, y_train)
    
    clf_NB_pred_train = clf_NB.predict(X_train)
    clf_NB_pred_test = clf_NB.predict(X_test)
    
    report_train = sklearn.metrics.classification_report(y_train, clf_NB_pred_train)
    report_test = sklearn.metrics.classification_report(y_test, clf_NB_pred_test)

    print('Training Results')
    print(report_train)
    print('Test Results')
    print(report_test)
    return None



































