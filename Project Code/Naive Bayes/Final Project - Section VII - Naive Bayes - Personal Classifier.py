# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 09:52:25 2017

@author: Chris.Cirelli
"""

import pandas as pd

'''Calculation

'''

####    FILES
File_fraud = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Project Code\Naive Bayes\Fraud Example from Book.xlsx')
df1_fraud = pd.DataFrame(File_fraud)




Fraud_count = df1_fraud.groupby('Fraud')['Fraud'].count()
Prob_fraud = Fraud_count / len(df1_fraud['Fraud'])

Prob_fraud_true = Prob_fraud.loc['T']
Prob_fraud_false = Prob_fraud.loc['F']



CreditHistory_given_fraud = df1_fraud.groupby(['Fraud', 'Credit History'])['Credit History'].count()


CreditHistory_given_fraud_true = CreditHistory_given_fraud.loc['T']
CreditHistory_given_fraud_false = CreditHistory_given_fraud.loc['F']


Prob_CreditHistory_given_fraud_false = CreditHistory_given_fraud_false / len(df1_fraud)
Prob_CreditHistory_given_fraud_true = CreditHistory_given_fraud_true / len(df1_fraud)

Conditional_prob_CH_given_fraud_false = Prob_CreditHistory_given_fraud_false * Prob_fraud_false
Conditional_prob_CH_given_fraud_true = Prob_CreditHistory_given_fraud_true * Prob_fraud_true


def get_conditional_probabilities(dataframe, feature, target):
    grouped_feature = dataframe.groupby([target, feature])[feature].count()
    grouped_feature_true = grouped_feature['T'] / grouped_feature.sum()
    grouped_feature_false = grouped_feature['F'] / grouped_feature.sum()
    print(grouped_feature_true)
    print(grouped_feature_false)
