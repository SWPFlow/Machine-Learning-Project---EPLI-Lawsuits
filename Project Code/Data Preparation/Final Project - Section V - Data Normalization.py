# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 05:39:09 2017

@author: Chris.Cirelli
"""

###########     PREPARING DATA FOR MACHING LEARNING ALGORITHM   ###################

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer #does both steps in one. 
from sklearn.preprocessing import MinMaxScaler

def write_to_excel(dataframe, filename):
    writer = pd.ExcelWriter(filename+'.xlsx')
    dataframe.to_excel(writer, 'Data')
    writer.save()

File = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final ABT.xlsx')
df_ABT = pd.DataFrame(File)


# Transform Categories to Numbers - Encoding

def encode_datset():
    encoder = LabelEncoder()
    df_ABT_state = df_ABT['State']
    df_ABT_Broker = df_ABT['Broker']
    df_ABT_Industry = df_ABT['Industry']
    df_ABT['State_Encoded'] = encoder.fit_transform(df_ABT_state)
    df_ABT['Broker_Encoded'] = encoder.fit_transform(df_ABT_Broker)
    df_ABT['Industry Encoded'] = encoder.fit_transform(df_ABT_Industry)
    df_ABT['Claims Count'] = df_ABT['Claim Count']
    df_ABT_Encoded = df_ABT.drop(labels = ['Claim Count', 'State', 'Broker', 'Industry', 'Revenues', 'Employees'], axis = 1)
    write_to_excel(df_ABT_Encoded, 'Final Project - Section V - ABT - Features Encoded')

encode_datset()

# One Hot Encoding

File_2 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\df_ABT_Encoded_11082017.xlsx')

def hot_encoder():
    df_ABT_Encoded = pd.DataFrame(File_2)
    df_ABT_Encoded_Industry = df_ABT_Encoded
    encoder = OneHotEncoder()
    df_ABT_Encoded_Industry_1hot = encoder.fit_transform(df_ABT_Encoded_Industry.values.reshape(-1,1))
    Hot_array = df_ABT_Encoded_Industry_1hot.toarray()
    df_test = pd.DataFrame(Hot_array)
    return df_test


# Feature Scaling - Encoded Data (Not! Hot)

'''Approaches

There are two types of approaches to scaling features - Min-max and standardization. 

Min Max = Values are shifted and rescaled so that they end up ranginf rom 0 to 1. We do this by subtracting the min value and deviding by the max minus the min (x - min) / max - min.  Note to self:  Min minus min / Max minus min will give us our 0.  Max minus min / Max minus Min will give us our 1.  Everything in between will be a value between these two points. 

Standardization = a.) subtract mean value - so standardized values always have a zero mean. b.) divide by the variance so that the resulting distribution has unit variance. Unlike min/max, the resulting values are not bounded, which may pose issues for algorithms. 

'''

# Max-Min approach

df_ABT_Encoded = pd.DataFrame(File_2)

def scale_dataframe():
    df_start = df_ABT_Encoded
    min_max = MinMaxScaler()
    df_start['Change Revenues MinMax'] = min_max.fit_transform(df_ABT_Encoded[['Change Revenues']])
    df_start['Change Employees MinMax'] = min_max.fit_transform(df_ABT_Encoded[['Change Employees']])
    df_start['Employee Count MinMax'] = min_max.fit_transform(df_ABT_Encoded[['Replace E-Count(0) w-Median']])
    df_start['Revenue MinMax'] = min_max.fit_transform(df_ABT_Encoded[['Replace Rev(0) w-Median']])
    df_end = df_start.drop(labels = ['Change Revenues', 'Change Employees', 'Replace E-Count(0) w-Median', 'Replace Rev(0) w-Median'], axis = 1)
    write_to_excel(df_end, 'df_ABT_Encoded_MinMax')
    return df_end

















