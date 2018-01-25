# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 07:32:25 2017

@author: Chris.Cirelli
"""

##############  PAGCKAGES   ##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##############  DATA FILES   #################################
'''Policy Data'''
File_1 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Policy_Information_10.28.2017.xls.xlsx.xlsm')
df1_policy_data = pd.DataFrame(File_1) 

'''Claims Data'''
File_2 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - FLPL Daily ITD Loss Run Report 071017.xlsx')
df2_claims_data = pd.DataFrame(File_2)

# Shape of Original Dataframes:

def get_columns(dataframe):
    df_columns = dataframe.columns
    List1 = []
    for x in df_columns:
        List1.append(x)
    df1 = pd.DataFrame(List1)
    return df1

def alphabet_soup_test(dataframe):
    Dictionary = {}
    List1 = []
    for x in dataframe:
        Dictionary[x] = [type(x) for x in dataframe[x]]
    df_beg = pd.DataFrame(Dictionary)
    for x in df_beg:
        List1.append(df_beg[x].value_counts())
    df_end = pd.DataFrame(List1)
    return df_end

df_test = df1_policy_data.replace(0, np.nan)




############  PROJECT DESCRIPTION  ######################

'''
Objective:      Try to train a machine learning model to see if it can be trained to categorize companies by their likelihood to incur a EPLI claim.   

Datasets:       5,500 policies
                Time period 2014 to 2017
                Claims 15,330 
                
Target:         1.) Incidence of a claim 2.) Claim payout $

Features:       1.) State  / Legal jurisdiction; 2.) Placing Broker; 3.) Industry / SIC Code; 4.) Revenues & Revenue^;  5.) Employee Count & EC^
                6.) Entity Type = Private, NFP, Government 

Generalization: Larger dataset of companies provided by data.com

Confirm with Beth that it is ok. 

'''

'''Merged Dataframes'''


'''Narrow Dataframe - Exclude Certain Data'''



##############  CH:1 CROSS INDUSTRY STANDARD PROCESS FOR DATA MINING (CRISP)  #############################

'''

1. Business Understanding - fully understand the business problem being addressed. 
2. Data Understanding -     fully understand the different data sources within an organization. 
3. Data Preparation -       combining desparate information into an ABT
4. Modeling -               when the maching learning work occurs. 
5. Evaluation -             model needs to be evaluated as being fit for the organizations purposes. 
6. Deployment -             deployment. 

'''





#   7. Profit Center Feature
'''
Steps:
    1. Write a formula to determine if this is a uniform attribute. 
    2. If not uniform, return a statement indicating not uniform, keep in dataframe. 
    3. If uniform, return statement that it is uniform.  If uniform, remove from dataframe. 
'''

def profit_center_test_uniformity():
    if True:
        for x in df1_start['Product Profit Center']:
            if x == 'Commercial Private - 31510':
                return 'Uniform descriptive feature'
            else:
                return 'Descriptive feature is not uniform'

def modify_dataframe():
    if True:
        profit_center_test_uniformity() == 'Uniform descriptive feature'
        df2_modified_data = df1_original_data.drop('Product Profit Center', axis = 1)
        print('Shape of original dataframe => ' + str(df1_original_data.shape))
        print('Shape of new dataframe      => ' + str(df2_modified_data.shape))
        return df2_modified_data
    else:
        return 'Nothing to do'

##### For now, dont use this file.  We will create final dataframes based on the information in the above project decription. 
        
def write_new_dataframe():
    writer = pd.ExcelWriter('df2_policy data.xlsx')
    df_start = modify_dataframe()
    df_start.to_excel(writer, 'Data')
    writer.save()
    return df_start

        
#   8. Coverages Feature
'''
Steps:
    1. Check data types
    2. Remove any that are not strings
    3. Create a new dataframe that with policies that include the EPLI coverage section. 
'''



##############  Section IV: DATA EXPLORATION  ##################################




















