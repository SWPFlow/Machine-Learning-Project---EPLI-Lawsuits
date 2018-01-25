# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:11:27 2017

@author: Chris.Cirelli
"""

##############  PAGCKAGES   ##################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def get_columns(dataframe):
    List1 = []
    [List1.append(x) for x in dataframe.columns]
    df_columns = pd.DataFrame(List1)
    print(df_columns)

##############  DATA FILES   #################################

File_merged_data = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section II - File Merger - Merged File.xlsx')
'note that information from this file was minipulated outside of python' 
df1_merged_data = pd.DataFrame(File_merged_data)


##############  FEATURES & TARGET  #############################

'''Features (Dependent Variables)

1.  SIC Code                                 Categorical. Indicates the Industry
2.  Billing State/Province                   Categorical. State location of client. 
3.  Placing Broker Ultimate Parent Producer  Categorical. Intermediary between client and insurance carrier. 
4.  Industry                                 Same as SIC but more narrow.  Ultimately will make a determination which feature creates a shallower 
5.  PriorYearRevenue                         Use to calculate Revenue Change.  Remove after calc generated. 
6.  Annual Revenue                           Group into quartiles.   
7.  Revenue Change                           Change in revenue from prior and current year.  Test 100% present. 
8.  PriorYearEmployees                       Use to calculate employee change.  Remove after calc generated. 
9.  Employees                                Group into quartiles. 
10. Employee Count Change                    Change in employees over prior year. Could indicate layoffs.    
'''

'''Target (Independent Variable)

Test I:
1.  Claim (1/0)                              1 = claim incurred, 0 = no claim. 

Test II:
1. Total Claims                              Count of claims for a given policy

Test III:
1.                                           Sum of paid claims ($).  Will require different training and testing datasets.  Exclude 2017 data.  
    
'''


##############  ANALYTICS BASED TABLE (ABT)  #############################

'''
Step1:                                      Limit dataframe by verifier == 1
Step2:                                      Create columns for year / year Revenue & Employee ^. 
Step3:                                      Remove all unecessary columns. 
Step4:                                      Define final dataframe as ABT
'''
                     
def final_dataframe():
    df1_limit_cov_section = df1_merged_data['Coverage Section Verifier'] == 1
    df2_limit_cov_EPLI = df1_merged_data[df_step1_limit_cov_section]
    df_final = df2_limit_cov_EPLI.copy()
    #Change in revenues
    Current_yr_rev = df_final['Annual Revenue']
    Prior_yr_rev = df_final['PriorYearRevenue']
    Change_rev = Current_yr_rev - Prior_yr_rev
    df_final['Change Revenues'] = Change_rev
    #Change in employees
    Current_yr_employees = df_final['Employees']
    Prior_yr_employees =  df_final['PriorYearEmployees']
    Change_employees = Current_yr_employees - Prior_yr_employees
    df_final['Change Employees'] = Change_employees
    #limit columns to defined features
    df_final = df_final.drop(['Policy Effective Date','Billing City', 'Coverage Section Verifier', 'Assigned Production Region','Policy Number - Current', 'Submission Name', 'DUNS Number', 'Placing Broker','Product Profit Center', 'PriorYearRevenue', 'PriorYearEmployees', 'Ownership', 'Pol_num_str', 'Policy Number', 'Client Description', '100 Percent Policy Booked Premium Currency','100 Percent Policy Booked Premium', 'Annual Revenue Currency'], axis = 1)
    df_final.columns = ['State', 'Broker', 'Industry', 'SIC Code', 'Revenues', 'Employees', 'Claim Count','Change Revenues', 'Change Employees']
    df_final = df_final[['State', 'Broker', 'Industry', 'SIC Code', 'Revenues', 'Employees', 'Change Revenues', 'Change Employees','Claim Count']]
    return df_final

def convert_claims_count_binary():
    df_final = final_dataframe().copy()
    df_claim_count = df_final['Claim Count']
    List1 = []
    for x in df_claim_count:
        if x > 0:
            x = 1
        else:
            x = 0
        List1.append(x)
    df_final['Claim Count'] = List1
    return df_final


df_final = convert_claims_count_binary()

def get_missing_values_industry():
    Count = 0
    for x in df_final['Industry']:
        if isinstance(x, float):
            x = 1
            Count += x
    return Count

df_final = df_final.dropna()    



##############  WRITE TO FILE  #############################

writer = pd.ExcelWriter('Final Project - Section III - Analytics Based Table.xlsx')
df_final.to_excel(writer, 'Data')
writer.save()


