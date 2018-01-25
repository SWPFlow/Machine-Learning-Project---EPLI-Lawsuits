# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 07:32:25 2017

@author: Chris.Cirelli
"""

##############  PAGCKAGES   ##################################
import pandas as pd
import matplotlib.pyplot as plt
import datetime 

##############  TEST FILES   #################################
 
File_1 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Policy_Information_10.25.2017.xls.xlsx')
df1_policy_info = pd.DataFrame(File_1)

File_2 = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\FLPL Daily ITD Loss Run Report 071017.xlsx')
df2_claims_info = pd.DataFrame(File_2)


##############  REUSABLE FORMULAS   #################################

def shit_head(dataframe):
    Dictionary = {}
    List1 = []
    for x in dataframe:
        Dictionary[x] = [type(x) for x in dataframe[x]]
    df_beg = pd.DataFrame(Dictionary)
    for x in df_beg:
        List1.append(df_beg[x].value_counts())
    df_end = pd.DataFrame(List1)
    return df_end

A = shit_head(df1_policy_info)
B = shit_head(df2_claims_info)

print(A)
print('crap')
print(B)

#writer = pd.ExcelWriter('Policy_datatype.xlsx')
#shit_head(df1_policy_info).to_excel(writer, 'Data')
#writer.save() 