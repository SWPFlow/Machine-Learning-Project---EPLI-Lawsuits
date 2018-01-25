# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 22:33:42 2017

@author: Chris.Cirelli
"""
import pandas as pd

def write_to_excel(dataframe):
    writer = pd.ExcelWriter('Dataframe.xlsx')
    dataframe.to_excel(writer, 'Data')
    writer.save()

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


####################    SECTION 2:  DATA CLEAN UP & MERGER  #####################################

'''Claims Dataset

1.) Limit claims data to those filed to policies after 2013 (so 2014-2017).
2.) Limit to only claims that include EPLI in the type. 
3.) Since there are multiple claims per policy number, group claims by policy.  Policy either has a claim or no (1/0)
4.) Convert claim dataset number to string. 
5.) Convert policy dataset policy # to string. 
    
'''

######      FILES       #######
#   Policies

File_policies = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Policy_Information_10.28.2017.xls.xlsx.xlsm')
df1_policies = pd.DataFrame(File_policies)

File_claims = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\FLPL Daily ITD Loss Run Report 071017.xlsx')
df2_claims = pd.DataFrame(File_claims)


# STEP 1:   LIMIT DATA FRAME TO POST 2013

def claims_post_2014():
    dataframe = df2_claims
    df2_claims_year = dataframe['Policy Year'] > 2013
    df_claims_post_2014 = dataframe[df2_claims_year]
    return df_claims_post_2014

claims_post_2014()

# Compare original and new data frame. 

def claims_data_group_plot():
    df_all = df2_claims.groupby('Policy Year')['Policy Year'].count()
    df_claims_post_2014 = claims_post_2014().groupby('Policy Year')['Policy Year'].count()
    print(df_all)
    print(df_claims_post_2014)
    df_claims_post_2014.plot(kind = 'bar', use_index = True)


# STEP 2 & 3: DROP NON-EPLI COVERAGES AND NON-EPLI CLAIMS 

def limit_2_EPLI():
    df_begin = claims_post_2014()
    df_claims_claimantIndex = df_begin.set_index('Claimant Type')
    df_end = df_claims_claimantIndex.drop(['CUMIS FAC', 'Crime (Excess)', 'Crime (Primary)', 'Cyber Liability (P)', 'Cyber Liability (XS)', 'E&O Fin Lines Excess', 'E&O Fin Lines Primary', 'Excess Prof Liability', 'PL Class Action (P)', 'PL Class Action (XS)', 'E&O Fin Lines Excess', 'Fidelity (Excess)', 'Fidelity (Primary)', 'Fiduciary (Excess)', 'Fiduciary (Primary)', 'Professional Liability', 'XS Blended (Crime/Fidelity)', 'D&O Liab (Excess)', 'D&O Liab (Primary)', 'Securities N/CL (P)', 'Securities N/CL (XS)'])
    df_EPLI_claims_only = df_end.reset_index()
    df_claims_policy_num = df_EPLI_claims_only[['Policy Number', 'Insured']].groupby('Policy Number')['Insured'].count()
    df_claims_final = pd.DataFrame(df_claims_policy_num).reset_index()    
    return df_claims_final


def limit_pol_data_EPLI():
    df1_limit_EPLI = df1_policies['Coverage Section Verifier'] == 1
    df1_pol_dataframe_limited_EPLI = df1_policies[df1_limit_EPLI].copy()
    return df1_pol_dataframe_limited_EPLI


# STEP 4: CONVERT CLAIM POLICY NUMBER TO STRING
    
def convert_claims_data_string():
    df_begin_claims_data = limit_2_EPLI()
    df_end_claims_data_string = df_begin_claims_data['Policy Number'].astype(str)
    return df_end_claims_data_string

def modify_pol_dataframe_pol_num_str():
    df1_pol_dataframe_limited_EPLI['Pol_num_str'] = [str(x) for x in df1_pol_dataframe_limited_EPLI['Policy Number - Current']]
    return df1_pol_dataframe_limited_EPLI

######      MERGE CLAIMS & POLICY DATA       #######

def merge_claims_policy_dataframes():
    df_policies = modify_pol_dataframe_pol_num_str()
    df_claims = convert_claims_data_string()
    df_merged_pol_claims = pd.merge(left = df_policies, left_on = 'Pol_num_str', right = df_claims, right_on = 'Policy Number', how = 'outer')
    writer = pd.ExcelWriter('Project - Section II - File Merger - Merged File.xlsx')
    df_merged_pol_claims.to_excel(writer, 'Data')
    writer.save()

#   STEP 5: TEST MERGE

def test_merger():
    File_merged = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section        II - File Merger - Merged File - Copy.xlsx')
    df_merged = pd.DataFrame(File_merged)
    df_transposed = pd.DataFrame.transpose(df_merged)
    write_to_excel(df_transposed)

##############################  DATA ISSUES    ####################################

'''
1.) Policy Info 10/25/2017:     1153 of the 5,600 policies do not have current policy numbers.  Are they in the bound status in Salesforce?
2.) Policy# 1000055015161 was found in the claims set and in the policy set, yet they were not matched in the merge.  Check data types. Maybe all should be str. 
3.) After merge_claims data run, eliminated via Excel all claims numbers at the end that did not match.  
    
    
1370 unique policy numbers for the claims data, yet only 400 matches. 
Could be we are not matching claims withe the right profit center. 
Need to review data file in Excel. 
Data could be incorrect in Salesforce. Take a few pol num for claims not match and refer to them in Salesforce.  Then your pol doc. 

'''











