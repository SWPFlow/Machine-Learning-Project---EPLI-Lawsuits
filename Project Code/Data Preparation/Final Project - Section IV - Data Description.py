# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 07:06:01 2017

@author: Chris.Cirelli
"""

##############  PACKAGES   ##################################
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
#you could probably add a none statement here as a if then statement. 

def get_columns(dataframe):
    List1 = []
    [List1.append(x) for x in dataframe.columns]
    df_columns = pd.DataFrame(List1)
    return df_columns

def write_to_excel(dataframe, filename):
    writer = pd.ExcelWriter(filename+'.xlsx')
    dataframe.to_excel(writer, 'Data')
    writer.save()

##############  DATA FILES   #################################

def create_df_ABT():
    File_ABT = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section III - Analytics Based Table - Fixed Industry Values.xlsx')
    df_ABT = pd.DataFrame(File_ABT)
    Median_employee_count = df_ABT['Employees'].median()
    Median_revenues = df_ABT['Revenues'].median()
    df_ABT['Replace E-Count(0) w-Median'] = df_ABT['Employees'].replace(0, value = Median_employee_count)
    df_ABT['Replace Rev(0) w-Median'] = df_ABT['Revenues'].replace(0, value = Median_revenues)
    df_ABT_drop_rev = df_ABT.drop('Revenues', axis = 1)
    df_ABT_drop_empl = df_ABT_drop_rev.drop('Employees', axis = 1)
    return df_ABT_drop_empl

def create_SIC_Dataframe():
    File_SIC = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\SIC-Code-List.xls')
    df_SIC = pd.DataFrame(File_SIC)
    return df_SIC

df_ABT = create_df_ABT()

write_to_excel(df_ABT, 'Final Project - Section IV - Final ABT')



##############  PLAN  #############################
'''
1.)                         Describe dataframe
2.)                         Discpersion of Revenues, Employee Count, SIC Code
3.)                         Quartiles for revenue and employee count
4.)                         Graphs: Bar for groupings, scatter for ordered nums, stdev for nums. 
                            Calculate denominator for state, broker, industry, SIC to get proportional representation. 
'''         

##############  STEP I: DESCRIBE DATAFRAME  #############################

# ORGANIZE BY FEATURE

#   ALL FEATURES

print(df_ABT.columns)

def plot_hist_all():
    for x in df_ABT.columns:
        plt.hist(df_ABT.groupby(x)[x].count(), bins = 40, log = True)
        plt.title(str(x))
        plt.show()

def correlation_claims(column):
    corr_matrix = df_ABT.corr()
    return corr_matrix[column].sort_values(ascending = False)

def scatter_matrix():
    attributes = ['Claim Count', 'Employees', 'Revenues', 'Change Revenues', 'Change Employees']
    pd.plotting.scatter_matrix(df_ABT[attributes], figsize = (20,15))
    plot.show()

def get_claim_ratio_table(dataframe, feature):
    Dictionary = {}
    df1 = pd.DataFrame(Dictionary)
    df1[feature] = pd.Series(dataframe.groupby(feature)[feature].count())
    df1['Claim Count'] = pd.Series(dataframe.groupby(feature)['Claim Count'].sum())
    df1['Ratio %'] = ((df1['Claim Count'] / df1[feature])*100).round()
    df1_sorted = df1.sort_values('Ratio %', ascending = False)
    df_final = df1_sorted.head(20)
    print('')
    return df_final
      
def get_ABT_info(dataframe):
    print(df_ABT.describe().round())
    print('')
    print(df_ABT.info())

def get_ABT_info_to_excel(dataframe):
    writer = pd.ExcelWriter('ABT.xlsx')
    get_columns(df_ABT).to_excel(writer, 'Data')
    pd.DataFrame(df_ABT.describe()).to_excel(writer, 'Data', startrow = 14)
    writer.save()
      
def get_zeros(dataframe, column):
    List1 = []
    for x in dataframe[column]:
        if x == 0:
            List1.append(x)
    print(List1.count(0))

def graph_series_MEDIAN(dataframe, column):
    # calculate median
    Median = dataframe[column].median()
    # split dataframe by median upper & lower
    List1 = []
    for x in dataframe[column]:
        if x < Median:
            x = 'Lower'
        else:
            x = 'Upper'
        List1.append(x)
    dataframe['Median Split'] = List1
    # calculate # of claims in each group
    df_Median_Split = dataframe.groupby('Median Split')['Claim Count'].sum()
    df_sum = df_ABT.groupby('Claim Count')['Claim Count'].sum()
    df_Pol_count_Split_Median = dataframe.groupby('Median Split')['Claim Count'].count()
    df_claim_ratio = (df_Median_Split / df_sum[1]) * 100
    print('Policy Split on Median', df_Pol_count_Split_Median)
    print('Ratio of Claims on Median', df_claim_ratio.round())
    df_plot = df_Median_Split.plot.bar(title = 'Median Split => ' + str(column), figsize = (12,12), fontsize = 15)
    return df_plot

def graph_series_MEAN(dataframe, column):
    Mean = dataframe[column].mean()
    List1 = []
    for x in dataframe[column]:
        if x < Mean:
            x = 'Lower'
        else:
            x = 'Upper'
        List1.append(x)
    dataframe['Mean Split'] = List1
    df_Mean_Split = dataframe.groupby('Mean Split')['Claim Count'].sum()
    return df_Mean_Split.plot.bar(title = 'Mean Split' + str(column), figsize = (12,12), fontsize = 15)

def ratio_upper_lower_series_MEAN(dataframe, column):
    Mean = dataframe[column].mean()
    List1 = []
    for x in dataframe[column]:
        if x < Mean:
            x = 'Lower'
        else:
            x = 'Upper'
        List1.append(x)
    df_name = str('Mean Split' + ': ' + column)
    dataframe[df_name] = List1
    df_Mean_Split = dataframe.groupby(df_name)['Claim Count'].sum()
    df_sum = df_ABT.groupby('Claim Count')['Claim Count'].sum()
    df_claim_ratio = (df_Mean_Split / df_sum[1]) * 100
    df_plot = df_claim_ratio.plot.bar()
    return df_plot
    
def ratio_upper_lower_series_MEDIAN(dataframe, column):
    Median = dataframe[column].median()
    List1 = []
    for x in dataframe[column]:
        if x < Median:
            x = 'Lower'
        else:
            x = 'Upper'
        List1.append(x)
    df_name = str('Median Split' + ': ' + column)
    dataframe[df_name] = List1
    df_Median_Split = dataframe.groupby(df_name)['Claim Count'].sum()
    df_sum = df_ABT.groupby('Claim Count')['Claim Count'].sum()
    df_claim_ratio = (df_Median_Split / df_sum[1]) * 100
    df_plot = df_claim_ratio.plot.bar(figsize = (12,12), fontsize = 15)
    df_plot.set_ylabel(' Percentage (%) of Total Claims', size = 15)
    df_plot.set_title('% Claims on Median' + '  Feature = ' + str(column), size = 20)
    return df_plot



#   CLAIMS

def calc_perct_records_with_claims():
    dataframe = df_ABT
    Num_records = len(df_ABT)
    Num_claims = sum(df_ABT['Claim Count'])
    Percentage_records_with_claims = (Num_claims / Num_records)*100
    print('Total number of records => ' + str(Num_records))
    print('Total number of claims => ' + str(Num_claims))
    print('Percentage of records with claims => ' + str(Percentage_records_with_claims))

#   INDUSTRY 

def graph_Claims_Industry():
    df_groupby_Industry = df_ABT.groupby('Industry')['Industry'].count()
    df_groupby_claims = df_ABT.groupby('Industry')['Claim Count'].sum()
    df_claims_div_count = df_groupby_claims.div(df_groupby_Industry, level = 'Claim Count') * 100
    df_final = df_claims_div_count.sort_values()
    df_plot = df_final.plot.bar(figsize = (12,12), fontsize = 15)
    df_plot.set_ylabel(' Percentage (%) of Total Claims', size = 15)
    df_plot.set_title('% EPLI Claims By Industry', size = 20)

#   SIC    

def graph_Claims_SIC():
    df_groupby_SIC = df_ABT.groupby('SIC Code')['SIC Code'].count()
    df_groupby_Claims = df_ABT.groupby('SIC Code')['Claim Count'].sum()
    df_per_claims = df_groupby_Claims.div(df_groupby_SIC, level = 'Claim Count') * 100
    df_per_Claims_high = df_per_claims > 40
    df_final = df_per_claims[df_per_Claims_high]
    df_final.plot.bar(title = '% EPLI Claims by SIC Code', figsize = (12, 12), fontsize = 15)
    #The SIC code provides more granualar information, but its probably not a good approach for this study. 

#   STATE

def graph_Claims_State():
    df_start = df_ABT.groupby('State')['State'].count() 
    df_claims = df_ABT.groupby('State')['Claim Count'].sum()
    df_ratio_claims_pol = (df_claims / df_start) * 100
    df_final = df_ratio_claims_pol.sort_values()
    df_plot = df_final.plot.bar(figsize = (12,12), fontsize = 15)
    df_plot.set_title('Ratio # Claims / # Policies Per State', size = 20)
    df_plot.set_ylabel('Percentage (Claims / Policies', size = 15)
    return df_plot    


#   BROKER
    
def graph_Claims_Broker():
    df_groupby_Broker = df_ABT.groupby('Broker')['Broker'].count()
    df_groupby_claims = df_ABT.groupby('Broker')['Claim Count'].sum()
    df_claims_div_count = df_groupby_claims.div(df_groupby_Broker, level = 'Claim Count') * 100
    df_sort = df_claims_div_count.sort_values()
    df_high = df_sort > 0
    df_final = df_sort[df_high]     
    df_final.plot.bar(title = '% EPLI Claims By Broker', figsize = (12,12), fontsize = 15)
    #add as second graph where you only look at brokers with more than x policies in force. 


#   EMPLOYEE 

def graph_employee_count_dist():
    df_Employees = df_ABT['Employees'] < 600
    df_ABT_new = df_ABT[df_Employees]
    df_ABT_new.hist(column = 'Employees', bins = 300, figsize = (12,12))

def graph_quartiles_Employees_group_claims(dataframe, column):
    List1 = []
    for x in dataframe[column]:
        if x < 7 or x == 7:
            x = 'Q_I'
        elif x > 7 and x < 68 or x == 68:
            x = 'Q_II'
        elif x > 68 and x < 400 or x == 400:
            x = 'Q_III'
        elif x > 400:
            x = 'Q_IV'
        List1.append(x)
    df_name = str('Quartiles' + ': ' + column)
    dataframe[df_name] = List1
    df_quartile_Split = dataframe.groupby(df_name)['Claim Count'].sum()
    return df_quartile_Split.plot.bar(title = df_name + str(column), figsize = (12,12), fontsize = 15)


def ratios_quartiles_Employees_group_claims(dataframe, column):
    List1 = []
    for x in dataframe[column]:
        if x < 15 or x == 15:
            x = 'Q_I'
        elif x > 15 and x < 68 or x == 68:
            x = 'Q_II'
        elif x > 68 and x < 400 or x == 400:
            x = 'Q_III'
        elif x > 400:
            x = 'Q_IV'
        List1.append(x)
    df_name = str('Quartiles' + ': ' + column)
    dataframe[df_name] = List1
    df_quartile_Split = dataframe.groupby(df_name)['Claim Count'].sum()
    df_total_claims = df_quartile_Split.sum()
    df_ratios = (df_quartile_Split / df_total_claims)*100
    df_plot = df_ratios.plot.bar(figsize = (12, 12), fontsize = 20, legend = True, grid = True)
    df_plot.set_ylabel(' Percentage (%) of Total Claims', size = 15)
    df_plot.set_title('Ratio Claims in Employee-Count Quartile / Total Claims', size = 20)
    return df_plot


#   REVENUE

def graph_revenue_count_dist():
    df_Revenues = df_ABT['Revenues'] < 10000000
    df_ABT_new = df_ABT[df_Revenues]
    df_ABT_new.hist(column = 'Revenues', bins = 20, figsize = (12,12))

def ratios_quartiles_Revenues_group_claims(dataframe, column):
    List1 = []
    for x in dataframe[column]:
        if x < 1400000 or x == 1400000:
            x = 'Q_I'
        elif x > 1400000 and x < 12130156 or x == 12130156:
            x = 'Q_II'
        elif x > 12130156 and x < 77517687 or x == 77517687:
            x = 'Q_III'
        elif x > 77517687:
            x = 'Q_IV'
        List1.append(x)
    df_name = str('Quartiles' + ': ' + column)
    dataframe[df_name] = List1
    df_quartile_Split = dataframe.groupby(df_name)['Claim Count'].sum()
    df_total_claims = df_quartile_Split.sum()
    df_ratios = (df_quartile_Split / df_total_claims)*100
    df_plot = df_ratios.plot.bar(legend = True, grid = True, figsize = (12, 12), fontsize = 20)
    df_plot.set_ylabel(' Percentage (%) of Total Claims', size = 15)
    df_plot.set_title('Ratio Claims in Rev Quartile / Total Claims', size = 20)
    return df_plot

def plot_relationship_EEcount_Revenues():
    df_plot = df_ABT.plot.scatter(x = 'Replace Rev = 0 w-Median', y = 'Replace EE = 0 w-Median', figsize = (15,15), loglog = True)
    df_plot.set_title('Relationship of Employee Count to Revenues', size = 20)
    df_plot.set_ylable('Employee Count', size = 15)
    df_plot.set_xlable('Revenues', size = 15)
    return df_plot

#   Revenue & Employee Change


def graph_claims_by_EPLI_Rev_change(dataframe, feature):
    feature_text = str(feature)
    List1 = []
    for x in dataframe[feature]:
        if x > 0:
            x = 'Positive'
            List1.append(x)
        elif x == 0:
            x = 'No Change'
            List1.append(x)
        elif x < 0:
            x = 'Negative'
            List1.append(x)
        else:
            x = 'No Match'
            List1.append(x)
    # create new column
    dataframe['Feature Change'] = List1
    # create dataframe grouping new feature
    df_policy_count = dataframe.groupby('Feature Change')['Claim Count'].count()
    df_claim_count = dataframe.groupby('Feature Change')['Claim Count'].sum()
    print(df_policy_count)
    print('')
    #create ratio    
    df_ratio = (df_claim_count / df_policy_count)*100
    print(df_ratio.round())
    df_plot = df_ratio.plot.bar(figsize = (12,12), fontsize = 15)
    df_plot.set_title('% Category With Claims => ' + feature_text, size = 15)
    return df_plot

'''both employee count and revenues show more companies with claims in the positive grouping'''
'''Note that this is not yet broken out as a ratio'''



'''
policy_count = df_ABT.groupby('State')['State'].count().sort_values(ascending = False)
claim_count = df_ABT.groupby('State')['Claim Count'].sum().sort_values(ascending = False)

df1 = pd.DataFrame(policy_count)
df2 = pd.DataFrame(claim_count)

df3 = df1.merge(df2, how = 'outer', left_index = True, right_index = True)
df3['Ratio'] = (df3['Claim Count'] / df3['State'])*100
df4 = df3.sort_values('Ratio', ascending = False)
print(df4[:15].round(0))
#df1.merge(df2, how = 'outer', )
'''
