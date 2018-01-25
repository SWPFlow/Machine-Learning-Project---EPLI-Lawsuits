# -*- coding: utf-8 -*-

import pandas as pd


"""
Created on Wed Nov 22 17:37:24 2017

@author: Chris.Cirelli
"""

'''Naive Bayes Classifier

Characteristics:            Supervised learning
                            Based on Bayes theorum w/ the naive assumption of the independence between 
                            every pair of dependent variables. 

Concept                     We use estimates of likelihoods to determine the most likely predictions. 
                            We revised these predictions when new information becomes available. 
                            
                            
Classifier Strenghts        Work well on small datasets
                            Can be continuouslly updated with new data. 
                            Speed of calculation / prediction. 

Bayes Theorum               P(h|d) = P(d|h) * (P(h)) / P(d))
                           
                            P(h|d) is the probability of hypothesis h given the data d. This is called the 
                            posterior probability.
                            P(d|h) is the probability of data d given that the hypothesis h was true.
                            P(h) is the probability of hypothesis h being true (regardless of the data).    
                            This is called the prior probability of h.
                            P(d) is the probability of testshe data (regardless of the hypothesis).

Spam Example                Every email has a probability of being Spam
                            Given that it 'is' spam, every email has a probability of cotaining certain     
                            words (condition probabiity)
                            
                            Words = Viagra, Prince, Udacity
                            
                            Prob Spam                           0.4
                            Prob Non-Spam                       0.6
                            
                            Prob Viagra in Spam Emails          0.3
                            Prob Viagra in Non-Spam Emails      0.001
                            
                            Prob Prince in Spam emails          0.2
                            Prob Prince in Non-Spam emails      0.1
                            
                            Prob Udacity in Spam emails         0.001
                            Prob Udacity in Non-Spam emails     0.1
                            
                            Test:  What is the probability an email is spam given it contains Viagra, but 
                            not prince and not Udacity?
                            
                            *The solution is the product of these three probabilities. 
                            
                            P(Viagra/Spam) * P(Not Prince / Spam) & * P(Not Udadicity/Spam)
                            
                            P(Viagra/When Spam) = 0.3
                            P(Not Price/ When Spam) = 1- 0.2 or 0.8
                            P(Not Udacity / When Spam) = 1 - 0.1 = 0.999
                            
                            Solution = 0.3 * 0.8 * 0.999
                            
Salesforce Implementation   Create probabilities for groups within features. 
                            1 additional column for each feature that assigns a probability
                            1 final column that runs the formula and assigns the final grade 
                            'High', 'Medium', 'Low'. 
                            

MACHINE LEARNING - APPENDIX B - INTRODUCTION TO PROBABILITY FOR MACHINE LEARNING

Contents:                   Calculating probabilities based on relative frequencies,
                            Calculating conditional probabilities
                            Probability Product rule
                            Probability Chain rule
                            Theorum of Total Probability

PROBABILITY BASICS

Probability                 Branch of mathmatics that deals with the likelyhood or unlikelyhood of an               
                            event occuring. 

Predict & Event             In machine learning we use relative frequency to predict future events.  How 
                            many times did this event occur in the past over total possible events. 

Terminology                 Domain of interest is represented by a set of random variables. 
                            
                            Random Variable -  has a set of possible outcomes equal to the domain for 
                            the event we are studying.  For example, the domain for a die would 1-6,    
                            which we could call D sub 1.  If we had two die, we would create two random 
                            variables D sub 1 and D sub 2, each with values 1-6. 
                            
                            Experiment - In the context of our 2 dice, the experiment would constitute 
                            rolling both dice. 

                            Sample Space - defines all outcomes for the experiment.  Set of all possible 
                            combinations of assignments of values to features. 

                            Experiment (known) - whose outcome has already been recoded is a row in a   
                            dataset. 

                            Experiment (unknown) - is the prediction task for which we are building the 
                            model. 

                            Event - Any subset of an experiment. 

Probability Function        A function that takes an event (an assignment of values to a feature) as a 
                            parameter and returns the likelihood of that event. 

Probability Mass Function   Defining a probability for a categorical feature because it can be 
                            understood as returning a discrete probability mass. 

Probability Mass            Simply the probability of the event. 

Probability Density Function
                            Terminology used when the feature we are dealing with is continous. 

Probability Mass Features   Always return a value of between 0.0 and 1.0
                            The sum of the probabilities over the set of events covering all possible 
                            assignments of values to features must equal 1. 

Probability of an Event     Relative frequency of that event in the dataset. 

Relative Frequency          How often an event happened relative to how often it could have happened. 
                            Note that this depends on the scope.  Are we talking about all policies or 
                            for a given attribute of a feature (example, California and State). 

                            Relative frequency is similar to the calculation of the number of claims for                    
                            x feature divided by total claims. 
                            Wrt to the dice, the Prob (D3) (see book) = 2  /5 = 0.4
                            This is different from asking, what is the probability that the dice roles 
                            to a 3 given 1 roll?  That would be 1/6 = .166

Joint Probability           Defined as the relative frequency of the joint event within the dataset. 
                            Ex: Probability of a target feature taking a particular value (1/0) and one 
                            of the descriptive features taking a particular value at the same time.  Ex:    
                            Claim = 1, State = CA. 
                            
                            In terms of a dataset, = num_rows event occurs / num_total_rows in dataset. 
                            
Conditional Probability     Probability of an event in the context where one or more other events are 
                            known to have happened    
                            *When we want to express this type of probability, we use a vertical |
                            'given that'. 
                            Ex: Prob Dice1[6] | Dice2[5] = probability Dice1 will roll a 6 'given that' 
                            Dice2 rolled a 5.
                            
                            Calculation - dividing the number of rows in the dataset where both events 
                            occured by the number of rows where just the given (already occured) event 
                            is true. 
                            
                            Ex: The Conditional Probability that a claim will occur for a company with > 
                            100 employees | 'given that' it is California = 
                            P(>100|CA) = P>100 * P(CA) / P(CA)

Probability Distribution    Probability of a feature taking a value for all the possible values the 
                            feature can take. 
                            Notation - P() bold vs P() normal, which denotes normal probability

                            Probability distribution of Meningitis being true with a prob of 0.3 being 
                            true: P(M) = (0.3, 0.7) 

Joint Probability Dist      Multidimensional matrix where the cell in the matrix lists the probability 
                            for one of the events in the sample space defined by the combination of the 
                            feature values. 
                            Sum of all cells must be 1
                            *Given a full joint probability distribution, we can compute the probability 
                            of any event in a domain by summing over the cells in the distribution where 
                            the event is true. 

                            Calculating a conditional probability from a Joint Probability Distribution 
                            by summing the values in the cells where h and f are the case (true/false)

Example                     P(h) = 0.7
                            P(m, h) = 0.2 (2 times both m and h were true together)
                            P(m|h) = P(m,h) / P(h)
                            Probability of meningitis given a headache is the probability of when they 
                            both occur together devided by the probability of a headache. 
                            All we are doing is calculating the probability of a new denominator, which 
                            is headache. 

Product Rule                P(m,h) = P(m|h) * P(h)
                            0.2 = 0.285 * 0.7  


Theorum Total Prob          P(X) = Sum P(X|Yi) * (Yi)*P(Yi)
                            Yi defines eac row of a dataset. 
                            
Bayes Theorum               'the probability that an event has happened, given a set of evidence for it, 
                            is equal to the probability of the evidence being cause by the event 
                            multiplied by the probability of the event itself. 
                            
                            P(X|Y) = [P(Y|X) * P(X)] / P(Y)
                            Conditional probability of (X) given some evidence (Y), in terms of
                            the product of the inverse conditional probability P(Y|X) and the prior     
                            probability of the event. 
                            Note P(Y) is the prior probability. 
                            Note that P(Y) acts to normalize the data such that 0<= P(X|Y) <= 1.   
                            
                            P(Y|X) = 1- P(X|Y)
Cal from data set           P(Y) = [rows where Y is the case] / [rows in the dataset] 
                            

Predictions Based on Bayes 
                            
Naive Bayes Model          Assumption of conditional independence between features  

'''


####    FRAUD EXAMPLE

File_fraud = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Project Code\Naive Bayes\Fraud Example from Book.xlsx')
df1_fraud = pd.DataFrame(File_fraud)

'''Step1

a.) Calculate prior probabilities of the target features taking each level in its domain
b.) Conditional probability of each feature taking each level in its domain conditioned for each level that a target can take. 

Levels:
    Target feature = 2 levels
    Credit History = 4 levels
    Guarantor = 3 levels
    Accomodation = 3 levels
'''

print(df1_fraud)
print('')

def Probability_feature(Dataframe, Symptom, Level):
    Count = 0
    List = []
    for x in Dataframe[Symptom]:
        if x == Level:
            x = int(1)
            List.append(x)
            Count += x
        else:
            List.append(0)
    Prob_symptom = (Count / len(List))
    return  Prob_symptom

def Joint_Probability(Symptom, Target):
    df['Match'] = np.where((df[Symptom] == 'true') & (df[Target] == 'true'), 1,0)
    Joint_prob = df['Match'].sum() / len(df['Match'])
    return Joint_prob

def Conditional_probability(Symtpom, Target):
    Prob_symptom = Probability_feature(Symptom)
    Prob_h_and_m = Joint_Probability('Headache', 'Meningitis')
    Prob_m_given_h = (Prob_h_and_m / Prob_h) 
    return Prob_m_given_h


#   Probabilitis (Fraud)
P_fraud = Probability_feature(df1_fraud, 'Fraud', 'T')
P_CH_given_fraud = Probability_feature(df1_fraud, 'Credit History', 'none')

#to make a prediction you would multiple the conditional probability of the features in question (separately for True and False of the Target) times the probability of the event (Fraud or Not Fraud) and then compare which percentage is higher. 



#   Probabilities (Not-Fraud)

P_not_fraud = Probability_feature(df1_fraud, 'Fraud', 'F')



####    MENINGITIS EXAMPLE



Dict = {}
Dict['Headache'] = ['true', 'false', 'true', 'true', 'false', 'true', 'true', 'true', 'false', 'true']
Dict['Fever'] = ['true', 'true', 'false', 'false', 'true', 'false', 'false', 'false', 'true', 'false']
Dict['Vomiting'] = ['false', 'false', 'true', 'true', 'false', 'true', 'true', 'true', 'false', 'true']
Dict['Meningitis'] = ['false', 'false', 'false', 'false', 'true', 'false', 'false', 'true', 'false', 'true' ]

df = pd.DataFrame(Dict)

def Probability_symptom(Dataframe, Symptom):
    Count = 0
    List = []
    for x in Dataframe[Symptom]:
        if x == 'true':
            x = int(1)
            List.append(x)
            Count += x
        else:
            List.append(0)
    Prob_symptom = (Count / len(List))
    return  Prob_symptom

Probability_symptom(df1






######  CLAIMS INCIDENCE BY STATE




#   Import ABT
File = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Project Code\Naive Bayes\Final Project - Section IV - Final ABT_Naive Bayes Experiment.xlsx')

#   Create Dataframe
df = pd.DataFrame(File)

#   Limit to Feature = State, Target = Claim
ABT = df[['State', 'Claim Count']]

#   Group claims by State
ABT_groupby_state = ABT.groupby('State').sum()

ABT_reset_index = ABT_groupby_state.reset_index()

#   Calculate Num claims by state
Total_claim_count = ABT_reset_index['Claim Count'].sum()

#   Calculate total policy count
Total_policy_count = len(df)

#   Calculate Percentage of Claim Count by State / Total Pol with CLaims
ABT_reset_index['Relative Probability'] = [(x/Total_claim_count)*100 for x in ABT_reset_index['Claim Count']]

#   Calculate Percentage of Claim Count by State / Total Policies 
ABT_reset_index['Joint Probability'] = [(x/Total_policy_count)*100 for x in ABT_reset_index['Claim Count']] #Joint Probability - It was a CA policy and it had a claim. Also, think about P(CA) = Count(CA) / Count(T) * P(CA Claim) / Count(CA)


ABT_count_policies_state = ABT.groupby('State')['State'].count()

df_pol_count = pd.DataFrame(ABT_count_policies_state)

df_merged = pd.merge(left = df_pol_count, right = ABT_reset_index, left_index = True, right_on = 'State')

df_merged_final = df_merged.drop('State_y', axis = 1)

df_merged_sorted = df_merged_final.sort_values('Relative Probability', ascending = False)

# Note:  You'll need to create a Relative & Joint Probability for every feature in the dataframe. 
# Rename State_x to Policy Count


P_CA = 700/3444
P_Claim_CA = 143/700

#print(P_CA * P_Cliam_CA)


































                            
                            
   
       
                            
                            
                            
                            
                            
                            
                            
                            
                            
                            