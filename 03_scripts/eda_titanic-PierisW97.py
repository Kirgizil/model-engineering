# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:23:29 2023

@author: eichner
"""

import sys
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import math


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.stats import pearsonr
from collections import Counter
from matplotlib.colors import to_rgba


## read in csv file
df = pd.read_csv('train.csv')


## Feature Engineering: Title
df['Title'] = [i.split(',')[1].split('.')[0].strip() for i in df['Name']]


## Feature Engineering: Cabin Deck
deck = []

for i in df['Cabin']:
    
    if type(i) ==  str:
        
        if i[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            
            deck.append(i[0])
        
        else:
            
            deck.append(np.nan)
    
    else:
        
        deck.append(i)

df['Deck'] = deck

## Compute missing values based on Ticket Number
lookup = {df.Ticket[i]: df.Deck[i] for i in range(len(df.Ticket)) if pd.notnull(df.Deck[i])}

deck = []
for i in df.Ticket:
    
    try:
        
        deck.append(lookup[i])
    
    except:
        
        deck.append(np.nan)

df['Deck'] = deck

        

## Feature Engineering: Cabin Number
cabin_num = []

for i in df['Cabin']:
    
    if type(i) ==  str:

        if [int(j[1:]) for j in i.split() if len(j) > 1]:
            
            cabin_num.append(np.mean([int(j[1:]) for j in i.split() if len(j) > 1]))
        
        else:
            
            cabin_num.append(np.nan)
            
    else:
        
        cabin_num.append(i)

df['Number'] = cabin_num

## Compute missing values based on Ticket Number
lookup = {df.Ticket[i]: df.Number[i] for i in range(len(df.Ticket)) if pd.notnull(df.Number[i])}

cabin_num = []
for i in df.Ticket:
    
    try:
        
        cabin_num.append(lookup[i])
    
    except:
        
        cabin_num.append(np.nan)

df['Number'] = cabin_num


## Feature Engineering: Number of Relatives
df['Relatives'] = [i+j for i,j in zip(df['Parch'], df['SibSp'])]


## Feature Engineering: FarePerson
tickets = dict(Counter(df['Ticket']))

df['FarePerson'] = [j/tickets[i] for i,j in zip(df['Ticket'], df['Fare'])]
    

## drop attributes
df = df.drop(columns=[

    # 'Embarked',
    'Cabin',
    'PassengerId',
    'Name',
    # 'Parch',
    # 'SibSp',
    'Ticket',

    ])


## Imputation of Age attribute via median based on Title
age, age_flag = [], []
for i in range(len(df)):

    if pd.notnull(df['Age'].iloc[i]):
        
        age.append(df['Age'].iloc[i])
        age_flag.append('measured')
    
    else:
               
        for t in ['Mr', 'Mrs', 'Master', 'Miss', 'Dr']:
            
            if df['Title'].iloc[i] == t:
                
                age.append(statistics.median(df.loc[
                    
                    (df['Title'] == t) & 
                    (pd.notnull(df['Age'])), 'Age'])
                    
                    )
                
                age_flag.append('imputed')
                                
df['Age'] = age
df['Age_flag'] = age_flag


## round down Age attribute to integer value
df['Age'] = df['Age'].astype(int)


## One-hot encoding
tbl = pd.get_dummies(
    
    df, 
    columns=[
        
        'Sex',
        'Embarked',
        'Title',

        ], 
    drop_first=True,
    
    )


## Imputation of mh_fam and mh_bad
imputbr = IterativeImputer(BayesianRidge())

tbl[[

#    'Survived',
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Relatives',
    'FarePerson',
    'Sex_male',
    'Embarked_Q', ## to be imputed
    'Embarked_S', ## to be imputed
    'Title_Col',
    'Title_Don',
    'Title_Dr',
    'Title_Jonkheer',
    'Title_Lady',
    'Title_Major',
    'Title_Master',
    'Title_Miss',
    'Title_Mlle',
    'Title_Mme',
    'Title_Mr',
    'Title_Mrs',
    'Title_Ms',
    'Title_Rev',
    'Title_Sir',
    'Title_the Countess',
     
     ]] = pd.DataFrame(imputbr.fit_transform(tbl[[
         
#         'Survived',
         'Pclass',
         'Age',
         'SibSp',
         'Parch',
         'Fare',
         'Relatives',
         'FarePerson',
         'Sex_male',
         'Embarked_Q', ## to be imputed
         'Embarked_S', ## to be imputed
         'Title_Col',
         'Title_Don',
         'Title_Dr',
         'Title_Jonkheer',
         'Title_Lady',
         'Title_Major',
         'Title_Master',
         'Title_Miss',
         'Title_Mlle',
         'Title_Mme',
         'Title_Mr',
         'Title_Mrs',
         'Title_Ms',
         'Title_Rev',
         'Title_Sir',
         'Title_the Countess',
         
         ]]))


## Reverse OHE after imputation
embarked = []         
for i,j in zip(tbl['Embarked_Q'], tbl['Embarked_S']):
    
    if i == 1.0:
        
        embarked.append('Queenstown')
    
    elif j == 1.0:
        
        embarked.append('Southampton')
        
    else:
        
        embarked.append('Cherbourg')


embarked_flag = []
for i,j in zip(df['Embarked'], embarked):
    
    if i == j:
        
        embarked_flag.append('measured')
    
    else:
        
        embarked_flag.append('imputed')


df['Embarked'] = embarked  
df['Embarked_flag'] = embarked_flag


## Transform Pclass to 1st, 2nd, 3rd
p = {
     
     1:'1st',
     2:'2nd',
     3:'3rd',
     
     }

df['Pclass'] = [p[i] for i in df.Pclass]


## descriptive statistics for table    
pd.set_option('max_columns', None)
# print(df.describe(include='all'))
# print(df['Number'].describe())
# print(statistics.mode(df['Number']))

## calculate interquartile range 
# x = 'Number'
# q3, q1 = np.nanpercentile(df[x], [75 ,25])
# iqr = q3 - q1

# print(df[x].describe())
# print(df[x].value_counts())
# print(iqr)


# print(statistics.mean(df.dropna().Number))
# sys.exit()

## calculate the mode
# mode = df.Number.mode()
# print('Mode:', mode)

## calculate the percentage of the mode
# mode_count = df.Number.value_counts()[mode]
# total_count = 211#len(df)
# mode_percentage = (mode_count / total_count) * 100
# print('Mode Percentage:', mode_percentage)
# sys.exit()

## calculating the percentage within a range
# print(
      
#       100/891*df.loc[
          
#           (df['Age'] < 21) | 
#           (df['Age'] > 35), 'Age'
          
#           ].count()
      
      
      
#       )

# print(
      
#       100/891*
#       df.loc[
          
#           (df['Fare'] < 7.9) | 
#           (df['Fare'] > 31), 'Fare'
          
#           ].count()
      
      
      
#       )


## population versus sample StDev
x = 'Number'

s = df[x].std()
# print(s)
pop = (df[x].std(ddof=0))

SE = s / math.sqrt(211)#len(df[x]))
# print(SE)


## distributions

# sns.lmplot('Age', 'Age', data=df, fit_reg=False)
# sns.kdeplot(df.Age)
# sns.distplot(df.Age)
# plt.hist(df.Age, alpha=.3)
# sns.rugplot(df.Age)

# sns.boxplot(x=df.Age, fliersize=0, whis=1.5)


# fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)


# sns.histplot(df['Fare'], bins=10, ax=ax1, kde=True)
# sns.histplot(df['FarePerson'], bins=10, ax=ax2, kde=True)
# sns.histplot(df['Age'], bins=10, ax=ax3, kde=True)
# sns.histplot(df['SibSp'], bins=10, ax=ax4, kde=True)
# sns.histplot(df['Parch'], bins=10, ax=ax5, kde=True)
# sns.histplot(df['Relatives'], bins=10, ax=ax6, kde=True)

# ax1.set_xlabel('Fare', fontsize=10)
# ax2.set_xlabel('FarePerson', fontsize=10)
# ax3.set_xlabel('Age', fontsize=10)
# ax4.set_xlabel('SibSp', fontsize=10)
# ax5.set_xlabel('Parch', fontsize=10)
# ax6.set_xlabel('Relatives', fontsize=10)

# # ax3.xticks(range(0,90,10))

# ax2.set(ylabel=None)
# ax3.set(ylabel=None)
# ax5.set(ylabel=None)
# ax6.set(ylabel=None)

# # adjust space between plots
# plt.subplots_adjust(hspace = 0.5)

# plt.subplots_adjust(wspace=0.5)



# creating histogram and violin subplots of numerical data
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(17, 12))

# plt.subplot(2, 2, 1)
# ax = sns.histplot(df['Fare'], bins=200, kde=True, color='darkblue', alpha=1)
# ax.lines[0].set_color('crimson')
# ax.lines[0].set_linewidth(3)
# ax.lines[0].set_alpha(.75)
# ax.plot(linewidth=7.0)
# ax.legend(('probability density function', 'measures'), loc='upper right', shadow=True, prop={'size': 20})
# plt.xticks(range(0, 550, 100))
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=14)
# plt.xlabel(None)#'British Pound [£]', fontsize=16)
# plt.ylabel("Count", fontsize=16, labelpad=10)
# plt.xlim([-50, 550])
# plt.ylim([0, 275])
# plt.title('Ticket Fare', fontsize=25, pad=20, weight='bold')

# plt.subplot(2, 2, 2)
# ax = sns.histplot(df['Age'], bins=20, kde=True, color='darkblue', alpha=1)
# ax.lines[0].set_color('crimson')
# ax.lines[0].set_linewidth(3)
# ax.lines[0].set_alpha(.75)
# ax.plot(linewidth=7.0)
# # ax.legend(('probability', 'data'), loc='upper right', shadow=True)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xlabel(None)#'Age [yrs]', fontsize=16)
# plt.ylabel(None)#"Count", fontsize=16)
# plt.xlim([-10, 90])
# plt.ylim([0, 275])
# plt.title('Passenger Age', fontsize=25, pad=20, weight='bold')

# plt.subplot(2, 2, 3)
# ax = sns.violinplot(x=df.Fare, linewidth=3)
# ax.get_children()[1].set_color('white') # median
# ax.get_children()[2].set_color('black') # whisker
# ax.get_children()[3].set_color('black') # iqr
# ax.collections[0].set_facecolor(to_rgba('crimson', 0.1))
# ax.collections[0].set_edgecolor(to_rgba('crimson', .75))
# sns.stripplot(data=df, x='Fare', color='darkblue', size=2.5, marker="X", alpha=0.5)
# plt.xticks(range(0, 550, 100))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('British Pound [£]', fontsize=16, labelpad=10)
# plt.xlim([-50, 550])
# plt.ylabel('Probability', fontsize=16, labelpad=33)
# # plt.title('Ticket Fare', fontsize=25, pad=20, weight='bold')

# plt.subplot(2, 2, 4)
# ax = sns.violinplot(x=df.Age, linewidth=3)
# ax.get_children()[1].set_color('white') # median
# ax.get_children()[2].set_color('black') # whisker
# ax.get_children()[3].set_color('black') # iqr
# ax.collections[0].set_facecolor(to_rgba('crimson', 0.1))
# ax.collections[0].set_edgecolor(to_rgba('crimson', .75))
# sns.stripplot(data=df, x='Age', color='darkblue', size=2.5, marker="X", alpha=0.5)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Age [yrs]', fontsize=16, labelpad=10)
# plt.xlim([-10, 90])
# # plt.ylabel('Density', fontsize=16, labelpad=15)
# # plt.title('Passenger Age', fontsize=25, pad=20, weight='bold')

# # space between subplots
# plt.subplots_adjust(hspace = 0.05) 
# plt.subplots_adjust(wspace=0.05)

# # set the DPI to 300
# plt.figure(dpi=300)

# plt.show()


# creating stacked histogram subplots of numerical data
# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(16, 12))

# ## Sex
# plt.subplot(2, 3, 1)
# ax = sns.histplot(df, x='Age', hue='Sex', bins=40, kde=True, multiple='stack', palette='bright', alpha=.9)
# ax.lines[0].set_linewidth(6)
# ax.lines[1].set_linewidth(6)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel("Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nby Sex', fontsize=20, pad=20, weight='bold')

# ## Passenger Class
# plt.subplot(2, 3, 2)
# ax = sns.histplot(df.sort_values(by=['Pclass']), x='Age', hue='Pclass', bins=40, multiple='stack', palette='bright', alpha=1)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=0)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel(None)#"Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nby Class', fontsize=20, pad=20, weight='bold')

# ## Embarked
# plt.subplot(2, 3, 3)
# ax = sns.histplot(df, x='Age', hue='Embarked', bins=40, multiple='stack', palette='bright', alpha=1)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=0)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel(None)#"Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nby Embarkation', fontsize=20, pad=20, weight='bold')

# ## Title
# plt.subplot(2, 3, 4)
# t = ['Mr', 'Mrs', 'Master', 'Miss']
# ax = sns.histplot(df.loc[(df['Title'].isin(t)) & (df['Age_flag'].isin(['measured','imputed']))], x='Age', hue='Title', bins=40, multiple='stack', palette='bright', alpha=1)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel("Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nby Title', fontsize=20, pad=20, weight='bold')

# ## Imputation
# plt.subplot(2, 3, 5)
# t = ['Mr', 'Mrs', 'Master', 'Miss']
# ax = sns.histplot(df.loc[(df['Title'].isin(t))], x='Age', hue='Age_flag', bins=40, multiple='stack', palette='bright', alpha=1)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=0)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel(None)#"Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nby Imputation', fontsize=20, pad=20, weight='bold')

# ## Without Imputation
# plt.subplot(2, 3, 6)
# ax = sns.histplot(df.loc[(df.Age_flag == 'measured')], x='Age', hue='Sex', kde=True, bins=40, multiple='stack', palette='bright', alpha=.9)
# ax.lines[0].set_linewidth(6)
# ax.lines[1].set_linewidth(6)
# plt.xticks(range(0, 90, 10))
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=0)
# plt.xlabel('Age [yrs]', fontsize=16)
# plt.ylabel(None)#"Count", fontsize=16)
# plt.xlim([-5, 85])
# plt.ylim([0, 175])
# plt.title('Passenger Age\nw/o Imputation by Sex', fontsize=20, pad=20, weight='bold')

# # space between subplots
# plt.subplots_adjust(hspace = 0.5) 
# plt.subplots_adjust(wspace=0.05)

# # set the DPI to 300
# plt.figure(dpi=300)

# plt.show()

# x = [10,11,12,13]

# print(df.loc[df['Age'].isin(x)]['Age'].value_counts())

print(statistics.median(df.loc[df['Title'] == 'Dr']['Age']))





sys.exit()



ax = sns.violinplot(x=df.Age, linewidth=3)
sns.stripplot(x=df.Age, color='red', size=2.5, marker="X")
# sns.distplot(df.Age)

ax.collections[0].set_facecolor(to_rgba('red', 0.1))
ax.collections[0].set_edgecolor(to_rgba('black', 0.3))
ax.set_yticklabels(['Density'], rotation=0)

plt.xticks(range(0,90,10))











          


# print(df.head(100))


## location ##

## mean
print(statistics.mean(df[  # female 3rd, 2nd, 1st == 21.6, 28.7, 34.6
                           # male 3rd, 2nd, 1st == 26.2, 31.0, 41.1
    (df['Pclass'] == 1) & 
    (df['Sex_ohe_male'] == 0)
    
    ]['Age']))


print(statistics.median(df[  # female 3rd, 2nd, 1st == 22, 28.5, 35
                             # male 3rd, 2nd, 1st == 26, 31, 41
    (df['Pclass'] == 3) & 
    (df['Sex_ohe_male'] == 0)
    
    ]['Age']))


## variance
print(statistics.variance(df[  # female 3rd, 2nd, 1st == 115, 161, 168
                             # male 3rd, 2nd, 1st == 108, 202, 191
    (df['Pclass'] == 1) & 
    (df['Sex_ohe_male'] == 1)
    
    ]['Age']))

print(statistics.stdev(df[  # female 3rd, 2nd, 1st == 10.7, 12.7, 12.9
                             # male 3rd, 2nd, 1st == 10.4, 14.2, 13.8
    (df['Pclass'] == 1) & 
    (df['Sex_ohe_male'] == 1)
    
    ]['Age']))


## distributions

# sns.lmplot('Age', 'Age', data=df, fit_reg=False)
# sns.kdeplot(df.Age)
# sns.distplot(df.Age)
# plt.hist(df.Age, alpha=.3)
# sns.rugplot(df.Age)

# sns.boxplot(x=df.Age, fliersize=0, whis=1.5)
# sns.violinplot(df.Age)
# sns.stripplot(x=df.Age, color='red', size=1)

# sns.displot(
    
#     data=df[(df['Pclass'] == 3)],
#     x='Age',
#     hue='Sex',
#     multiple='stack'
    
#     )

# sns.displot(
    
#     data=df[(df['Sex'] == 'male')],
#     x='Age',
#     hue='Pclass',
#     multiple='stack',
#     palette='bright'
    
#     )

# sns.displot(
    
#     data=df,#[(df['Sex'] == 'female')],
#     x='Pclass',
#     hue='Survived',
#     multiple='stack',
#     palette='bright'
    
#     )

# sns.kdeplot(
    
#     data=df[(df['Pclass'] == 3)],
#     x='Age',
#     hue='Sex',
#     fill=True,
#     common_norm=False,
#     palette='pastel',
#     alpha=.5,
#     linewidth=0,
    
#     )

# sns.kdeplot(
    
#     data=df[(df['Sex'] == 'male')],
#     x='Age',
#     hue='Pclass',
#     fill=True,
#     common_norm=False,
#     palette='pastel',
#     alpha=.5,
#     linewidth=0,
    
#     )

# sns.displot(
    
#     data=df[(df['Sex'] == 'male')],
#     x='Age',
#     hue='Pclass',
#     multiple='stack',
#     palette='bright'
    
#     )


## co-variance
X = np.stack((        ## Age and Pclass are negatively correlated (male/female)
    
    df[(df['Sex_ohe_male'] == 0)]['Age'],
    df[(df['Sex_ohe_male'] == 0)]['Pclass']
    
    ), axis=0)

# print(np.cov(X))

X = np.stack((        ## Age and Survived is negatively correlated (male); female is slightly positive correlated
    
    df[(df['Sex_ohe_male'] == 1)]['Age'],
    df[(df['Sex_ohe_male'] == 1)]['Survived']
    
    ), axis=0)

# print(np.cov(X))

X = np.stack((        ## Age and Survived is slightly negatively correlated (male/female)
    
    df[(df['Sex_ohe_male'] == 0)]['Pclass'],
    df[(df['Sex_ohe_male'] == 0)]['Survived']
    
    ), axis=0)

# print(np.cov(X))


## correlation

corr = pearsonr(
    
    df[(df['Sex_ohe_male'] == 0)]['Pclass'],
    df[(df['Sex_ohe_male'] == 0)]['Fare']
    
    )

print(corr)

sns.set_theme(style='dark')

sns.heatmap(df[['Pclass', 'Age', 'Survived', 'Fare', 'Sex_ohe_male']].corr())







sys.exit()

# print(df.head(10))

# print(df[['Pclass', 'Age', 'Sex']].value_counts())
# print(df[['Pclass', 'Age', 'Sex']].describe())

upper_class = df[
    
    (df['Pclass'] == 1) & 
    (df['Sex'] == 'female') &
    (df['Age'] < 18)
    
    ]
# upper_class = df[df['Sex'] == 'female']

print(upper_class['Survived'].value_counts())
print(upper_class['Survived'].describe()) 


# print(upper_class['Age'].value_counts())
# print(upper_class['Age'].describe())

print(df[
    
    (df['Pclass'] == 2) & 
    (df['Sex'] == 'female')
    
    ]['Age'].median())