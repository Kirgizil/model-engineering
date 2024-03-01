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
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Rectangle


## read in csv file
df = pd.read_csv('../train.csv')


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
#    'Ticket',

    ])


## Imputation of Age attribute via median based on Title

# set the seed to ensure reproducibility
np.random.seed(42)

age, age_flag, age_gauss = [], [], []
for i in range(len(df)):

    if pd.notnull(df['Age'].iloc[i]):
        
        age.append(df['Age'].iloc[i])
        age_flag.append('measured')
        age_gauss.append(df['Age'].iloc[i])
    
    else:
        
        for t in ['Mr', 'Mrs', 'Master', 'Miss', 'Dr']:
            
            if df['Title'].iloc[i] == t:
                
                
                mean = statistics.mean(
                    
                    df.loc[
                        
                        (df['Title'] == t) & 
                        (pd.notnull(df['Age'])), 'Age'
                        
                        ])
                    
                
                median = statistics.median(
                    
                    df.loc[
                        
                        (df['Title'] == t) & 
                        (pd.notnull(df['Age'])), 'Age'
                        
                        ])
                
                stdev = statistics.stdev(
                    
                    df.loc[
                        
                        (df['Title'] == t) & 
                        (pd.notnull(df['Age'])), 'Age'
                        
                        ])
                
                age.append(median)
                age_flag.append('imputed')
                age_gauss.append(int(np.random.normal(mean,stdev,1)[0]))
                

df['Age'] = age
df['Age_flag'] = age_flag
df['Age_gauss'] = age_gauss


## round down Age attribute to integer value
df['Age'] = df['Age'].astype(int)


## One-hot encoding
tbl = pd.get_dummies(
    
    df, 
    columns=[
        
        'Sex',
        'Embarked',
        'Title',
        'Deck',

        ], 
#    drop_first=True,
    
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
    
    if i == j[:1]:
        
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
# x = 'Number'

# s = df[x].std()
# print(s)
# pop = (df[x].std(ddof=0))

# SE = s / math.sqrt(211)#len(df[x]))
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



## creating histogram and violin subplots of numerical data
sns.set_theme(style="whitegrid")
plt.figure(figsize=(17, 12))

plt.subplot(2, 2, 1)
ax = sns.histplot(df['Fare'], bins=200, kde=True, color='darkblue', alpha=1)
ax.lines[0].set_color('crimson')
ax.lines[0].set_linewidth(6)
ax.lines[0].set_alpha(.75)
ax.plot(linewidth=7.0)
ax.legend(('probability density function', 'measures'), loc='upper right', shadow=True, prop={'size': 20})
plt.xticks(range(0, 550, 100))
plt.xticks(fontsize=0)
plt.yticks(fontsize=18)
plt.xlabel(None)#'British Pound [£]', fontsize=16)
plt.ylabel("Count", fontsize=20, labelpad=10)
plt.xlim([-50, 550])
plt.ylim([0, 275])
plt.title('Ticket Fare', fontsize=25, pad=20, weight='bold')

plt.subplot(2, 2, 2)
ax = sns.histplot(df['Age'], bins=40, kde=True, color='darkblue', alpha=1)
ax.lines[0].set_color('crimson')
ax.lines[0].set_linewidth(6)
ax.lines[0].set_alpha(.75)
ax.plot(linewidth=7.0)
# ax.legend(('probability', 'data'), loc='upper right', shadow=True)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.xlabel(None)#'Age [yrs]', fontsize=16)
plt.ylabel(None)#"Count", fontsize=16)
plt.xlim([-10, 90])
plt.ylim([0, 275])
plt.title('Passenger Age', fontsize=25, pad=20, weight='bold')

plt.subplot(2, 2, 3)
ax = sns.violinplot(x=df.Fare, linewidth=6)
ax.get_children()[1].set_color('white') # median
ax.get_children()[2].set_color('black') # whisker
ax.get_children()[3].set_color('black') # iqr
ax.collections[0].set_facecolor(to_rgba('crimson', 0.1))
ax.collections[0].set_edgecolor(to_rgba('crimson', .75))
sns.stripplot(data=df, x='Fare', color='darkblue', size=5, marker="o", alpha=0.3)
plt.xticks(range(0, 550, 100))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('British Pound [£]', fontsize=20, labelpad=10)
plt.xlim([-50, 550])
plt.ylabel('Probability', fontsize=20, labelpad=33)
# plt.title('Ticket Fare', fontsize=25, pad=20, weight='bold')

plt.subplot(2, 2, 4)
ax = sns.violinplot(x=df.Age, linewidth=6)
ax.get_children()[1].set_color('white') # median
ax.get_children()[2].set_color('black') # whisker
ax.get_children()[3].set_color('black') # iqr
ax.collections[0].set_facecolor(to_rgba('crimson', 0.1))
ax.collections[0].set_edgecolor(to_rgba('crimson', .75))
sns.stripplot(data=df, x='Age', color='darkblue', size=5, marker="o", alpha=0.3)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Age [yrs]', fontsize=20, labelpad=10)
plt.xlim([-10, 90])
# plt.ylabel('Density', fontsize=16, labelpad=15)
# plt.title('Passenger Age', fontsize=25, pad=20, weight='bold')

# space between subplots
plt.subplots_adjust(hspace = 0.05) 
plt.subplots_adjust(wspace=0.05)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()


# creating stacked histogram subplots of numerical data
sns.set_theme(style="whitegrid")
plt.figure(figsize=(16, 12))

## Sex
plt.subplot(2, 3, 1)
ax = sns.histplot(df, x='Age', hue='Sex', bins=40, kde=True, multiple='stack', palette='bright', alpha=.9)
ax.lines[0].set_linewidth(6)
ax.lines[1].set_linewidth(6)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel("Count", fontsize=18)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nby Sex', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

## Passenger Class
plt.subplot(2, 3, 2)
ax = sns.histplot(df.sort_values(by=['Pclass']), x='Age', hue='Pclass', bins=40, multiple='stack', palette='bright', alpha=1)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=0)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel(None)#"Count", fontsize=16)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nby Class', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

## Embarked
plt.subplot(2, 3, 3)
ax = sns.histplot(df, x='Age', hue='Embarked', bins=40, multiple='stack', palette='bright', alpha=1)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=0)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel(None)#"Count", fontsize=16)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nby Embarkation', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

## Title
plt.subplot(2, 3, 4)
t = ['Mr', 'Mrs', 'Master', 'Miss']
ax = sns.histplot(df.loc[(df['Title'].isin(t)) & (df['Age_flag'].isin(['measured','imputed']))], x='Age', hue='Title', bins=40, multiple='stack', palette='bright', alpha=1)
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel("Count", fontsize=18)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nby Title', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

## Imputation
plt.subplot(2, 3, 5)
t = ['Mr', 'Mrs', 'Master', 'Miss']
ax = sns.histplot(df.loc[(df['Title'].isin(t))], x='Age', kde=True, hue='Age_flag', bins=40, multiple='stack', palette='bright', alpha=1)
ax.lines[0].set_linewidth(0)
ax.lines[1].set_linewidth(6)
ax.lines[1].set_color('crimson')
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=0)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel(None)#"Count", fontsize=16)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nwith Median Imputation', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

## Without Imputation
plt.subplot(2, 3, 6)
# ax = sns.histplot(df.loc[(df.Age_flag == 'measured')], x='Age', hue='Sex', kde=True, bins=40, multiple='stack', palette='bright', alpha=.9)
ax = sns.histplot(df, x='Age_gauss', hue='Age_flag', kde=True, bins=40, multiple='stack', palette='bright', alpha=.9)
ax.lines[0].set_linewidth(0)
ax.lines[1].set_linewidth(6)
ax.lines[1].set_color('crimson')
plt.xticks(range(0, 90, 10))
plt.xticks(fontsize=18)
plt.yticks(fontsize=0)
plt.xlabel('Age [yrs]', fontsize=18)
plt.ylabel(None)#"Count", fontsize=16)
plt.xlim([-5, 85])
plt.ylim([0, 175])
plt.title('Passenger Age\nwith Gauss Imputation', fontsize=20, pad=20, weight='bold')

plt.setp(ax.get_legend().get_texts(), fontsize='16')
plt.setp(ax.get_legend().get_title(), fontsize='16')

# space between subplots
plt.subplots_adjust(hspace = 0.5) 
plt.subplots_adjust(wspace=0.05)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()


## find out about age gap
# x = [10,11,12,13]
# print(df.loc[df['Age'].isin(x)]['Age'].value_counts())
# print(statistics.median(df.loc[df['Title'] == 'Dr']['Age']))


## for reference to understand color options
# palette = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 
# 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
# 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 
# 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
# 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
# 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 
# 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 
# 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 
# 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 
# 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
# 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
# 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 
# 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 
# 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 
# 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 
# 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 
# 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 
# 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 
# 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 
# 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 
# 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 
# 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 
# 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 
# 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 
# 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 
# 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 
# 'vlag_r', 'winter', 'winter_r']


## creating heatmaps
plt.figure(figsize=(16, 8))

## Percentage Surviver: Survived vs. Sex
plt.subplot(2, 4, 1)

ax = sns.heatmap(np.array([
    
    [list(df.loc[(df.Sex == 'male')].Survived.value_counts())[0]/sum(list(df.loc[(df.Sex == 'male')].Survived.value_counts())),
    list(df.loc[(df.Sex == 'male')].Survived.value_counts())[1]/sum(list(df.loc[(df.Sex == 'male')].Survived.value_counts()))],
    
    [list(df.loc[(df.Sex == 'female')].Survived.value_counts())[0]/sum(list(df.loc[(df.Sex == 'female')].Survived.value_counts())),
    list(df.loc[(df.Sex == 'female')].Survived.value_counts())[1]/sum(list(df.loc[(df.Sex == 'female')].Survived.value_counts()))]
    
          
    ]), annot=True, fmt=".0%", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['killed', 'lived'])
ax.set_yticklabels(['male', 'female'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('PERCENTAGE SURVIVER\nSurvived vs. Sex', fontsize=14, pad=15, weight='bold')

## Percentage Surviver: Survived vs. Class
plt.subplot(2, 4, 2)

ax = sns.heatmap(np.array([
    
    [i/len(df.loc[(df.Pclass == '1st')].Survived) for i in list(df.loc[(df.Pclass == '1st')].Survived.value_counts())],
    [i/len(df.loc[(df.Pclass == '2nd')].Survived) for i in list(df.loc[(df.Pclass == '2nd')].Survived.value_counts())],
    [i/len(df.loc[(df.Pclass == '3rd')].Survived) for i in list(df.loc[(df.Pclass == '3rd')].Survived.value_counts())],
          
    ]), annot=True, fmt=".0%", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['killed', 'lived'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('PERCENTAGE SURVIVER\nSurvived vs. Class', fontsize=14, pad=15, weight='bold')

## Percentage Passenger: Sex vs. Class
plt.subplot(2, 4, 3)

ax = sns.heatmap(np.array([
    
    [i/len(df.loc[(df.Pclass == '1st')].Sex) for i in list(df.loc[(df.Pclass == '1st')].Sex.value_counts())],
    [i/len(df.loc[(df.Pclass == '2nd')].Sex) for i in list(df.loc[(df.Pclass == '2nd')].Sex.value_counts())],
    [i/len(df.loc[(df.Pclass == '3rd')].Sex) for i in list(df.loc[(df.Pclass == '3rd')].Sex.value_counts())],
          
    ]), annot=True, fmt=".0%", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['male', 'female'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('PERCENTAGE PASSENGER\nSex vs. Class', fontsize=14, pad=15, weight='bold')

## Percentage Housed: Deck vs. Class
plt.subplot(2, 4, 4)

ax = sns.heatmap(np.array([
    
    [len(df.loc[(df.Pclass == '1st') & (df.Deck == 'A')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'B')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'C')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'D')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'E')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'F')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '1st') & (df.Deck == 'G')])/len(df.loc[(df.Pclass == '1st') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))])],
    
    [len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'A')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'B')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'C')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'D')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'E')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'F')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '2nd') & (df.Deck == 'G')])/len(df.loc[(df.Pclass == '2nd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))])],
    
    [len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'A')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'B')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'C')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'D')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'E')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'F')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))]),
      len(df.loc[(df.Pclass == '3rd') & (df.Deck == 'G')])/len(df.loc[(df.Pclass == '3rd') & (df.Deck.isin(['A', 'B', 'C', 'D', 'E', 'F', 'G']))])],
     
          
    ]), annot=False, fmt=".0%", cmap='viridis', annot_kws={"size": 8}
    
    )

ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('PRECENTAGE HOUSED\nDeck vs. Class', fontsize=14, pad=15, weight='bold')


## Median Age: Sex vs. Class
plt.subplot(2, 4, 5)

ax = sns.heatmap(np.array([
    
    [statistics.median(df.loc[(df.Pclass == '1st') & (df.Sex == 'male')].Age),
      statistics.median(df.loc[(df.Pclass == '1st') & (df.Sex == 'female')].Age)],
    
    [statistics.median(df.loc[(df.Pclass == '2nd') & (df.Sex == 'male')].Age),
      statistics.median(df.loc[(df.Pclass == '2nd') & (df.Sex == 'female')].Age)],
    
    [statistics.median(df.loc[(df.Pclass == '3rd') & (df.Sex == 'male')].Age),
      statistics.median(df.loc[(df.Pclass == '3rd') & (df.Sex == 'female')].Age)],
          
    ]), annot=True, fmt=".0f", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['male', 'female'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('MEDIAN AGE\nSex vs. Class', fontsize=14, pad=15, weight='bold')

## Mean Relatives: Sex vs. Class
# plt.subplot(2, 4, 6)

# ax = sns.heatmap(np.array([
    
#     [statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'male')].Relatives),
#       statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'female')].Relatives)],
    
#     [statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'male')].Relatives),
#       statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'female')].Relatives)],
    
#     [statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'male')].Relatives),
#       statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'female')].Relatives)],
          
#     ]), annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 20}
    
#     )

# ax.set_xticklabels(['male', 'female'])
# ax.set_yticklabels(['1st', '2nd', '3rd'])
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.title('MEAN RELATIVES\nSex vs. Class', fontsize=14, pad=15, weight='bold')

## Mean Siblings/Spouses: Sex vs. Class
plt.subplot(2, 4, 6)

ax = sns.heatmap(np.array([
    
    [statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'male')].SibSp),
      statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'female')].SibSp)],
    
    [statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'male')].SibSp),
      statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'female')].SibSp)],
    
    [statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'male')].SibSp),
      statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'female')].SibSp)],
          
    ]), annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['male', 'female'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('MEAN SIBLINGS/SPOUSES\nSex vs. Class', fontsize=14, pad=15, weight='bold')

## Mean Parents/Children: Sex vs. Class

print(len(df.loc[(df.Pclass == '1st') & (df.Parch == 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '1st') & (df.Parch == 0) & (df.Age_gauss < 21)]),
      
      len(df.loc[(df.Pclass == '1st') & (df.Parch > 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '1st') & (df.Parch > 0) & (df.Age_gauss < 21)])
      
      )

## Mean Parents/Children: Sex vs. Class
plt.subplot(2, 4, 7)

ax = sns.heatmap(np.array([
    
    [statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'male')].Parch),
      statistics.mean(df.loc[(df.Pclass == '1st') & (df.Sex == 'female')].Parch)],
    
    [statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'male')].Parch),
      statistics.mean(df.loc[(df.Pclass == '2nd') & (df.Sex == 'female')].Parch)],
    
    [statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'male')].Parch),
      statistics.mean(df.loc[(df.Pclass == '3rd') & (df.Sex == 'female')].Parch)],
          
    ]), annot=True, fmt=".2f", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['male', 'female'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('MEAN PARENTS/CHILDREN\nSex vs. Class', fontsize=14, pad=15, weight='bold')

## Percentage <21 Surviver: Class vs. Guardian
plt.subplot(2, 4, 8)

ax = sns.heatmap(np.array([
    
    [len(df.loc[(df.Pclass == '1st') & (df.Parch == 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '1st') & (df.Parch == 0) & (df.Age_gauss < 21)]),
          
      len(df.loc[(df.Pclass == '1st') & (df.Parch > 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '1st') & (df.Parch > 0) & (df.Age_gauss < 21)])],
    
    [len(df.loc[(df.Pclass == '2nd') & (df.Parch == 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '2nd') & (df.Parch == 0) & (df.Age_gauss < 21)]),
          
      len(df.loc[(df.Pclass == '2nd') & (df.Parch > 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '2nd') & (df.Parch > 0) & (df.Age_gauss < 21)])],
    
    [len(df.loc[(df.Pclass == '3rd') & (df.Parch == 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '3rd') & (df.Parch == 0) & (df.Age_gauss < 21)]),
          
      len(df.loc[(df.Pclass == '3rd') & (df.Parch > 0) & (df.Age_gauss < 21) & (df.Survived == 1)])/
      len(df.loc[(df.Pclass == '3rd') & (df.Parch > 0) & (df.Age_gauss < 21)])],
    
          
    ]), annot=True, fmt=".0%", cmap='viridis', annot_kws={"size": 20}
    
    )

ax.set_xticklabels(['alone', 'guardian'])
ax.set_yticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('PERCENTAGE <21 SURVIVER\nClass vs. Guardian', fontsize=14, pad=15, weight='bold')

# space between subplots
plt.subplots_adjust(hspace = 0.5) 
plt.subplots_adjust(wspace=0.25)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()


tbl = pd.get_dummies(
    
    df, 
    columns=[
        
        'Pclass',
        'Sex',
        'Title',
        'Embarked',
        'Deck',

        ]
    
    )

# ## creating correlation heatmaps (pearson, spearman)
plt.figure(figsize=(14, 6))

## Numerical values only --> Pearson
plt.subplot(1, 2, 1)

att = {
       
        'Fare' : 'Ticket Fare', 
        'FarePerson' : 'Passenger Fare', 
        'Age_gauss' : 'Passenger Age', 
        'Relatives' : '#Relatives', 
        'SibSp' : '#Siblings/Spouses', 
        'Parch' : '#Parents/Children',
       
        }

p = tbl[att.keys()]

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(p)

# Compute the Pearson correlation matrix
ax = sns.heatmap(
    
    pd.DataFrame(X_normalized).corr(method='pearson'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":20},
    cmap='viridis',
    cbar=False,
    
    )

ax.set_xticklabels(att.values(), rotation=90)
ax.set_yticklabels(att.values(), rotation=0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('PEARSON CORRELATION\nnumerical features', fontsize=20, pad=15, weight='bold')

## highlighting
ax.add_patch(Rectangle((0.054, 3.04), 1.9, 2.89, fill=False, edgecolor='lightblue', lw=5, linestyle='-'))
ax.add_patch(Rectangle((3.04, 0.04), 2.9, 1.9, fill=False, edgecolor='lightblue', lw=5, linestyle='-'))
ax.add_patch(Rectangle((2.04, 3.03), 0.9, 2.9, fill=False, edgecolor='magenta', lw=5))
ax.add_patch(Rectangle((3.04, 2.04), 2.9, 0.9, fill=False, edgecolor='magenta', lw=5))
ax.add_patch(Rectangle((0.054, 0.04), 1.9, 1.9, fill=False, edgecolor='darkgreen', lw=5, linestyle='dotted'))
ax.add_patch(Rectangle((3.048, 3.03), 2.9, 2.9, fill=False, edgecolor='darkgreen', lw=5, linestyle='--'))



## Survivier Features
plt.subplot(1, 2, 2)

att = {
       
        'Survived' : 'Survived', 
        'Sex_female' : 'Female', 
        'Title_Mrs' : 'Mrs', 
        'Pclass_1st' : '1st Class',
        'Fare' : 'Ticket Fare',
        'Embarked_Cherbourg' : 'Cherbourg',
        # 'Parch' : '#Parents/Children',
       
        }

p = tbl[att.keys()]

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(p)

# Compute the Pearson correlation matrix
ax = sns.heatmap(
    
    pd.DataFrame(X_normalized).corr(method='spearman'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":20},
    cmap='viridis',
    cbar=False,
    
    )

ax.set_xticklabels(att.values(), rotation=90)
ax.set_yticklabels(att.values(), rotation=0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('SPEARMAN CORRELATION\nsurviver features', fontsize=20, pad=15, weight='bold')

## highlighting
ax.add_patch(Rectangle((4.054, 1.04), 0.9, 0.89, fill=False, edgecolor='lightblue', lw=5, linestyle='-'))
ax.add_patch(Rectangle((1.04, 4.054), 0.89, 0.9, fill=False, edgecolor='lightblue', lw=5, linestyle='-'))


# space between subplots
plt.subplots_adjust(hspace = 0.5) 
plt.subplots_adjust(wspace=0.4)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()


## Covariance Computation of numerical values
# Ticket_Fare = np.array([df.Fare])
# Passenger_Fare = np.array([df.FarePerson])
# Passenger_Age = np.array([df.Age_gauss])
# Siblings_Spouses = np.array([df.SibSp])
# Parents_Children =  np.array([df.Parch])
# Relatives = np.array([df.Relatives])


# scaler = StandardScaler()

# Ticket_Fare_normalized = scaler.fit_transform(Ticket_Fare.reshape(-1, 1)).flatten()
# Passenger_Fare_normalized = scaler.fit_transform(Passenger_Fare.reshape(-1, 1)).flatten()
# Passenger_Age_normalized = scaler.fit_transform(Passenger_Age.reshape(-1, 1)).flatten()
# Siblings_Spouses_normalized = scaler.fit_transform(Siblings_Spouses.reshape(-1, 1)).flatten()
# Parents_Children_normalized = scaler.fit_transform(Parents_Children.reshape(-1, 1)).flatten()
# Relatives_normalized = scaler.fit_transform(Relatives.reshape(-1, 1)).flatten()

# l = [
    
#     Ticket_Fare_normalized,
#     Passenger_Fare_normalized,
#     Passenger_Age_normalized,
#     Siblings_Spouses_normalized,
#     Parents_Children_normalized,
#     Relatives_normalized,
    
#     ]

# cov_matrix = np.cov(l)

# l_name = [
    
#     'Ticket_Fare',
#     'Passenger_Fare',
#     'Passenger_Age',
#     '#Siblings_Spouses',
#     '#Parents_Children',
#     '#Relatives',
    
#     ]


# ## Print the covariance for each pair of variables
# for i in range(len(l)):

#     for j in range(i+1, len(l)):

#         print(f"Covariance between {l_name[i]} and {l_name[j]}: {cov_matrix[i,j]:.2f}")




## Interesting aspects
data = []

for q in ['1st', '2nd', '3rd']:
    
    test = df.loc[(df.Age_gauss < 21) & (df.Pclass == q)].values.tolist()
    
    guard1, guard2 = [], []
    
    for i in test:
        
        tmp = {}
        tmp2 = {}
        
        for j in df[['Ticket', 'Age_gauss', 'Sex', 'Survived', 'SibSp']].values.tolist():
                       
            if i[6] == j[0] and i[5] > 0 and j[1] > i[15]:
                
                i.append(j[1])
                tmp[j[1]] = j[2]
                tmp2[j[1]] = j[3]
        
        if len(i) > 17 and i[5] == 1:
            
            l = max(i[17:])
            
            guard1.append([i[1], l, tmp[l], i[15], i[0], tmp2[l], i[4]])
        
        elif len(i) > 17 and i[5] > 1:
            
            l = max(i[17:])
            # i.remove(l)
            # s = max(i[17:])
            
            guard2.append([i[1], l, tmp[l], i[15], i[0], tmp2[l], i[4]])
    
    
    data.append([
        
        len(guard1)/(len(guard1)+len(guard2)), 
        np.median([i[3] for i in guard1]), 
        np.median([i[1] for i in guard1]), 
        [i[2] for i in guard1].count('female')/len(guard1),
        np.average([i[6] for i in guard1]), ## number of siblings
        sum([i[4] for i in guard1])/len(guard1),
        # sum([i[4] for i in guard2])/len(guard2),
        sum([i[5] for i in guard1])/len(guard1),
        # sum([i[5] for i in guard2])/len(guard2),
        
        ])


# Create a uniform color colormap with a single color
color1 = 'whitesmoke'
color2 = 'whitesmoke'
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("", [color1, color2])


## creating heatmaps
plt.figure(figsize=(14, 8))

## Percentage Surviver: Survived vs. Sex
plt.subplot(1, 2, 1)

ax = sns.heatmap(np.array(   
    
    list(zip(*data))
    
    ), annot=True, fmt=".1f", annot_kws={"size": 30}, cmap=cmap, cbar=False, linecolor='black', linewidths=3
    
    )

ax.set_yticklabels([
    
    'one guardian\nonly ratio', 
    'median\nward age', 
    'median\nguardian age', 
    'guardian\nfemale ratio',
    'average\n #siblings',
    'ward\nsurviver ratio',
    # 'multiple guardian\nsurviver ratio',
    'guardian\nsurviver ratio',
    # 'multiple guardian\nsurviver ratio',
    
    
    ], rotation=0,)# ha='left')

# find the maximum width of the label on the major ticks
# yax = ax.get_yaxis()
# pad = max(T.label.get_window_extent().width for T in yax.majorTicks)*1.5
# yax.set_tick_params(pad=pad)

ax.set_xticklabels(['1st', '2nd', '3rd'])
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('UNDERAGED CHILDREN (<21yrs)\nguarded by one parent only', fontsize=20, pad=15, weight='bold')

ax.add_patch(Rectangle((0.05, 0.075), 0.90, 0.85, fill=True, facecolor='yellow'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((0.05, 1.075), 0.90, 0.85, fill=True, facecolor='yellow'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((0.05, 2.075), 0.90, 0.85, fill=True, facecolor='yellow'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((0.05, 5.075), 0.90, 0.85, fill=True, facecolor='lime'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((1.05, 5.075), 0.90, 0.85, fill=True, facecolor='lime'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((2.05, 5.075), 0.90, 0.85, fill=True, facecolor='red'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((0.05, 6.075), 0.90, 0.85, fill=True, facecolor='lime'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((1.05, 6.075), 0.90, 0.85, fill=True, facecolor='lime'))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((2.05, 6.075), 0.90, 0.85, fill=True, facecolor='red'))#, edgecolor='black', lw=1))

ax.add_patch(Rectangle((0.05, 4.075), 0.90, 0.85, fill=True, facecolor='magenta', alpha=0.2))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((1.05, 4.075), 0.90, 0.85, fill=True, facecolor='magenta', alpha=0.5))#, edgecolor='black', lw=1))
ax.add_patch(Rectangle((2.05, 4.075), 0.90, 0.85, fill=True, facecolor='magenta'))#, edgecolor='black', lw=1))

## spearman correlation of 3rd class passenger
plt.subplot(1, 2, 2)

data = []

test = df.loc[(df.Age_gauss < 21) & (df.Pclass == '3rd')].values.tolist()

guard3 = []

for i in test:
    
    tmp = {}
    tmp2 = {}
    
    for j in df[['Ticket', 'Age_gauss', 'Sex', 'Survived']].values.tolist():
                   
        if i[6] == j[0] and i[5] > 0 and j[1] > i[15]:
            
            i.append(j[1])
            tmp[j[1]] = j[2]
            tmp2[j[1]] = j[3]
    
    
    if len(i) > 17 and i[5] == 1:
        
        l = max(i[17:])
        
        guard3.append([i[0], tmp2[l]])
    
    elif len(i) > 17 and i[5] == 2:
        
        l = max(i[17:])
        i.remove(l)
        
        ## fix incomplete dataset
        if len(i) > 17:
            
            s = max(i[17:])
        
        else:
            
            s = l

        guard3.append([i[0], max([tmp2[l], tmp2[s]])])


ax = sns.heatmap(
    
    pd.DataFrame(guard3).corr(method='spearman'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":45},
    cmap='viridis',
    cbar=False,
    
    )

ax.set_xticklabels(['ward survived', 'guardian survived'], rotation=0)
ax.set_yticklabels(['ward survived', 'guardian survived'], rotation=90)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('SPEARMAN CORRELATION\n3rd class surviver', fontsize=20, pad=15, weight='bold')


# space between subplots
# plt.subplots_adjust(hspace = 0.5) 
# plt.subplots_adjust(wspace=0.25)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()


# 3rd class female correlation with relatives
# creating heatmaps
plt.figure(figsize=(14, 4))

## Percentage Surviver: Survived vs. Sex
plt.subplot(1, 3, 1)

female3rd = df.loc[(df.Sex == 'female') & (df.Pclass == '1st') & (df.Age > 20)][['Survived', 'SibSp', 'Parch']].values.tolist()

## Normalize the data
scaler = StandardScaler()
female3rd_normalized = scaler.fit_transform(female3rd)

## Compute the Pearson correlation matrix
ax = sns.heatmap(
    
    pd.DataFrame(female3rd_normalized).corr(method='spearman'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":30},
    cmap='viridis',
    cbar = False,
    
    )

ax.set_xticklabels(['survived', '#Siblings/\nSpouse', '#Parents/\nChildren'], rotation=0)
ax.set_yticklabels(['survived', '#Siblings/\nSpouse', '#Parents/\nChildren'], rotation=0)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.title('1ST FEMALE SURVIVER\nvs. #Relationships', fontsize=20, pad=15, weight='bold')


plt.subplot(1, 3, 2)

female3rd = df.loc[(df.Sex == 'female') & (df.Pclass == '2nd') & (df.Age > 20)][['Survived', 'SibSp', 'Parch']].values.tolist()

## Normalize the data
scaler = StandardScaler()
female3rd_normalized = scaler.fit_transform(female3rd)

## Compute the Pearson correlation matrix
ax = sns.heatmap(
    
    pd.DataFrame(female3rd_normalized).corr(method='spearman'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":30},
    cmap='viridis',
    yticklabels=False,
    cbar = False,
    
    )

ax.set_xticklabels(['survived', '#Siblings/\nSpouse', '#Parents/\nChildren'], rotation=0)
plt.xticks(fontsize=18)
plt.title('2ND FEMALE SURVIVER\nvs. #Relationships', fontsize=20, pad=15, weight='bold')


plt.subplot(1, 3, 3)

female3rd = df.loc[(df.Sex == 'female') & (df.Pclass == '3rd') & (df.Age > 20)][['Survived', 'SibSp', 'Parch']].values.tolist()

## Normalize the data
scaler = StandardScaler()
female3rd_normalized = scaler.fit_transform(female3rd)

## Compute the Pearson correlation matrix
ax = sns.heatmap(
    
    pd.DataFrame(female3rd_normalized).corr(method='spearman'),
    annot=True,
    fmt=".1f",
    annot_kws={"size":30},
    cmap='viridis',
    yticklabels=False,
    cbar = False,
    
    )

ax.set_xticklabels(['survived', '#Siblings/\nSpouse', '#Parents/\nChildren'], rotation=0)
plt.xticks(fontsize=18)
plt.title('3RD FEMALE SURVIVER\nvs. #Relationships', fontsize=20, pad=15, weight='bold')

ax.add_patch(Rectangle((0.05, 1.05), 0.9, 1.9, fill=False, edgecolor='red', lw=8))
ax.add_patch(Rectangle((1.05, 0.05), 1.9, 0.9, fill=False, edgecolor='red', lw=8))

# space between subplots
# plt.subplots_adjust(hspace = 0.01) 
plt.subplots_adjust(wspace=0.1)

# set the DPI to 300
plt.figure(dpi=300)

plt.show()








sys.exit()

## Calculate covariance
X = np.stack((
    
    tbl['Age_gauss'],
    tbl['Age_gauss'],
    
    ), axis=0)

print(X)

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

print(X_normalized)

print(np.cov(X)[0][1])


sys.exit()




c = ['Survived', 'Title_Mr', 'Title_Master', 'Title_Mrs', 'Title_Miss', 'Title_Dr']
c = ['Survived', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E', 'Deck_F', 'Deck_G']
c = ['Survived', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

tbl = pd.get_dummies(
    
    df, 
    columns=[
        
        'Pclass',

        ]
    
    )


c = ['Survived', 'Pclass_1st', 'Pclass_2nd', 'Pclass_3rd']

p= tbl[c]

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(p)

# Compute the Pearson correlation matrix
corr = pd.DataFrame(X_normalized).corr()

sns.heatmap(pd.DataFrame(X_normalized, columns=[c]).corr(method = 'spearman'))#, cmap='coolwarm')#'spearman'))

plt.figure(dpi=300)


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