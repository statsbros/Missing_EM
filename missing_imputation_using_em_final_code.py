#Importing necessary packages
import random
import numpy as np
import pandas as pd
#For data exploration
import missingno as msno #A simple library to view completeness of data
import matplotlib.pyplot as plt
from numpy import random
# %matplotlib inline

#creating synthetic data for EM example
import sklearn.datasets as dt
seed = 11
rand_state = 11
noise = 0.1
df_x, df_y = dt.make_regression(n_samples=100000,
                             n_features=1,
                             noise=noise,
                             random_state=rand_state) 

df = pd.DataFrame(df_x)
df.columns=['age1'] 
df['age1'] = abs(df['age1']*1000)
df = df[(df['age1']>=20) & (df['age1']<=90)]
df.head()

print(df['age1'].max(),df['age1'].min(),df['age1'].mean())


# Changing data type for all required variables
df['age1'] = df['age1'].astype(float)

#Creating copy of the actual df
act_df = df.copy()
df_theme_missing = df.copy()
df_rand_missing = df.copy()

# creating missing data (underlying theme: age between 25 to 35 will be missing, which is typically unknown to a data scientist)
import numpy as np
df_theme_missing['age'] = np.where((df_theme_missing['age1'] >= 25) & (df_theme_missing['age1'] <=35), np.nan, df_theme_missing['age1'])
df_theme_missing = df_theme_missing.drop(['age1'],axis=1)
df_theme_missing[df_theme_missing['age'].isnull()]

print("Percentage of missing values:",round(len(df_theme_missing[df_theme_missing['age'].isnull()])/df_theme_missing['age'].count()*100,2))

# creating similar number of missing values in the other data, randomly without any theme
#Number of Observations
obs = df_rand_missing.shape[0]
#Simulation
remove_n = 827
miss_indices = np.random.choice(df_rand_missing.index, remove_n, replace=False)
df_rand_missing.iloc[miss_indices]

df_rand_missing['flag'] = 0

for i in miss_indices:
  df_rand_missing['flag'].iloc[i] = 1

print(df_rand_missing['flag'].sum())

df_rand_missing['age'] = np.where(df_rand_missing['flag'] == 1,np.nan,df_rand_missing['age1'])
df_rand_missing = df_rand_missing.drop(['age1','flag'],axis=1)
print(df_rand_missing)

print("Percentage of missing values:",round(len(df_rand_missing[df_rand_missing['age'].isnull()])/df_rand_missing['age'].count()*100,2))

len(df_rand_missing[df_rand_missing['age'].isnull()])


msno.matrix(df_theme_missing)

msno.matrix(df_rand_missing)

#The following code will help us to impute the missing values using EM algorithm.

#Isolating the columns for which we want to impute
X_theme = df_theme_missing[['age']]
X_rand = df_rand_missing[['age']]

#Creating another copy for imputation exercise
df_theme_missing1 = df_theme_missing.copy()
df_rand_missing1 = df_rand_missing.copy()

# importing the package
import impyute as impy
# imputing the missing value but ensure that the values are in matrix form
df_theme_missing1[['age']] = impy.em(df_theme_missing1[['age']].values, loops=1000)
df_rand_missing1[['age']] = impy.em(df_rand_missing1[['age']].values, loops=1000)

#Now, let's have a look at the age variable

print(df_theme_missing1[df_theme_missing1.age<0]['age'])
print('\n')
print("==================================")
print('\n')
print(df_rand_missing1[df_rand_missing1.age<0]['age'])

df_theme_missing2 = df_theme_missing.copy()
df_rand_missing2 = df_rand_missing.copy()

# imputing the missing value but ensure that the values are in matrix form
# np.seterr(divide = 'ignore') 
df_theme_missing2[['age']] = np.exp(impy.em(np.log(df_theme_missing2[['age']].values), loops=1000))
df_rand_missing2[['age']] = np.exp(impy.em(np.log(df_rand_missing2[['age']].values), loops=1000))

#Simulate New Comparison Container
comparison_df = pd.concat([act_df[['age1']], X_theme,X_rand], axis=1) 
#Rename so We can Compare Across Datasets
comparison_df.columns = ["age_orig",  "age_MNAR", "age_MCAR"]
cols = comparison_df.columns.to_list()
#Creating final comparison dataset
comparison_df = pd.concat([comparison_df, df_theme_missing1[['age']],df_rand_missing1[['age']]], axis=1)
comparison_df.columns =  [*cols,'age_MNAR_EM_imp', 'age_MCAR_EM_imp']

# Joining transformed and imputed variables into our previous comparison dataset
cols = comparison_df.columns.to_list()
comparison_df_2 = pd.concat([comparison_df, df_theme_missing2[['age']],df_rand_missing2[['age']]], axis=1)
comparison_df_2.columns =  [*cols,'trans_age_MNAR_EM_imp','trans_age_MCAR_EM_imp' ]
comparison_df_2.head()

#Looking at the previously negatively imputed values

comparison_df_2[(comparison_df_2.age_MNAR_EM_imp<0) | (comparison_df_2.age_MCAR_EM_imp<0)]

#Creating mean and median imputation
#Import the imputer
from sklearn.impute import SimpleImputer
#Initiate the imputer object
imp1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp2 = SimpleImputer(missing_values=np.nan, strategy='median')

imp3 = SimpleImputer(missing_values=np.nan, strategy='mean')
imp4 = SimpleImputer(missing_values=np.nan, strategy='median')

#Isolate the columns where we want to import
df_theme_missing3 = df_theme_missing.copy()
df_rand_missing3 = df_rand_missing.copy()
y_theme = df_theme_missing3[['age']]
y_rand = df_rand_missing3[['age']]
#Fit to learn the mean
imp1.fit(y_theme)
imp2.fit(y_theme)

imp3.fit(y_rand)
imp4.fit(y_rand)
#Impute
imput1 = imp1.transform(y_theme)
imput2 = imp2.transform(y_theme)

imput3 = imp3.transform(y_rand)
imput4 = imp4.transform(y_rand)


# Joining transformed and imputed variables into our previous comparison dataset
cols = comparison_df_2.columns.to_list()
comparison_df_3 = pd.concat([comparison_df_2, pd.DataFrame(imput1, columns=['age_MNAR_mean_imp']),pd.DataFrame(imput2, columns=['age_MNAR_med_imp']),
                             pd.DataFrame(imput3, columns=['age_MCAR_mean_imp']),pd.DataFrame(imput4, columns=['age_MCAR_med_imp'])
                             ],axis = 1)
# comparison_df_3.columns =  [*cols,'age_mean_imp', 'fnlwgt_mean_imp']
comparison_df_3.head()

#Mean Comparisons
print(comparison_df_3[['age_orig', 'age_MNAR', 'age_MNAR_EM_imp','trans_age_MNAR_EM_imp','age_MNAR_mean_imp','age_MNAR_med_imp',
                       'age_MCAR',  'age_MCAR_EM_imp', 'trans_age_MCAR_EM_imp','age_MCAR_mean_imp','age_MCAR_med_imp']].mean())


#STD Comparisons
np.sqrt(comparison_df_3[['age_orig', 'age_MNAR', 'age_MNAR_EM_imp','trans_age_MNAR_EM_imp','age_MNAR_mean_imp','age_MNAR_med_imp',
                       'age_MCAR',  'age_MCAR_EM_imp', 'trans_age_MCAR_EM_imp','age_MCAR_mean_imp','age_MCAR_med_imp']].var())

import numpy as np
print("Comparison of imputation methods for missing not at random data")
print("====================================================================")
print("MAPE between transformed EM imputed and original data is", round(np.mean(abs((comparison_df_3['trans_age_MNAR_EM_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
print("MAPE between mean imputed and original data is", round(np.mean(abs((comparison_df_3['age_MNAR_mean_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
print("MAPE between median imputed and original data is", round(np.mean(abs((comparison_df_3['age_MNAR_med_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
print("\n")
print("Comparison of imputation methods for missing completely at random data")
print("====================================================================")
print("MAPE between transformed EM imputed and original data is", round(np.mean(abs((comparison_df_3['trans_age_MCAR_EM_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
print("MAPE between mean imputed and original data is", round(np.mean(abs((comparison_df_3['age_MCAR_mean_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
print("MAPE between median imputed and original data is", round(np.mean(abs((comparison_df_3['age_MCAR_med_imp'] - comparison_df_3['age_orig'])/comparison_df_3['age_orig']))*100,2),"%")
