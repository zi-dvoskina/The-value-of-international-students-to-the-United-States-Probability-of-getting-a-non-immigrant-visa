# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

# data import and cleaning
df_= pd.read_csv("/Users/zinaidadvoskina/Documents/NORTHEASTERN UNIVERSITY/MSBA/MISM 6212/International Students/2020_new.csv", encoding='cp1252')

df = df_.dropna( how='any',
                    subset=['PW_UNIT_OF_PAY'])

# removes ~ 2000 lines

# convert units to numeric
df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].replace(to_replace = "Year", value = 1)
df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].replace(to_replace = "Month", value = 12)
df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].replace(to_replace = "Bi-Weekly", value = 12)
df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].replace(to_replace = "Week", value = 51)
df['PW_UNIT_OF_PAY'] = df['PW_UNIT_OF_PAY'].replace(to_replace = "Hour", value = 2087)

df_.info()
# must ensure int
df['PW'] = df['PW_UNIT_OF_PAY'] * df['PREVAILING_WAGE']
# remove extra columns
df = df.drop(['PW_UNIT_OF_PAY','PREVAILING_WAGE','CASE_NUMBER','YEAR','RECEIVED_DATE','DECISION_DATE','APPX_A_NAME_OF_INSTITUTION','APPX_A_FIELD_OF_STUDY','APPX_A_DATE_OF_DEGREE'], axis = 1)
# Much easier to deal with data set

# review unique values for status
df.CASE_STATUS.unique()
# ['Certified', 'Certified - Withdrawn', 'Denied', 'Withdrawn']
# have to alter Cert - withdrawn to cert, remove withdrawn

df.CASE_STATUS[df['CASE_STATUS']=='Certified - Withdrawn'] = 'Certified'
df.CASE_STATUS.unique()
# ['Certified', 'Denied', 'Withdrawn']
# remove withdrawn
df = df.drop(df[df.CASE_STATUS == 'Withdrawn'].index)
df.CASE_STATUS.unique()
# down to two target variables['Certified', 'Denied']

df.describe()
print(df['CASE_STATUS'].value_counts())
# Certified    560725
# Denied         3983

deny_pct = 3983/(3983+560725)
print(100 * deny_pct)
# 0.7053202717156477

count_nan = len(df) - df.count()
print(count_nan)
'''
CASE_STATUS                        0
JOB_TITLE                          1
SOC_TITLE                          0
FULL_TIME_POSITION                 0
EMPLOYER_NAME                      4
AGENT_REPRESENTING_EMPLOYER        0
WORKSITE_CITY                     11
WORKSITE_STATE                     0
WILLFUL_VIOLATOR               12468
PW                                 0
'''
# most nan values within willfull violator
print(df['WILLFUL_VIOLATOR'].value_counts())
# N    552019
# Y       221
# mostly N, replace with most used (N)
# apply same logic to job title, worksite city, empoloyer name
df['WILLFUL_VIOLATOR'] = df['WILLFUL_VIOLATOR'].fillna(df['WILLFUL_VIOLATOR'].mode()[0])
df['WORKSITE_CITY'] = df['WORKSITE_CITY'].fillna(df['WORKSITE_CITY'].mode()[0])
df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].fillna(df['EMPLOYER_NAME'].mode()[0])
df['JOB_TITLE'] = df['JOB_TITLE'].fillna(df['JOB_TITLE'].mode()[0])

# one way to check
count_nan = len(df) - df.count()
print(count_nan)

# second way to check
# ensure nulls are gone
assert pd.notnull(df['WILLFUL_VIOLATOR']).all().all()
assert pd.notnull(df['WORKSITE_CITY']).all().all()
assert pd.notnull(df['EMPLOYER_NAME']).all().all()
assert pd.notnull(df['JOB_TITLE']).all().all()

# review PW
np.nanpercentile(df.PW,98)
# 98th percentil is 170019.0
df.PW.median()
# 90376.0
df.PW.mean()
# 95311.50252519887
# may need to cap 

foo1 = df['FULL_TIME_POSITION']=='Y'
foo2 = df['CASE_STATUS']=='Certified'
print(len(df[foo1])/len(df))
print(len(df[foo2])/len(df))
# 98% of jobs are full time
# 99% of cases are certifed

df['NEW_EMPLOYER'] = np.nan
df.shape

df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].str.lower()
df.NEW_EMPLOYER[df['EMPLOYER_NAME'].str.contains('university')] = 'university'
df['NEW_EMPLOYER']= df.NEW_EMPLOYER.replace(np.nan, 'non university', regex=True)

df['OCCUPATION'] = np.nan
df['SOC_TITLE'] = df['SOC_TITLE'].str.lower()
df.OCCUPATION[df['SOC_TITLE'].str.contains('computer','programmer')] = 'computer occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('software','web developer')] = 'computer occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('database')] = 'computer occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('math','statistic')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('predictive model','stats')] = 'Mathematical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('teacher','linguist')] = 'Education Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('professor','Teach')] = 'Education Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('school principal')] = 'Education Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('medical','doctor')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('physician','dentist')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('Health','Physical Therapists')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('surgeon','nurse')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('psychiatr')] = 'Medical Occupations'
df.OCCUPATION[df['SOC_TITLE'].str.contains('chemist','physicist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_TITLE'].str.contains('biology','scientist')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_TITLE'].str.contains('biologi','clinical research')] = 'Advance Sciences'
df.OCCUPATION[df['SOC_TITLE'].str.contains('public relation','manage')] = 'Management Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('management','operation')] = 'Management Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('chief','plan')] = 'Management Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('executive')] = 'Management Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('advertis','marketing')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('promotion','market research')] = 'Marketing Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('business','business analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('business systems analyst')] = 'Business Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('accountant','finance')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('financial')] = 'Financial Occupation'
df.OCCUPATION[df['SOC_TITLE'].str.contains('engineer','architect')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_TITLE'].str.contains('surveyor','carto')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_TITLE'].str.contains('technician','drafter')] = 'Architecture & Engineering'
df.OCCUPATION[df['SOC_TITLE'].str.contains('information security','information tech')] = 'Architecture & Engineering'
df['OCCUPATION']= df.OCCUPATION.replace(np.nan, 'Others', regex=True)

# convert to binary
class_mapping = {'Certified':0, 'Denied':1}
df["CASE_STATUS"] = df["CASE_STATUS"].map(class_mapping)

test1 = pd.Series(df['JOB_TITLE'].ravel()).unique()
print(pd.DataFrame(test1))

df = df.drop('EMPLOYER_NAME', axis = 1)
df = df.drop('SOC_TITLE', axis = 1)
df = df.drop('JOB_TITLE', axis = 1)
df = df.drop('WORKSITE_CITY', axis = 1)
df = df.drop('WORKSITE_STATE', axis = 1)
df = df.drop('AGENT_REPRESENTING_EMPLOYER', axis = 1)
df = df.drop('WILLFUL_VIOLATOR', axis = 1)
df = df.drop('OCCUPATION', axis = 1)

df1 = df.copy()
df1[['CASE_STATUS', 'FULL_TIME_POSITION','NEW_EMPLOYER','PW']] = df1[['CASE_STATUS', 'FULL_TIME_POSITION','NEW_EMPLOYER','PW']].apply(lambda x: x.astype('category'))

df1.info()
### modelling

x = df1.drop('CASE_STATUS', axis = 1)
y = df1['CASE_STATUS']

# split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 20)

# get dummies for categorical, takes a while, need smaller subsets
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

# summarize class distribution
print("Before undersampling: ", Counter(y_train))
#Before undersampling:  Counter({0: 392514, 1: 2781})

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
x_train_under, y_train_under = undersample.fit_resample(x_train, y_train)
x_test_under, y_test_under = undersample.fit_resample(x_test, y_test)
# summarize class distribution
print("After undersampling: ", Counter(y_train_under))
# After undersampling:  Counter({0: 2781, 1: 2781})

########### SVC MODEL
from sklearn.svm import SVC
model = SVC()
model.fit(x_train_under, y_train_under)

# predictions
y_pred = model.predict(x_test_under)

### evaluate
from sklearn.metrics import f1_score, recall_score
f1_score(y_test_under, y_pred)
recall_score(y_test_under, y_pred)


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
c_mat = pd.DataFrame(confusion_matrix(y_test_under, y_pred, labels = [0,1]), index = ["Actual:0", "Actual:1"],
                     columns = ["Pred:0","Pred:1"])

print(c_mat)
print('Accuracy is', accuracy_score(y_test_under, y_pred))
print('Recall is', recall_score(y_test_under, y_pred))
print('Precision is', precision_score(y_test_under, y_pred))
'''
          Pred:0  Pred:1
Actual:0     753     449
Actual:1     349     853
Accuracy is 0.668053244592346
Recall is 0.7096505823627288
Precision is 0.6551459293394777
'''

# grid search to find best C, gamma and kernel
param_grid = {'C':[1,10,100], 'gamma':[1,0.1,0.01], 'kernel':['rbf','linear']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid, verbose= 3, scoring = 'f1')

grid.fit(x_train_under, y_train_under)

grid.best_params_

# make predictions

y_pred = model.predict(x_test_under)

# evaluate

from sklearn.metrics import f1_score, recall_score
f1_score(y_test, y_pred)
recall_score(y_test, y_pred)

########### Regression model

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression(solver="liblinear")
logmodel.fit(x_train_under,y_train_under)

y_pred=logmodel.predict(x_test_under)

y_prob=logmodel.predict_proba(x_test_under)

from sklearn.metrics import confusion_matrix, f1_score,recall_score,precision_score

C_mat=pd.DataFrame(confusion_matrix(y_test_under, y_pred, labels=[0,1]), index=["Actual:0","Actual:1"], columns=["Pred:0", "Pred:1"])

print (C_mat)
print("F-score is", f1_score(y_test_under,y_pred))
print ("Recall is", recall_score(y_test_under, y_pred))
print ("Precision is", precision_score(y_test_under, y_pred))

"""          Pred:0  Pred:1
Actual:0     591     611
Actual:1     294     908    

Accuracy is 0.668053244592346
Recall is 0.7554076539101497
Precision is 0.597761685319289
"""


###### KNN model

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train_under)
x_train_scaled = scaler.transform(x_train_under)
x_test_scaled = scaler.transform(x_test_under)


knn = KNeighborsClassifier()
knn.fit(x_train_scaled, y_train_under)

# predictions

y_pred = knn.predict(x_test_scaled)
#f1
print(f1_score(y_test_under, y_pred))

# 0.6367472190257001

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test_under, y_pred))
print(classification_report(y_test_under, y_pred))

"""[[627 575]
 [372 830]]
              precision    recall  f1-score   support

           0       0.63      0.52      0.57      1202
           1       0.59      0.69      0.64      1202

    accuracy                           0.61      2404
   macro avg       0.61      0.61      0.60      2404
weighted avg       0.61      0.61      0.60      2404"""