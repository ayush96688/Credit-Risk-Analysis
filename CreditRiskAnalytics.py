# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:35:01 2018

@author: hp
"""
#%%
import pandas as pd
import numpy as np
train_data=pd.read_csv(r'C:\Users\hp\Desktop\DSP\24nov18\risk_analytics_train.csv',
               header=0)
test_data=pd.read_csv(r'C:\Users\hp\Desktop\DSP\24nov18\risk_analytics_test.csv',
               header=0)
#%%
print(train_data.shape)
train_data.head()
#%%
print(train_data.isnull().sum())
#%%
#Replacing NA with mode
colname1=['Gender','Married','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)#inplace true will make changes permanent
#%%
#Replacing NA with mean
train_data['LoanAmount'].fillna(train_data['LoanAmount'].mean(),inplace=True)
#%%
#Replacing NA with manual input
train_data['Credit_History'].fillna(value=0,inplace=True)
#%%
#converting categorial to numerical
from sklearn import preprocessing
colname=['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    train_data[x]=le[x].fit_transform(train_data[x])
#%%
train_data.head()
#loan status as Y-->1 and N-->0
#%%
print(test_data.shape)
test_data.head()
#%%
print(test_data.isnull().sum())

#%%
colname1=['Gender','Dependents','Self_Employed','Loan_Amount_Term']
for x in colname1:
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)#inplace true will make changes permanent
#%%
test_data['Credit_History'].fillna(value=0,inplace=True)
#%%
test_data['LoanAmount'].fillna(test_data['LoanAmount'].mean(),inplace=True)
#%%
#converting categorial to numerical
from sklearn import preprocessing
colname=['Gender','Married','Education','Self_Employed','Property_Area']
le={}

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    test_data[x]=le[x].fit_transform(test_data[x])
#%%
print(test_data.head())
#%%
X_train=train_data.values[:,1:-1]
Y_train=train_data.values[:,-1]
Y_train=Y_train.astype(int)
#%%
X_test=test_data.values[:,1:]
#%%
#SCALING 
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#%%
#MODEL BUILDING
from sklearn import svm
svc_model=svm.SVC(kernel='rbf',C=1.0,gamma=0.1)
svc_model.fit(X_train,Y_train)
Y_pred=svc_model.predict(X_test)
print(list(Y_pred))
#%%
#creating column
Y_pred_col=list(Y_pred)
#%%
test_data=pd.read_csv(r'C:\Users\hp\Desktop\DSP\24nov18\risk_analytics_test.csv',
               header=0)
test_data['Y_predictions']=Y_pred_col
test_data.head()
#%%
test_data.to_csv('test_data.csv')
#%%
#CROSS VALIDATION
#no y test
#tweaking values of C,gamma
classifier=svm.SVC(kernel='rbf',C=100,gamma=0.001)
#classifier=LogisticRegression()
from sklearn import cross_validation
#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

#for going ahead with the cross validation model
for train_value, test_value in kfold_cv: 
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])


Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))