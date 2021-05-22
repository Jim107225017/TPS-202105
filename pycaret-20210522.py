# -*- coding: utf-8 -*-
"""
Created on Sat May 22 12:40:28 2021

@author: Chen
"""

import os
import numpy as np
import pandas as pd

import catboost
import lightgbm
import xgboost
from sklearn.preprocessing import LabelEncoder
from joblib import load, dump

os.chdir(r'C:\Users\Chen\Desktop\Kaggle\Classifier\Tabular Playground Series - May 2021')

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Ratio of missing values
# No missing
NAratio_in_train = (df_train.isnull().sum() / len(df_train)) * 100
NAratio_in_train = NAratio_in_train.sort_values(ascending=False)
print(NAratio_in_train.head())

# No missing
NAratio_in_test = (df_test.isnull().sum() / len(df_test)) * 100
NAratio_in_test = NAratio_in_test.sort_values(ascending=False)
print(NAratio_in_test.head())

stat_table = df_train.describe()
df_train['target'].value_counts()

# target labelling
f = LabelEncoder()
f.fit(df_train['target'])
y = f.transform(df_train['target'])
y = pd.DataFrame(y).astype('int64')

col = [i for i in df_train.columns if i not in ['id', 'target']]
x = df_train[col]
x = x.astype('int64')






