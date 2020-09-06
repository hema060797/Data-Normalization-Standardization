# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 21:53:25 2020

@author: hemahemu
"""

import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',header=None,usecols=[0,1,2])
#df=pd.read_csv('https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv',header=None,usecols=[0,1,2])
print(df)
df.columns=["Class","Alcohol","Malic"]
print(df.head())


from sklearn.preprocessing import MinMaxScaler
print('USING MINMAX SCALAR')
scaling=MinMaxScaler()
df1=scaling.fit_transform(df[['Alcohol','Malic']])
print(df1)
