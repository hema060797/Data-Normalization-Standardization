import pandas as pd
df=pd.read_csv('https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',header=None,usecols=[0,1,2])
print(df)
df.columns=["Class","Alcohol","Malic"]
print(df.head())

print(df['Alcohol'].max())
print(df['Alcohol'].min())
print(df['Alcohol'].mean())

from sklearn.preprocessing import MinMaxScaler
print('USING MINMAX SCALAR')
scaling=MinMaxScaler()
df1=scaling.fit_transform(df[['Alcohol','Malic']])
print(df1)


print('Z-score normalization')
from sklearn.preprocessing import StandardScaler
scalingg=StandardScaler()
df2=scalingg.fit_transform(df[['Alcohol','Malic']])
print(df2)