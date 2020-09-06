# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 17:28:29 2020

@author: hemahemu
"""


import numpy as np
import pandas as pd
df=pd.DataFrame([-5,10,15])
print(df)
from sklearn.preprocessing import MinMaxScaler
# sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(df))
print(scaler.transform(df))
