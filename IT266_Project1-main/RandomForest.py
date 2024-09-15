import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
import tensorflow as tf

df = pd.read_csv("rainfall in india 1901-2022.csv")
df.head()

df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df['SUBDIVISION'].value_counts()
df.mean(numeric_only=True)

# filling na values with mean
df = df.fillna(df.mean(numeric_only=True))
df.isnull().any()
df.YEAR.unique()
df.YEAR.unique()

V = df.loc[((df['SUBDIVISION'] == 'Gujarat Region'))]


df["SUBDIVISION"].nunique()
group = df.groupby('SUBDIVISION')[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
df=group.get_group(('Gujarat Region'))


df2=df.melt(['YEAR']).reset_index()

df2= df2[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])

df2.YEAR.unique()
df2.columns=['Index','Year','Month','Avg_Rainfall']

Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df2['Month']=df2['Month'].map(Month_map)


df2.drop(columns="Index",inplace=True)


X=np.asanyarray(df2[['Year','Month']]).astype('int')
y=np.asanyarray(df2['Avg_Rainfall']).astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

X_train
y_train

from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)

