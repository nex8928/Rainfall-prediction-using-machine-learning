import pickle
from sklearn import metrics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics


df = pd.read_csv("rainfall in india 1901-2015.csv")
df = df.fillna(df.mean(numeric_only=True))


df2=df.melt(['YEAR']).reset_index()

df2= df2[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])


df2.columns=['Index','Year','Month','Avg_Rainfall']
print(df2.columns)

Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
   'OCT':10,'NOV':11,'DEC':12}
df2['Month']=df2['Month'].map(Month_map)

df2.drop(columns="Index",inplace=True)

X=np.asanyarray(df2[['Year','Month']]).astype('int')
y=np.asanyarray(df2['Avg_Rainfall']).astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)


from sklearn.ensemble import RandomForestRegressor
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                      min_samples_split=10, n_estimators=800)
random_forest_model.fit(X_train, y_train)


from sklearn import svm
svm_regr = svm.SVC(kernel='rbf')
svm_regr.fit(X_train, y_train)


from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(random_state=0,solver='lbfgs',class_weight='balanced', max_iter=10000)
logreg = LogisticRegression(random_state=0,solver='lbfgs')
logreg.fit(X_train,y_train)


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)


from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0)
gbr.fit(X_train, y_train)




file = open("model.pkl","wb")
pickle.dump(random_forest_model,file)
file.close()