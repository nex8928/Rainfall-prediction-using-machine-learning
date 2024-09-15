import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras import layers, models

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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000000)

# Define the TensorFlow model architecture
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
