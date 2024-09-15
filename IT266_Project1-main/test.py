from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app=Flask(__name__)

# Initialize model as None
model = None

# @app.route('/train', methods=['POST'])
# def train_model():
#     global model
    
#     df = pd.read_csv("rainfall in india 1901-2022.csv")
#     df.head()

#     df.info()
#     df.describe()
#     df.isnull().sum()
#     df.duplicated().sum()
#     df['SUBDIVISION'].value_counts()
#     df.mean(numeric_only=True)

#     # filling na values with mean
#     df = df.fillna(df.mean(numeric_only=True))
#     df.isnull().any()
#     df.YEAR.unique()
#     df.YEAR.unique()

#     V = df.loc[((df['SUBDIVISION'] == 'Gujarat Region'))]


#     df["SUBDIVISION"].nunique()
#     group = df.groupby('SUBDIVISION')[['YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]
#     df=group.get_group(('Gujarat Region'))


#     df2=df.melt(['YEAR']).reset_index()

#     df2= df2[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])

#     df2.YEAR.unique()
#     df2.columns=['Index','Year','Month','Avg_Rainfall']

#     Month_map={'JAN':1,'FEB':2,'MAR' :3,'APR':4,'MAY':5,'JUN':6,'JUL':7,'AUG':8,'SEP':9,
#     'OCT':10,'NOV':11,'DEC':12}
#     df2['Month']=df2['Month'].map(Month_map)


#     df2.drop(columns="Index",inplace=True)


#     X=np.asanyarray(df2[['Year','Month']]).astype('int')
#     y=np.asanyarray(df2['Avg_Rainfall']).astype('int')

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

#     X_train
#     y_train

#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
#                         min_samples_split=10, n_estimators=800)
#     model.fit(X_train, y_train)
   

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    df = pd.read_csv("rainfall in india 1901-2022.csv")
    df.head()

    # df.info()
    # df.describe()
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
    model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                        min_samples_split=10, n_estimators=800)
    model.fit(X_train, y_train)
    
    if model is None:
        return jsonify({'error': 'Model not trained yet'})

    # Get data from the request
    data = request.json
    
    # Extract feature values from the request data
    features = data.get('features')  # Assuming the JSON object has a 'features' key
    
    if features is None:
        return jsonify({'error': 'Features not provided'})
    
    # Convert features to numpy array for prediction
    X = np.array(features).reshape(1, -1)  # Convert features to 2D array
    
    # Make predictions
    prediction = model.predict(X)
    print("Successfull")
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
