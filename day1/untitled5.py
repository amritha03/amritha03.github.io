import numpy as np
import pandas as pd
data = pd.read_csv('50_Startups.csv')
data.columns
x = data.iloc[:,:4].values
y = data.iloc[:,4].values
#x1=data.iloc[:,3:4].values


from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()

x[:,3] = lEncoder.fit_transform(x[:,3])
#lEncoder.fit(x1)
#lEncoder.transform(x1)

from sklearn.preprocessing import OneHotEncoder
ohEncoder = OneHotEncoder(categorical_features=[3])
x = ohEncoder.fit_transform(x).toarray()

x = x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
mRegressor= LinearRegression()

mRegressor.fit(x_train,y_train)
y_pred = mRegressor.predict(x_test)

score = mRegressor.score(x_test,y_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_pred)**(1/2)






