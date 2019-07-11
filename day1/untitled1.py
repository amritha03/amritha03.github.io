import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')
x = data.iloc[:,0:1].values
y = data.iloc[:,1].values

#plt.scatter(x,y,color='red')
#%matplotlib auto

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

regressor.fit(x_train,y_train)
m = regressor.coef_
c = regressor.intercept_

y75 = (m*7.5) +c
yP75 = regressor.predict([[7.5]])

y_pred = regressor.predict(x_test)

plt.scatter(x_train,y_train,color='red')
plt.scatter(x_test,regressor.predict(x_test),color='green')
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='blue')


a=input("enter the exp sep by coma")
a=a.split(',')
result=[]
for x in a:
    result.append(float(x))
result
result=np.array(result)
result=result.reshape((len(result),1))
regressor.predict(result)
#a=np.array(a).reshape((-1,1))


from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
rmse

sample=0
for i in range(0,len(y_test)):
    sample += (y_test[i] -y_pred[i])**2
    
res = sample/len(y_test)
import math
math.sqrt(res)
