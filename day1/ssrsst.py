import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('headbrain.csv')
x_train=numpy.array(data.iloc[:100,2])
y_train=numpy.array(data.iloc[:100,3])
x_test=numpy.array(data.iloc[:100,2])
y_test=numpy.array(data.iloc[:100,3])

xm,ym=x_train.mean(),y_train.mean()
b1_num,b1_denom=0,0
for i in range (len(x_train)):
    b1_num += (x_train[i]-xm)*(y_train[i]-ym)
    b1_denom+=(x_train[i]-xm)**2
b1=b1_num/b1_denom
b0=ym-b1*xm

y_pred = b1*x_test + b0

plt.scatter(x_train,y_train,color='red',label='training')
plt.scatter(x_test,y_pred,color='blue',label='predict')
plt.scatter(x_test,y_pred,color='green',label='actual values')
plt.legend()
plt.show()

ssr,sst=0,0
for i in range(len(y_test)):
    sst +=(y_test[i] - y_test.mean())**2
    ssr +=(y_test[i] - y_pred[i])**2
r= 1 - (float(ssr)/float(sst))

print('score of linearmodel:',r)
