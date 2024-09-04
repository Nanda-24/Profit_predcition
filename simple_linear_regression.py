import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr

data = pd.read_csv('Salary_Data.csv')
#print(data.head(2))
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
#print(x.head(2))
#print(y.head(2))

x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2,random_state =1)
#print(x_train.head(2))
#print(x_test.head(2))
x_train.reshape(-1,1)
y_train.reshape(-1,1)

plt.plot(x_train,y_train,'r.',label ='salary projection')
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.show()

reg = lr()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
plt.scatter(x_test,y_test,color ='red')
#plt.plot(x_train,reg.predict(x_train),'g.-',label='original val')
plt.plot(x_train,reg.predict(x_train),'b-',label = 'predicted val')
plt.title('prediction of  salaries')
plt.legend()
plt.show()

print(y_pred)
print('\n')
print(y_test)