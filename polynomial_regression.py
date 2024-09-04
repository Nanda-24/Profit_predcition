import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures as PF
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

poly_transform = PF(degree = 3)
x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2,random_state=1)

x_poly_transform = poly_transform.fit_transform(x)
l_reg = LR()
l_reg.fit(x_poly_transform,y)  # type: ignore

x_poly_test = poly_transform.fit_transform(x_test)
y_pred = l_reg.predict(x_poly_test)

score = r2_score(y_test,y_pred)
score2 = mean_absolute_error(y_test,y_pred)
score3 = mean_squared_error(y_test,y_pred)

pickle.dump(l_reg,open('model_poly.pkl','wb'))
pickle.load(open('model_poly.pkl','rb'))




