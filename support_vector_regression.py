
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pickle

data = pd.read_csv("50_Startups.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2,random_state=1)

reg = SVR(kernel = 'linear')
reg.fit(x,y) # type: ignore

y_pred =reg.predict(x_test)
sv_score = r2_score(y_test,y_pred)
score2 = mean_absolute_error(y_test,y_pred)
score3 = mean_squared_error(y_test,y_pred)


pickle.dump(reg,open('model_svr.pkl','wb'))
pickle.load(open('model_svr.pkl','rb'))