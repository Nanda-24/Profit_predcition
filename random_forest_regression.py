import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as RFR
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

x_train,x_test,y_train,y_test = tts(x,y,test_size = 0.2,random_state= 0)
f_reg = RFR(n_estimators=73,random_state=0)
f_reg.fit(x,y) # type: ignore

y_pred = f_reg.predict(x_test)

score = r2_score(y_test,y_pred)
score2 = mean_absolute_error(y_test,y_pred)
score3 = mean_squared_error(y_test,y_pred)

pickle.dump(f_reg,open('model_forest.pkl','wb'))
pickle.load(open('model_forest.pkl','rb'))