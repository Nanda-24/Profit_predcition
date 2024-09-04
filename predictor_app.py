
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures as pf
import scores
  
model_multi = pickle.load(open('model_multi.pkl','rb'))
model_poly = pickle.load(open('model_poly.pkl','rb'))
model_svr = pickle.load(open('model_svr.pkl','rb'))
model_tree = pickle.load(open('model_tree.pkl','rb'))
model_forest = pickle.load(open('model_forest.pkl','rb'))

preg = pf(degree=3)

def predict_sal_poly(r_d,admin,market):
    input = np.array([[r_d,admin,market]]).astype(np.float64)
    input1 = preg.fit_transform(input)
    prediction = model_poly.predict(input1).round(3)
    return (prediction.item())

def predict_sal_svr(r_d,admin,market):
    input = [[r_d,admin,market]]
    input1 = np.array(input).astype(np.float64)
    prediction = model_svr.predict(input1).round(3)
    return (prediction.item())

def predict_sal_multi(r_d,admin,market): 
    input = [[r_d,admin,market]]
    input = np.array(input).astype(np.float64)
    prediction = model_multi.predict(input).round(3)
    return (prediction.item())

def predict_sal_dtr(r_d,admin,market):
    input = [[r_d,admin,market]]
    input = np.array(input).astype(np.float64)
    prediction = model_tree.predict(input).round(3)
    return (prediction.item())

def predict_sal_rfr(r_d,admin,market):
    input = [[r_d,admin,market]]
    input = np.array(input).astype(np.float64)
    prediction = model_forest.predict(input).round(3)
    return (prediction.item())
 
def metrics(regessor,parameters):
    col1,col2,col3 = st.columns(3)
    
    if regessor == "Polynomial Regression(Recommended)":
      if "R Squared" in parameters:
         col1.metric("R2 score:",scores.score_poly) # type: ignore
      if "Mean Absolute Error" in parameters:
         col2.metric("MAE Score:",scores.score_poly2) # type: ignore
      if "Mean Squared Error" in parameters:
         mse = np.sqrt(scores.score_poly3).round(5)
         col3.metric("MSE Score:",mse)  # type: ignore
 
    elif regessor == "Multiple Linear Regression":
      if "R Squared" in parameters:
         col1.metric("R2 score:",scores.score_multi) # type: ignore
      if "Mean Absolute Error" in parameters:
         col2.metric("MAE Score:",scores.score_multi2) # type: ignore
      if "Mean Squared Error" in parameters:
         mse = np.sqrt(scores.score_multi3).round(5)
         col3.metric("MSE Scorre:",mse) # type: ignore

    elif regessor == "Support Vector Regression":
       if "R Squared" in parameters:
         col1.metric("R2 score:",scores.score_svr) # type: ignore
       if "Mean Absolute Error" in parameters:
         col2.metric("MAE Score:",scores.score_svr2) # type: ignore
       if "Mean Squared Error" in parameters:
         mse = np.sqrt(scores.score_svr3).round(5)
         col3.metric("MSE Score:",mse)   # type: ignore

    elif regessor == "Decision Tree Regression":
       if "R Squared" in parameters:
         col1.metric("R2 score:",scores.score_tree) # type: ignore
       if "Mean Absolute Error" in parameters:
         col2.metric("MAE Score:",scores.score_tree2) # type: ignore
       if "Mean Squared Error" in parameters:
         mse = np.sqrt(scores.score_tree3).round(4)
         col3.metric("MSE Score:",mse)   # type: ignore 

    elif regessor == "Random Forest Regression":
       if "R Squared" in parameters:
         col1.metric("R2 score:",scores.score_forest) # type: ignore
       if "Mean Absolute Error" in parameters:
         col2.metric("MAE Score:",scores.score_forest2) # type: ignore
       if "Mean Squared Error" in parameters:
         mse = np.sqrt(scores.score_forest3).round(4)
         col3.metric("MSE Score:",mse)  # type: ignore         
      
def main():
    html_temp = """
        <div style="background-color:#025246 ;padding:2px">
        <h1 style="color:white;text-align:center;">Profit Predictor</h1>"""
    st.markdown(html_temp, unsafe_allow_html=True)

    html_temp2 = """
        <div style="background-color:#025246 ;padding:0px">
       <h2 style = "color:white;text-align:center;">Model Selection</h2></div>"""
    
    st.sidebar.markdown(html_temp2,unsafe_allow_html=True)

    regressor = st.sidebar.selectbox("Regressors",("--","Random Forest Regression(Recommended)","Polynomial Regression","Decision Tree Regression","Support Vector Regression","Multiple Linear Regression") )

    html_temp3 = """ <div style="background-color:#025246 ;padding:2px">
       <h2 style = "color:white;text-align:center;">Regression Parameters</h2></div>"""
    
    st.sidebar.markdown(html_temp3,unsafe_allow_html=True)

    parameters = st.sidebar.multiselect("Paramters:",("R Squared","Mean Absolute Error","Mean Squared Error"))

    if(regressor == "--"):
       st.header("Select a regressor!!")

    if(regressor != "--"):
        o1 = st.text_input("R&D Spend","00")
        o2 = st.text_input("Adminsitration Spend","00")
        o3 = st.text_input("Marketing Spend","00")

    if(regressor == "Multiple Linear Regression"):
        if st.button("Predict"):
         output=(predict_sal_multi(o1,o2,o3)) # type: ignore
         st.header(f'Profit: {output}')
         metrics("Multiple Linear Regression",parameters)


    elif(regressor == "Polynomial Regression"):
        if st.button("Predict"):
         output=float(predict_sal_poly(o1,o2,o3)) # type: ignore
         st.header(f'Profit: {output}')
         metrics("Polynomial Regression(Recommended)",parameters)  

    elif(regressor == "Support Vector Regression"):
        if st.button("Predict"):
         output=float(predict_sal_svr(o1,o2,o3)) # type: ignore
         st.header(f'Profit: {output}')
         metrics("Support Vector Regression",parameters)

    elif(regressor == "Decision Tree Regression"):
       if st.button("Predict"):
         output=float(predict_sal_dtr(o1,o2,o3)) # type: ignore
         st.header(f'Profit: {output}')
         metrics("Decision Tree Regression",parameters)

    elif(regressor == "Random Forest Regression(Recommended)"):
       if st.button("Predict"):
         output=float(predict_sal_rfr(o1,o2,o3)) # type: ignore
         st.header(f'Profit: {output}')
         metrics("Random Forest Regression",parameters)              
            
   
st.set_page_config(page_title="Profit Predictor")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

if __name__=='__main__':
    main() 