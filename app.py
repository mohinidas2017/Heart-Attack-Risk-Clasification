import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
model=pickle.load(open('model.pkl','rb'))
#@app.route('/')
def welcome():
    return "Welcome All"
#@app.route('/predict',methods=["Get"])
def predict_heart_attack(age,sex,cp,restecg,thalach,exang,oldpeak,slope,ca,thal):
    input=np.array([[age,sex,cp,restecg,thalach,exang,oldpeak,slope,ca,thal]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][1], 2)
    return float(pred)

def main():
    st.title("Heart Attack Predictor App")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Heart Attack Prediction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.text_input("Age","Type Here")
    sex = st.text_input("Sex","Type Here")
    cp = st.text_input("chest pain type","Type Here")
    restecg = st.text_input("Resting blood pressure","Type Here")
    thalach = st.text_input("Maximum heart rate achieved","Type Here")
    exang = st.text_input("Exercise induced angina","Type Here")
    oldpeak = st.text_input("oldpeak = ST depression induced by exercise relative to rest","Type Here")
    slope = st.text_input("The slope of the peak exercise ST segment","Type Here")
    ca = st.text_input("Number of major vessels (0-3) colored by flourosopy","Type Here")
    thal = st.text_input("Thal: 1 = normal; 2 = fixed defect; 3 = reversable defect","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> You are Safe!!!</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> You May have Heart Attack, Consult Doctor</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_heart_attack(age,sex,cp,restecg,thalach,exang,oldpeak,slope,ca,thal)
        st.success('The probability of getting a heart attack is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()