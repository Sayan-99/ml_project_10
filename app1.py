#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in = open("regressor.pkl","rb")
regressor=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(LotArea,YearBuilt,stFlrSF,ndFlrSF,FullBath,BedroomAbvGr,TotRmsAbvGrd):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=regressor.predict([[LotArea,YearBuilt,stFlrSF,ndFlrSF,FullBath,BedroomAbvGr,TotRmsAbvGrd]])
    print(prediction)
    return prediction



def main():
    st.title("Housing Price Estimator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Housing Price Estimator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    LotArea = st.text_input("LotArea","Type Here")
    YearBuilt = st.text_input("YearBuilt","Type Here")
    stFlrSF = st.text_input("1st Floor Square Feet","Type Here")
    ndFlrSF = st.text_input("2nd Floor Square Feet","Type Here")
    FullBath = st.text_input("Number of FullBath","Type Here")
    BedroomAbvGr = st.text_input("Bedroom Above Grade","Type Here")
    TotRmsAbvGrd = st.text_input("Total rooms","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(LotArea,YearBuilt,stFlrSF,ndFlrSF,FullBath,BedroomAbvGr,TotRmsAbvGrd)
    st.success('The price of your dream house is ${}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()


# In[ ]:





# In[ ]:




