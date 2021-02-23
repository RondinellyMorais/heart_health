import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Simple Heart Diseases Prediction App

This app predicts the **Heart Diseases** !
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    cholesterol = st.sidebar.selectbox('cholesterol',('1','2','3'))
    gluc = st.sidebar.selectbox('gluc',('1','2','3'))
    gender = st.sidebar.selectbox('gender',('0','1'))
    smoke = st.sidebar.selectbox('smoke',('0','1'))
    alco =  st.sidebar.selectbox('alco',('0','1'))
    active = st.sidebar.selectbox('active',('0','1'))
    age= st.sidebar.slider('age', 10798, 23713, 15000)
    height = st.sidebar.slider('height', 55, 250, 100)
    weight = st.sidebar.slider('weight', 10.0, 200.0, 100.0)
    ap_hi = st.sidebar.slider('ap_hi', 93, 169, 120)
    ap_lo = st.sidebar.slider('ap_lo', 52, 115, 80)
    data = {'cholesterol': cholesterol,
                'gluc': gluc,
                'smoke': smoke,
                'alco': alco,
                'active': active,
                'age': age,
                'height': height,
                'weight': weight,
                'ap_hi': ap_hi,
                'ap_lo': ap_lo,
                'gender': gender}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

heart_mas = pd.read_csv('heart_care.csv')
heart_mas  = heart_mas.drop('Unnamed: 0', axis= 1)
X = heart_mas.drop('cardio', axis=1)
Y = heart_mas['cardio']

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

heart_cardio = np.array([0, 1])

st.subheader('Class labels and their corresponding index number')
st.write(heart_cardio)

st.subheader('Prediction')
st.write(heart_cardio[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)