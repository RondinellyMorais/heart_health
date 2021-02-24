import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Heart sick Prediction App

This app predicts the **some heart sick type** species!

""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

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

hert = pd.read_csv('heart_care.csv')
X = hert.drop('cardio', axis= 1)
Y = hert['cardio']

clf = RandomForestClassifier()
clf.fit(X, Y)

heart_cardio = np.array([0, 1])

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(heart_cardio)

st.subheader('Prediction')
st.write(heart_cardio[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


