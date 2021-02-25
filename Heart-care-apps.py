import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier

st.write("""
# Heart sick Prediction App

This app predicts the **some heart sick type** species!

""")

st.sidebar.header('User Input Features')


st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/RondinellyMorais/heart_health/main/df_exemple.csv)
""")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
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
    input_df = user_input_features()

hert = pd.read_csv('heart.csv')
herts = hert.drop(columns=['cardio'])
df = pd.concat([input_df, herts],axis=0)

df = df[:1]

st.subheader('User Input parameters')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('heart_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
heart_cardio = np.array([0, 1])
st.write(heart_cardio[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
