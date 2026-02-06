# app.py - Mushroom Classification Streamlit App (Regression Model for Diabetes Prediction)

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open("diabetes_model.pkl", 'rb') as f:  # Load your saved model
        return pickle.load(f)

model = load_model()

# Set page configuration
st.set_page_config(page_title="Diabetes Progression Prediction", page_icon="⚖️")

st.title("⚖️ Diabetes Progression Prediction")

# Input features from the user
st.header("Input Features")

# Input fields for the features
age = st.number_input("Age", min_value=-1.0, max_value=100.0, value=25.0)
sex = st.number_input("Sex (0 = Female, 1 = Male)", min_value=0, max_value=1, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=22.0)
bp = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0)
s1 = st.number_input("S1", min_value=-10.0, max_value=10.0, value=0.0)
s2 = st.number_input("S2", min_value=-10.0, max_value=10.0, value=0.0)
s3 = st.number_input("S3", min_value=-10.0, max_value=10.0, value=0.0)
s4 = st.number_input("S4", min_value=-10.0, max_value=10.0, value=0.0)
s5 = st.number_input("S5", min_value=-10.0, max_value=10.0, value=0.0)
s6 = st.number_input("S6", min_value=-10.0, max_value=10.0, value=0.0)

# Create a DataFrame from the inputs
features = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'bp': [bp],
    's1': [s1],
    's2': [s2],
    's3': [s3],
    's4': [s4],
    's5': [s5],
    's6': [s6]
})

# Make prediction when the button is clicked
if st.button("Predict"):
    prediction = model.predict(features)
    
    st.subheader("Predicted Diabetes Progression:")
    st.write(f"The predicted diabetes progression value is: {prediction[0]:.2f}")

