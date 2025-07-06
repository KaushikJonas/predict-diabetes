import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Progression Prediction App")

# Input fields for all features in load_diabetes
features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
input_data = []

for feature in features:
    val = st.number_input(f"Input for {feature}", step=0.01)
    input_data.append(val)

if st.button("Predict Disease Progression"):
    data = np.array([input_data])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    st.success(f"Predicted Disease Progression: {prediction[0]:.2f}")