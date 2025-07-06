import streamlit as st
import numpy as np
import joblib

# ✅ Load Model & Scaler from Specific Path
model_path = ("C:\\Users\\KAUSHIK\\OneDrive\\Documents\\lr.pkl")
scaler_path = ("C:\\Users\\KAUSHIK\\OneDrive\\Documents\\scaler.pkl")

lr = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("Diabetes Disease Progression Predictor")
st.write("Enter the following patient details:")

# Input Features
features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
inputs = []

for feature in features:
    val = st.number_input(f"{feature}", value=0.0, step=0.01, format="%.2f")
    inputs.append(val)

# Predict Button
if st.button("Predict Disease Progression"):
    data = np.array([inputs])
    scaled_data = scaler.transform(data)
    prediction = lr.predict(scaled_data)
    st.success(f"Predicted Disease Progression Score: {prediction[0]:.2f}")
