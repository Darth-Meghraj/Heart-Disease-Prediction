import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Enter patient details to predict the risk of heart disease.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.selectbox("Gender", ["Male", "Female"])
impluse = st.number_input("Heart Rate (impluse)", min_value=30, max_value=200, value=70)
pressurehight = st.number_input("Systolic BP (pressurehight)", min_value=50, max_value=250, value=120)
pressurelow = st.number_input("Diastolic BP (pressurelow)", min_value=30, max_value=150, value=80)
glucose = st.number_input("Blood Glucose", min_value=50, max_value=400, value=100)
kcm = st.number_input("KCM", min_value=0.0, max_value=100.0, value=10.0)
troponin = st.number_input("Troponin Level",min_value=0.0, max_value=100.0,value=0.03,step=0.001,format="%.3f")

# Feature processing (match training pipeline)
impluse_clean = np.clip(impluse, 40, 180)
glucose_capped = np.clip(glucose, None, 300)
kcm_log = np.log1p(kcm)
troponin_log = np.log1p(troponin)
gender_encoded = 1 if gender == 'Male' else 0

# Input formatting
input_data = np.array([[
    age,
    gender_encoded,
    impluse_clean,
    pressurehight,
    pressurelow,
    glucose_capped,
    kcm_log,
    troponin_log
]])

# Scale features
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    st.subheader("🔍 Prediction Result:")
    if prediction == 0:
        st.success("🟢 Low Risk of Heart Disease")
    else:
        st.error("🔴 High Risk of Heart Disease")
