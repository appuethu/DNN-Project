import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
from tensorflow.keras.models import load_model                  
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="HeartGuard+", layout="centered")
st.title("❤️ HeartGuard+ — Heart Disease Prediction (DNN)")
st.write("Deep Neural Network model with clinical + lifestyle features (2-input model) and SHAP explainability.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/heart_combined_synthetic.csv")

MODEL_DIR = os.path.join(BASE_DIR, "../model/artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
SC_CLIN_PATH = os.path.join(MODEL_DIR, "scaler_clinical.joblib")
SC_LIFE_PATH = os.path.join(MODEL_DIR, "scaler_lifestyle.joblib")


# ---------------------------------------------------------
# LOAD MODEL + SCALERS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        model = load_model(MODEL_PATH)
        sc_clin = joblib.load(SC_CLIN_PATH)
        sc_life = joblib.load(SC_LIFE_PATH)
        return model, sc_clin, sc_life
    except Exception as e:
        st.error("❌ Error loading model or scalers")
        st.write(e)
        return None, None, None


model, sc_clin, sc_life = load_artifacts()

if model is None:
    st.stop()


# ---------------------------------------------------------
# USER INPUT — CLINICAL
# ---------------------------------------------------------
st.header("🧬 Clinical Features")

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.slider("Resting BP (mmHg)", 80, 200, 130)
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.slider("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])


# Convert sex
sex_val = 1 if sex == "Male" else 0


clinical_features = np.array([[ 
    age, sex_val, cp, trestbps, chol, 
    fbs, restecg, thalach, exang, oldpeak,
    slope, ca, thal
]])


# ---------------------------------------------------------
# USER INPUT — LIFESTYLE
# ---------------------------------------------------------
st.header("🏃 Lifestyle Features")

steps = st.slider("Daily Steps", 1000, 20000, 8000)
sleep_hours = st.slider("Sleep Hours", 3, 12, 7)
smoker = st.selectbox("Smoking Habit", ["No", "Yes"])
alcohol_level = st.selectbox("Alcohol Level", [0, 1, 2])
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
bmi = st.slider("BMI", 10.0, 45.0, 25.0)

smoker_val = 1 if smoker == "Yes" else 0

lifestyle_features = np.array([[
    steps, sleep_hours, smoker_val,
    alcohol_level, stress_level, bmi
]])


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔍 Predict Heart Disease Risk"):

    # scale separately
    clin_scaled = sc_clin.transform(clinical_features)
    life_scaled = sc_life.transform(lifestyle_features)

    # ⭐ IMPORTANT FIX ⭐
    pred = float(model.predict([clin_scaled, life_scaled])[0][0])
    risk_percent = round(pred * 100, 2)

    st.subheader(f"💓 Predicted Risk: **{risk_percent}%**")

    if risk_percent > 60:
        st.error("⚠ High Risk — Please consult a cardiologist.")
    elif risk_percent > 30:
        st.warning("🟡 Medium Risk — Lifestyle changes recommended.")
    else:
        st.success("🟢 Low Risk — Keep maintaining healthy habits!")


    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------
    st.header("📊 SHAP Explainability")

    try:
        explainer = shap.DeepExplainer(model, [clin_scaled, life_scaled])
        shap_values = explainer.shap_values([clin_scaled, life_scaled])

        st.write("SHAP values generated successfully!")

    except Exception as e:
        st.error("⚠ SHAP could not generate explanation.")
        st.write(e)
