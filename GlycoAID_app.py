import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="GlycoAID", layout="centered")

st.title("🩺 GlycoAID - Diabetes Risk Prediction System")
st.markdown("AI-powered early screening tool for diabetes risk")


# INPUT FORM

with st.form("input_form"):
    preg = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    skin = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
    age = st.number_input("Age", 1, 120)

    submitted = st.form_submit_button("Predict Risk")

# PREDICTION
if submitted:
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    # RESULT DISPLAY
  
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes ({(1-probability)*100:.2f}%)")

  
    # RISK GAUGE (BONUS)
    st.subheader("Risk Level")
    st.progress(int(probability * 100))

   
    # EXPLAINABILITY (BONUS)
    st.subheader("Why this prediction?")

    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DPF", "Age"
    ]

    input_df = pd.DataFrame(input_data, columns=feature_names)

    if hasattr(model, "coef_"):
        influence = model.coef_[0]
        explanation = pd.Series(influence, index=feature_names)
        explanation = explanation.sort_values()

        st.bar_chart(explanation)

    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        explanation = pd.Series(importance, index=feature_names)
        explanation = explanation.sort_values()

        st.bar_chart(explanation)