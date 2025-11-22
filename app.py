import streamlit as st
import pandas as pd
import joblib

# ===========================
# Load trained models
# ===========================
pipe_diab = joblib.load("diabetes_model.joblib")
pipe_ckd  = joblib.load("ckd_model.joblib")
pipe_cvd  = joblib.load("cvd_model.joblib")

st.set_page_config(page_title="Non-Dietary Chronic Disease Risk Tool", page_icon="🧬")

st.title("Non-Dietary Chronic Disease Risk Assessment Tool")
st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

# ===========================
# UI inputs
# ===========================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 90, 45)
    bmi = st.number_input("BMI (kg/m²)", 15.0, 60.0, 27.0)
    waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
    sbp = st.number_input("Avg systolic BP (mmHg)", 80, 220, 120)

with col2:
    dbp = st.number_input("Avg diastolic BP (mmHg)", 40, 140, 75)
    hr = st.number_input("Resting heart rate (bpm)", 40, 140, 70)
    smoker = st.selectbox("Current smoker?", ["No", "Yes"])
    activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

income = st.number_input("Family income-to-poverty ratio", 0.0, 10.0, 2.0)

col3, col4 = st.columns(2)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col4:
    race = st.selectbox("Race/ethnicity", ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"])

# ===========================
# Encoding maps
# ===========================
smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}
gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4
}

# ===========================
# Prediction button
# ===========================
if st.button("Estimate risks"):
    
    # Build inference row
    X = pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity],
        "smoking": smoke_map[smoker],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income,
        "Education": None,                 # you did not collect in UI
        "Race": race_map[race],
        "Gender": gender_map[gender],
    }])

    # Compute probabilities
    p_diab = float(pipe_diab.predict_proba(X)[0, 1])
    p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
    p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

    # Display results in columns
    r1, r2, r3 = st.columns(3)

    r1.metric("Diabetes Risk", f"{p_diab*100:.1f}%")
    r2.metric("CKD Risk", f"{p_ckd*100:.1f}%")
    r3.metric("CVD Risk", f"{p_cvd*100:.1f}%")

    st.caption("These estimates are based solely on non-dietary predictors "
               "(anthropometrics, vital signs, sociodemographics).")

# Footer
st.markdown("---")
st.markdown("**Model info**")
st.markdown("- Bagged decision trees with preprocessing pipeline")
st.markdown("- Inputs mirror NHANES preprocessing pipeline exactly")
