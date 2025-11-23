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

# ============================================================
#               HEIGHT + WEIGHT INPUT (with unit choice)
# ============================================================
st.subheader("Anthropometrics")

colA, colB = st.columns(2)

with colA:
    height_unit = st.selectbox("Height unit", ["cm", "inches"])
    if height_unit == "cm":
        height_val = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
        height_m = height_val / 100
    else:
        height_val = st.number_input("Height (inches)", 40.0, 100.0, 70.0)
        height_m = height_val * 0.0254

with colB:
    weight_unit = st.selectbox("Weight unit", ["kg", "lbs"])
    if weight_unit == "kg":
        weight_val = st.number_input("Weight (kg)", 30.0, 300.0, 75.0)
        weight_kg = weight_val
    else:
        weight_val = st.number_input("Weight (lbs)", 60.0, 600.0, 165.0)
        weight_kg = weight_val * 0.453592

bmi = weight_kg / (height_m ** 2)
st.write(f"**Calculated BMI:** {bmi:.1f} kg/m²")

# ============================================================
#      FAMILY INCOME → INCOME-TO-POVERTY RATIO (FIPR)
# ============================================================
st.subheader("Socioeconomic Variables")

# Federal poverty guidelines for 48 contiguous states & D.C.
BASE_POVERTY_48 = {
    1: 15650,
    2: 21150,
    3: 26650,
    4: 32150,
    5: 37650,
    6: 43150,
    7: 48650,
    8: 54150,
}
EXTRA_PER_PERSON_48 = 5500

colF1, colF2 = st.columns(2)

with colF1:
    family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)

with colF2:
    household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

# Compute poverty threshold automatically (48 contiguous states)
if household_size <= 8:
    poverty_threshold = BASE_POVERTY_48[household_size]
else:
    poverty_threshold = BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

st.write(f"**Estimated poverty threshold (48 states, {household_size} people):** "
         f"${poverty_threshold:,.0f}")

income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0
st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

# ============================================================
#                 DEMOGRAPHICS & CLINICAL MEASUREMENTS
# ============================================================
st.subheader("Demographics & Clinical Measurements")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 90, 45)
    waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
    sbp = st.number_input("Avg systolic BP (mmHg)", 80, 220, 120)
    smoker = st.selectbox("Current smoker?", ["No", "Yes"])

with col2:
    dbp = st.number_input("Avg diastolic BP (mmHg)", 40, 140, 75)
    hr = st.number_input("Resting heart rate (bpm)", 40, 140, 70)
    activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

# ============================================================
#                      EDUCATION (NHANES)
# ============================================================
education = st.selectbox(
    "Education level (NHANES categories)",
    [
        "Less than 9th grade",
        "9-11th grade (Includes 12th w/o diploma)",
        "High school graduate/GED or equivalent",
        "Some college or AA degree",
        "College graduate or above"
    ]
)

education_map = {
    "Less than 9th grade": 1,
    "9-11th grade (Includes 12th w/o diploma)": 2,
    "High school graduate/GED or equivalent": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5,
}

# ============================================================
#                      RACE & GENDER (NHANES)
# ============================================================
gender = st.selectbox("Gender", ["Male", "Female"])
race = st.selectbox(
    "Race/ethnicity",
    ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
)

gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4
}

smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

# ============================================================
#                      BUILD FEATURE ROW & PREDICT
# ============================================================
if st.button("Estimate risks"):
    
    X = pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity],
        "smoking": smoke_map[smoker],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income_ratio,
        "Education": education_map[education],
        "Race": race_map[race],
        "Gender": gender_map[gender],
    }])

    p_diab = float(pipe_diab.predict_proba(X)[0, 1])
    p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
    p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

    r1, r2, r3 = st.columns(3)
    r1.metric("Diabetes Risk", f"{p_diab*100:.1f}%")
    r2.metric("CKD Risk", f"{p_ckd*100:.1f}%")
    r3.metric("CVD Risk", f"{p_cvd*100:.1f}%")

    st.caption("These estimates are based solely on non-dietary predictors.")

# Footer
st.markdown("---")
st.markdown("**Model info**")
st.markdown("- Bagged decision trees with preprocessing pipeline")
st.markdown("- Inputs mirror NHANES preprocessing pipeline")
