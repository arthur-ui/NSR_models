import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# ===========================
# Load trained models
# ===========================
pipe_diab = joblib.load("diabetes_model.joblib")
pipe_ckd  = joblib.load("ckd_model.joblib")
pipe_cvd  = joblib.load("cvd_model.joblib")

st.set_page_config(page_title="Non-Dietary Chronic Disease Risk Tool", page_icon="🧬")

# -------------------------------------------------
# Shared constants / helper functions
# -------------------------------------------------
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

education_map = {
    "Less than 9th grade": 1,
    "9-11th grade (Includes 12th w/o diploma)": 2,
    "High school graduate/GED or equivalent": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5,
}
gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4,
}
smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

def compute_poverty_threshold(household_size: int) -> int:
    if household_size <= 8:
        return BASE_POVERTY_48[household_size]
    return BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

def build_feature_df(
    bmi, age, waist, activity_label, smoker_label,
    sbp, dbp, hr, income_ratio, education_label,
    race_label, gender_label
) -> pd.DataFrame:
    return pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity_label],
        "smoking": smoke_map[smoker_label],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income_ratio,
        "Education": education_map[education_label],
        "Race": race_map[race_label],
        "Gender": gender_map[gender_label],
    }])

def predict_three(X: pd.DataFrame):
    """Return probabilities for diabetes, CKD, CVD as tuple of 1D numpy arrays."""
    p_diab = pipe_diab.predict_proba(X)[:, 1]
    p_ckd  = pipe_ckd.predict_proba(X)[:, 1]
    p_cvd  = pipe_cvd.predict_proba(X)[:, 1]
    return p_diab, p_ckd, p_cvd

# ===========================
# Tabs
# ===========================
tab_calc, tab_research = st.tabs(["Risk calculator", "Researcher tools"])


# ============================================================
#                     TAB 1: RISK CALCULATOR
# ============================================================
with tab_calc:
    st.title("Non-Dietary Chronic Disease Risk Assessment Tool")
    st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

    # ---------------- Anthropometrics ----------------
    st.subheader("Anthropometrics")

    colA, colB = st.columns(2)

    with colA:
        height_unit = st.selectbox("Height unit", ["cm", "inches"])
        if height_unit == "cm":
            height_val = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
            height_m = height_val / 100.0
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

    # ---------------- Socioeconomic ----------------
    st.subheader("Socioeconomic variables")

    colF1, colF2 = st.columns(2)
    with colF1:
        family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)
    with colF2:
        household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

    poverty_threshold = compute_poverty_threshold(household_size)
    st.write(
        f"**Estimated poverty threshold (48 states, {household_size} people):** "
        f"${poverty_threshold:,.0f}"
    )
    income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0.0
    st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

    # ---------------- Demographics & clinical ----------------
    st.subheader("Demographics & clinical measurements")

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

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox(
        "Race/ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
    )

    # ---------------- Predict ----------------
    if st.button("Estimate risks", key="calc_button"):
        X = build_feature_df(
            bmi=bmi, age=age, waist=waist, activity_label=activity,
            smoker_label=smoker, sbp=sbp, dbp=dbp, hr=hr,
            income_ratio=income_ratio, education_label=education,
            race_label=race, gender_label=gender
        )

        p_diab, p_ckd, p_cvd = predict_three(X)

        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes risk", f"{p_diab[0]*100:.1f}%")
        r2.metric("CKD risk", f"{p_ckd[0]*100:.1f}%")
        r3.metric("CVD risk", f"{p_cvd[0]*100:.1f}%")

        st.caption("These estimates are based solely on non-dietary predictors.")

    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown("- Bagged decision trees with preprocessing pipeline")
    st.markdown("- Inputs mirror NHANES preprocessing pipeline")


# ============================================================
#                 TAB 2: RESEARCHER TOOLS
# ============================================================
with tab_research:
    st.title("Researcher tools & sensitivity analysis")
    st.caption(
        "Explore how changes in individual predictors affect modelled risk for "
        "diabetes, CKD, and CVD. All calculations use the same non-laboratory "
        "models as the main risk calculator."
    )

    # ---------------- Baseline profile for research tools ----------------
    st.subheader("Baseline profile (held constant for sensitivity analyses)")
    colB1, colB2, colB3 = st.columns(3)

    with colB1:
        age_r = st.number_input("Age (years)", 18, 90, 50, key="age_r")
        bmi_r = st.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, key="bmi_r")
        waist_r = st.number_input("Waist circumference (cm)", 50.0, 200.0, 100.0, key="waist_r")

    with colB2:
        sbp_r = st.number_input("Systolic BP (mmHg)", 80, 220, 135, key="sbp_r")
        dbp_r = st.number_input("Diastolic BP (mmHg)", 40, 140, 80, key="dbp_r")
        hr_r  = st.number_input("Resting heart rate (bpm)", 40, 140, 72, key="hr_r")

    with colB3:
        family_income_r = st.number_input("Annual family income (USD)", 0, 300000, 60000, key="inc_r")
        household_size_r = st.selectbox("Household size", list(range(1, 13)), index=3, key="hh_r")

    poverty_threshold_r = compute_poverty_threshold(household_size_r)
    income_ratio_r = family_income_r / poverty_threshold_r if poverty_threshold_r > 0 else 0.0
    st.write(f"Baseline income-to-poverty ratio: **{income_ratio_r:.2f}**")

    colB4, colB5, colB6 = st.columns(3)
    with colB4:
        education_r = st.selectbox(
            "Education (NHANES categories)", list(education_map.keys()),
            index=3, key="edu_r"
        )
    with colB5:
        gender_r = st.selectbox("Gender", ["Male", "Female"], key="gender_r")
    with colB6:
        race_r = st.selectbox(
            "Race/ethnicity",
            ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
            key="race_r"
        )

    baseline_df = build_feature_df(
        bmi=bmi_r, age=age_r, waist=waist_r,
        activity_label="Moderate",  # default; can extend later
        smoker_label="No",
        sbp=sbp_r, dbp=dbp_r, hr=hr_r,
        income_ratio=income_ratio_r,
        education_label=education_r,
        race_label=race_r, gender_label=gender_r
    )

    base_diab, base_ckd, base_cvd = predict_three(baseline_df)
    st.markdown(
        f"Baseline predicted risks – Diabetes: **{base_diab[0]*100:.1f}%**, "
        f"CKD: **{base_ckd[0]*100:.1f}%**, CVD: **{base_cvd[0]*100:.1f}%**."
    )

    st.markdown("---")

    # ============================================================
    # 1. One-dimensional sensitivity curves (all 3 diseases)
    # ============================================================
    st.subheader("One-dimensional sensitivity analysis")

    var_options = {
        "Age (years)": ("AgeYears", 20, 85, 40),
        "BMI (kg/m²)": ("bmi", 18, 45, 40),
        "Waist circumference (cm)": ("waist_circumference", 60, 140, 40),
        "Systolic BP (mmHg)": ("avg_systolic", 90, 180, 40),
        "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110, 40),
        "Resting HR (bpm)": ("avg_HR", 50, 110, 40),
        "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0, 40),
    }

    sens_label = st.selectbox(
        "Choose variable to vary",
        list(var_options.keys()),
        index=0
    )
    var_col, vmin, vmax, n_points = var_options[sens_label]
    vals = np.linspace(vmin, vmax, n_points)

    # Build grid of inputs
    sens_X = pd.concat([baseline_df] * n_points, ignore_index=True)
    sens_X[var_col] = vals

    s_diab, s_ckd, s_cvd = predict_three(sens_X)

    sens_df = pd.DataFrame({
        "Value": vals,
        "Diabetes": s_diab,
        "CKD": s_ckd,
        "CVD": s_cvd,
    }).melt("Value", var_name="Disease", value_name="Risk")

    sens_chart = (
        alt.Chart(sens_df)
        .mark_line(point=False)
        .encode(
            x=alt.X("Value:Q", title=sens_label),
            y=alt.Y("Risk:Q", title="Predicted risk"),
            color=alt.Color("Disease:N", title=None),
        )
        .properties(height=300)
    )

    st.altair_chart(sens_chart, use_container_width=True)
    st.caption("Risk curves are generated by varying one predictor while holding the baseline profile constant.")

    st.markdown("---")

    # ============================================================
    # 2. SBP × DBP heatmaps (all 3 diseases)
    # ============================================================
    st.subheader("SBP × DBP interaction heatmaps")

    sbp_vals = np.arange(90, 181, 5)
    dbp_vals = np.arange(50, 111, 5)
    SBP_grid, DBP_grid = np.meshgrid(sbp_vals, dbp_vals)
    n_grid = SBP_grid.size

    grid_X = pd.concat([baseline_df] * n_grid, ignore_index=True)
    grid_X["avg_systolic"] = SBP_grid.ravel()
    grid_X["avg_diastolic"] = DBP_grid.ravel()

    h_diab, h_ckd, h_cvd = predict_three(grid_X)

    dfs = []
    for disease, arr in zip(
        ["Diabetes", "CKD", "CVD"],
        [h_diab, h_ckd, h_cvd]
    ):
        dfs.append(
            pd.DataFrame({
                "SBP": SBP_grid.ravel(),
                "DBP": DBP_grid.ravel(),
                "Disease": disease,
                "Risk": arr
            })
        )
    heat_df = pd.concat(dfs, ignore_index=True)
    max_risk = float(heat_df["Risk"].max())

    heat_chart = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("SBP:O", title="Systolic BP (mmHg)"),
            y=alt.Y("DBP:O", title="Diastolic BP (mmHg)"),
            color=alt.Color(
                "Risk:Q",
                title="Predicted risk",
                scale=alt.Scale(domain=[0, max_risk])
            ),
        )
        .properties(width=220, height=220)
        .facet(column=alt.Column("Disease:N", title=None))
    )
    st.altair_chart(heat_chart, use_container_width=True)
    st.caption("Shared color scale is set from 0 to the maximum risk observed across all three diseases.")

    st.markdown("---")

    # ============================================================
    # 3. Pulse pressure & MAP heatmaps
    # ============================================================
    st.subheader("Pulse pressure and mean arterial pressure")

    PP = SBP_grid - DBP_grid
    MAP = (SBP_grid + 2 * DBP_grid) / 3.0

    dfs_pp = []
    for disease, arr in zip(
        ["Diabetes", "CKD", "CVD"],
        [h_diab, h_ckd, h_cvd]
    ):
        dfs_pp.append(
            pd.DataFrame({
                "Pulse pressure": PP.ravel(),
                "MAP": MAP.ravel(),
                "Disease": disease,
                "Risk": arr
            })
        )
    pp_df = pd.concat(dfs_pp, ignore_index=True)
    max_risk_pp = float(pp_df["Risk"].max())

    pp_chart = (
        alt.Chart(pp_df)
        .mark_rect()
        .encode(
            x=alt.X("Pulse pressure:O", title="Pulse pressure (mmHg)"),
            y=alt.Y("MAP:O", title="Mean arterial pressure (mmHg)"),
            color=alt.Color(
                "Risk:Q",
                title="Predicted risk",
                scale=alt.Scale(domain=[0, max_risk_pp])
            ),
        )
        .properties(width=220, height=220)
        .facet(column=alt.Column("Disease:N", title=None))
    )
    st.altair_chart(pp_chart, use_container_width=True)

    st.markdown("---")

    # ============================================================
    # 4. BMI / weight-loss projection curves
    # ============================================================
    st.subheader("BMI / weight-change projections")

    # Interpret baseline BMI as reference; vary +/- 10 units
    bmi_min = max(15.0, bmi_r - 10)
    bmi_max = min(60.0, bmi_r + 10)
    bmi_range = np.linspace(bmi_min, bmi_max, 40)

    bmi_X = pd.concat([baseline_df] * len(bmi_range), ignore_index=True)
    bmi_X["bmi"] = bmi_range

    w_diab, w_ckd, w_cvd = predict_three(bmi_X)
    bmi_df = pd.DataFrame({
        "BMI": bmi_range,
        "Diabetes": w_diab,
        "CKD": w_ckd,
        "CVD": w_cvd
    }).melt("BMI", var_name="Disease", value_name="Risk")

    bmi_chart = (
        alt.Chart(bmi_df)
        .mark_line()
        .encode(
            x=alt.X("BMI:Q", title="BMI (kg/m²)"),
            y=alt.Y("Risk:Q", title="Predicted risk"),
            color=alt.Color("Disease:N", title=None)
        )
        .properties(height=300)
    )
    st.altair_chart(bmi_chart, use_container_width=True)
    st.caption("Use this plot to visualise how weight loss or gain may change risk across diseases.")

    st.markdown("---")

    # ============================================================
    # 5. Simple intervention simulator
    # ============================================================
    st.subheader("Intervention simulator (baseline vs modified profile)")

    st.markdown("Specify changes from the baseline profile and compare risks before and after.")

    colI1, colI2, colI3 = st.columns(3)
    with colI1:
        delta_bmi = st.number_input("Δ BMI (kg/m²)", -15.0, 15.0, -3.0, step=0.5)
        delta_sbp = st.number_input("Δ systolic BP (mmHg)", -40, 40, -10)
    with colI2:
        delta_dbp = st.number_input("Δ diastolic BP (mmHg)", -30, 30, -5)
        delta_hr  = st.number_input("Δ resting HR (bpm)", -40, 40, 0)
    with colI3:
        quit_smoking = st.checkbox("Smoking cessation (set smoker = No)", value=True)
        increase_activity = st.checkbox("Increase activity to High", value=False)

    if st.button("Simulate intervention", key="intervention_button"):
        # Baseline prediction already computed
        base_risks = np.array([base_diab[0], base_ckd[0], base_cvd[0]])

        # Build modified profile
        mod_df = baseline_df.copy()
        mod_df["bmi"] = np.clip(bmi_r + delta_bmi, 15.0, 60.0)
        mod_df["avg_systolic"] = np.clip(sbp_r + delta_sbp, 80, 220)
        mod_df["avg_diastolic"] = np.clip(dbp_r + delta_dbp, 40, 140)
        mod_df["avg_HR"] = np.clip(hr_r + delta_hr, 40, 140)

        if quit_smoking:
            mod_df["smoking"] = 0
        if increase_activity:
            mod_df["activity_level"] = activity_map["High"]

        m_diab, m_ckd, m_cvd = predict_three(mod_df)
        mod_risks = np.array([m_diab[0], m_ckd[0], m_cvd[0]])

        sim_df = pd.DataFrame({
            "Disease": ["Diabetes", "CKD", "CVD"],
            "Risk type": ["Baseline"]*3 + ["After intervention"]*3,
            "Risk": np.concatenate([base_risks, mod_risks])
        })

        sim_chart = (
            alt.Chart(sim_df)
            .mark_bar()
            .encode(
                x=alt.X("Disease:N", title=None),
                y=alt.Y("Risk:Q", title="Predicted risk"),
                color=alt.Color("Risk type:N", title=None),
                column=alt.Column("Risk type:N", title=None)
            )
        )
        st.altair_chart(sim_chart, use_container_width=True)

        abs_diff = (base_risks - mod_risks) * 100
        st.markdown(
            f"Absolute risk reductions (percentage points): "
            f"Diabetes **{abs_diff[0]:.1f}**; "
            f"CKD **{abs_diff[1]:.1f}**; "
            f"CVD **{abs_diff[2]:.1f}**."
        )
