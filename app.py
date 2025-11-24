import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ===========================
# Load trained models (cached)
# ===========================
@st.cache_resource
def load_models():
    pipe_diab = joblib.load("diabetes_model.joblib")
    pipe_ckd  = joblib.load("ckd_model.joblib")
    pipe_cvd  = joblib.load("cvd_model.joblib")
    return pipe_diab, pipe_ckd, pipe_cvd

pipe_diab, pipe_ckd, pipe_cvd = load_models()

st.set_page_config(page_title="Non-Laboratory Chronic Disease Risk Tool", page_icon="🧬")

st.title("Non-Laboratory Chronic Disease Risk Assessment Tool")
st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

# ===========================
# Shared constants / mappings
# ===========================

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

# Default baseline profile (used if user hasn't run the risk calculator yet)
DEFAULT_BASELINE = {
    "bmi": 27.0,
    "AgeYears": 45,
    "waist_circumference": 95.0,
    "activity_level": 1,                 # Moderate
    "smoking": 0,                        # No
    "avg_systolic": 120,
    "avg_diastolic": 75,
    "avg_HR": 70,
    "FamIncome_to_poverty_ratio": 2.0,
    "Education": 4,                      # Some college/AA
    "Race": 1,                           # NH White
    "Gender": 1,                         # Male
}

# Friendly labels for researcher plots
VAR_LABELS = {
    "AgeYears": "Age (years)",
    "bmi": "Body mass index (kg/m²)",
    "waist_circumference": "Waist circumference (cm)",
    "avg_systolic": "Systolic BP (mmHg)",
    "avg_diastolic": "Diastolic BP (mmHg)",
    "avg_HR": "Heart rate (bpm)",
    "FamIncome_to_poverty_ratio": "Income-to-poverty ratio",
}

# Recommended ranges for sensitivity analysis
SENSITIVITY_RANGES = {
    "AgeYears": (20, 80),
    "bmi": (18, 40),
    "waist_circumference": (60, 130),
    "avg_systolic": (90, 180),
    "avg_diastolic": (50, 110),
    "avg_HR": (50, 110),
    "FamIncome_to_poverty_ratio": (0.5, 5.0),
}

# Helper: build DataFrame from baseline dict
def df_from_baseline(baseline_dict):
    return pd.DataFrame([baseline_dict])


# ===========================
# Tabs
# ===========================
tab_calc, tab_research = st.tabs(["Risk calculator", "Researcher tools"])

# ============================================================
#                      TAB 1: RISK CALCULATOR
# ============================================================
with tab_calc:
    st.subheader("Individual risk calculator")
    st.markdown(
        "Enter your information below. The model uses only non-laboratory predictors "
        "to estimate your 10-year risk of diabetes, chronic kidney disease (CKD), "
        "and cardiovascular disease (CVD)."
    )

    # ------------------ Anthropometrics ------------------
    st.markdown("### Anthropometrics")

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

    # ------------------ Socioeconomic variables ------------------
    st.markdown("### Socioeconomic variables")

    colF1, colF2 = st.columns(2)

    with colF1:
        family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)

    with colF2:
        household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

    if household_size <= 8:
        poverty_threshold = BASE_POVERTY_48[household_size]
    else:
        poverty_threshold = BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

    st.write(
        f"**Estimated poverty threshold (48 contiguous states, {household_size} people):** "
        f"${poverty_threshold:,.0f}"
    )

    income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0.0
    st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

    # ------------------ Demographics & clinical ------------------
    st.markdown("### Demographics & clinical measurements")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 90, 45)
        waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
        sbp = st.number_input("Average systolic BP (mmHg)", 80, 220, 120)
        smoker = st.selectbox("Current smoker?", ["No", "Yes"])

    with col2:
        dbp = st.number_input("Average diastolic BP (mmHg)", 40, 140, 75)
        hr = st.number_input("Resting heart rate (bpm)", 40, 140, 70)
        activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

    education = st.selectbox(
        "Education level (NHANES categories)",
        [
            "Less than 9th grade",
            "9-11th grade (Includes 12th w/o diploma)",
            "High school graduate/GED or equivalent",
            "Some college or AA degree",
            "College graduate or above",
        ],
    )

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox(
        "Race/ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
    )

    # ------------------ Build row & predict ------------------
    if st.button("Estimate risks", type="primary"):
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

        # Save baseline for researcher tools
        st.session_state["baseline_features"] = X.iloc[0].to_dict()

        p_diab = float(pipe_diab.predict_proba(X)[0, 1])
        p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
        p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes risk", f"{p_diab * 100:.1f} %")
        r2.metric("CKD risk", f"{p_ckd * 100:.1f} %")
        r3.metric("CVD risk", f"{p_cvd * 100:.1f} %")

        st.caption(
            "These estimates are based solely on non-laboratory predictors and are "
            "intended for research and public health exploration, not for clinical decision-making."
        )

    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown("- Bagged decision trees with preprocessing pipeline")
    st.markdown("- Inputs mirror NHANES preprocessing pipeline")


# ============================================================
#                      TAB 2: RESEARCHER TOOLS
# ============================================================
with tab_research:
    st.subheader("Researcher tools: sensitivity analysis and risk landscapes")

    st.markdown(
        "This section lets you explore how predicted risks change as you vary key predictors. "
        "You can either start from the last profile you entered on the *Risk calculator* tab "
        "or directly specify a baseline profile here."
    )

    # --- get starting baseline (from session or default) ---
    base_from_calc = st.session_state.get("baseline_features", DEFAULT_BASELINE)
    baseline = DEFAULT_BASELINE.copy()
    baseline.update(base_from_calc)

    # ------------------ Baseline editor ------------------
    st.markdown("### Baseline profile for analyses")

    with st.expander("Edit baseline profile used in analyses", expanded=False):
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            age_r = st.number_input("Age (years)", 18, 90, int(baseline["AgeYears"]), key="rf_age")
            bmi_r = st.number_input("BMI (kg/m²)", 15.0, 60.0, float(baseline["bmi"]), key="rf_bmi")
            waist_r = st.number_input("Waist circumference (cm)", 50.0, 200.0,
                                      float(baseline["waist_circumference"]), key="rf_waist")
            sbp_r = st.number_input("Systolic BP (mmHg)", 80, 220,
                                    int(baseline["avg_systolic"]), key="rf_sbp")
            dbp_r = st.number_input("Diastolic BP (mmHg)", 40, 140,
                                    int(baseline["avg_diastolic"]), key="rf_dbp")
        with col_b2:
            hr_r = st.number_input("Heart rate (bpm)", 40, 140,
                                   int(baseline["avg_HR"]), key="rf_hr")
            fipr_r = st.number_input("Income-to-poverty ratio", 0.1, 10.0,
                                     float(baseline["FamIncome_to_poverty_ratio"]), key="rf_fipr")
            smoking_r = st.selectbox(
                "Smoking (0=No, 1=Yes)",
                options=[0, 1],
                index=int(baseline["smoking"]),
                key="rf_smoke",
            )
            activity_r = st.selectbox(
                "Activity (0=Low, 1=Moderate, 2=High)",
                options=[0, 1, 2],
                index=int(baseline["activity_level"]),
                key="rf_act",
            )
            education_r = st.selectbox(
                "Education (1–5)",
                options=[1, 2, 3, 4, 5],
                index=int(baseline["Education"]) - 1,
                key="rf_edu",
            )

        col_b3, col_b4 = st.columns(2)
        with col_b3:
            race_r = st.selectbox(
                "Race (1–4)",
                options=[1, 2, 3, 4],
                index=int(baseline["Race"]) - 1,
                key="rf_race",
            )
        with col_b4:
            gender_r = st.selectbox(
                "Gender (1=Male, 2=Female)",
                options=[1, 2],
                index=int(baseline["Gender"]) - 1,
                key="rf_gender",
            )

    # build research baseline from editor values
    research_baseline = {
        "AgeYears": age_r if "age_r" in locals() else baseline["AgeYears"],
        "bmi": bmi_r if "bmi_r" in locals() else baseline["bmi"],
        "waist_circumference": waist_r if "waist_r" in locals() else baseline["waist_circumference"],
        "avg_systolic": sbp_r if "sbp_r" in locals() else baseline["avg_systolic"],
        "avg_diastolic": dbp_r if "dbp_r" in locals() else baseline["avg_diastolic"],
        "avg_HR": hr_r if "hr_r" in locals() else baseline["avg_HR"],
        "FamIncome_to_poverty_ratio": fipr_r if "fipr_r" in locals() else baseline["FamIncome_to_poverty_ratio"],
        "smoking": smoking_r if "smoking_r" in locals() else baseline["smoking"],
        "activity_level": activity_r if "activity_r" in locals() else baseline["activity_level"],
        "Education": education_r if "education_r" in locals() else baseline["Education"],
        "Race": race_r if "race_r" in locals() else baseline["Race"],
        "Gender": gender_r if "gender_r" in locals() else baseline["Gender"],
    }

    with st.expander("View current baseline profile as JSON", expanded=False):
        show_dict = {
            "AgeYears": research_baseline["AgeYears"],
            "BMI": research_baseline["bmi"],
            "WaistCircumference": research_baseline["waist_circumference"],
            "SystolicBP": research_baseline["avg_systolic"],
            "DiastolicBP": research_baseline["avg_diastolic"],
            "HeartRate": research_baseline["avg_HR"],
            "IncomeToPovertyRatio": research_baseline["FamIncome_to_poverty_ratio"],
            "Smoking(0=No,1=Yes)": research_baseline["smoking"],
            "Activity(0=Low,1=Moderate,2=High)": research_baseline["activity_level"],
            "Education(1–5)": research_baseline["Education"],
            "Race(1–4)": research_baseline["Race"],
            "Gender(1=Male,2=Female)": research_baseline["Gender"],
        }
        st.json(show_dict)

    # ------------------ 1D sensitivity analysis ------------------
    st.markdown("### 1D sensitivity analysis (all three diseases)")

    st.markdown(
        "Select a variable to vary while holding the rest of the baseline profile constant. "
        "The plot shows predicted risk for **diabetes, CKD, and CVD** as that variable changes."
    )

    var_options = list(SENSITIVITY_RANGES.keys())
    var_choice = st.selectbox(
        "Variable to vary",
        options=var_options,
        format_func=lambda v: VAR_LABELS.get(v, v),
    )

    vmin_default, vmax_default = SENSITIVITY_RANGES[var_choice]

    col_range1, col_range2 = st.columns(2)
    with col_range1:
        vmin = st.number_input(
            f"Minimum {VAR_LABELS.get(var_choice, var_choice)}",
            value=float(vmin_default),
        )
    with col_range2:
        vmax = st.number_input(
            f"Maximum {VAR_LABELS.get(var_choice, var_choice)}",
            value=float(vmax_default),
        )

    default_step = 2.0 if var_choice != "FamIncome_to_poverty_ratio" else 0.25
    step = st.number_input("Step size", value=default_step, min_value=0.01)

    if vmin >= vmax:
        st.error("Minimum must be less than maximum.")
    else:
        if st.button("Run 1D sensitivity analysis", key="run_1d_sens"):
            x_vals = np.arange(vmin, vmax + 1e-9, step)

            records = []
            for val in x_vals:
                row = research_baseline.copy()
                row[var_choice] = float(val)
                X = df_from_baseline(row)

                p_diab = float(pipe_diab.predict_proba(X)[0, 1])
                p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
                p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

                records.append({"Variable": val, "Disease": "Diabetes", "Risk": p_diab})
                records.append({"Variable": val, "Disease": "CKD",      "Risk": p_ckd})
                records.append({"Variable": val, "Disease": "CVD",      "Risk": p_cvd})

            df_plot = pd.DataFrame(records)
            max_risk = float(df_plot["Risk"].max()) if len(df_plot) > 0 else 1.0
            ymax = min(1.0, max_risk * 1.05) if max_risk > 0 else 1.0

            fig_line = px.line(
                df_plot,
                x="Variable",
                y="Risk",
                color="Disease",
                labels={
                    "Variable": VAR_LABELS.get(var_choice, var_choice),
                    "Risk": "Predicted risk (probability)",
                    "Disease": "Outcome",
                },
            )
            fig_line.update_layout(
                yaxis=dict(range=[0, ymax]),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,      # move legend further down so it doesn't overlap axes
                    xanchor="center",
                    x=0.5,
                ),
            )
            st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ------------------ 2D heatmap ------------------
    st.markdown("### 2D risk heatmap")

    st.markdown(
        "Generate a risk heatmap by varying **two predictors** at once for a selected disease, "
        "holding all other characteristics at the baseline profile."
    )

    disease_choice = st.selectbox("Disease for heatmap", ["Diabetes", "CKD", "CVD"])

    heatmap_vars = ["AgeYears", "bmi", "waist_circumference", "avg_systolic", "avg_diastolic"]

    col_hx, col_hy = st.columns(2)
    with col_hx:
        x_var = st.selectbox(
            "X-axis variable",
            options=heatmap_vars,
            format_func=lambda v: VAR_LABELS.get(v, v),
            key="x_var_heat",
        )
    with col_hy:
        y_var = st.selectbox(
            "Y-axis variable",
            options=heatmap_vars,
            format_func=lambda v: VAR_LABELS.get(v, v),
            key="y_var_heat",
        )

    if x_var == y_var:
        st.warning("X and Y variables must be different to generate a heatmap.")
    else:
        xmin, xmax = SENSITIVITY_RANGES[x_var]
        ymin, ymax = SENSITIVITY_RANGES[y_var]

        st.markdown("Specify ranges for the heatmap axes:")

        col_xr1, col_xr2 = st.columns(2)
        with col_xr1:
            x_min_val = st.number_input(
                f"X min ({VAR_LABELS.get(x_var, x_var)})",
                value=float(xmin),
                key="x_min_heat",
            )
        with col_xr2:
            x_max_val = st.number_input(
                f"X max ({VAR_LABELS.get(x_var, x_var)})",
                value=float(xmax),
                key="x_max_heat",
            )

        col_yr1, col_yr2 = st.columns(2)
        with col_yr1:
            y_min_val = st.number_input(
                f"Y min ({VAR_LABELS.get(y_var, y_var)})",
                value=float(ymin),
                key="y_min_heat",
            )
        with col_yr2:
            y_max_val = st.number_input(
                f"Y max ({VAR_LABELS.get(y_var, y_var)})",
                value=float(ymax),
                key="y_max_heat",
            )

        grid_res = st.slider(
            "Grid resolution (higher = smoother, slower)",
            min_value=15,
            max_value=60,
            value=35,
        )

        if x_min_val >= x_max_val or y_min_val >= y_max_val:
            st.error("Minimums must be less than maximums for both axes.")
        else:
            if st.button("Generate heatmap", key="run_heatmap"):
                # Vectorized grid generation
                x_values = np.linspace(x_min_val, x_max_val, grid_res)
                y_values = np.linspace(y_min_val, y_max_val, grid_res)
                X_grid, Y_grid = np.meshgrid(x_values, y_values)

                # Build a DataFrame of all grid points at once
                n_points = X_grid.size
                base_array = np.repeat(
                    pd.DataFrame([research_baseline]).values,
                    n_points,
                    axis=0,
                )
                base_df = pd.DataFrame(base_array, columns=research_baseline.keys())

                # Override x_var and y_var with flattened grid
                base_df[x_var] = X_grid.ravel()
                base_df[y_var] = Y_grid.ravel()

                # Choose disease model
                if disease_choice == "Diabetes":
                    model = pipe_diab
                elif disease_choice == "CKD":
                    model = pipe_ckd
                else:
                    model = pipe_cvd

                # Single batched predict_proba call (much faster)
                probs = model.predict_proba(base_df)[:, 1]
                risk_grid = probs.reshape(X_grid.shape)
                max_risk = float(risk_grid.max()) if risk_grid.size > 0 else 1.0

                # Heatmap with z-range from 0 to max risk
                fig_hm = px.imshow(
                    risk_grid,
                    x=np.round(x_values, 1),
                    y=np.round(y_values, 1),
                    labels=dict(
                        x=VAR_LABELS.get(x_var, x_var),
                        y=VAR_LABELS.get(y_var, y_var),
                        color="Predicted risk",
                    ),
                    origin="lower",
                    aspect="auto",
                    color_continuous_scale="Viridis",
                )
                fig_hm.update_coloraxes(cmin=0, cmax=max_risk)
                st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(
        "These researcher tools help visualize the model's behavior across the predictor space "
        "for both public health applications (e.g., weight loss or blood pressure reduction scenarios) "
        "and methodological exploration."
    )
