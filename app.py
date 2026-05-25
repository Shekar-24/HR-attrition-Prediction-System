import streamlit as st
import joblib
import numpy as np
import os

# Page config 
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="centered",
)

# Load model 
MODEL_PATH = "attrition_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file `{MODEL_PATH}` not found.\n\n"
            "Please run `python train_model.py` first to generate it."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# Header 
st.title("👥 HR Employee Attrition Predictor")
st.markdown(
    "Enter employee details below to predict whether they are likely to **leave (Yes)** "
    "or **stay (No)**."
)
st.divider()

# Input form 
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=65,
        value=35,
        step=1,
        help="Employee's current age in years.",
    )

    total_working_years = st.number_input(
        "🏢 Total Working Years",
        min_value=0,
        max_value=40,
        value=10,
        step=1,
        help="Total years the employee has worked in their career.",
    )

with col2:
    monthly_income = st.number_input(
        "💰 Monthly Income (USD)",
        min_value=1_000,
        max_value=20_000,
        value=5_000,
        step=100,
        help="Employee's monthly salary.",
    )

    years_at_company = st.number_input(
        "📅 Years at Company",
        min_value=0,
        max_value=40,
        value=5,
        step=1,
        help="Number of years the employee has worked at this company.",
    )

st.divider()

# Predict 
if st.button("🔍 Predict Attrition", use_container_width=True, type="primary"):
    features = np.array([[age, monthly_income, total_working_years, years_at_company]])
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    attrition_prob = probability[1] * 100   # probability of "Yes"
    stay_prob      = probability[0] * 100   # probability of "No"

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ **High Attrition Risk — The employee is likely to LEAVE.**\n\n"
            f"Probability of leaving: **{attrition_prob:.1f}%**"
        )
    else:
        st.success(
            f"✅ **Low Attrition Risk — The employee is likely to STAY.**\n\n"
            f"Probability of staying: **{stay_prob:.1f}%**"
        )

    # Probability bar
    st.markdown("#### Confidence Breakdown")
    col_a, col_b = st.columns(2)
    col_a.metric("🟢 Probability — Stay (No)",  f"{stay_prob:.1f}%")
    col_b.metric("🔴 Probability — Leave (Yes)", f"{attrition_prob:.1f}%")

    st.progress(int(attrition_prob), text=f"Attrition risk: {attrition_prob:.1f}%")

    # Feature summary
    st.markdown("#### Input Summary")
    st.table(
        {
            "Feature": [
                "Age",
                "Monthly Income",
                "Total Working Years",
                "Years at Company",
            ],
            "Value": [
                age,
                f"${monthly_income:,}",
                total_working_years,
                years_at_company,
            ],
        }
    )

# Footer 
st.divider()
st.caption(
    "Model: Gradient Boosting Classifier | "
    "Features: Age, Monthly Income, Total Working Years, Years at Company"
)
