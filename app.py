import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request

# List of all expected one-hot columns
EXPECTED_COLUMNS = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
    "InterestRate", "LoanTerm", "DTIRatio",
    "Education_High School", "Education_Bachelor's", "Education_Master's", "Education_PhD",
    "EmploymentType_Full-time", "EmploymentType_Part-time", "EmploymentType_Unemployed", "EmploymentType_Self-employed",
    "MaritalStatus_Single", "MaritalStatus_Married", "MaritalStatus_Divorced",
    "HasMortgage_Yes", "HasMortgage_No",
    "HasDependents_Yes", "HasDependents_No",
    "LoanPurpose_Auto", "LoanPurpose_Business", "LoanPurpose_Education", "LoanPurpose_Home", "LoanPurpose_Other",
    "HasCoSigner_Yes", "HasCoSigner_No"
]

# Set up the page
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Loan Default Risk Predictor")

st.markdown("### Enter Applicant Information")
with st.form("loan_form"):
    age = st.slider("Age", 18, 100, 30)
    income = st.number_input("Annual Income", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=10000)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    months_employed = st.number_input("Months Employed", min_value=0, value=12)
    num_credit_lines = st.slider("Number of Credit Lines", 0, 20, 4)
    interest_rate = st.slider("Interest Rate (%)", 0.0, 50.0, 12.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])
    dti_ratio = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)

    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Unemployed", "Self-employed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
    has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    has_cosigner = st.selectbox("Has Co-signer?", ["Yes", "No"])

    submit = st.form_submit_button("Predict Default Risk")

# Raw input DataFrame
raw_input = pd.DataFrame([{
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "CreditScore": credit_score,
    "MonthsEmployed": months_employed,
    "NumCreditLines": num_credit_lines,
    "InterestRate": interest_rate,
    "LoanTerm": loan_term,
    "DTIRatio": dti_ratio,
    "Education": education,
    "EmploymentType": employment_type,
    "MaritalStatus": marital_status,
    "HasMortgage": has_mortgage,
    "HasDependents": has_dependents,
    "LoanPurpose": loan_purpose,
    "HasCoSigner": has_cosigner
}])

# One-hot encode categorical variables
categorical_cols = ["Education", "EmploymentType", "MaritalStatus", "HasMortgage", 
                    "HasDependents", "LoanPurpose", "HasCoSigner"]
input_encoded = pd.get_dummies(raw_input, columns=categorical_cols)

# Add missing columns with 0 values
for col in EXPECTED_COLUMNS:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder columns to match training data
input_encoded = input_encoded[EXPECTED_COLUMNS]

MODEL_PATH = "random_forest_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

# Predict
if submit and pipeline is not None:
    prediction = pipeline.predict(input_encoded)[0]
    proba = pipeline.predict_proba(input_encoded)[0][1]

    if prediction == 1:
        st.error(f"❌ Prediction: Likely to Default (Risk Score: {proba:.2f})")
    else:
        st.success(f"✅ Prediction: Likely to Repay (Risk Score: {proba:.2f})")

# Tips section
st.markdown("---")
st.markdown("### 📌 Tips:")
st.markdown("- Aim for a low Debt-to-Income ratio and higher credit score.")
st.markdown("- Stable employment and income improve approval chances.")
st.markdown("- Co-signers and mortgage-free profiles are seen positively.")