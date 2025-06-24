import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("model.pkl")

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")
st.title("Credit Score & Loan Default Risk Estimator")
st.write("Fill in your details to check your credit risk and eligibility status.")

with st.form("credit_form"):
    st.subheader("Applicant Information")

    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    Income = st.number_input("Monthly Income (₹)", min_value=0, value=40000)
    LoanAmount = st.number_input("Desired Loan Amount (₹)", min_value=0, value=150000)
    CreditScore = st.number_input("Your Credit Score", min_value=300, max_value=900, value=720)
    MonthsEmployed = st.number_input("Months of Employment", min_value=0, value=24)
    NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=2)
    InterestRate = st.number_input("Expected Interest Rate (%)", min_value=0.0, value=12.5)
    LoanTerm = st.number_input("Loan Term (in months)", min_value=6, value=60)
    DTIRatio = st.number_input("Debt-to-Income Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.25)

    st.subheader("Background Information")

    Education = st.selectbox("Highest Education Level", ["High School", "Bachelors", "Masters", "Doctorate"])
    EmploymentType = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married"])
    HasMortgage = st.selectbox("Do you have a mortgage?", ["No", "Yes"])
    HasDependents = st.selectbox("Do you have dependents?", ["No", "Yes"])
    LoanPurpose = st.selectbox("Purpose of the Loan", ["Home", "Car", "Education", "Business", "Other"])
    HasCoSigner = st.selectbox("Do you have a co-signer?", ["No", "Yes"])

    submit = st.form_submit_button("Predict Credit Risk")

if submit:
    map_dict = {
        "Education": {"High School": 0, "Bachelors": 1, "Masters": 2, "Doctorate": 3},
        "EmploymentType": {"Salaried": 0, "Self-Employed": 1},
        "MaritalStatus": {"Single": 0, "Married": 1},
        "HasMortgage": {"No": 0, "Yes": 1},
        "HasDependents": {"No": 0, "Yes": 1},
        "LoanPurpose": {"Home": 0, "Car": 1, "Education": 2, "Business": 3, "Other": 4},
        "HasCoSigner": {"No": 0, "Yes": 1}
    }

    input_data = pd.DataFrame([[
        Age,
        Income,
        LoanAmount,
        CreditScore,
        MonthsEmployed,
        NumCreditLines,
        InterestRate,
        LoanTerm,
        DTIRatio,
        map_dict["Education"][Education],
        map_dict["EmploymentType"][EmploymentType],
        map_dict["MaritalStatus"][MaritalStatus],
        map_dict["HasMortgage"][HasMortgage],
        map_dict["HasDependents"][HasDependents],
        map_dict["LoanPurpose"][LoanPurpose],
        map_dict["HasCoSigner"][HasCoSigner]
    ]], columns=[
        "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm", "DTIRatio", "Education", "EmploymentType", "MaritalStatus",
        "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
    ])

    try:
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # class 1 = default
        default_score = probability * 100

        st.success("Prediction Complete")
        st.write(f"Probability of Default: {default_score:.2f}%")

        threshold = 0.13

        if probability > threshold:
            st.markdown("### High Default Risk")
            st.error("This applicant shows a high likelihood of defaulting on the loan.")

        else:
            st.markdown("### No Default Risk")
            st.info("This applicant is likely to repay the loan successfully.")

    except Exception as e:
        st.error(f"Prediction failed due to: {e}")
