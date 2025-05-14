import streamlit as st
import pandas as pd
import joblib
#First time using streamlit so site is pretty basic, further enhancements can be done 

model = joblib.load("loan_model.pkl")
le_dict = joblib.load("encoders.pkl")

st.title("🏦 Loan Approval Predictor")


Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", [0, 1, 2, 3])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0)
Loan_Amount_Term = st.selectbox("Loan Amount Term", [360.0, 120.0, 180.0, 240.0, 300.0, 84.0, 60.0, 12.0])
Credit_History = st.selectbox("Credit History (1 = good, 0 = bad)", [1.0, 0.0]) #Yeah, techincally I am pretty sure credit history has more than only two values but from the csv file I took it like that
Property_Area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])


input_dict = {
    "Gender": le_dict["Gender"].transform([Gender])[0],
    "Married": le_dict["Married"].transform([Married])[0],
    "Dependents": int(Dependents),
    "Education": le_dict["Education"].transform([Education])[0],
    "Self_Employed": le_dict["Self_Employed"].transform([Self_Employed])[0],
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": le_dict["Property_Area"].transform([Property_Area])[0]
}

input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    label = le_dict["Loan_Status"].inverse_transform([prediction])[0]
    result = "✅ Loan Approved" if label == 'Y' else "❌ Loan Rejected"
    st.subheader(f"Prediction: {result}")
