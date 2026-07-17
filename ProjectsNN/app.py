# Updated Customer Churn Dashboard
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("customer_churn_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_assets()

st.markdown('''
<style>
.hero{
padding:3rem;
border-radius:25px;
background:linear-gradient(135deg,#1e3a8a,#2563eb,#06b6d4);
color:white;
text-align:center;
margin-bottom:30px;
box-shadow:0px 10px 30px rgba(0,0,0,.35);
}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="hero">
<h1>📊 Customer Churn Prediction Dashboard</h1>
<p>AI Powered Customer Retention Analysis using AdaBoost</p>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Customer Inputs")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0,1])
    partner = st.selectbox("Partner", ["Yes","No"])
    dependents = st.selectbox("Dependents", ["Yes","No"])
    tenure = st.slider("Tenure (Months)", 0, 72, 24)

    phone_service = st.selectbox("Phone Service", ["Yes","No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

    online_security = st.selectbox("Online Security", ["No","Yes","No internet service"])
    online_backup = st.selectbox("Online Backup", ["No","Yes","No internet service"])
    device_protection = st.selectbox("Device Protection", ["No","Yes","No internet service"])
    tech_support = st.selectbox("Tech Support", ["No","Yes","No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes","No"])

    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
    )

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

total_charges = tenure * monthly_charges

st.info(f"💰 Estimated Total Charges: ₹{total_charges:.2f}")

if st.button("🚀 Predict Churn", use_container_width=True):

    data = {col:0 for col in feature_columns}

    data["gender"] = 1 if gender=="Male" else 0
    data["SeniorCitizen"] = senior
    data["Partner"] = 1 if partner=="Yes" else 0
    data["Dependents"] = 1 if dependents=="Yes" else 0
    data["tenure"] = tenure
    data["PhoneService"] = 1 if phone_service=="Yes" else 0
    data["PaperlessBilling"] = 1 if paperless=="Yes" else 0
    data["MonthlyCharges"] = monthly_charges
    data["TotalCharges"] = total_charges

    mappings = {
        "MultipleLines_No phone service": multiple_lines == "No phone service",
        "MultipleLines_Yes": multiple_lines == "Yes",
        "InternetService_Fiber optic": internet_service == "Fiber optic",
        "InternetService_No": internet_service == "No",
        "OnlineSecurity_No internet service": online_security == "No internet service",
        "OnlineSecurity_Yes": online_security == "Yes",
        "OnlineBackup_No internet service": online_backup == "No internet service",
        "OnlineBackup_Yes": online_backup == "Yes",
        "DeviceProtection_No internet service": device_protection == "No internet service",
        "DeviceProtection_Yes": device_protection == "Yes",
        "TechSupport_No internet service": tech_support == "No internet service",
        "TechSupport_Yes": tech_support == "Yes",
        "StreamingTV_No internet service": streaming_tv == "No internet service",
        "StreamingTV_Yes": streaming_tv == "Yes",
        "StreamingMovies_No internet service": streaming_movies == "No internet service",
        "StreamingMovies_Yes": streaming_movies == "Yes",
        "Contract_One year": contract == "One year",
        "Contract_Two year": contract == "Two year",
        "PaymentMethod_Credit card (automatic)": payment_method == "Credit card (automatic)",
        "PaymentMethod_Electronic check": payment_method == "Electronic check",
        "PaymentMethod_Mailed check": payment_method == "Mailed check",
    }

    for k, v in mappings.items():
        data[k] = int(v)

    X = pd.DataFrame([data])

    pred = model.predict(X)[0]
    prob = float(model.predict_proba(X)[0][1]) * 100

    st.subheader("📊 Prediction Dashboard")

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Churn Probability", f"{prob:.2f}%")
    m2.metric("Tenure", f"{tenure} Months")
    m3.metric("Monthly Charges", f"₹{monthly_charges:.2f}")
    m4.metric("Total Charges", f"₹{total_charges:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={"text":"Customer Churn Risk"},
        gauge={"axis":{"range":[0,100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    if prob < 30:
        st.success("🟢 Low Churn Risk")
    elif prob < 70:
        st.warning("🟡 Medium Churn Risk")
    else:
        st.error("🔴 High Churn Risk")

    if pred == 1:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    st.subheader("📋 Customer Summary")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Contract:** {contract}")
    st.write(f"**Internet Service:** {internet_service}")
    st.write(f"**Tenure:** {tenure} Months")

st.markdown("---")
st.markdown('''
<center>
<h4>Developed by Deepak Kumar</h4>
<p>Machine Learning • Streamlit • AdaBoost</p>
</center>
''', unsafe_allow_html=True)
