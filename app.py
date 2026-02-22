import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
with open("final_churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract feature importance (Logistic Regression coefficients)

try:
    # Get classifier
    classifier = model.named_steps["classifier"]

    # Get preprocessor
    preprocessor = model.named_steps["preprocessor"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Get coefficients
    coefficients = classifier.coef_[0]

    # Create dataframe
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    # Sort
    importance_df = importance_df.sort_values(by="Coefficient", ascending=False)

except:
    importance_df = None

st.title("📊 Customer Churn Prediction System")

st.write("Upload a customer dataset to predict churn risk.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing (must match training preprocessing)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Get probabilities
    probabilities = model.predict_proba(df)[:, 1]

    threshold = 0.35
    predictions = (probabilities > threshold).astype(int)

    # Risk classification
    def risk_category(prob):
        if prob < 0.35:
            return "Low"
        elif prob < 0.65:
            return "Medium"
        else:
            return "High"

    risk_levels = [risk_category(p) for p in probabilities]

    df["Churn_Probability"] = probabilities
    df["Predicted_Churn"] = predictions
    df["Risk_Level"] = risk_levels

    st.subheader("🔍 Prediction Results")

    high_risk = (df["Risk_Level"] == "High").sum()
    medium_risk = (df["Risk_Level"] == "Medium").sum()
    low_risk = (df["Risk_Level"] == "Low").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("High Risk Customers", high_risk)
    col2.metric("Medium Risk Customers", medium_risk)
    col3.metric("Low Risk Customers", low_risk)

    display_columns = [
        "Churn_Probability",
        "Predicted_Churn",
        "Risk_Level"
    ]

    st.dataframe(df[display_columns])

    st.subheader("📊 Risk Distribution")
    st.bar_chart(df["Risk_Level"].value_counts())

    st.subheader("📈 Top Churn Drivers")

    if importance_df is not None:

        col1, col2 = st.columns(2)

        # Top 5 features increasing churn
        top_positive = importance_df.head(5)

        # Top 5 features reducing churn
        top_negative = importance_df.tail(5)

        with col1:
            st.markdown("### 🔺 Increases Churn Risk")
            st.dataframe(top_positive)

        with col2:
            st.markdown("### 🔻 Reduces Churn Risk")
            st.dataframe(top_negative)