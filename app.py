import streamlit as st
import pandas as pd
import pickle
import numpy as np
import altair as alt

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

# --- Threshold Slider ---
st.markdown("---")
threshold = st.slider("Select Threshold", 0.1, 0.9, 0.35)

if uploaded_file is not None:
    original_df = pd.read_csv(uploaded_file)
    df = original_df.copy()

    # Preprocessing (must match training preprocessing)
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Get probabilities
    probabilities = model.predict_proba(df)[:, 1]

    predictions = (probabilities > threshold).astype(int)

    # Risk classification
    def risk_category(prob):
        if prob < threshold:
            return "Low"
        elif prob < threshold + ((1-threshold)/2):
            return "Medium"
        else:
            return "High"

    risk_levels = [risk_category(p) for p in probabilities]

    original_df["Churn_Probability"] = probabilities
    original_df["Risk_Level"] = risk_levels

    # --- Summary Metrics ---
    st.markdown("---")
    st.subheader("� Summary Metrics")

    total_customers = len(original_df)
    churn_percentage = (predictions.sum() / total_customers) * 100
    average_risk = probabilities.mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("High Risk %", f"{churn_percentage:.1f}%")
    col3.metric("Average Risk", f"{average_risk:.2f}")

    # --- Prediction Table ---
    st.markdown("---")
    st.subheader("🔍 Prediction Table")

    display_columns = []
    if "customerID" in original_df.columns:
        display_columns.append("customerID")

    display_columns.extend(["Churn_Probability", "Risk_Level"])

    st.dataframe(original_df[display_columns], use_container_width=True)

    # --- Risk Distribution ---
    st.markdown("---")
    st.subheader("📊 Risk Distribution")
    st.bar_chart(original_df["Risk_Level"].value_counts())

    # --- Key Drivers Section ---
    st.markdown("---")
    st.subheader("📈 Top Features Contributing to Churn")

    if importance_df is not None:
        importance_df['Abs_Coefficient'] = importance_df['Coefficient'].abs()
        top_10_features = importance_df.sort_values(by="Abs_Coefficient", ascending=False).head(10)

        top_10_features['Impact'] = np.where(top_10_features['Coefficient'] > 0, 'Increases Risk', 'Decreases Risk')
        
        chart = alt.Chart(top_10_features).mark_bar().encode(
            x=alt.X('Coefficient:Q', title='Coefficient Value'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            color=alt.Color('Impact:N', scale=alt.Scale(domain=['Increases Risk', 'Decreases Risk'], range=['#dc3545', '#198754'])),
            tooltip=['Feature', 'Coefficient', 'Impact']
        ).properties(height=400)
        
        st.altair_chart(chart, use_container_width=True)