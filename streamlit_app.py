import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load all models
model_raw = joblib.load("model_raw_scaled.pkl")
model_transformed = joblib.load("model_transformed_skew.pkl")
model_balanced = joblib.load("model_final_balanced.pkl")

# title
st.set_page_config(page_title="Intrusion Detection Model Dashboard", layout="wide")
st.title("🚨 Intrusion Detection System - Model Comparison")

# Sidebar: Model selection
st.sidebar.header("🔍 Select a Model Version")
model_choice = st.sidebar.radio("Model Version", 
    ["Raw Scaled", "Transformed Skew", "Final Balanced"])

# File upload
st.subheader("📁 Upload Test Data")
uploaded_file = st.file_uploader("Upload a CSV file with test rows", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)

    st.write("### 🧾 Uploaded Data Preview")
    st.dataframe(df_input.head())

    # Store label if available
    has_label = "Label" in df_input.columns
    if has_label:
        true_labels = df_input["Label"].values

    # Drop label and unwanted columns
    for col in ["Label", "Unnamed: 0"]:
        if col in df_input.columns:
            df_input.drop(columns=col, inplace=True)

    # Ensure the dataframe has more than 0 features
    if df_input.shape[1] <= 0:
        st.error(f"❌ The file should have at least one feature. Please upload a valid file.")
        st.stop()
    else:
        st.success(f"✔️ File uploaded with {df_input.shape[1]} features.")

        input_data = df_input.values

        # Model selection
        if model_choice == "Raw Scaled":
            model = model_raw
        elif model_choice == "Transformed Skew":
            model = model_transformed
        else:
            model = model_balanced

        # Predict
        try:
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
            st.stop()

        # Output predictions
        st.subheader("✅ Predictions")
        output = pd.DataFrame({
            "Prediction": prediction
        })

        if proba is not None:
            output["Confidence (Class 0)"] = np.round(proba[:, 0], 4)
            output["Confidence (Class 1)"] = np.round(proba[:, 1], 4)

        st.dataframe(output)

        # Evaluation
        if has_label:
            st.subheader("📊 Model Evaluation")

            # Accuracy
            acc = accuracy_score(true_labels, prediction)
            st.write(f"**Accuracy:** `{acc:.4f}`")

            # Classification report
            st.write("**Classification Report:**")
            report_str = classification_report(true_labels, prediction)
            st.text(report_str)


            # Confusion matrix
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(true_labels, prediction)
            cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
            st.dataframe(cm_df)

else:
    st.info("📤 Please upload a CSV file to begin.")
