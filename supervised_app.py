
# supervised_app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

st.set_page_config(page_title="Supervised Customer Segmentation", layout="wide")
st.title("🎯 Supervised Customer Segmentation Dashboard")

uploaded_file = st.file_uploader("Upload labeled customer CSV file (with 'Segment' column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df.head())

    features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']
    target = 'Segment'

    if all(col in df.columns for col in features + [target]):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        st.subheader("✅ Model Accuracy")
        st.write(f"Accuracy: {accuracy:.2f}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        joblib.dump(model, "customer_segment_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        st.success("Model and scaler saved as .pkl files")

    else:
        st.error(f"CSV must contain the columns: {features + [target]}")
else:
    st.info("Please upload a labeled CSV file containing: Age, Income, SpendingScore, PurchaseFrequency, Segment")
