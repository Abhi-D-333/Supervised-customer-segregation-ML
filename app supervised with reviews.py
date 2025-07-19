import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Customer Segment Classifier", layout="wide")
st.title("🧠 Customer Segment Prediction (Supervised Learning)")

uploaded_file = st.file_uploader("Upload a CSV file with customer data including reviews", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    required_columns = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'Review', 'Segment']

    if all(col in df.columns for col in required_columns):
        st.subheader("📊 Segment Distribution")
        st.bar_chart(df['Segment'].value_counts())

        X = df[['Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'Review']]
        y = df['Segment']

        numeric_features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']
        text_feature = 'Review'

        preprocessor = ColumnTransformer(transformers=[
            ('num', make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), numeric_features),
            ('text', TfidfVectorizer(), text_feature)
        ])

        model = make_pipeline(preprocessor, RandomForestClassifier(random_state=42))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.subheader("✅ Model Accuracy")
        st.metric(label="Test Accuracy", value=f"{acc:.2%}")

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("🔍 Predict Segment for a New Customer")
        age = st.number_input("Age", 18, 100, 30)
        income = st.number_input("Income", 0, 1000000, 50000)
        score = st.slider("Spending Score", 0, 100, 50)
        frequency = st.slider("Purchase Frequency", 1, 30, 5)
        review = st.text_area("Customer Review")

        if st.button("Predict Segment"):
            new_data = pd.DataFrame([{
                'Age': age,
                'Income': income,
                'SpendingScore': score,
                'PurchaseFrequency': frequency,
                'Review': review
            }])
            prediction = model.predict(new_data)[0]
            st.success(f"🎯 Predicted Segment: {prediction}")

    else:
        st.error(f"CSV must contain columns: {', '.join(required_columns)}")
else:
    st.info("Please upload a CSV file.")

