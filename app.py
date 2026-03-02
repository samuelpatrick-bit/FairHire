import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

st.set_page_config(page_title="FairHire - AI Bias Detector", layout="wide")
st.title("🚀 FairHire – AI Recruitment Bias Auditor")
st.title("Welcome to Our Bias Auditor")
st.sidebar.header("Upload Hiring Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(
    uploaded_file,
    encoding='ISO-8859-1',
    sep=None,
    engine='python',
    on_bad_lines='skip'
)  
    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    if all(col in data.columns for col in ['Gender','Experience','TestScore','Selected']):

        # Convert gender to numeric
        data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})

        X = data[['Gender','Experience','TestScore']]
        y = data['Selected']

        model = LogisticRegression()
        model.fit(X,y)
        predictions = model.predict(X)
        from sklearn.metrics import accuracy_score

        # Accuracy Calculation
        accuracy = accuracy_score(y, predictions)

        st.subheader("📊 Model Accuracy")
        st.write("Accuracy:", round(accuracy, 3))

        st.subheader("📈 Model Performance")

        # Bias Metrics
        dp = demographic_parity_difference(y, predictions, sensitive_features=data['Gender'])
        eo = equalized_odds_difference(y, predictions, sensitive_features=data['Gender'])

        col1, col2 = st.columns(2)
        col1.metric("Demographic Parity Difference", round(dp,3))
        col2.metric("Equalized Odds Difference", round(eo,3))

        if abs(dp) > 0.1:
            st.error("⚠ Gender Bias Detected")
        else:
            st.success("✅ No Significant Gender Bias")

        # Visualization
        st.subheader("📊 Selection Rate by Gender")
        selection_rate = data.groupby('Gender')['Selected'].mean()

        fig, ax = plt.subplots()
        ax.bar(['Male','Female'], selection_rate)
        ax.set_ylabel("Selection Rate")
        st.pyplot(fig)

        # Download Report
        report = f"""
        FairHire Bias Report

        Demographic Parity Difference: {dp}
        Equalized Odds Difference: {eo}
        """

        st.download_button("Download Bias Report", report, file_name="bias_report.txt")

    else:
        st.warning("Dataset must contain: Gender, Experience, TestScore, Selected columns")

# Resume Bias Checker
st.sidebar.header("Resume Bias Checker")
resume_text = st.sidebar.text_area("Paste Resume Text")

if resume_text:
    masculine_words = ["leader", "dominant", "competitive"]
    feminine_words = ["supportive", "empathetic", "nurturing"]

    masculine_count = sum(word in resume_text.lower() for word in masculine_words)
    feminine_count = sum(word in resume_text.lower() for word in feminine_words)

    st.subheader("📝 Resume Bias Analysis")
    st.write("Masculine-coded words:", masculine_count)
    st.write("Feminine-coded words:", feminine_count)

    if masculine_count > feminine_count:
        st.warning("⚠ Resume contains more masculine-coded words")
    elif feminine_count > masculine_count:
        st.warning("⚠ Resume contains more feminine-coded words")
    else:
        st.success("✅ Balanced Language Detected")
        