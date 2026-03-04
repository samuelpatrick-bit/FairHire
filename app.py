import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

# Page config (TOP la mattum irukanum)
st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")

st.title("🚀 FairHire – AI Recruitment Bias Auditor")
st.sidebar.header("Upload Hiring Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    data = pd.read_csv(uploaded_file, engine="python")

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    if all(col in data.columns for col in ['Gender','Experience','TestScore','Selected']):

        # Convert gender to numeric
        data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})

        X = data[['Gender','Experience','TestScore']]
        y = data['Selected']
        x# Remove rows with missing values
        data = data.dropna()

        X = data[['Gender','Experience','TestScore']]
        y = data['Selected']
        model = LogisticRegression()
        model.fit(X,y)
        predictions = model.predict(X)

        # -------------------------
        # 📈 PERFORMANCE METRICS
        # -------------------------
        st.subheader("📈 Model Performance")

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(accuracy,3))
        col2.metric("Precision", round(precision,3))
        col3.metric("Recall", round(recall,3))
        col4.metric("F1 Score", round(f1,3))

        # -------------------------
        # 🔍 CONFUSION MATRIX
        # -------------------------
        st.subheader("🔍 Confusion Matrix")

        cm = confusion_matrix(y, predictions)

        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        st.pyplot(fig)

        # -------------------------
        # ⚖ BIAS METRICS
        # -st.markdown("## ⚖ AI Bias Analysis Dashboard")

dp = demographic_parity_difference(y, predictions, sensitive_features=data['Gender'])
eo = equalized_odds_difference(y, predictions, sensitive_features=data['Gender'])

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Demographic Parity")
    st.metric(label="Difference Score", value=round(dp,3))
    
    if abs(dp) > 0.1:
        st.error("High Risk of Gender Bias 🚨")
    else:
        st.success("Bias Within Safe Limit ✅")

with col2:
    st.markdown("### 🎯 Equalized Odds")
    st.metric(label="Difference Score", value=round(eo,3))
    
    if abs(eo) > 0.1:
        st.error("Unequal Error Rates Detected 🚨")
    else:
        st.success("Balanced Error Distribution ✅")

        st.markdown("---")
        
        col5, col6 = st.columns(2)
        col5.metric("Demographic Parity Difference", round(dp,3))
        col6.metric("Equalized Odds Difference", round(eo,3))

        if abs(dp) > 0.1:
            st.error("⚠ High Gender Bias Detected")
        else:
            st.success("✅ Bias Within Safe Range")

        # -------------------------
        # 📊 SELECTION RATE GRAPH
        # -------------------------
        st.subheader("📊 Selection Rate by Gender")

        selection_rate = data.groupby('Gender')['Selected'].mean()

        fig2, ax2 = plt.subplots()
        ax2.bar(['Male','Female'], selection_rate)
        ax2.set_ylabel("Selection Rate")
        st.pyplot(fig2)

        # -------------------------
        # 📥 DOWNLOAD REPORT
        # -------------------------
        report = f"""
        FairHire Bias Report

        Accuracy: {accuracy}
        Precision: {precision}
        Recall: {recall}
        F1 Score: {f1}

        Demographic Parity Difference: {dp}
        Equalized Odds Difference: {eo}
        """

        st.download_button("Download Bias Report", report, file_name="bias_report.txt")

    else:
        st.warning("Dataset must contain: Gender, Experience, TestScore, Selected")

# -------------------------
# 📝 NLP Resume Bias Checker
# -------------------------

from textblob import TextBlob

st.sidebar.header("Resume Bias Checker (NLP)")
resume_text = st.sidebar.text_area("Paste Resume Text")

if resume_text:

    st.subheader("🧠 NLP Resume Analysis")

    blob = TextBlob(resume_text)

    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    col1, col2 = st.columns(2)

    col1.metric("Sentiment Polarity", round(polarity,3))
    col2.metric("Subjectivity", round(subjectivity,3))

    if polarity > 0.3:
        st.success("Positive & Confident Tone ✅")
    elif polarity < -0.3:
        st.warning("Negative Tone Detected ⚠")
    else:
        st.info("Neutral Tone ℹ")

    if subjectivity > 0.6:
        st.warning("Highly Subjective Language")
    else:
        st.success("Professional & Objective Language")
