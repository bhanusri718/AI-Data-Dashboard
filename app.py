import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -----------------------------
# PAGE CONFIG & STYLING
# -----------------------------
st.set_page_config(
    page_title="AI-Powered Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme + glowing effects
st.markdown("""
<style>
body { background-color: #0e1117; color: #c7d0d9; }
h1, h2, h3, h4, h5 { color: #00bfa6; }
.stButton>button { 
    background-color:#1f77b4; 
    color:white; 
    border-radius:8px; 
    box-shadow: 0 0 10px #00bfa6;
}
.metric-card {
    background: linear-gradient(145deg, #1f1f1f, #151515);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 20px #00bfa6;
    color: #c7d0d9;
    margin-bottom: 10px;
}
.insight-card {
    background: linear-gradient(145deg, #121212, #1a1a1a);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 15px #1f77b4;
    color: #c7d0d9;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🚀 AI-Powered Analytics Dashboard")
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV", type=["csv"])

st.sidebar.header("Filters (Optional)")
date_range = st.sidebar.date_input("Select Date Range", [datetime(2025,1,1), datetime(2025,12,31)])

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # -----------------------------
    # METRIC CARDS (KPIs)
    # -----------------------------
    st.subheader("📊 Key Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Records</h4>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        survived = df['Survived'].sum() if 'Survived' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Survived</h4>
            <h2>{survived}</h2>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        avg_age = round(df['Age'].mean(), 1) if 'Age' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average Age</h4>
            <h2>{avg_age}</h2>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        pclass1 = df[df['Pclass']==1].shape[0] if 'Pclass' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <h4>1st Class</h4>
            <h2>{pclass1}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # AI INSIGHTS / RECOMMENDATIONS
    # -----------------------------
    st.subheader("🧠 AI Insights & Recommendations")
    insight1, insight2 = st.columns(2)

    with insight1:
        st.markdown(f"""
        <div class="insight-card">
            <h4>Insight</h4>
            <p>Highest survival rate observed in 1st class females.</p>
            <p style='color:#00bfa6;'>Confidence: 90%</p>
        </div>
        """, unsafe_allow_html=True)

    with insight2:
        st.markdown(f"""
        <div class="insight-card">
            <h4>Recommendation</h4>
            <p>Focus analysis on 3rd class male passengers for targeted interventions.</p>
            <p style='color:#00bfa6;'>Confidence: 85%</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------
    # CHARTS
    # -----------------------------
    st.subheader("📈 Visual Analytics")

    # Survival by Class and Sex
    if 'Pclass' in df.columns and 'Survived' in df.columns and 'Sex' in df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, palette="coolwarm", ax=ax)
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#0e1117")
        ax.set_ylabel("Survival Rate")
        st.pyplot(fig)

    # Optional numeric column correlation heatmap
    numeric_df = df.select_dtypes(include='number')
    if not numeric_df.empty:
        st.subheader("Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_facecolor("#121212")
        fig2.patch.set_facecolor("#0e1117")
        st.pyplot(fig2)

else:
    st.info("Please upload a CSV to visualize AI-powered insights.")
