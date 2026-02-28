import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AquaSight AI", layout="wide", page_icon="💧")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open('water_potability.pkl', 'rb'))

# ---------------- SIDEBAR ----------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=100)
st.sidebar.title("AquaSight AI")
choice = st.sidebar.radio("Navigation", ["Citizen Portal", "Authority Dashboard"])

# ---------------- WQI FUNCTION ----------------
def calculate_wqi(ph, solids, turbidity, conductivity, organic_carbon):
    score = 0
    score += 20 if 6.5 <= ph <= 8.5 else 5
    score += 20 if solids <= 500 else 5
    score += 20 if turbidity <= 5 else 5
    score += 20 if conductivity <= 400 else 5
    score += 20 if organic_carbon <= 10 else 5
    return score


# =========================================================
# ================== CITIZEN PORTAL =======================
# =========================================================

if choice == "Citizen Portal":

    st.title("💧 Personal Water Quality Analyst")
    st.write("Hybrid AI system combining Machine Learning + WQI scoring.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")

        ph = st.slider("pH Level", 0.0, 14.0, 7.0)
        solids = st.number_input("Total Dissolved Solids (TDS)", 0, 60000, 500)
        turbidity = st.slider("Turbidity", 0.0, 10.0, 3.0)
        conductivity = st.number_input("Conductivity", 0, 1000, 400)
        organic = st.slider("Organic Carbon", 0.0, 30.0, 10.0)

        # Remaining parameters (fixed demo values)
        hardness = 150
        chloramines = 7.0
        sulfate = 300
        trihalomethanes = 60

    with col2:
        if st.button("Run Hybrid Analysis"):

            # -------- Prepare Input --------
            features = [ph, hardness, solids, chloramines,
                        sulfate, conductivity, organic,
                        trihalomethanes, turbidity]

            input_array = np.array(features).reshape(1, -1)

            # -------- ML Prediction --------
            prob = model.predict_proba(input_array)[0][0] * 100

            # -------- WQI --------
            wqi = calculate_wqi(ph, solids, turbidity, conductivity, organic)

            # -------- Hybrid Calculation --------
            ml_component = prob * 0.6
            wqi_component = (100 - wqi) * 0.4
            hybrid_risk = ml_component + wqi_component

            # -------- Results Display --------
            st.subheader("Analysis Results")

            st.metric("Hybrid Risk Score", f"{hybrid_risk:.2f}%")

            st.write(f"ML Unsafe Risk: {prob:.2f}%")
            st.write(f"WQI Score: {wqi}")

            st.write("### Breakdown")
            st.write(f"AI Contribution (60% weight): {ml_component:.2f}")
            st.write(f"WQI Contribution (40% weight): {wqi_component:.2f}")

            # -------- Final Status --------
            if hybrid_risk < 40:
                st.success("✅ Status: Safe / Low Risk")
            elif hybrid_risk < 70:
                st.warning("⚠️ Status: Moderate Risk (Filtration Recommended)")
            else:
                st.error("🚨 Status: High Risk (DO NOT CONSUME)")


# =========================================================
# ================= AUTHORITY DASHBOARD ===================
# =========================================================

else:

    st.title("🏛️ Water Authority Command Center")

    df = pd.read_csv('water_potability.csv')

    st.write("### Regional Potability Distribution")

    fig = px.pie(df,
                 names='Potability',
                 color='Potability',
                 color_discrete_map={0: 'red', 1: 'green'},
                 title="Overall Safety Statistics")

    st.plotly_chart(fig)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Parameter Correlation Heatmap")

    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig2)