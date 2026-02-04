# =========================================================
# APLIKASI WEB OPTIMASI BLENDING BATUBARA
# IDENTIK DENGAN SKRIP PYTHON (LP & NLP)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import pulp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimasi Blending Batubara", layout="wide")
st.title("Selamat Datang di BEST (Blending Estimation Strategic Technology)")
st.markdown("Disini kita menggunakan Metode **Linear Programming (LP)** dan **Non-Linear Programming (NLP)**")

# =========================================================
# 1. INPUT DATA
# =========================================================
st.subheader("📋 Data Kualitas Batubara")

df = st.data_editor(
    pd.DataFrame({
        "Jenis": [
            "MT 47-STOCK 1",
            "MT 47-STOCK 3",
            "BB 51-STOCK 2",
            "BB 51-STOCK 4"
        ],
        "Kalori (ar)": [4528, 4449, 5010, 5026],
        "TM (%)": [27.87, 28.96, 27.75, 27.78],
        "Ash (%)": [5.15, 5.66, 4.83, 4.14],
        "TS (%)": [0.62, 0.55, 0.64, 0.65],
        "Stok (ton)": [255100, 305900, 194850, 200950]
    }),
    num_rows="dynamic",
    use_container_width=True
)

if df.isnull().values.any():
    st.warning("⚠️ Data belum lengkap")
    st.stop()

# =========================================================
# 2. TARGET BLENDING
# =========================================================
st.subheader("🎯 Target Spesifikasi")

col1, col2, col3 = st.columns(3)

with col1:
    Target_CV_min = st.number_input("Target Kalori Minimum", 4500, 6000, 4800)
    Total_ton = st.number_input("Total Tonase (ton)", 10000, 500000, 55000)

with col2:
    Target_TM_max = st.number_input("TM Maksimum (%)", 20.0, 35.0, 28.0)
    Target_Ash_max = st.number_input("Ash Maksimum (%)", 2.0, 15.0, 8.0)

with col3:
    Target_TS_max = st.number_input("TS Maksimum (%)", 0.2, 2.0, 0.8)
    min_fraction = st.slider("Fraksi Minimum LP (%)", 0, 20, 10) / 100

# =========================================================
# 3. PROSES OPTIMASI
# =========================================================
if st.button("🚀 Jalankan Optimasi"):

    CV = df["Kalori (ar)"].values
    TM = df["TM (%)"].values
    Ash = df["Ash (%)"].values
    TS = df["TS (%)"].values
    Stock = df["Stok (ton)"].values
    names = df["Jenis"].values
    n = len(df)

    # =====================================================
    # A. NLP (IDENTIK DENGAN SKRIP PYTHON)
    # =====================================================
    def objective_nlp(x):
        total_cv = np.sum(CV * x) / np.sum(x)
        penalty = 0.10 * np.sum((x - np.mean(x))**2)
        return -(total_cv - penalty)

    constraints_nlp = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - Total_ton},
        {'type': 'ineq', 'fun': lambda x: Target_TM_max - np.sum(TM*x)/np.sum(x)},
        {'type': 'ineq', 'fun': lambda x: Target_Ash_max - np.sum(Ash*x)/np.sum(x)},
        {'type': 'ineq', 'fun': lambda x: Target_TS_max - np.sum(TS*x)/np.sum(x)},
        {'type': 'ineq', 'fun': lambda x: np.sum(CV*x)/np.sum(x) - Target_CV_min}
    ]

    bounds_nlp = [(0, Stock[i]) for i in range(n)]
    x0 = np.array([Total_ton / n] * n)

    result_nlp = minimize(
        objective_nlp,
        x0,
        bounds=bounds_nlp,
        constraints=constraints_nlp,
        method="SLSQP"
    )

    # =====================================================
    # B. LP (IDENTIK DENGAN SKRIP PYTHON)
    # =====================================================
    model_lp = pulp.LpProblem("Blending_Batubara_LP", pulp.LpMaximize)
    x_lp = [
        pulp.LpVariable(f"x_{i}", lowBound=min_fraction*Total_ton, upBound=Stock[i])
        for i in range(n)
    ]

    model_lp += pulp.lpSum(x_lp[i] * CV[i] for i in range(n))
    model_lp += pulp.lpSum(x_lp) == Total_ton
    model_lp += pulp.lpSum(x_lp[i] * TM[i] for i in range(n)) <= Target_TM_max * Total_ton
    model_lp += pulp.lpSum(x_lp[i] * Ash[i] for i in range(n)) <= Target_Ash_max * Total_ton
    model_lp += pulp.lpSum(x_lp[i] * TS[i] for i in range(n)) <= Target_TS_max * Total_ton
    model_lp += pulp.lpSum(x_lp[i] * CV[i] for i in range(n)) >= Target_CV_min * Total_ton

    model_lp.solve(pulp.PULP_CBC_CMD(msg=0))

    # =====================================================
    # 4. OUTPUT
    # =====================================================
    st.subheader("📊 Hasil Optimasi")
    colA, colB = st.columns(2)

    # NLP OUTPUT
    with colA:
        st.markdown("### 🔵 NLP (Non-Linear Programming)")
        if result_nlp.success:
            x = result_nlp.x
            st.dataframe(pd.DataFrame({
                "Jenis": names,
                "Tonase (ton)": x,
                "Persentase (%)": x / Total_ton * 100
            }), use_container_width=True)

            st.write("**Kualitas Campuran NLP**")
            st.write(f"Kalori : {np.sum(CV*x)/np.sum(x):.2f}")
            st.write(f"TM     : {np.sum(TM*x)/np.sum(x):.2f}")
            st.write(f"Ash    : {np.sum(Ash*x)/np.sum(x):.2f}")
            st.write(f"TS     : {np.sum(TS*x)/np.sum(x):.2f}")
        else:
            st.error("NLP tidak menemukan solusi")

    # LP OUTPUT
    with colB:
        st.markdown("### 🟢 LP (Linear Programming)")
        if pulp.LpStatus[model_lp.status] == "Optimal":
            x = np.array([v.value() for v in x_lp])
            st.dataframe(pd.DataFrame({
                "Jenis": names,
                "Tonase (ton)": x,
                "Persentase (%)": x / Total_ton * 100
            }), use_container_width=True)

            st.write("**Kualitas Campuran LP**")
            st.write(f"Kalori : {np.sum(CV*x)/Total_ton:.2f}")
            st.write(f"TM     : {np.sum(TM*x)/Total_ton:.2f}")
            st.write(f"Ash    : {np.sum(Ash*x)/Total_ton:.2f}")
            st.write(f"TS     : {np.sum(TS*x)/Total_ton:.2f}")
        else:
            st.error("LP tidak menemukan solusi")

