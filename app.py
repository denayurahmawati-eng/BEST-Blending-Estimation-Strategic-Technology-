import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
from pulp import *

# ================= CONFIG =================
st.set_page_config(
    page_title="BEST - Coal Blending",
    layout="wide",
    page_icon="⛏️"
)

# ================= CSS =================
st.markdown("""
<style>
body {background-color: #0e1117;}
h1, h2, h3 {color: #00ADB5;}
.block-container {padding-top: 2rem;}

.stButton>button {
    background-color: #00ADB5;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #007B83;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<h1 style='text-align: center;'>⛏️ BEST</h1>
<h4 style='text-align: center;'>Blending Estimation Strategic Technology</h4>
<hr>
""", unsafe_allow_html=True)

# ================= DEFAULT DATA =================
default_data = pd.DataFrame({
    "Jenis": ["MT 47-1", "MT 47-3", "BB 51-2", "BB 51-4"],
    "Kalori": [4528, 4449, 5010, 5026],
    "TM": [27.87, 28.96, 27.75, 27.78],
    "Ash": [5.15, 5.66, 4.83, 4.14],
    "TS": [0.62, 0.55, 0.64, 0.65],
    "Stok": [255100, 305900, 194850, 200950]
})

# ================= SESSION STATE =================
if "df" not in st.session_state:
    st.session_state.df = default_data.copy()

# ================= LAYOUT =================
col1, col2 = st.columns([2,1])

# ================= DATA INPUT =================
with col1:
    st.markdown("### 📊 Data Batubara")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)

    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True
    )

    if st.button("🔄 Reset Data"):
        st.session_state.df = default_data.copy()

# ================= PARAMETER =================
with col2:
    st.markdown("### ⚙️ Parameter Blending")

    target_kalori = st.number_input("Target Kalori", 4000, 6000, 4800)
    max_tm = st.number_input("Max TM (%)", 0.0, 50.0, 30.0)
    max_ash = st.number_input("Max Ash (%)", 0.0, 20.0, 6.0)
    max_ts = st.number_input("Max TS (%)", 0.0, 5.0, 0.7)

# ================= VALIDASI =================
df = st.session_state.df

if df.empty:
    st.error("❌ Data kosong!")
    st.stop()

if df.isnull().values.any():
    st.error("❌ Data tidak boleh kosong!")
    st.stop()

# ================= BUTTON =================
if st.button("🚀 Jalankan Optimasi"):

    with st.spinner("🔄 Menghitung blending terbaik..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

    try:
        # ================= MODEL =================
        model = LpProblem("Coal_Blending", LpMaximize)

        x = LpVariable.dicts("blend", range(len(df)), lowBound=0)

        total = lpSum([x[i] for i in range(len(df))])

        model += total

        model += lpSum(x[i]*df.loc[i, "Kalori"] for i in range(len(df))) / total >= target_kalori
        model += lpSum(x[i]*df.loc[i, "TM"] for i in range(len(df))) / total <= max_tm
        model += lpSum(x[i]*df.loc[i, "Ash"] for i in range(len(df))) / total <= max_ash
        model += lpSum(x[i]*df.loc[i, "TS"] for i in range(len(df))) / total <= max_ts

        for i in range(len(df)):
            model += x[i] <= df.loc[i, "Stok"]

        model.solve()

        # ================= OUTPUT =================
        if model.status == 1:

            result = np.array([x[i].varValue for i in range(len(df))])
            total_ton = result.sum()

            persen = result / total_ton * 100

            st.markdown("## 📈 Hasil Optimasi")

            col3, col4 = st.columns(2)

            with col3:
                st.success(f"Kalori: {(result*df['Kalori']).sum()/total_ton:.2f}")
                st.info(f"TM: {(result*df['TM']).sum()/total_ton:.2f}%")
                st.warning(f"Ash: {(result*df['Ash']).sum()/total_ton:.2f}%")
                st.error(f"TS: {(result*df['TS']).sum()/total_ton:.2f}%")

            with col4:
                fig = px.pie(
                    names=df["Jenis"],
                    values=result,
                    title="Komposisi Blending"
                )
                st.plotly_chart(fig, use_container_width=True)

            hasil_df = pd.DataFrame({
                "Jenis": df["Jenis"],
                "Tonase": result,
                "Persentase (%)": persen
            })

            st.markdown("### 📋 Detail Komposisi")
            st.dataframe(hasil_df, use_container_width=True)

        else:
            st.error("❌ Tidak ditemukan solusi blending")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
