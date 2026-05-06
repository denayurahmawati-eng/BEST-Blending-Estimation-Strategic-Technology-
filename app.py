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

.metric-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.status-ok {
    color: #00ff9f;
    font-weight: bold;
}

.status-bad {
    color: #ff4b4b;
    font-weight: bold;
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

if "df" not in st.session_state:
    st.session_state.df = default_data.copy()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Parameter Blending")

target_kalori = st.sidebar.number_input("Target Kalori", 4000, 6000, 4800)
max_tm = st.sidebar.number_input("Max TM (%)", 0.0, 50.0, 30.0)
max_ash = st.sidebar.number_input("Max Ash (%)", 0.0, 20.0, 6.0)
max_ts = st.sidebar.number_input("Max TS (%)", 0.0, 5.0, 0.7)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

if st.sidebar.button("🔄 Reset Data"):
    st.session_state.df = default_data.copy()

run = st.sidebar.button("🚀 Jalankan Optimasi")

# ================= MAIN =================
st.markdown("### 📊 Data Batubara")

st.session_state.df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True
)

df = st.session_state.df

# ================= VALIDASI =================
if df.empty:
    st.error("❌ Data kosong!")
    st.stop()

if df.isnull().values.any():
    st.error("❌ Data tidak boleh kosong!")
    st.stop()

# ================= RUN =================
if run:

    with st.spinner("🔄 Menghitung..."):
        time.sleep(1)

    model = LpProblem("Coal_Blending", LpMaximize)

    x = LpVariable.dicts("blend", range(len(df)), lowBound=0)
    total = lpSum(x[i] for i in range(len(df)))

    model += total

    model += lpSum(x[i]*df.loc[i,"Kalori"] for i in range(len(df))) / total >= target_kalori
    model += lpSum(x[i]*df.loc[i,"TM"] for i in range(len(df))) / total <= max_tm
    model += lpSum(x[i]*df.loc[i,"Ash"] for i in range(len(df))) / total <= max_ash
    model += lpSum(x[i]*df.loc[i,"TS"] for i in range(len(df))) / total <= max_ts

    for i in range(len(df)):
        model += x[i] <= df.loc[i,"Stok"]

    model.solve()

    if model.status == 1:

        result = np.array([x[i].varValue for i in range(len(df))])
        total_ton = result.sum()

        if total_ton == 0:
            st.error("❌ Total tonase 0")
            st.stop()

        kalori = (result*df['Kalori']).sum()/total_ton
        tm = (result*df['TM']).sum()/total_ton
        ash = (result*df['Ash']).sum()/total_ton
        ts = (result*df['TS']).sum()/total_ton

        # ================= KPI =================
        st.markdown("## 📈 Hasil Blending")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Kalori", f"{kalori:.2f}")
        c2.metric("TM (%)", f"{tm:.2f}")
        c3.metric("Ash (%)", f"{ash:.2f}")
        c4.metric("TS (%)", f"{ts:.2f}")

        # ================= STATUS =================
        st.markdown("### 🧪 Status")

        if kalori >= target_kalori and tm <= max_tm and ash <= max_ash and ts <= max_ts:
            st.markdown("<p class='status-ok'>✅ MEMENUHI SPESIFIKASI</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='status-bad'>❌ TIDAK MEMENUHI SPESIFIKASI</p>", unsafe_allow_html=True)

        # ================= CHART =================
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(names=df["Jenis"], values=result, title="Komposisi Blending")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(x=df["Jenis"], y=result, title="Distribusi Tonase")
            st.plotly_chart(fig2, use_container_width=True)

        # ================= TABLE =================
        hasil_df = pd.DataFrame({
            "Jenis": df["Jenis"],
            "Tonase": result,
            "Persentase (%)": result/total_ton*100
        })

        st.markdown("### 📋 Detail")
        st.dataframe(hasil_df, use_container_width=True)

        # ================= DOWNLOAD =================
        csv = hasil_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "hasil_blending.csv", "text/csv")

    else:
        st.error("❌ Tidak ditemukan solusi")import streamlit as st
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

.metric-card {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.status-ok {
    color: #00ff9f;
    font-weight: bold;
}

.status-bad {
    color: #ff4b4b;
    font-weight: bold;
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

if "df" not in st.session_state:
    st.session_state.df = default_data.copy()

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Parameter Blending")

target_kalori = st.sidebar.number_input("Target Kalori", 4000, 6000, 4800)
max_tm = st.sidebar.number_input("Max TM (%)", 0.0, 50.0, 30.0)
max_ash = st.sidebar.number_input("Max Ash (%)", 0.0, 20.0, 6.0)
max_ts = st.sidebar.number_input("Max TS (%)", 0.0, 5.0, 0.7)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)

if st.sidebar.button("🔄 Reset Data"):
    st.session_state.df = default_data.copy()

run = st.sidebar.button("🚀 Jalankan Optimasi")

# ================= MAIN =================
st.markdown("### 📊 Data Batubara")

st.session_state.df = st.data_editor(
    st.session_state.df,
    num_rows="dynamic",
    use_container_width=True
)

df = st.session_state.df

# ================= VALIDASI =================
if df.empty:
    st.error("❌ Data kosong!")
    st.stop()

if df.isnull().values.any():
    st.error("❌ Data tidak boleh kosong!")
    st.stop()

# ================= RUN =================
if run:

    with st.spinner("🔄 Menghitung..."):
        time.sleep(1)

    model = LpProblem("Coal_Blending", LpMaximize)

    x = LpVariable.dicts("blend", range(len(df)), lowBound=0)
    total = lpSum(x[i] for i in range(len(df)))

    model += total

    model += lpSum(x[i]*df.loc[i,"Kalori"] for i in range(len(df))) / total >= target_kalori
    model += lpSum(x[i]*df.loc[i,"TM"] for i in range(len(df))) / total <= max_tm
    model += lpSum(x[i]*df.loc[i,"Ash"] for i in range(len(df))) / total <= max_ash
    model += lpSum(x[i]*df.loc[i,"TS"] for i in range(len(df))) / total <= max_ts

    for i in range(len(df)):
        model += x[i] <= df.loc[i,"Stok"]

    model.solve()

    if model.status == 1:

        result = np.array([x[i].varValue for i in range(len(df))])
        total_ton = result.sum()

        if total_ton == 0:
            st.error("❌ Total tonase 0")
            st.stop()

        kalori = (result*df['Kalori']).sum()/total_ton
        tm = (result*df['TM']).sum()/total_ton
        ash = (result*df['Ash']).sum()/total_ton
        ts = (result*df['TS']).sum()/total_ton

        # ================= KPI =================
        st.markdown("## 📈 Hasil Blending")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Kalori", f"{kalori:.2f}")
        c2.metric("TM (%)", f"{tm:.2f}")
        c3.metric("Ash (%)", f"{ash:.2f}")
        c4.metric("TS (%)", f"{ts:.2f}")

        # ================= STATUS =================
        st.markdown("### 🧪 Status")

        if kalori >= target_kalori and tm <= max_tm and ash <= max_ash and ts <= max_ts:
            st.markdown("<p class='status-ok'>✅ MEMENUHI SPESIFIKASI</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='status-bad'>❌ TIDAK MEMENUHI SPESIFIKASI</p>", unsafe_allow_html=True)

        # ================= CHART =================
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(names=df["Jenis"], values=result, title="Komposisi Blending")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.bar(x=df["Jenis"], y=result, title="Distribusi Tonase")
            st.plotly_chart(fig2, use_container_width=True)

        # ================= TABLE =================
        hasil_df = pd.DataFrame({
            "Jenis": df["Jenis"],
            "Tonase": result,
            "Persentase (%)": result/total_ton*100
        })

        st.markdown("### 📋 Detail")
        st.dataframe(hasil_df, use_container_width=True)

        # ================= DOWNLOAD =================
        csv = hasil_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "hasil_blending.csv", "text/csv")

    else:
        st.error("❌ Tidak ditemukan solusi")
