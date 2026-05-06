# =========================================================
# OPTIMASI BLENDING BATUBARA UNTUK EKSPOR BANGLADESH
# Gabungan Model LP & NLP DENGAN VISUALISASI HASIL
# =========================================================

import streamlit as st
import time
import numpy as np
import pulp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Data Batubara
# ---------------------------------------------------------
# Data untuk NLP (numpy array)
CV_NLP = np.array([4528, 4449, 5010, 5026])      # Kalori (ar)
TM_NLP = np.array([27.87, 28.96, 27.75, 27.78])  # Total Moisture (%)
Ash_NLP = np.array([5.15, 5.66, 4.83, 4.14])     # Abu (%)
TS_NLP = np.array([0.62, 0.55, 0.64, 0.65])      # Total Sulfur (%)
Stock_NLP = np.array([255100, 305900, 194850, 200950])  # Stok tersedia (ton)
Batubara_Names = ['MT 47-STOCK 1', 'MT 47-STOCK 3','BB 51-STOCK 2', 'BB 51-STOCK 4']

# Data untuk LP (Pandas DataFrame)
data_LP = {
    'Nama': Batubara_Names,
    'Kalori_ar': [4528, 4449, 5010, 5026],
    'TM_ar': [27.87, 28.96, 27.75, 27.78],
    'Ash_adb': [5.15, 5.66, 4.83, 4.14],
    'TS_adb': [0.62, 0.55, 0.64, 0.65],
    'Stock': [255100, 305900, 194850, 200950] 
}
df_LP = pd.DataFrame(data_LP)

# Target spesifikasi (konsisten untuk kedua model)
Target_CV_min = 4800
Target_TM_max = 28.0
Target_Ash_max = 8.0
Target_TS_max = 0.8
Total_ton = 55000
min_fraction_LP = 0.10 # Batasan minimal proporsi untuk model LP

# Variabel global untuk menyimpan hasil
results_comp = {}

# =========================================================
# A. OPTIMASI NON-LINEAR PROGRAMMING (NLP)
# =========================================================

# ---------------------------------------------
# 2. Fungsi Tujuan NLP (Memaksimalkan Kalori rata-rata - Penalti Variasi)
# ---------------------------------------------
def objective_nlp(x):
    total_cv = np.sum(CV_NLP * x) / np.sum(x)
    penalty = 0.10 * np.sum((x - np.mean(x))**2)
    return -(total_cv - penalty)

# ---------------------------------------------
# 3. Fungsi Kendala NLP
# ---------------------------------------------
def total_constraint(x):
    return np.sum(x) - Total_ton

def tm_constraint_nlp(x):
    return Target_TM_max - (np.sum(TM_NLP * x) / np.sum(x))

def ash_constraint_nlp(x):
    return Target_Ash_max - (np.sum(Ash_NLP * x) / np.sum(x))

def ts_constraint_nlp(x):
    return Target_TS_max - (np.sum(TS_NLP * x) / np.sum(x))

def cv_constraint_nlp(x):
    return (np.sum(CV_NLP * x) / np.sum(x)) - Target_CV_min

constraints_nlp = [
    {'type': 'eq', 'fun': total_constraint},
    {'type': 'ineq', 'fun': tm_constraint_nlp},
    {'type': 'ineq', 'fun': ash_constraint_nlp},
    {'type': 'ineq', 'fun': ts_constraint_nlp},
    {'type': 'ineq', 'fun': cv_constraint_nlp} 
]

bounds_nlp = [(0, Stock_NLP[i]) for i in range(len(Stock_NLP))]
x0_nlp = np.array([Total_ton / len(CV_NLP)] * len(CV_NLP)) 

# ---------------------------------------------
# 4. Jalankan Optimasi NLP
# ---------------------------------------------
result_nlp = minimize(objective_nlp, x0_nlp, bounds=bounds_nlp, constraints=constraints_nlp, method='SLSQP')

print("="*60)
print("             HASIL OPTIMASI BLENDING (NON-LINEAR)")
print("============================================================")

if result_nlp.success:
    x_opt_nlp = result_nlp.x
    total_cv_nlp = np.sum(CV_NLP * x_opt_nlp) / np.sum(x_opt_nlp)
    total_tm_nlp = np.sum(TM_NLP * x_opt_nlp) / np.sum(x_opt_nlp)
    total_ash_nlp = np.sum(Ash_NLP * x_opt_nlp) / np.sum(x_opt_nlp)
    total_ts_nlp = np.sum(TS_NLP * x_opt_nlp) / np.sum(x_opt_nlp)
    
    # Simpan hasil NLP
    results_comp['NLP'] = {
        'CV': total_cv_nlp, 'TM': total_tm_nlp, 
        'Ash': total_ash_nlp, 'TS': total_ts_nlp
    }

    print("\n=== Proporsi Batubara (NLP) ===")
    for i, ton in enumerate(x_opt_nlp):
        nama = f"{Batubara_Names[i]} (Stok: {Stock_NLP[i]:,})"
        print(f"{nama}: {ton:,.2f} ton ({ton/Total_ton*100:.2f}%)")

    print("\n=== KUALITAS CAMPURAN AKHIR (NLP) ===")
    print(f"Kalori (ar): {total_cv_nlp:.2f} kcal/kg (Target: >= {Target_CV_min})")
    print(f"Total Moisture: {total_tm_nlp:.2f} % (Target: <= {Target_TM_max})")
    print(f"Ash: {total_ash_nlp:.2f} % (Target: <= {Target_Ash_max})")
    print(f"Total Sulfur: {total_ts_nlp:.2f} % (Target: <= {Target_TS_max})")

else:
    print("❌ Tidak ditemukan solusi optimal NLP. Periksa batasan atau data input.")
    results_comp['NLP'] = {'CV': np.nan, 'TM': np.nan, 'Ash': np.nan, 'TS': np.nan}


# =========================================================
# B. OPTIMASI LINEAR PROGRAMMING (LP)
# =========================================================

# ---------------------------------------------------------
# 5. Inisialisasi dan Kendala LP
# ---------------------------------------------------------
model_lp = pulp.LpProblem("Blending_Batubara_LP", pulp.LpMaximize)
n_batubara = len(df_LP)
x_lp = [pulp.LpVariable(f"x_{i+1}", lowBound=0, upBound=df_LP.loc[i, 'Stock']) for i in range(n_batubara)]

# Fungsi Tujuan LP: Maksimalkan Total Kalori
model_lp += pulp.lpSum(x_lp[i] * df_LP.loc[i, 'Kalori_ar'] for i in range(n_batubara)), "Maksimalkan_Total_Kalori"

# Kendala
model_lp += pulp.lpSum(x_lp[i] for i in range(n_batubara)) == Total_ton, "Total_Tonase"
model_lp += pulp.lpSum(x_lp[i] * df_LP.loc[i, 'TM_ar'] for i in range(n_batubara)) <= Target_TM_max * Total_ton, "Batas_TM"
model_lp += pulp.lpSum(x_lp[i] * df_LP.loc[i, 'Ash_adb'] for i in range(n_batubara)) <= Target_Ash_max * Total_ton, "Batas_Ash"
model_lp += pulp.lpSum(x_lp[i] * df_LP.loc[i, 'TS_adb'] for i in range(n_batubara)) <= Target_TS_max * Total_ton, "Batas_TS"
model_lp += pulp.lpSum(x_lp[i] * df_LP.loc[i, 'Kalori_ar'] for i in range(n_batubara)) >= Target_CV_min * Total_ton, "Batas_Kalori"

min_ton_lp = min_fraction_LP * Total_ton
for i in range(n_batubara):
    model_lp += x_lp[i] >= min_ton_lp, f"Minimal_{df_LP.loc[i, 'Nama']}"

# ---------------------------------------------------------
# 6. Jalankan Optimasi LP
# ---------------------------------------------------------
model_lp.solve(pulp.PULP_CBC_CMD(msg=0))

# ---------------------------------------------------------
# 7. Tampilkan Hasil LP
# ---------------------------------------------------------
print("\n" + "="*60)
print("              HASIL OPTIMASI BLENDING (LINEAR)")
print("============================================================")

if pulp.LpStatus[model_lp.status] == 'Optimal':
    x_opt_lp = [x.value() for x in x_lp]

    total_cv_lp = sum(x_opt_lp[i] * df_LP.loc[i, 'Kalori_ar'] for i in range(n_batubara)) / Total_ton
    total_tm_lp = sum(x_opt_lp[i] * df_LP.loc[i, 'TM_ar'] for i in range(n_batubara)) / Total_ton
    total_ash_lp = sum(x_opt_lp[i] * df_LP.loc[i, 'Ash_adb'] for i in range(n_batubara)) / Total_ton
    total_ts_lp = sum(x_opt_lp[i] * df_LP.loc[i, 'TS_adb'] for i in range(n_batubara)) / Total_ton

    # Simpan hasil LP
    results_comp['LP'] = {
        'CV': total_cv_lp, 'TM': total_tm_lp, 
        'Ash': total_ash_lp, 'TS': total_ts_lp
    }
    
    print("\n=== Proporsi Batubara (LP) ===")
    for i in range(n_batubara):
        tonase = x_opt_lp[i]
        persen = (tonase / Total_ton) * 100
        nama = f"{df_LP.loc[i, 'Nama']} (Stok: {df_LP.loc[i, 'Stock']:,})"
        print(f"{nama}: {tonase:,.2f} ton ({persen:.2f}%)")

    print("\n=== KUALITAS BLENDING CAMPURAN (LP) ===")
    print(f"Kalori (ar): {total_cv_lp:.2f} kcal/kg (Target: >= {Target_CV_min})")
    print(f"Total Moisture (ar): {total_tm_lp:.2f} % (Target: <= {Target_TM_max})")
    print(f"Ash (adb): {total_ash_lp:.2f} % (Target: <= {Target_Ash_max})")
    print(f"Total Sulfur (adb): {total_ts_lp:.2f} % (Target: <= {Target_TS_max})")
    print("\n✅ Optimasi LP dan NLP selesai.")
    
else:
    print("❌ Tidak ditemukan solusi optimal LP. Periksa batasan atau data input.")
    results_comp['LP'] = {'CV': np.nan, 'TM': np.nan, 'Ash': np.nan, 'TS': np.nan}


# =========================================================
# C. VISUALISASI PERBANDINGAN HASIL
# =========================================================

def plot_comparison(results, targets):
    """Membuat 4 grafik batang terpisah untuk perbandingan kualitas."""
    
    if not all(k in results for k in ['LP', 'NLP']):
        print("\n⚠️ Tidak dapat membuat grafik: Hasil dari salah satu model tidak tersedia.")
        return

    df_results = pd.DataFrame(results).T
    
    # 1. CV (Maksimalkan, target minimum)
    plt.figure(figsize=(10, 5))
    plt.bar(df_results.index, df_results['CV'], color=['#4CAF50', '#2196F3'])
    plt.axhline(targets['CV_min'], color='r', linestyle='--', label=f'Target Min ({targets["CV_min"]})')
    plt.title('1. Perbandingan Kalori (CV) Campuran (kcal/kg)')
    plt.ylabel('Kalori (ar)')
    plt.ylim(min(df_results['CV']) - 50, max(df_results['CV']) + 50)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    for i, val in enumerate(df_results['CV']):
        plt.text(i, val + 5, f'{val:.2f}', ha='center', fontweight='bold')
    
    # 2. TM (Minimalkan, target maksimum)
    plt.figure(figsize=(10, 5))
    plt.bar(df_results.index, df_results['TM'], color=['#FF9800', '#FFC107'])
    plt.axhline(targets['TM_max'], color='r', linestyle='--', label=f'Target Max ({targets["TM_max"]})')
    plt.title('2. Perbandingan Total Moisture (TM) Campuran (%)')
    plt.ylabel('Total Moisture (%)')
    plt.ylim(min(df_results['TM']) - 1, targets['TM_max'] + 1)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    for i, val in enumerate(df_results['TM']):
        plt.text(i, val + 0.1, f'{val:.2f}', ha='center', fontweight='bold')
        
    # 3. TS (Minimalkan, target maksimum)
    plt.figure(figsize=(10, 5))
    plt.bar(df_results.index, df_results['TS'], color=['#9C27B0', '#E1BEE7'])
    plt.axhline(targets['TS_max'], color='r', linestyle='--', label=f'Target Max ({targets["TS_max"]})')
    plt.title('3. Perbandingan Total Sulfur (TS) Campuran (%)')
    plt.ylabel('Total Sulfur (%)')
    plt.ylim(0.5, targets['TS_max'] + 0.1)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    for i, val in enumerate(df_results['TS']):
        plt.text(i, val + 0.01, f'{val:.2f}', ha='center', fontweight='bold')

    # 4. Ash (Minimalkan, target maksimum)
    plt.figure(figsize=(10, 5))
    plt.bar(df_results.index, df_results['Ash'], color=['#00BCD4', '#80DEEA'])
    plt.axhline(targets['Ash_max'], color='r', linestyle='--', label=f'Target Max ({targets["Ash_max"]})')
    plt.title('4. Perbandingan Ash Campuran (%)')
    plt.ylabel('Ash (%)')
    plt.ylim(min(df_results['Ash']) - 1, targets['Ash_max'] + 1)
    plt.legend()
    plt.grid(axis='y', linestyle=':')
    for i, val in enumerate(df_results['Ash']):
        plt.text(i, val + 0.1, f'{val:.2f}', ha='center', fontweight='bold')

    plt.show()

# Panggil fungsi visualisasi
targets = {
    'CV_min': Target_CV_min, 'TM_max': Target_TM_max, 
    'Ash_max': Target_Ash_max, 'TS_max': Target_TS_max
}
plot_comparison(results_comp, targets)
