# =========================================================
# PERBANDINGAN LP vs NLP BLENDING BATUBARA
# + VISUALISASI PARAMETER KUALITAS
# =========================================================

import numpy as np
import pulp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =========================================================
# 1. DATA BATUBARA
# =========================================================
nama = ["Batubara A", "Batubara B"]

CV = np.array([5100, 4700])        # kcal/kg
TM = np.array([25.41, 28.69])      # %
Ash = np.array([3.79, 5.15])       # %
TS = np.array([0.77, 0.61])        # %
Stock = np.array([150000, 250000]) # ton

Total_ton = 55000

Target = {
    "CV_min": 4800,
    "TM_max": 28,
    "Ash_max": 8,
    "TS_max": 0.8
}

# =========================================================
# 2. MODEL NLP (Soft Constraint + Penalty λ)
# =========================================================
lambda_penalty = 0.10

def objective_nlp(x):
    cv_blend = np.sum(CV * x) / np.sum(x)
    penalty = lambda_penalty * np.sum((x - np.mean(x))**2)
    return -(cv_blend - penalty)

constraints_nlp = [
    {"type": "eq",   "fun": lambda x: np.sum(x) - Total_ton},
    {"type": "ineq", "fun": lambda x: Target["TM_max"]  - (np.sum(TM  * x) / np.sum(x))},
    {"type": "ineq", "fun": lambda x: Target["Ash_max"] - (np.sum(Ash * x) / np.sum(x))},
    {"type": "ineq", "fun": lambda x: Target["TS_max"]  - (np.sum(TS  * x) / np.sum(x))}
]

bounds = [(0, Stock[i]) for i in range(len(Stock))]
x0 = np.array([Total_ton / 2] * 2)

res_nlp = minimize(
    objective_nlp, x0,
    bounds=bounds,
    constraints=constraints_nlp,
    method="SLSQP"
)

x_nlp = res_nlp.x

# =========================================================
# 3. MODEL LP (Hard Constraint)
# =========================================================
model = pulp.LpProblem("LP_Blending", pulp.LpMaximize)

x_lp = [
    pulp.LpVariable(f"x_{i}", lowBound=0, upBound=Stock[i])
    for i in range(len(Stock))
]

# Fungsi tujuan
model += pulp.lpSum(x_lp[i] * CV[i] for i in range(2))

# Kendala
model += pulp.lpSum(x_lp) == Total_ton
model += pulp.lpSum(x_lp[i] * TM[i]  for i in range(2)) <= Target["TM_max"]  * Total_ton
model += pulp.lpSum(x_lp[i] * Ash[i] for i in range(2)) <= Target["Ash_max"] * Total_ton
model += pulp.lpSum(x_lp[i] * TS[i]  for i in range(2)) <= Target["TS_max"]  * Total_ton
model += pulp.lpSum(x_lp[i] * CV[i]  for i in range(2)) >= Target["CV_min"]  * Total_ton

model.solve(pulp.PULP_CBC_CMD(msg=0))
x_lp = np.array([v.value() for v in x_lp])

# =========================================================
# 4. FUNGSI HITUNG KUALITAS
# =========================================================
def kualitas(x):
    return {
        "CV":  np.sum(CV  * x) / np.sum(x),
        "TM":  np.sum(TM  * x) / np.sum(x),
        "Ash": np.sum(Ash * x) / np.sum(x),
        "TS":  np.sum(TS  * x) / np.sum(x)
    }

q_nlp = kualitas(x_nlp)
q_lp  = kualitas(x_lp)

# =========================================================
# 5. CETAK HASIL NUMERIK
# =========================================================
print("\n=========== HASIL NLP ===========")
for i in range(2):
    print(f"{nama[i]}: {x_nlp[i]:,.2f} ton ({x_nlp[i]/Total_ton*100:.2f}%)")

print("\nKualitas NLP:")
for k, v in q_nlp.items():
    print(f"{k}: {v:.2f}")

print("\n=========== HASIL LP ===========")
for i in range(2):
    print(f"{nama[i]}: {x_lp[i]:,.2f} ton ({x_lp[i]/Total_ton*100:.2f}%)")

print("\nKualitas LP:")
for k, v in q_lp.items():
    print(f"{k}: {v:.2f}")

# =========================================================
# 6. VISUALISASI PER PARAMETER (4 GRAFIK TERPISAH)
# =========================================================
labels = ["LP", "NLP"]
x = np.arange(len(labels))

# ---- CV ----
plt.figure()
plt.bar(x, [q_lp["CV"], q_nlp["CV"]])
plt.xticks(x, labels)
plt.ylabel("kcal/kg")
plt.title("Perbandingan Kalori (CV)")
plt.show()

# ---- TM ----
plt.figure()
plt.bar(x, [q_lp["TM"], q_nlp["TM"]])
plt.xticks(x, labels)
plt.ylabel("%")
plt.title("Perbandingan Total Moisture (TM)")
plt.show()

# ---- TS ----
plt.figure()
plt.bar(x, [q_lp["TS"], q_nlp["TS"]])
plt.xticks(x, labels)
plt.ylabel("%")
plt.title("Perbandingan Total Sulfur (TS)")
plt.show()

# ---- Ash ----
plt.figure()
plt.bar(x, [q_lp["Ash"], q_nlp["Ash"]])
plt.xticks(x, labels)
plt.ylabel("%")
plt.title("Perbandingan Ash")
plt.show()
