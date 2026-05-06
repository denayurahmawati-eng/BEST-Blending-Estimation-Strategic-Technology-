"""
coal_blending_lp.py
Model LP blending batubara: baca input.xlsx -> solve -> simpan hasil ke output.xlsx
Dependencies: pandas, pulp, openpyxl
Install: pip install pandas pulp openpyxl
"""

import pandas as pd
import pulp
import os

# ---------- Config / File names ----------
INPUT_FILE = "input.xlsx"
OUTPUT_FILE = "output_results.xlsx"

# ---------- Example data fallback (used if input.xlsx tidak ada) ----------
example_stocks = pd.DataFrame([
    {"Name": "A", "Calorific": 5100, "Ash": 3.79, "Sulfur": 0.77, "Moisture": 27.41, "AvailableTon": 10000},
    {"Name": "B", "Calorific": 4700, "Ash": 4.17,  "Sulfur": 0.61, "Moisture": 31.69, "AvailableTon": 15000},

])

example_targets = {
    "DemandTon": 55000,
    "Calorific_min": 4800,
    "Ash_max": 8.0,
    "Sulfur_max": 0.8,
    "Moisture_max": 12.0
}

# ---------- Read input.xlsx if exists ----------
if os.path.exists(INPUT_FILE):
    try:
        stocks = pd.read_excel(INPUT_FILE, sheet_name="stocks")
    except Exception as e:
        raise RuntimeError(f"Gagal membaca sheet 'stocks' dari {INPUT_FILE}: {e}")
    # read targets sheet if present
    try:
        targets_df = pd.read_excel(INPUT_FILE, sheet_name="targets")
        targets = {row['TargetName']: row['Value'] for _, row in targets_df.iterrows()}
    except Exception:
        targets = example_targets.copy()
        print("Sheet 'targets' tidak ditemukan atau tidak valid -> menggunakan default contoh.")
else:
    print(f"{INPUT_FILE} tidak ditemukan. Menggunakan data contoh internal.")
    stocks = example_stocks.copy()
    targets = example_targets.copy()

# Basic validation
required_cols = {"Name", "Calorific", "Ash", "Sulfur", "Moisture", "AvailableTon"}
if not required_cols.issubset(set(stocks.columns)):
    missing = required_cols - set(stocks.columns)
    raise ValueError(f"Kolom hilang di sheet 'stocks': {missing}")

# ---------- Parameter ----------
names = list(stocks['Name'])
cal = dict(zip(names, stocks['Calorific']))
ash = dict(zip(names, stocks['Ash']))
sulfur = dict(zip(names, stocks['Sulfur']))
moist = dict(zip(names, stocks['Moisture']))
avail = dict(zip(names, stocks['AvailableTon']))

demand = float(targets.get("DemandTon", example_targets["DemandTon"]))
cal_min = float(targets.get("Calorific_min", example_targets["Calorific_min"]))
ash_max = float(targets.get("Ash_max", example_targets["Ash_max"]))
sulfur_max = float(targets.get("Sulfur_max", example_targets["Sulfur_max"]))
moist_max = float(targets.get("Moisture_max", example_targets["Moisture_max"]))

# ---------- Model LP ----------
model = pulp.LpProblem("Coal_Blending", pulp.LpMinimize)

# Decision variables: proportion by ton (tons of each coal to use)
# We will model in TON basis: y_i = ton of coal i used. Then sum y_i = demand
y = {i: pulp.LpVariable(f"ton_{i}", lowBound=0, upBound=avail[i], cat="Continuous") for i in names}

# Constraints:
# 1) Meet demand (tons)
model += pulp.lpSum([y[i] for i in names]) == demand, "DemandConstraint"

# 2) Quality constraints (weighted average)
# average_calorific = sum(cal_i * y_i) / demand >= cal_min
model += pulp.lpSum([cal[i] * y[i] for i in names]) >= cal_min * demand, "Calorific_min"

# ash average <= ash_max
model += pulp.lpSum([ash[i] * y[i] for i in names]) <= ash_max * demand, "Ash_max"

# sulfur average <= sulfur_max
model += pulp.lpSum([sulfur[i] * y[i] for i in names]) <= sulfur_max * demand, "Sulfur_max"

# moisture average <= moist_max
model += pulp.lpSum([moist[i] * y[i] for i in names]) <= moist_max * demand, "Moisture_max"

# (Optional) You can add min usage constraints, integer constraints, or blending group constraints here.

# ---------- Solve ----------
# Default solver is CBC (included with pulp). If you have Gurobi/CPLEX, pulp will use it if configured.
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=60)
res = model.solve(solver)

status = pulp.LpStatus[model.status]
print("Solver status:", status)

# ---------- Collect results ----------
if status != "Optimal":
    print("Peringatan: solusi tidak optimal. Status:", status)

result_tons = {i: y[i].value() if y[i].value() is not None else 0.0 for i in names}
# convert to proportions
result_prop = {i: (result_tons[i] / demand) if demand > 0 else 0.0 for i in names}

# compute blended qualities
cal_mix = sum(cal[i] * result_tons[i] for i in names) / demand
ash_mix = sum(ash[i] * result_tons[i] for i in names) / demand
sulfur_mix = sum(sulfur[i] * result_tons[i] for i in names) / demand
moist_mix = sum(moist[i] * result_tons[i] for i in names) / demand

# prepare output dataframe
out_df = pd.DataFrame.from_records([
    {"Name": i,
     "TonUsed": result_tons[i],
     "Proportion": result_prop[i],
     "Calorific": cal[i],
     "Ash": ash[i],
     "Sulfur": sulfur[i],
     "Moisture": moist[i],
     "AvailableTon": avail[i]}
    for i in names
])

summary = pd.DataFrame([
    {"Metric": "DemandTon", "Value": demand},
    {"Metric": "Blended_Calorific_kcal/kg", "Value": cal_mix},
    {"Metric": "Blended_Ash_%", "Value": ash_mix},
    {"Metric": "Blended_Sulfur_%", "Value": sulfur_mix},
    {"Metric": "Blended_Moisture_%", "Value": moist_mix},
    {"Metric": "SolverStatus", "Value": status}
])

# ---------- Save results ----------
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    out_df.to_excel(writer, sheet_name="blend_details", index=False)
    summary.to_excel(writer, sheet_name="summary", index=False)
    stocks.to_excel(writer, sheet_name="input_stocks", index=False)
print(f"Hasil disimpan ke {OUTPUT_FILE}")

# ---------- Print concise summary ----------
print("\nHasil ringkas:")
for i in names:
    print(f"  {i}: {result_tons[i]:.1f} ton ({result_prop[i]*100:.2f}%)")
print(f"Blended Calorific = {cal_mix:.1f} kcal/kg")
print(f"Blended Ash = {ash_mix:.2f} %")
print(f"Blended Sulfur = {sulfur_mix:.3f} %")
print(f"Blended Moisture = {moist_mix:.2f} %")
