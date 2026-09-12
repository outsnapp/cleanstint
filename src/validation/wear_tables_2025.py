"""validate_2025.py — bulletproof cross-year check on teammate's 2025 dataset."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

NAMES = ["year","circuit","session","driver","driver_number","team","lap_number",
         "stint","compound","tyre_age","flag_unknown","lap_time","s1","s2","s3",
         "pos_or_gap","track_status","col18","is_valid_window","is_pit_lap",
         "is_green","fuel_burned_kg","fuel_corrected_lap_time","harvest_limit_mj",
         "air_temp","track_temp","is_wet"]

# 1. Read CSV safely
df = pd.read_csv("laps.csv", header=None, names=NAMES, low_memory=False)

# 2. FORCE numeric types (fixes the str vs int error)
for col in ["lap_time", "tyre_age", "lap_number", "stint"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. FORCE boolean types safely
for col in ["is_wet", "is_pit_lap", "is_green"]:
    df[col] = df[col].astype(str).str.lower().isin(["true", "1.0", "1"])

# 4. Filter out the noise
df = df[df.lap_time.notna() & (~df.is_wet) & (~df.is_pit_lap) & df.is_green].copy()
race = df[df.session == "R"].copy()

print("2025 race laps used:", len(race), "| circuits:", sorted(race.circuit.unique()))

rows = []
for (circ, comp), g in race.groupby(["circuit","compound"]):
    g = g.sort_values(["driver","stint","tyre_age"])
    if g.tyre_age.max() < 10 or len(g) < 30:
        continue
    
    y = g.lap_time.values
    # Stint demeaning (removes driver pace + fuel load level)
    stint_means = g.groupby(["driver", "stint"])["lap_time"].transform("mean")
    y = y - stint_means.values
    
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(g.tyre_age.values, y)
    ages = np.arange(1, int(g.tyre_age.max())+1)
    vals = iso.predict(ages)
    
    rate = np.diff(vals, prepend=vals[0])
    rate[:5] = 0  # ignore warmup
    hi = min(25, len(ages))
    wear = (vals[hi-1]-vals[4])/(hi-1-4) if hi > 5 else 0
    
    # Cliff detection
    thr = np.median(rate[5:15]) + max(0.05, 3*np.std(rate[5:15])) if len(rate) > 15 else 0.1
    idx = np.where(rate > thr)[0]
    cliff = int(ages[idx[0]]) if len(idx) else None
    
    rows.append([circ, comp, len(g), int(g.tyre_age.max()), round(wear,4), cliff])

out = pd.DataFrame(rows, columns=["circuit","compound","laps","max_age","wear_s_per_lap","cliff_lap"])
print("\n=== 2025 Cross-Year Validation ===")
print(out.sort_values(["compound","circuit"]).to_string(index=False))
print("\nMean wear by compound (2025 races, stint-demeaned):")
print(out.groupby("compound")["wear_s_per_lap"].mean().round(4).to_string())
