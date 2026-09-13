"""Haas public-deg admissions vs CleanStint numbers (2026 AUS/SIL/MONZA races)."""
import json
import numpy as np, pandas as pd
from scipy.stats import theilslopes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT/"data/processed/laps_2026_Australia_R.csv",
         ROOT/"data/laps_2026_Silverstone_R.csv",
         ROOT/"data/laps_2026_Monza_R.csv"]
HAAS = ("OCO", "BEA", "31", "87")
try:
    metrics = json.loads((ROOT/"data/processed/metrics_2026.json").read_text())
except Exception:
    metrics = {}

def slope(g):
    g = g.sort_values("tyre_age")
    x = g["tyre_age"].to_numpy(float); y = g["lap_time_s"].to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 8: return None
    return float(theilslopes(y[m], x[m])[0])

rows = []
for f in FILES:
    if not f.exists(): continue
    df = pd.read_csv(f)
    if "is_representative" in df.columns:
        rep = df["is_representative"].astype(str).str.lower().isin(["true","1","yes"])
        if rep.any(): df = df[rep]
    ev = f.stem.replace("laps_2026_","").replace("_R","")
    st = []
    for (drv, stint, comp), g in df.groupby(["driver","stint","compound"]):
        s = slope(g)
        if s is None: continue
        st.append(dict(drv=drv, stint=stint, comp=comp, laps=len(g),
                       age_max=int(g["tyre_age"].max()), wear=s*1000.0,
                       start=int(g["tyre_age"].min())))
    if not st: continue
    S = pd.DataFrame(st)
    field = S[~S.drv.isin(HAAS)].groupby("comp")["wear"].median()
    stops = S.groupby("drv")["stint"].nunique() - 1
    for _, r in S[S.drv.isin(HAAS)].iterrows():
        fm = field.get(r.comp, np.nan)
        cliff = (metrics.get("cliff_windows") or {}).get(r.comp)
        crossed = (cliff is not None) and (r.age_max >= cliff[0])
        rows.append(dict(event=ev, driver=r.drv, stint=int(r.stint), comp=r.comp,
                         laps=r.laps, haas_wear_ms=round(r.wear,1),
                         field_wear_ms=round(fm,1), delta_ms=round(r.wear-fm,1),
                         haas_stops=int(stops.get(r.drv, np.nan)),
                         field_stops=float(stops[~stops.index.isin(HAAS)].median()) if len(stops[~stops.index.isin(HAAS)]) else np.nan,
                         cliff_window=f"{cliff[0]}-{cliff[1]}" if cliff else "-",
                         crossed_cliff="YES" if crossed else "no",
                         our_box_lap=cliff[0] if cliff else "-"))
out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
print("\nINTERPRET (pre-registered): delta_ms > +5 => Haas deg worse than field (matches Komatsu).")
print("crossed_cliff YES with actual stint running past our_box_lap => we would have called the stop earlier.")
