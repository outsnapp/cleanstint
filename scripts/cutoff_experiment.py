"""Phase-2 cutoff: freeze priors at Silverstone 2026 -> score held-out Monza 2026 Haas.
TRAIN = all-2025 (cleaned) + 2026 Australia R + 2026 Silverstone R.
TEST  = 2026 Monza R (Haas isolated). Estimator identical to team_prior v2.
Verdict rule is printed BEFORE any test number is computed (pre-registered)."""
import json, glob
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
HAAS_CODES = {"OCO", "BEA", "31", "87"}

def load(p):
    df = pd.read_csv(p, on_bad_lines="skip")
    if "event" not in df.columns:
        import re
        match = re.search(r"20\d{2}[_\-\s]*(.+?)[_\-\s]*(?:R|Race|FP\d|Q|Quali)", Path(p).stem, re.IGNORECASE)
        df["event"] = match.group(1).replace("_", " ").title() if match else Path(p).stem
    if "rainfall" in df.columns:
        df = df[~df["rainfall"].astype(str).str.lower().isin(["true", "1"])]
    if "is_green" in df.columns:
        df = df[df["is_green"].astype(str).str.lower().isin(["true", "1"])]
    if "is_representative" in df.columns:
        rep = df["is_representative"].astype(str).str.lower().isin(["true", "1", "yes"])
        if rep.any(): df = df[rep]
    if "team" not in df.columns or df["team"].isna().all():
        df = df.copy()
        df["team"] = df["driver"].astype(str).str.strip().apply(
            lambda d: "Haas F1 Team" if d in HAAS_CODES else "Field")
        print("[team column missing -> mapped from driver codes]")
    return df

def slopes(df, tag):
    tc = "lap_time_fuel_corr_s" if "lap_time_fuel_corr_s" in df.columns else "lap_time_s"
    ac = "tyre_life" if "tyre_life" in df.columns else "tyre_age"
    print(f"[{tag}] time col = {tc} | age col = {ac}")
    df = df[df[ac] >= 2].dropna(subset=[tc])
    rows = []
    for (ev, drv, st, comp, team), g in df.groupby(["event", "driver", "stint", "compound", "team"]):
        if len(g) < 6: continue
        x = g[ac].to_numpy(float); y = g[tc].to_numpy(float)
        m = y < np.percentile(y, 85)
        if m.sum() < 6: continue
        try: s = theilslopes(y[m], x[m])[0]
        except Exception: continue
        rows.append(dict(event=ev, driver=drv, compound=comp, team=team, slope=s, n=int(m.sum())))
    return pd.DataFrame(rows)

def is_haas(s): return s.astype(str).str.contains("Haas", case=False, na=False)

def offsets(sl):
    fm = sl.groupby("compound")["slope"].median()
    sl = sl.copy()
    sl["off_ms"] = (sl["slope"] - sl["compound"].map(fm)) * 1000.0
    return sl

# ---------------- TRAIN ----------------
files = [str(p) for p in dict.fromkeys(
    [str(D / "processed" / "laps_2025_all.csv")] +
    sorted(glob.glob(str(D / "**" / "laps_2026_Australia_R.csv"), recursive=True)) +
    sorted(glob.glob(str(D / "**" / "laps_2026_Silverstone_R.csv"), recursive=True))
) if Path(p).exists()]
print("TRAIN files:", files)
tr = offsets(slopes(pd.concat([load(p) for p in files], ignore_index=True), "train"))
h = tr[is_haas(tr.team)]
pred_comp = h.groupby("compound")["off_ms"].median()
pred = float(h["off_ms"].median()) if len(h) else np.nan
print(f"TRAIN stints: {len(tr)} | Haas stints: {len(h)}")
print("PREDICTED Haas offset (ms/lap) per compound, from history only:")
print(pred_comp.round(1).to_string())
print(f"PREDICTED composite: {pred:+.1f} ms/lap")
print()
print("PRE-REGISTERED RULE (fixed before seeing Monza):")
print("  PASS           = composite same sign as actual AND |pred-actual| <= 60 ms/lap")
print("  DIRECTION-ONLY = same sign, error > 60 ms/lap")
print("  MISS           = opposite sign")
print("  VALUE TEST     = |pred-actual| < |0-actual|  (team prior beats 'Haas = field' naive)")
print()

# ---------------- TEST ----------------
tfiles = sorted(glob.glob(str(D / "**" / "laps_2026_Monza_R.csv"), recursive=True))
print("TEST files:", tfiles)
te = offsets(slopes(load(tfiles[0]), "test"))
ht = te[is_haas(te.team)]
act_comp = ht.groupby("compound")["off_ms"].median()
act = float(ht["off_ms"].median()) if len(ht) else np.nan
print(f"TEST stints: {len(te)} | Haas stints: {len(ht)}")
print("ACTUAL Monza 2026 Haas offset (ms/lap) per compound:")
print(act_comp.round(1).to_string())
print(f"ACTUAL composite: {act:+.1f} ms/lap")
print()
rows = []
for c in sorted(set(pred_comp.index) | set(act_comp.index)):
    p = pred_comp.get(c, np.nan); a = act_comp.get(c, np.nan)
    rows.append(dict(compound=c, predicted=round(float(p), 1), actual=round(float(a), 1),
                     err=round(abs(p - a), 1) if pd.notna(p) and pd.notna(a) else np.nan,
                     naive_err=round(abs(a), 1) if pd.notna(a) else np.nan))
print(pd.DataFrame(rows).to_string(index=False))

err = abs(pred - act)
verdict = ("PASS" if err <= 60 else "DIRECTION-ONLY") if np.sign(pred) == np.sign(act) else "MISS"
value = "PRIOR BEATS NAIVE" if err < abs(act) else "NAIVE BEATS PRIOR"
print(f"\nCOMPOSITE: predicted {pred:+.1f} | actual {act:+.1f} | err {err:.1f} ms/lap")
print(f"VERDICT: {verdict} | VALUE: {value}")
json.dump(dict(pred_per_compound={k: float(v) for k, v in pred_comp.round(1).items()},
               act_per_compound={k: float(v) for k, v in act_comp.round(1).items()},
               pred_composite=round(pred, 1), act_composite=round(act, 1), err=round(err, 1),
               verdict=verdict, value=value, n_train_stints=int(len(tr)), n_train_haas=int(len(h)),
               n_test_stints=int(len(te)), n_test_haas=int(len(ht))),
          open(ROOT / "scripts" / "cutoff_result.json", "w"), indent=2)
print("saved scripts/cutoff_result.json")
