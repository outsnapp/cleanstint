import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import theilslopes
DATA, OUT = Path("data"), Path("data/processed")
stints = []
for ev in ["Monza", "Austria", "Silverstone"]:
    f = DATA / f"laps_2026_{ev}_R.csv"
    if not f.exists():
        print("skip", f); continue
    df = pd.read_csv(f)
    if "tyre_age" not in df.columns and "tyre_life" in df.columns:
        df["tyre_age"] = df["tyre_life"]
    if "lap_time_fuel_corr_s" in df.columns:
        df["lt"] = pd.to_numeric(df["lap_time_fuel_corr_s"], errors="coerce")
    elif "fuel_kg_burned" in df.columns:
        df["lt"] = pd.to_numeric(df["lap_time_s"], errors="coerce") + 0.03*pd.to_numeric(df["fuel_kg_burned"], errors="coerce")
    else:
        df["lt"] = pd.to_numeric(df["lap_time_s"], errors="coerce")
    if "is_representative" in df.columns:
        df = df[df["is_representative"].astype(str).str.lower().isin(["true","1","yes"])]
    df = df.dropna(subset=["lt","tyre_age"]); df = df[df["tyre_age"] >= 2]
    for (drv, st, comp), g in df.groupby(["driver","stint","compound"]):
        if len(g) < 8: continue
        x = g["tyre_age"].to_numpy(float); y = g["lt"].to_numpy(float)
        s, ic, _, _ = theilslopes(y, x)
        stints.append(dict(comp=str(comp).upper(), x=x, y=y, slope=float(s), ic=float(ic)))
S = pd.DataFrame([{k: v for k, v in s.items() if k in ("comp","slope")} for s in stints])
print(S.groupby("comp")["slope"].agg(["count","median"]).round(4).to_string())
metrics = {"wear_rate_s_per_lap_by_compound": {}, "cliff_windows": {},
           "cliff_not_observed_in_session": [], "clean_curves": {}}
for comp, g in S.groupby("comp"):
    wear = float(g["slope"].median()); sub = [s for s in stints if s["comp"] == comp]
    metrics["wear_rate_s_per_lap_by_compound"][comp] = round(wear, 4)
    best = (0.0, None)
    for k in range(8, 17):
        jumps = []
        for s in sub:
            e, l = s["x"] < k, s["x"] >= k
            if e.sum() >= 4 and l.sum() >= 4:
                se = theilslopes(s["y"][e], s["x"][e])[0]; sl = theilslopes(s["y"][l], s["x"][l])[0]
                jumps.append(float(sl - se))
        if jumps:
            med = float(np.median(jumps))
            if med > best[0]: best = (med, k)
    jump, k = best
    xs = list(range(1, 19))
    if jump >= 0.03 and k:
        metrics["cliff_windows"][comp] = [k, k + 3]
        early = float(np.median([theilslopes(s["y"][s["x"] < k], s["x"][s["x"] < k])[0] for s in sub if (s["x"] < k).sum() >= 4]))
        late = early + jump
    else:
        metrics["cliff_not_observed_in_session"].append(comp)
        early = late = wear
    base = float(np.median([s["ic"] for s in sub]))
    ys, cur = [], base
    for i, xv in enumerate(xs):
        if i > 0: cur += (late if (k and xv > k) else early)
        ys.append(round(cur, 3))
    metrics["clean_curves"][comp] = {"xs": xs, "vals": ys}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "metrics_2026.json").write_text(json.dumps(metrics, indent=1))
print("wrote", OUT / "metrics_2026.json")
print("cliffs:", metrics["cliff_windows"], "| not observed:", metrics["cliff_not_observed_in_session"])
