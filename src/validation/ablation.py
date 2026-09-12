"""ablation.py — proves whether the battery proxy earns its place."""
import glob, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

BASE = ["fuel_in_stint_kg", "te_index", "traffic_flag", "air_temp", "track_temp", "driver_id"]
BATT = ["e_deploy_lap_mj", "e_harvest_lap_mj", "soc_min_lap_mj",
        "empty_clip_duration_s", "superclip_duration_s", "lift_coast_duration_s"]


def load(gp):
    prac, race = [], None
    for f in glob.glob(f"data/laps_*_{gp.replace(' ', '')}_*.csv"):
        df = pd.read_csv(f)
        if df.empty:
            continue
        if str(df["session"].iloc[0]).startswith("FP"):
            prac.append(df)
        else:
            race = df
    if not prac:
        return None, None
    prac = pd.concat(prac, ignore_index=True)
    cats = sorted(prac["driver"].unique())
    prac["driver_id"] = pd.Categorical(prac["driver"], categories=cats).codes
    if race is not None:
        race["driver_id"] = pd.Categorical(race["driver"], categories=cats).codes
    if "is_representative" in prac.columns:
        prac = prac[prac["is_representative"] == 1].copy()
    if race is not None and "is_representative" in race.columns:
        race = race[race["is_representative"] == 1].copy()
    return prac, race


def race_mae(cols, prac, race):
    prac = prac.copy()
    prac[cols] = prac[cols].fillna(prac[cols].median())
    X = prac[cols].values
    y = prac["lap_time_s"].values
    mA = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    mA.fit(X, y)
    curves = {}
    for comp, g in prac.groupby("compound"):
        r = g["lap_time_s"].values - mA.predict(g[cols].values)
        curves[comp] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(
            g["tyre_age"].values, r)
    if race is None:
        return np.nan, 0
    race = race.copy()
    race[cols] = race[cols].fillna(prac[cols].median())
    clean = race[race["traffic_flag"] == 0]
    errs = []
    for comp, g in clean.groupby("compound"):
        if comp not in curves or len(g) < 5:
            continue
        resid = g["lap_time_s"].values - mA.predict(g[cols].values)
        pred = curves[comp].predict(g["tyre_age"].values)
        e = (resid - resid.mean()) - (pred - pred.mean())
        errs.extend(np.abs(e))
    return (float(np.mean(errs)) if errs else np.nan), len(errs)


if __name__ == "__main__":
    gp = sys.argv[1] if len(sys.argv) > 1 else "Australia"
    prac, race = load(gp)
    if prac is None:
        raise SystemExit("No data. Run build_dataset.py first.")
    mae_base, n = race_mae(BASE, prac, race)
    mae_full, n = race_mae(BASE + BATT, prac, race)
    print(f"\n=== ABLATION: {gp} | {n} clean race laps ===")
    print(f"Without battery proxy : MAE = {mae_base:.3f} s/lap")
    print(f"With    battery proxy : MAE = {mae_full:.3f} s/lap")
    d = mae_base - mae_full
    print(f"Delta                 : {d:+.3f} s/lap")
    if d > 0.01:
        print("VERDICT: battery proxy IMPROVES race prediction. Cite the delta.")
    elif d < -0.01:
        print("VERDICT: battery proxy HURTS here. Report it with the sensitivity table.")
    else:
        print("VERDICT: NEUTRAL on current data. Value = 2026-readiness, report honestly.")
