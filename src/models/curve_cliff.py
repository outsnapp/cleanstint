"""
causal_model.py v2 — real-data Double ML + cliff window + 5/10/15 lap forecast
Usage: python causal_model.py --gp Australia
"""
import argparse, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

CONTROLS = ["fuel_in_stint_kg", "te_index", "traffic_flag", "air_temp", "track_temp", "driver_id",
            "e_deploy_lap_mj", "e_harvest_lap_mj", "soc_min_lap_mj",
            "empty_clip_duration_s", "superclip_duration_s", "lift_coast_duration_s"]


def load(gp):
    prac, race = [], None
    import os
    for f in glob.glob(f"data/laps_*_{gp.replace(' ', '')}_*.csv"):
        if os.path.getsize(f) < 100:  # skip basically empty files
            print(f"  Skipping empty file: {f}")
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            if str(df["session"].iloc[0]).startswith("FP"):
                prac.append(df)
            else:
                race = df
        except Exception as e:
            print(f"  Error reading {f}: {e}")
    prac = pd.concat(prac, ignore_index=True) if prac else None
    return prac, race


def cliff_window(curve_ages, curve_vals):
    rate = np.diff(curve_vals, prepend=curve_vals[0])
    # Ignore the first 3 laps (out-lap and tyre warm-up phase)
    rate[:5] = 0.0  
    early = rate[5:min(13, len(rate))]
    base = np.median(early) if len(early) else 0.0
    thr = base + max(0.03, 3.0 * np.std(early)) if len(early) else 0.05
    idx = np.where(rate > thr)[0]
    if len(idx) == 0:
        return int(curve_ages[-1]), int(curve_ages[-1]) + 3
    s = int(curve_ages[idx[0]])
    return max(1, s - 1), s + 2


def main(gp):
    prac, race = load(gp)
    if prac is None:
        raise SystemExit("No practice CSVs. Run build_dataset.py first.")
    cats = sorted(prac["driver"].unique())
    prac["driver_id"] = pd.Categorical(prac["driver"], categories=cats).codes
    if race is not None:
        race["driver_id"] = pd.Categorical(race["driver"], categories=cats).codes
    if "is_representative" in prac.columns:
        prac = prac[prac["is_representative"] == 1].copy()
    if race is not None and "is_representative" in race.columns:
        race = race[race["is_representative"] == 1].copy()

    prac[CONTROLS] = prac[CONTROLS].fillna(prac[CONTROLS].median())
    X = prac[CONTROLS].values
    y = prac["lap_time_s"].values
    age = prac["tyre_age"].values

    # ---- Double ML residualization ----
    mA = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)
    mA.fit(X, y)
    resid_y = y - mA.predict(X)
    mT = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0)
    mT.fit(X, age)
    resid_t = age - mT.predict(X)
    lr = LinearRegression().fit(resid_t.reshape(-1, 1), resid_y)
    print(f"\nCausal wear rate (confounder-free): {lr.coef_[0]:+.4f} s/lap")

    # ---- clean monotone wear curve per compound ----
    curves, sigmas, cliffs = {}, {}, {}
    ages = np.arange(1, int(prac["tyre_age"].max()) + 1)
    for comp, g in prac.groupby("compound"):
        Xc = g[CONTROLS].fillna(0.0).values
        rc = g["lap_time_s"].values - mA.predict(Xc)
        rc = rc - pd.Series(rc).groupby(g["stint"].to_numpy()).transform("mean").to_numpy()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(g["tyre_age"].values, rc)
        curves[comp] = iso
        sigmas[comp] = float(np.std(rc - iso.predict(g["tyre_age"].values)))
        cliffs[comp] = cliff_window(ages, iso.predict(ages))
        print(f"{comp}: cliff window Lap {cliffs[comp][0]}–{cliffs[comp][1]} "
              f"(sigma {sigmas[comp]:.3f}s)")

    # ---- Q3: forecast +5/+10/+15 laps with 95% band ----
    curve_points = {c: {"ages": [int(a) for a in ages], "vals": [round(float(v), 3) for v in curves[c].predict(ages)]} for c in curves}
    wear_rates = {}
    for comp in curves:
        hi = min(16, int(ages.max()))
        wear_rates[comp] = round(float(curves[comp].predict([hi])[0] - curves[comp].predict([5])[0]) / (hi - 5), 4) if hi > 5 else None

    print("\nForecast table (current age -> +5/+10/+15 lap delta, 95% band):")
    forecast = {}
    for comp in curves:
        c = curves[comp]
        for a0 in [5, 10, 15]:
            row = {}
            for h in [5, 10, 15]:
                d = float(c.predict([a0 + h])[0] - c.predict([a0])[0])
                row[f"+{h}"] = [round(d, 3), round(d - 1.96 * sigmas[comp], 3),
                                round(d + 1.96 * sigmas[comp], 3)]
            forecast[f"{comp}@{a0}"] = row
            print(f"  {comp} age {a0}: " +
                  ", ".join(f"+{h}: {v[0]:+.2f}s [{v[1]:+.2f},{v[2]:+.2f}]"
                            for h, v in row.items()))

    # ---- frozen-model race validation ----
    not_obs = [c for c in cliffs if cliffs[c][0] == int(ages[-1])]
    metrics = {"n_practice_laps": int(len(prac)),
               "causal_wear_rate_s_per_lap": round(float(lr.coef_[0]), 4),
               "wear_rate_s_per_lap_by_compound": wear_rates,
               "clean_curves": curve_points,
               "cliff_windows": {c: list(cliffs[c]) for c in cliffs},
               "cliff_not_observed_in_session": not_obs}
    if race is not None:
        clean = race[race["traffic_flag"] == 0].copy()
        med = clean.groupby("driver")["lap_time_s"].transform("median")
        n_before = len(clean)
        clean = clean[(clean["lap_time_s"] < med + 5.0) & (clean["lap_time_s"] > med - 8.0)]
        n_anom = n_before - len(clean)
        Xr = clean[CONTROLS].fillna(prac[CONTROLS].median()).values
        clean["resid"] = clean["lap_time_s"].values - mA.predict(Xr)
        errs, base_errs, cliff_errs = [], [], []
        for (drv, stint), g in clean.groupby(["driver", "stint"]):
            g = g.sort_values("tyre_age")
            comp = g["compound"].iloc[0]
            if comp not in curves or len(g) < 6:
                continue
            pred = curves[comp].predict(g["tyre_age"].values)
            e = (g["resid"].values - g["resid"].values.mean()) - (pred - pred.mean())
            errs.extend(np.abs(e))
            # baseline: raw linear fit from practice
            gp_ = prac[prac["compound"] == comp]
            bl = LinearRegression().fit(gp_["tyre_age"].values.reshape(-1, 1),
                                        gp_["lap_time_s"].values)
            base_errs.extend(np.abs(g["lap_time_s"].values -
                                    bl.predict(g["tyre_age"].values.reshape(-1, 1))
                                    - (g["lap_time_s"].mean() -
                                       bl.predict(g["tyre_age"].values.reshape(-1, 1)).mean())))
            obs = cliff_window(g["tyre_age"].values, g["resid"].values)
            pr = cliffs[comp]
            cliff_errs.append(abs(0.5 * (obs[0] + obs[1]) - 0.5 * (pr[0] + pr[1])))
        metrics.update({
            "cleanstint_MAE_s_per_lap": round(float(np.mean(errs)), 3),
            "baseline_MAE_s_per_lap": round(float(np.mean(base_errs)), 3),
            "cliff_error_laps": round(float(np.mean(cliff_errs)), 2) if cliff_errs else None,
            "n_race_laps_validated": int(len(clean)),
            "n_anomaly_laps_excluded_SC_VSC": int(n_anom),
            "MAE_definition": "within-stint pace-shape error (level removed), both models",
        })
        print("\n--- FROZEN-MODEL RACE VALIDATION ---")
        for k, v in metrics.items():
            print(f"{k}: {v}")

        # plot for the deck (matches your submission slide)
        plt.figure(figsize=(10, 6))
        plt.scatter(clean["tyre_age"], clean["resid"], s=18, alpha=0.6,
                    label="Actual race laps (confounder-removed)")
        for comp in curves:
            plt.plot(ages, curves[comp].predict(ages), lw=2.5,
                     label=f"CleanStint predicted curve ({comp})")
            plt.axvspan(*cliffs[comp], alpha=0.12, color="red")
        plt.xlabel("Tyre age (laps)"); plt.ylabel("Clean lap-time signal (s)")
        plt.title("Predicted vs Actual Race Pace (frozen model)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig("predicted_vs_actual.png", dpi=200)
        print("[SAVED] predicted_vs_actual.png")

    json.dump(metrics, open("metrics.json", "w"), indent=2)
    json.dump(forecast, open("forecast.json", "w"), indent=2)
    print("[SAVED] metrics.json, forecast.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="Australia")
    args = ap.parse_args()
    main(args.gp)