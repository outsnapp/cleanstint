"""Battery-ablation inside the frozen Track-3 harness (2026 Silverstone+Monza).

Protocol (pre-registered, written before running):
  1. Reproduce his frozen baseline via core.test_2026 (core/ never edited).
  2. Attach CleanStint battery features (physics proxy) to 2026 race laps.
  3. Remove battery-attributable component from Model-A residuals (per-event Ridge).
  4. Re-score stints; compare wear slopes vs baseline.
Decision rule:
  corr(delta_wear, stint_deploy) < 0 and |mean delta| >= 2 ms/lap
      -> battery acts in physics-predicted direction: KEEP as electro-thermal correction
  within +/-2 ms/lap -> NEUTRAL: keep battery as reported 2026-readiness feature
  opposite direction -> battery OUT of wear path; keep for strategy layer only
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for p in ("core", "src/features", "src/ingest"):
    sys.path.insert(0, str(ROOT / p))

import test_2026 as t26

BATT = ["e_deploy_lap_mj", "e_harvest_lap_mj", "soc_min_lap_mj",
        "empty_clip_duration_s", "lift_coast_duration_s"]
KEYS = ["year", "event", "session", "driver", "stint", "compound"]


def attach_battery(df):
    out = df.copy()
    cache = ROOT / "data" / "battery_2026.csv"
    if cache.exists():
        batt = pd.read_csv(cache)
    else:
        import build_2026 as b26
        frames = []
        for ev in ["Silverstone", "Monza"]:
            try:
                p = b26.build_session(2026, ev, "R")
                d = pd.read_csv(p); d["event"] = ev; frames.append(d)
            except Exception as e:
                print(f"[warn] battery features failed {ev}: {e}")
        if not frames:
            return out, False
        batt = pd.concat(frames, ignore_index=True)
        batt.to_csv(cache, index=False)
    have = [c for c in BATT if c in batt.columns]
    out = out.merge(batt[["event", "driver", "lap_number"] + have],
                    on=["event", "driver", "lap_number"], how="left")
    return out, len(have) == len(BATT)


def battery_adjust(df):
    res = "residual_s" if "residual_s" in df.columns else \
          next((c for c in df.columns if "resid" in c), None)
    if res is None or not all(c in df.columns for c in BATT):
        print("STOP — available columns:", sorted(df.columns))
        raise SystemExit(1)
    X = df[BATT].fillna(0.0).to_numpy(float)
    y = df[res].to_numpy(float)
    eff = np.zeros(len(df))
    for _, idx in df.groupby("event").groups.items():
        i = np.sort(np.asarray(idx, dtype=int))
        mask = ~np.isnan(y[i])
        if mask.sum() < 2:
            eff[i] = 0.0
        else:
            eff[i] = Ridge(alpha=1.0).fit(X[i][mask], y[i][mask]).predict(X[i])
    out = df.copy()
    out[res] = y - eff
    if "wear_y" in out.columns:
        out["wear_y"] = out["wear_y"].to_numpy(float) - eff
    return out


def pred_col(st):
    return next((c for c in st.columns if "pred" in c), None)


def main():
    clean = t26.ingest_2026()
    # PATCH v3: pd.to_numeric with coerce for known object columns holding pd.NA
    for _c in ["lap_time_s", "sector1_s", "sector2_s", "sector3_s", "lap_time_fuel_corr_s"]:
        if _c in clean.columns:
            clean[_c] = pd.to_numeric(clean[_c], errors="coerce")
    practice, race = t26.score_a_2026(clean)
    st_base = t26.score_b_2026(practice, race)

    race_b, ok_r = attach_battery(race)
    prac_b, ok_p = attach_battery(practice)
    if not (ok_r and ok_p):
        print("Battery features unavailable (network?) — falling back to frozen-only report.")
        print(st_base.describe(include="number").T.to_string())
        return
    st_batt = t26.score_b_2026(battery_adjust(prac_b), battery_adjust(race_b))

    pc = pred_col(st_base)
    keys = [k for k in KEYS if k in st_base.columns and k in st_batt.columns]
    m = st_base[keys + [pc]].merge(st_batt[keys + [pc]], on=keys, suffixes=("_base", "_batt"))
    dep = race_b.groupby(keys)["e_deploy_lap_mj"].mean().reset_index()
    m = m.merge(dep, on=keys, how="left")
    m["delta_ms"] = (m[pc + "_batt"] - m[pc + "_base"]) * 1000.0

    corr = m["delta_ms"].corr(m["e_deploy_lap_mj"])
    mean_d = m["delta_ms"].mean()
    per = m.groupby("compound")["delta_ms"].mean().round(1).to_dict()
    print(f"\nstints scored: {len(m)} | pred column: {pc}")
    print(f"mean wear-slope shift after battery removal: {mean_d:+.1f} ms/lap")
    print(f"corr(shift, stint deploy MJ/lap): {corr:+.3f}")
    print(f"per-compound shift (ms/lap): {per}")
    if corr < 0 and abs(mean_d) >= 2:
        dec = "KEEP: battery adjustment acts in physics-predicted direction"
    elif abs(mean_d) < 2:
        dec = "NEUTRAL: battery kept as reported 2026-readiness feature"
    else:
        dec = "OUT: battery removed from wear path, kept for strategy layer"
    print("DECISION:", dec)
    json.dump({"n_stints": int(len(m)), "mean_delta_ms": round(float(mean_d), 1),
               "corr_delta_deploy": round(float(corr), 3), "per_compound_ms": per,
               "decision": dec}, open(ROOT / "scripts" / "merge_result.json", "w"), indent=2)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}")
        print("Paste this output + the STOP column list (if printed) for a one-round fix.")
        raise
