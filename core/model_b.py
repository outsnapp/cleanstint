"""Model B: tyre wear on A's leftover.

Reads model_a_train.csv and model_a_race.csv. Does not refit A.
Does not touch laps.csv or laps_model.csv.

A already applied fuel as physics on Y, so wear_y is residual_s.
Practice stints are too short for a wear slope. B uses:

1. Same-race, leave-one-driver-out field prior (event x compound).
2. Live update from the first 8 flying laps on long stints.
3. Cross-event LOEO only as a fallback.
4. SOFT shrinks toward MEDIUM when the field sample is thin.
5. No pit-lap call. Product is wear rate and +5/+10/+15.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor

from config import (
    CHECKPOINT_DRIVER,
    PREDICT_LAPS_AHEAD,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_A = ROOT / "data" / "processed" / "model_a_train.csv"
RACE_A = ROOT / "data" / "processed" / "model_a_race.csv"
STINT_OUT = ROOT / "data" / "processed" / "model_b_stints.csv"
PRIOR_OUT = ROOT / "data" / "processed" / "model_b_prior.csv"
BUNDLE_OUT = ROOT / "data" / "models" / "model_b.joblib"

CHECKPOINT_EVENT = "Bahrain"
T_COL = "tyre_life"
STINT_KEYS = ["year", "event", "session", "driver", "stint", "compound"]
DRY = ("HARD", "MEDIUM", "SOFT")
MIN_PRIOR_LAPS = 15
MIN_COMPOUND_LAPS = 20
MIN_STINT_LAPS = 8
LIVE_FIT_LAPS = 8
LIVE_MIN_TAIL = 4
LIVE_SHRINK_K = 16
SOFT_TO_MEDIUM = 1.15
MIN_SOFT_FIELD_LAPS = 40
CLIFF_SECONDS = 1.0
MIN_WEAR_FOR_CLIFF = 0.02
FLYING_PAD_S = 5.0


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _clip_wear(wear: float) -> float:
    if pd.isna(wear):
        return float("nan")
    return max(float(wear), 0.0)


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.unique(x).size < 2:
        return float("nan")
    try:
        model = TheilSenRegressor(random_state=0, max_iter=500)
        model.fit(x.reshape(-1, 1), y)
        return float(model.coef_[0])
    except ValueError:
        return float("nan")


def _cliff(wear: float) -> float:
    if pd.isna(wear) or wear < MIN_WEAR_FOR_CLIFF:
        return float("nan")
    return round(CLIFF_SECONDS / wear, 1)


def _stint_demean(series: pd.Series) -> pd.Series:
    return series - series.mean()


def load_scored() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = TRAIN_A if TRAIN_A.exists() else TRAIN_A.with_name(TRAIN_A.stem + "_new" + TRAIN_A.suffix)
    race_path = RACE_A if RACE_A.exists() else RACE_A.with_name(RACE_A.stem + "_new" + RACE_A.suffix)
    # Prefer the _new copy if it is newer (write was blocked on the original).
    train_new = TRAIN_A.with_name(TRAIN_A.stem + "_new" + TRAIN_A.suffix)
    race_new = RACE_A.with_name(RACE_A.stem + "_new" + RACE_A.suffix)
    if train_new.exists() and (not TRAIN_A.exists() or train_new.stat().st_mtime > TRAIN_A.stat().st_mtime):
        train_path = train_new
    if race_new.exists() and (not RACE_A.exists() or race_new.stat().st_mtime > RACE_A.stat().st_mtime):
        race_path = race_new
    if not train_path.exists() or not race_path.exists():
        raise FileNotFoundError("Missing Model A scores. Run model_a.py first.")
    train = pd.read_csv(train_path)
    race = pd.read_csv(race_path)
    if "residual_s" not in train.columns or "residual_s" not in race.columns:
        raise KeyError("A scores must contain residual_s. Re-run model_a.py.")
    if train["session"].isin(["Q", "R"]).any():
        raise ValueError("Q or R leaked into model_a_train.csv. B will not start.")
    if (race["session"] != "R").any():
        raise ValueError("model_a_race.csv must be race-only.")
    return train, race


def add_wear_y(df: pd.DataFrame) -> pd.DataFrame:
    """A already put fuel into y_fuel. Residual is the tyre leftover."""
    out = df.copy()
    out["wear_y"] = out["residual_s"]
    return out


def flying_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Drop in-stint outliers. Does not rewrite A CSVs."""
    d = df[df["wear_y"].notna() & pd.to_numeric(df[T_COL], errors="coerce").notna()].copy()
    med = d.groupby(STINT_KEYS)["lap_time_s"].transform("median")
    return d[d["lap_time_s"] <= med + FLYING_PAD_S].reset_index(drop=True)


def _is_wet(df: pd.DataFrame) -> pd.Series:
    if "rainfall" not in df.columns:
        return pd.Series(False, index=df.index)
    return _as_bool(df["rainfall"])


def _grouped_slope(df: pd.DataFrame, keys: list[str], min_laps: int) -> pd.DataFrame:
    d = df.copy()
    d["y_dm"] = d.groupby(STINT_KEYS, dropna=False)["wear_y"].transform(_stint_demean)
    rows = []
    for key, g in d.groupby(keys, dropna=False):
        if len(g) < min_laps:
            continue
        wear = _theil_sen(g[T_COL].to_numpy(float), g["y_dm"].to_numpy(float))
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(keys, key))
        row["laps"] = int(len(g))
        row["max_age"] = float(g[T_COL].max())
        row["wear_s_per_lap"] = wear
        rows.append(row)
    return pd.DataFrame(rows)


def fit_practice_prior(practice: pd.DataFrame) -> pd.DataFrame:
    dry = practice[practice["compound"].isin(DRY)].copy()
    by_event = _grouped_slope(dry, ["event", "compound"], MIN_PRIOR_LAPS)
    by_comp = _grouped_slope(dry, ["compound"], MIN_COMPOUND_LAPS)
    frames = []
    if not by_event.empty:
        e = by_event.copy()
        e["level"] = "practice_event_compound"
        frames.append(e)
    if not by_comp.empty:
        c = by_comp.copy()
        c["level"] = "practice_compound"
        c["event"] = "*"
        frames.append(c)
    if not frames:
        return pd.DataFrame(columns=["level", "event", "compound", "laps", "max_age", "wear_s_per_lap"])
    return pd.concat(frames, ignore_index=True)


def fit_loeo_prior(race: pd.DataFrame) -> pd.DataFrame:
    dry = race[race["compound"].isin(DRY) & ~_is_wet(race)].copy()
    rows = []
    for held in dry["event"].dropna().unique():
        other = dry[dry["event"] != held]
        tab = _grouped_slope(other, ["compound"], MIN_COMPOUND_LAPS)
        if tab.empty:
            continue
        tab["level"] = "loeo_compound"
        tab["event"] = held
        rows.append(tab)
    if not rows:
        return pd.DataFrame(columns=["level", "event", "compound", "laps", "max_age", "wear_s_per_lap"])
    return pd.concat(rows, ignore_index=True)


def fit_field_prior(race: pd.DataFrame) -> pd.DataFrame:
    """Same event x compound, leave this driver out. Other cars, this race."""
    dry = race[race["compound"].isin(DRY) & ~_is_wet(race)].copy()
    rows = []
    for (event, compound), g in dry.groupby(["event", "compound"], dropna=False):
        for driver, _ in g.groupby("driver", dropna=False):
            other = g[g["driver"] != driver]
            if len(other) < MIN_PRIOR_LAPS:
                continue
            other = other.copy()
            other["y_dm"] = other.groupby(STINT_KEYS, dropna=False)["wear_y"].transform(_stint_demean)
            wear = _theil_sen(other[T_COL].to_numpy(float), other["y_dm"].to_numpy(float))
            rows.append(
                {
                    "level": "field_lodo",
                    "event": event,
                    "compound": compound,
                    "driver": driver,
                    "laps": int(len(other)),
                    "max_age": float(other[T_COL].max()),
                    "wear_s_per_lap": wear,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["level", "event", "compound", "driver", "laps", "max_age", "wear_s_per_lap"]
        )
    return pd.DataFrame(rows)


def _lookup(prior: pd.DataFrame, level: str, event: str, compound: str) -> float:
    hit = prior[
        (prior["level"] == level)
        & (prior["compound"] == compound)
        & (prior["event"].isin([event, "*"]))
    ]
    if level.startswith("practice") and level.endswith("compound") and "event_compound" not in level:
        hit = prior[(prior["level"] == level) & (prior["compound"] == compound)]
    if hit.empty or pd.isna(hit["wear_s_per_lap"].iloc[0]):
        return float("nan")
    if event in set(hit["event"].astype(str)) and event != "*":
        hit = hit[hit["event"] == event]
    return float(hit["wear_s_per_lap"].iloc[0])


def _practice_lookup(prior: pd.DataFrame, event: str, compound: str) -> float:
    ev = _lookup(prior, "practice_event_compound", event, compound)
    if pd.notna(ev):
        return ev
    return _lookup(prior, "practice_compound", "*", compound)


def _loeo_lookup(prior: pd.DataFrame, event: str, compound: str) -> float:
    return _lookup(prior, "loeo_compound", event, compound)


def _field_lookup(prior: pd.DataFrame, event: str, compound: str, driver: str) -> tuple[float, int]:
    if "driver" not in prior.columns:
        return float("nan"), 0
    hit = prior[
        (prior["level"] == "field_lodo")
        & (prior["event"] == event)
        & (prior["compound"] == compound)
        & (prior["driver"] == driver)
    ]
    if hit.empty or pd.isna(hit["wear_s_per_lap"].iloc[0]):
        return float("nan"), 0
    return float(hit["wear_s_per_lap"].iloc[0]), int(hit["laps"].iloc[0])


def _soft_fallback(prior: pd.DataFrame, event: str, driver: str, compound: str, wear: float, laps: int) -> tuple[float, str]:
    if compound != "SOFT":
        return wear, ""
    weak = pd.isna(wear) or laps < MIN_SOFT_FIELD_LAPS
    if not weak:
        return wear, ""
    med, _ = _field_lookup(prior, event, "MEDIUM", driver)
    if pd.isna(med):
        med = _loeo_lookup(prior, event, "MEDIUM")
    if pd.isna(med):
        return wear, ""
    return float(med) * SOFT_TO_MEDIUM, "soft_from_medium"


def _blend(live: float, prior_wear: float, live_n: int, prior_name: str) -> tuple[float, str]:
    live_c = _clip_wear(live)
    prior_c = _clip_wear(prior_wear)
    if pd.notna(live_c) and pd.notna(prior_c):
        w = live_n / (live_n + LIVE_SHRINK_K)
        return w * live_c + (1.0 - w) * prior_c, f"live_{prior_name}"
    if pd.notna(live_c):
        return live_c, "live_window"
    if pd.notna(prior_c):
        return prior_c, prior_name
    return float("nan"), "none"


def score_stints(race: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    last = (
        race.groupby(["year", "event", "driver"], dropna=False)["stint"]
        .max()
        .rename("last_stint")
        .reset_index()
    )
    rows = []
    for key, g in race.groupby(STINT_KEYS, dropna=False):
        g = g.sort_values([T_COL, "lap_number"]).reset_index(drop=True)
        year, event, session, driver, stint, compound = key
        practice = _practice_lookup(prior, str(event), str(compound))
        loeo = _loeo_lookup(prior, str(event), str(compound))
        field, field_laps = _field_lookup(prior, str(event), str(compound), str(driver))
        field, soft_note = _soft_fallback(prior, str(event), str(driver), str(compound), field, field_laps)

        live = float("nan")
        tail = float("nan")
        full = _theil_sen(g[T_COL].to_numpy(float), g["wear_y"].to_numpy(float))
        if len(g) < MIN_STINT_LAPS:
            full = float("nan")

        use_live = len(g) >= LIVE_FIT_LAPS + LIVE_MIN_TAIL
        if use_live:
            head = g.iloc[:LIVE_FIT_LAPS]
            rest = g.iloc[LIVE_FIT_LAPS:]
            live = _theil_sen(head[T_COL].to_numpy(float), head["wear_y"].to_numpy(float))
            tail = _theil_sen(rest[T_COL].to_numpy(float), rest["wear_y"].to_numpy(float))

        base, base_name = (field, "field_lodo") if pd.notna(field) else (loeo, "loeo_compound")
        pred, source = _blend(live, base, LIVE_FIT_LAPS if use_live else 0, base_name)
        if source == "none":
            pred, source = _clip_wear(practice), "practice"

        actual = tail if use_live and pd.notna(tail) else full

        tyre_start = float(g[T_COL].iloc[0])
        tyre_end = float(g[T_COL].iloc[-1])
        lap_start = float(g["lap_number"].iloc[0])
        lap_end = float(g["lap_number"].iloc[-1])
        cliff = _cliff(pred)
        rec_pit = float("nan")

        driver_last = last[
            (last["year"] == year) & (last["event"] == event) & (last["driver"] == driver)
        ]
        last_stint = float(driver_last["last_stint"].iloc[0]) if not driver_last.empty else stint
        actual_pit = lap_end + 1.0 if float(stint) < last_stint else float("nan")

        notes = []
        if compound not in DRY:
            notes.append("wet compound")
        if _is_wet(g).any():
            notes.append("wet session")
        if source == "practice":
            notes.append("fell back to practice prior")
        if pd.notna(practice) and practice < 0:
            notes.append("practice slope negative")
        if soft_note:
            notes.append(soft_note)
        notes.append("no pit call from wear")

        ahead = {
            f"pred_plus_{n}_s": (pred * n if pd.notna(pred) else float("nan"))
            for n in PREDICT_LAPS_AHEAD
        }
        err = pred - actual if pd.notna(pred) and pd.notna(actual) else float("nan")
        rows.append(
            {
                "year": year,
                "event": event,
                "session": session,
                "driver": driver,
                "driver_number": g["driver_number"].iloc[0],
                "team": g["team"].iloc[0],
                "stint": stint,
                "compound": compound,
                "n_laps": int(len(g)),
                "stint_start_lap": lap_start,
                "stint_end_lap": lap_end,
                "tyre_age_start": tyre_start,
                "tyre_age_end": tyre_end,
                "pred_wear_s_per_lap": pred,
                "practice_wear_s_per_lap": practice,
                "loeo_wear_s_per_lap": loeo,
                "field_wear_s_per_lap": field,
                "live_wear_s_per_lap": live,
                "actual_wear_s_per_lap": actual,
                "actual_wear_full_s_per_lap": full,
                "wear_error": err,
                **ahead,
                "cliff_tyre_lap": cliff,
                "recommended_pit_lap": rec_pit,
                "actual_pit_lap": actual_pit,
                "prior_source": source,
                "notes": "; ".join(notes),
            }
        )
    return pd.DataFrame(rows).sort_values(["event", "driver", "stint"]).reset_index(drop=True)


def _fmt(v, digits=4) -> str:
    return "" if pd.isna(v) else f"{v:.{digits}f}"


def _print_prior(prior: pd.DataFrame) -> None:
    print("\n=== Model B priors (stint-demeaned Theil-Sen) ===")
    show = prior[prior["level"] != "field_lodo"].copy()
    show["wear_s_per_lap"] = show["wear_s_per_lap"].map(lambda v: _fmt(v))
    cols = [c for c in ["level", "event", "compound", "laps", "max_age", "wear_s_per_lap"] if c in show.columns]
    print(show[cols].to_string(index=False))
    field = prior[prior["level"] == "field_lodo"]
    if not field.empty:
        summ = (
            field.groupby(["event", "compound"], dropna=False)["wear_s_per_lap"]
            .median()
            .reset_index()
        )
        summ["wear_s_per_lap"] = summ["wear_s_per_lap"].map(lambda v: _fmt(v))
        print("\nField prior (median across drivers):")
        print(summ.to_string(index=False))
    print("Practice rows are diagnostic. Sunday predictions use field/live, not practice.")


def _print_checkpoint(stints: pd.DataFrame) -> None:
    demo = stints[
        (stints["event"] == CHECKPOINT_EVENT) & (stints["driver"] == CHECKPOINT_DRIVER)
    ].copy()
    print(f"\n=== Checkpoint: {CHECKPOINT_DRIVER} {CHECKPOINT_EVENT} race stints ===")
    if demo.empty:
        print("No demo stints.")
        return
    cols = [
        "stint",
        "compound",
        "n_laps",
        "tyre_age_start",
        "tyre_age_end",
        "pred_wear_s_per_lap",
        "actual_wear_s_per_lap",
        "pred_plus_5_s",
        "pred_plus_10_s",
        "pred_plus_15_s",
        "cliff_tyre_lap",
        "recommended_pit_lap",
        "actual_pit_lap",
        "prior_source",
        "notes",
    ]
    show = demo[cols].copy()
    for c in ["pred_wear_s_per_lap", "actual_wear_s_per_lap", "pred_plus_5_s", "pred_plus_10_s", "pred_plus_15_s"]:
        show[c] = show[c].map(lambda v: _fmt(v))
    for c in ["cliff_tyre_lap", "recommended_pit_lap", "actual_pit_lap"]:
        show[c] = show[c].map(lambda v: _fmt(v, 1))
    print(show.to_string(index=False))

    row = demo.sort_values("n_laps").iloc[-1]
    print(
        f"\nLongest stint: stint {int(row['stint'])} {row['compound']} "
        f"(tyre {row['tyre_age_start']:.0f}-{row['tyre_age_end']:.0f})"
    )
    print(f"  source: {row['prior_source']}")
    for n in PREDICT_LAPS_AHEAD:
        val = row[f"pred_plus_{n}_s"]
        print(f"  +{n} laps: " + ("" if pd.isna(val) else f"{val:+.3f}s"))
    cliff = row["cliff_tyre_lap"]
    print("  cliff: " + ("none" if pd.isna(cliff) else f"~{cliff:.1f} tyre laps to lose 1s"))
    rec = row["recommended_pit_lap"]
    act = row["actual_pit_lap"]
    print("  recommended pit: " + ("none" if pd.isna(rec) else f"lap {rec:.0f}"))
    print("  actual pit: " + ("ran to flag" if pd.isna(act) else f"lap {act:.0f}"))


def _print_score(stints: pd.DataFrame) -> None:
    dry = stints[
        stints["compound"].isin(DRY)
        & ~stints["notes"].str.contains("wet session", na=False)
        & stints["actual_wear_s_per_lap"].notna()
        & stints["pred_wear_s_per_lap"].notna()
    ].copy()
    print("\n=== Sunday score (dry stints) ===")
    print(f"Scored stints: {len(dry)}")
    if dry.empty:
        return
    mae = dry["wear_error"].abs().mean()
    rmse = float((dry["wear_error"] ** 2).mean() ** 0.5)
    ss_res = float((dry["wear_error"] ** 2).sum())
    ss_tot = float(((dry["actual_wear_s_per_lap"] - dry["actual_wear_s_per_lap"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    print(f"Wear-slope MAE: {mae:.4f} s/lap")
    print(f"Wear-slope RMSE: {rmse:.4f} s/lap")
    print(f"Wear-slope R2: {r2:.3f}")
    print("\nMean wear by compound (predicted vs actual vs full stint):")
    by_comp = (
        dry.groupby("compound")[
            ["pred_wear_s_per_lap", "actual_wear_s_per_lap", "actual_wear_full_s_per_lap"]
        ]
        .mean()
        .reindex(list(DRY))
    )
    print(by_comp.to_string())
    print("\nMAE by compound:")
    print(dry.groupby("compound")["wear_error"].apply(lambda s: s.abs().mean()).reindex(list(DRY)).to_string())
    print("\nPit laps are recorded for reference only. B does not call a pit lap.")
    print(f"Sources:\n{dry['prior_source'].value_counts().to_string()}")


if __name__ == "__main__":
    train, race = load_scored()
    train = flying_laps(add_wear_y(train))
    race = flying_laps(add_wear_y(race))
    practice_prior = fit_practice_prior(train)
    loeo_prior = fit_loeo_prior(race)
    field_prior = fit_field_prior(race)
    prior = pd.concat([practice_prior, loeo_prior, field_prior], ignore_index=True)
    if prior.empty:
        raise ValueError("B prior is empty.")
    stints = score_stints(race, prior)

    PRIOR_OUT.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    prior.to_csv(PRIOR_OUT, index=False)
    stints.to_csv(STINT_OUT, index=False)
    joblib.dump(
        {
            "prior": prior,
            "ahead": list(PREDICT_LAPS_AHEAD),
            "live_fit_laps": LIVE_FIT_LAPS,
            "field_prior": True,
        },
        BUNDLE_OUT,
    )

    print(f"Practice flying laps: {len(train)}  Race flying laps: {len(race)}")
    print("A was not refit. laps.csv and laps_model.csv were not touched.")
    _print_prior(prior)
    _print_checkpoint(stints)
    _print_score(stints)
    print(f"\nWrote {PRIOR_OUT}")
    print(f"Wrote {STINT_OUT}")
    print(f"Wrote {BUNDLE_OUT}")
    print("B is the wear model. A stays frozen.")
