"""Model A: nuisance learner (confounders only).

Reads data/processed/laps_model.csv. Does not write that file and never
touches data/processed/laps.csv. Fit stays on practice; Q/R are hold-out.

Practice fuel is always 0, so A must not learn fuel. Fuel is applied as
physics on the target:

    y_fuel = lap_time_s + TIME_GAIN_S_PER_KG * fuel_kg_burned

X is track temp, session, compound, event. No tyre_life. No fuel.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from config import CHECKPOINT_DRIVER, FIT_SESSIONS, HOLD_SESSIONS, TIME_GAIN_S_PER_KG

ROOT = Path(__file__).resolve().parents[1]
MODEL_CSV = ROOT / "data" / "processed" / "laps_model.csv"
TRAIN_OUT = ROOT / "data" / "processed" / "model_a_train.csv"
RACE_OUT = ROOT / "data" / "processed" / "model_a_race.csv"
BUNDLE_OUT = ROOT / "data" / "models" / "model_a.joblib"

Y_COL = "lap_time_s"
Y_FUEL_COL = "y_fuel_s"
T_COL = "tyre_life"
CHECKPOINT_EVENT = "Bahrain"
STINT_KEYS = ["year", "event", "session", "driver", "stint", "compound"]
NUMERIC_X = ["track_temp_c"]
SESSION_X = ["sess_FP1", "sess_FP2", "sess_FP3", "sess_Q", "sess_R"]
CAT_X = ["compound", "event"]
LEAVE_OUT = {
    "tyre_life",
    "stint",
    "lap_number",
    "lap_time_fuel_corr_s",
    "deleted",
    "assumed_harvest_mj",
    "fuel_kg_burned",
}


def load_clean() -> pd.DataFrame:
    if not MODEL_CSV.exists():
        raise FileNotFoundError(f"Missing {MODEL_CSV}. Run preprocess.py first.")
    return pd.read_csv(MODEL_CSV)


def split_track3(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on FP1/FP2/FP3. Hold out Q and R. Race is the exam."""
    missing = set(FIT_SESSIONS + HOLD_SESSIONS) - set(df["session"].unique())
    if missing:
        raise ValueError(f"laps_model.csv is missing sessions: {sorted(missing)}")

    train = df[df["session"].isin(FIT_SESSIONS)].copy()
    hold = df[df["session"].isin(HOLD_SESSIONS)].copy()

    if train.empty:
        raise ValueError("Fit set is empty. Check session labels in laps_model.csv.")
    if hold.empty:
        raise ValueError("Hold-out set is empty. Check session labels in laps_model.csv.")
    if train["session"].isin(HOLD_SESSIONS).any():
        raise ValueError("Q or R leaked into the fit set.")
    if hold["session"].isin(FIT_SESSIONS).any():
        raise ValueError("Practice leaked into the hold-out set.")

    return train.reset_index(drop=True), hold.reset_index(drop=True)


def add_fuel_target(df: pd.DataFrame) -> pd.DataFrame:
    """Undo fuel fade with the known 0.03 s/kg. Do not learn it from X."""
    out = df.copy()
    fuel = pd.to_numeric(out["fuel_kg_burned"], errors="coerce").fillna(0.0)
    out[Y_FUEL_COL] = out[Y_COL].astype(float) + TIME_GAIN_S_PER_KG * fuel
    return out


def build_features(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """X = track temp, session dummies, compound, event. No tyre_life. No fuel."""
    missing = [c for c in NUMERIC_X + SESSION_X + CAT_X + [Y_COL, T_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"laps_model.csv missing columns: {missing}")

    numeric = df[NUMERIC_X].astype(float)
    session = df[SESSION_X].astype(float)
    cats = pd.get_dummies(df[CAT_X], columns=CAT_X, prefix=CAT_X, dtype=float)
    X = pd.concat([numeric, session, cats], axis=1)

    leaked = sorted(LEAVE_OUT & set(X.columns))
    if leaked:
        raise ValueError(f"X must not contain {leaked}")

    if feature_columns is None:
        feature_columns = list(X.columns)
    else:
        X = X.reindex(columns=feature_columns, fill_value=0.0)

    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"X has NaNs in {bad}")
    return X, feature_columns


def fit_nuisance(
    train: pd.DataFrame,
) -> tuple[Ridge, Ridge, pd.DataFrame, list[str]]:
    """A_Y: X -> y_fuel_s. A_T: X -> tyre_life. Same X, two Ridge fits."""
    X, feature_columns = build_features(train)
    y = train[Y_FUEL_COL].astype(float)
    t = train[T_COL].astype(float)
    if y.isna().any() or t.isna().any():
        raise ValueError("Y or T has NaNs on the fit set.")

    a_y = Ridge().fit(X, y)
    a_t = Ridge().fit(X, t)
    return a_y, a_t, X, feature_columns


def add_residual(df: pd.DataFrame, a_y: Ridge, X: pd.DataFrame) -> pd.DataFrame:
    """residual_s = y_fuel_s - A_Y(X). Does not touch laps_model.csv."""
    out = add_fuel_target(df)
    out["a_y_pred"] = a_y.predict(X)
    out["residual_s"] = out[Y_FUEL_COL] - out["a_y_pred"]
    return out


def _safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        df.to_csv(alt, index=False)
        print(f"WARNING: {path.name} is open in another program. Wrote {alt.name} instead.")
        return alt


def save_fit(
    train_with_residual: pd.DataFrame,
    a_y: Ridge,
    a_t: Ridge,
    feature_columns: list[str],
) -> None:
    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(train_with_residual, TRAIN_OUT)
    joblib.dump(
        {
            "a_y": a_y,
            "a_t": a_t,
            "feature_columns": feature_columns,
            "fit_sessions": list(FIT_SESSIONS),
            "fuel_on_y": True,
            "time_gain_s_per_kg": TIME_GAIN_S_PER_KG,
        },
        BUNDLE_OUT,
    )


def pick_checkpoint_stint(train: pd.DataFrame) -> pd.DataFrame:
    """Longest Bahrain FP stint for CHECKPOINT_DRIVER (VER)."""
    pool = train[
        (train["event"] == CHECKPOINT_EVENT) & (train["driver"] == CHECKPOINT_DRIVER)
    ].copy()
    if pool.empty:
        raise ValueError(
            f"No {CHECKPOINT_EVENT} FP laps for {CHECKPOINT_DRIVER} in the fit set."
        )
    sizes = pool.groupby(["session", "stint", "compound"], dropna=False).size()
    session, stint, compound = sizes.idxmax()
    rows = pool[
        (pool["session"] == session)
        & (pool["stint"] == stint)
        & (pool["compound"] == compound)
    ].sort_values(["tyre_life", "lap_number"])
    if len(rows) < 5:
        raise ValueError(
            f"{CHECKPOINT_DRIVER} {CHECKPOINT_EVENT} checkpoint stint is too short ({len(rows)} laps)."
        )
    return rows.reset_index(drop=True)


def run_checkpoint(train: pd.DataFrame, a_y: Ridge, feature_columns: list[str]) -> dict:
    """One Bahrain FP stint. Fuel must not be in X. Residual must still trend up."""
    stint = pick_checkpoint_stint(train)
    leaked_fuel = "fuel_kg_burned" in feature_columns
    slope = float(np.polyfit(stint[T_COL].to_numpy(), stint["residual_s"].to_numpy(), 1)[0])
    corr = float(stint["residual_s"].corr(stint[T_COL]))
    residual_up = (slope > 0) and (corr > 0)

    print("\n=== Checkpoint 6: one Bahrain FP stint ===")
    print(
        f"{CHECKPOINT_DRIVER} {CHECKPOINT_EVENT} {stint['session'].iloc[0]} "
        f"stint {int(stint['stint'].iloc[0])} {stint['compound'].iloc[0]} "
        f"({len(stint)} laps)"
    )
    show = stint[
        ["lap_number", T_COL, "fuel_kg_burned", Y_COL, Y_FUEL_COL, "a_y_pred", "residual_s"]
    ]
    print(show.to_string(index=False))

    print("\nFuel handling:")
    print(f"  physics on Y: +{TIME_GAIN_S_PER_KG} s/kg")
    print(f"  fuel in X: {leaked_fuel}")
    fuel_verdict = "FAIL: fuel leaked into X" if leaked_fuel else "PASS: fuel not learned, applied as physics"
    print(f"  {fuel_verdict}")

    print("\nResidual vs tyre_life (this stint):")
    print(f"  slope {slope:.4f} s per tyre lap")
    print(f"  corr  {corr:.4f}")
    residual_verdict = (
        "PASS: leftover still trends up (older tyre -> slower)"
        if residual_up
        else "FAIL: residual flat or down - A ate the tyre"
    )
    print(f"  {residual_verdict}")

    hard_fail = leaked_fuel or not residual_up
    return {
        "fuel_in_x": leaked_fuel,
        "residual_slope": slope,
        "residual_corr": corr,
        "residual_up": residual_up,
        "hard_fail": hard_fail,
    }


def score_laps(df: pd.DataFrame, a_y: Ridge, feature_columns: list[str]) -> pd.DataFrame:
    """Apply frozen A. Does not write files and does not refit."""
    X, _ = build_features(df, feature_columns)
    return add_residual(df, a_y, X).reset_index(drop=True)


def score_sunday(
    hold: pd.DataFrame, a_y: Ridge, feature_columns: list[str]
) -> pd.DataFrame:
    """Step 7: freeze A. Score race only. Do not refit."""
    race = hold[hold["session"] == "R"].copy()
    if race.empty:
        raise ValueError("No race laps in the hold-out set.")
    scored = score_laps(race, a_y, feature_columns)
    RACE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _safe_to_csv(scored, RACE_OUT)
    return scored


def _print_sunday(race: pd.DataFrame) -> None:
    print("\n=== Freeze 7: score A_Y on Sunday (not refit) ===")
    print(f"Race rows: {len(race)}")
    print(race.groupby("event").size().to_string())
    abs_mae = (race[Y_FUEL_COL] - race["a_y_pred"]).abs().mean()
    dm = race.groupby(STINT_KEYS, dropna=False)[Y_FUEL_COL].transform(lambda s: s - s.mean())
    pred_dm = race.groupby(STINT_KEYS, dropna=False)["a_y_pred"].transform(lambda s: s - s.mean())
    within_mae = (dm - pred_dm).abs().mean()
    print(f"\nAbsolute MAE on y_fuel (practice pace vs race): {abs_mae:.3f}s")
    print(f"Within-stint MAE (level removed): {within_mae:.3f}s")
    print("Absolute MAE is the practice/race offset. Wear uses the within-stint leftover.")
    print("\nRace residual_s:")
    print(race["residual_s"].describe().to_string())
    print(
        f"\ncorr residual vs tyre_life: {race['residual_s'].corr(race[T_COL]):.4f}"
    )
    print(
        f"corr residual vs fuel_kg_burned: {race['residual_s'].corr(race['fuel_kg_burned']):.4f}"
    )

    ver = race[
        (race["event"] == CHECKPOINT_EVENT) & (race["driver"] == CHECKPOINT_DRIVER)
    ].sort_values("lap_number")
    if ver.empty:
        print(f"\nNo {CHECKPOINT_DRIVER} {CHECKPOINT_EVENT} race laps to preview.")
        return
    print(f"\n{CHECKPOINT_DRIVER} {CHECKPOINT_EVENT} race (first/last 6):")
    cols = [
        "lap_number",
        "stint",
        "compound",
        T_COL,
        "fuel_kg_burned",
        Y_COL,
        Y_FUEL_COL,
        "a_y_pred",
        "residual_s",
    ]
    preview = pd.concat([ver.head(6), ver.tail(6)])
    print(preview[cols].to_string(index=False))
    print(
        f"  residual vs tyre_life: {ver['residual_s'].corr(ver[T_COL]):.4f}"
    )
    print(
        f"  residual vs fuel:      {ver['residual_s'].corr(ver['fuel_kg_burned']):.4f}"
    )


if __name__ == "__main__":
    table = load_clean()
    train, hold = split_track3(table)
    train = add_fuel_target(train)
    a_y, a_t, X, feature_columns = fit_nuisance(train)
    train = add_residual(train, a_y, X)
    save_fit(train, a_y, a_t, feature_columns)

    print(f"Loaded {len(table)} rows from {MODEL_CSV}")
    print(f"Fit on {FIT_SESSIONS} ({len(train)} rows). Hold-out {HOLD_SESSIONS} unused in fit.")
    print(f"Wrote {TRAIN_OUT}")
    print(f"Wrote {BUNDLE_OUT}")

    checkpoint = run_checkpoint(train, a_y, feature_columns)
    if checkpoint["hard_fail"]:
        print("\nCheckpoint failed. A is not frozen. Model B must not start.")
        raise SystemExit(1)

    race = score_sunday(hold, a_y, feature_columns)
    _print_sunday(race)
    print(f"\nWrote Sunday scores to {RACE_OUT}")
    print("laps_model.csv was not overwritten. laps.csv was not touched.")
    print("A is frozen. Model B has not started.")
