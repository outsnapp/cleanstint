"""Build 2026 test data and score frozen 2025 Model A and B.

Does not refit A or B. Does not touch laps.csv or laps_model.csv.
2026 calendar: Silverstone and Monza have run. Imola is not on the
calendar. Bahrain 2026 is Sepang in October — skipped.
"""

from pathlib import Path

import joblib
import pandas as pd

from config import (
    CHECKPOINT_DRIVER,
    EVENTS_2026,
    FIT_SESSIONS,
    HOLD_SESSIONS,
    SESSIONS,
    YEAR_TEST,
)
from ingest import enable_cache, load_session, session_to_rows
from model_a import (
    BUNDLE_OUT,
    SESSION_X,
    STINT_KEYS,
    T_COL,
    Y_FUEL_COL,
    _safe_to_csv,
    score_laps,
)
from model_b import (
    DRY,
    add_wear_y,
    fit_field_prior,
    fit_loeo_prior,
    fit_practice_prior,
    flying_laps,
    score_stints,
    _print_checkpoint,
    _print_prior,
    _print_score,
)
from preprocess import clean_laps

ROOT = Path(__file__).resolve().parents[1]
RAW_2026 = ROOT / "data" / "processed" / "laps_2026.csv"
CLEAN_2026 = ROOT / "data" / "processed" / "laps_model_2026.csv"
TRAIN_A_2026 = ROOT / "data" / "processed" / "model_a_2026_train.csv"
RACE_A_2026 = ROOT / "data" / "processed" / "model_a_2026_race.csv"
STINT_B_2026 = ROOT / "data" / "processed" / "model_b_stints_2026.csv"
PRIOR_B_2026 = ROOT / "data" / "processed" / "model_b_prior_2026.csv"


def ingest_2026() -> pd.DataFrame:
    enable_cache()
    frames = []
    skipped = []
    for event in EVENTS_2026:
        for session_name in SESSIONS:
            label = f"{YEAR_TEST} {event} {session_name}"
            print(f"Loading {label}...")
            try:
                session = load_session(YEAR_TEST, event, session_name)
                rows = session_to_rows(session, YEAR_TEST, event, session_name)
                print(f"  OK - {len(rows)} laps")
                frames.append(rows)
            except Exception as exc:
                skipped.append((label, f"{type(exc).__name__}: {exc}"))
                print(f"  SKIP - {type(exc).__name__}: {exc}")
    if not frames:
        raise RuntimeError("No 2026 sessions loaded.")
    table = pd.concat(frames, ignore_index=True)
    _safe_to_csv(table, RAW_2026)
    print(f"\nWrote {len(table)} rows -> {RAW_2026}")
    print(table.groupby(["year", "event", "session"]).size().to_string())
    if skipped:
        print("\nSkipped sessions:")
        for label, err in skipped:
            print(f"  {label}: {err}")
    return table


def ensure_a_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SESSION_X:
        if col not in out.columns:
            out[col] = 0.0
    if out["track_temp_c"].isna().any():
        fill = out.groupby("event")["track_temp_c"].transform("median")
        out["track_temp_c"] = out["track_temp_c"].fillna(fill).fillna(out["track_temp_c"].median())
    return out


def load_frozen_a():
    if not BUNDLE_OUT.exists():
        raise FileNotFoundError(f"Missing {BUNDLE_OUT}. Run model_a.py first.")
    bundle = joblib.load(BUNDLE_OUT)
    return bundle["a_y"], bundle["feature_columns"]


def score_a_2026(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a_y, feature_columns = load_frozen_a()
    table = ensure_a_columns(clean)
    missing = set(FIT_SESSIONS + ["R"]) - set(table["session"].unique())
    if missing:
        print(f"WARNING: 2026 data missing sessions {sorted(missing)}")

    practice = table[table["session"].isin(FIT_SESSIONS)].copy()
    race = table[table["session"] == "R"].copy()
    if practice.empty or race.empty:
        raise ValueError("2026 set needs practice and race laps.")

    practice = score_laps(practice, a_y, feature_columns)
    race = score_laps(race, a_y, feature_columns)
    _safe_to_csv(practice, TRAIN_A_2026)
    _safe_to_csv(race, RACE_A_2026)

    abs_mae = (race[Y_FUEL_COL] - race["a_y_pred"]).abs().mean()
    dm = race.groupby(STINT_KEYS, dropna=False)[Y_FUEL_COL].transform(lambda s: s - s.mean())
    pred_dm = race.groupby(STINT_KEYS, dropna=False)["a_y_pred"].transform(lambda s: s - s.mean())
    within_mae = (dm - pred_dm).abs().mean()
    print("\n=== Frozen A on 2026 race (not refit) ===")
    print(f"Practice rows: {len(practice)}  Race rows: {len(race)}")
    print(race.groupby("event").size().to_string())
    print(f"Absolute MAE on y_fuel: {abs_mae:.3f}s")
    print(f"Within-stint MAE: {within_mae:.3f}s")
    print(f"corr residual vs tyre_life: {race['residual_s'].corr(race[T_COL]):.4f}")
    print(f"corr residual vs fuel: {race['residual_s'].corr(race['fuel_kg_burned']):.4f}")

    for event in EVENTS_2026:
        ver = race[(race["event"] == event) & (race["driver"] == CHECKPOINT_DRIVER)]
        if ver.empty:
            print(f"\nNo {CHECKPOINT_DRIVER} {event} 2026 race laps.")
            continue
        print(
            f"\n{CHECKPOINT_DRIVER} {event} 2026: "
            f"residual vs tyre {ver['residual_s'].corr(ver[T_COL]):.4f}, "
            f"vs fuel {ver['residual_s'].corr(ver['fuel_kg_burned']):.4f}"
        )
    return practice, race


def score_b_2026(practice: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    train = flying_laps(add_wear_y(practice))
    race_f = flying_laps(add_wear_y(race))
    prior = pd.concat(
        [fit_practice_prior(train), fit_loeo_prior(race_f), fit_field_prior(race_f)],
        ignore_index=True,
    )
    stints = score_stints(race_f, prior)
    _safe_to_csv(prior, PRIOR_B_2026)
    _safe_to_csv(stints, STINT_B_2026)

    print("\n=== Frozen-A leftover, B scored on 2026 (A not refit) ===")
    print(f"Practice flying laps: {len(train)}  Race flying laps: {len(race_f)}")
    _print_prior(prior)
    import model_b

    old = model_b.CHECKPOINT_EVENT
    for event in EVENTS_2026:
        model_b.CHECKPOINT_EVENT = event
        _print_checkpoint(stints)
    model_b.CHECKPOINT_EVENT = old
    _print_score(stints)
    return stints


if __name__ == "__main__":
    print(f"2026 test events: {EVENTS_2026}")
    print("2025 laps.csv / laps_model.csv will not be written.")
    if RAW_2026.exists():
        raw = pd.read_csv(RAW_2026)
        print(f"Reusing {RAW_2026} ({len(raw)} rows). Delete that file to re-download.")
    else:
        raw = ingest_2026()
    clean = clean_laps(raw)
    _safe_to_csv(clean, CLEAN_2026)
    print(f"\nClean 2026 rows: {len(clean)} -> {CLEAN_2026}")
    print(clean.groupby(["event", "session"]).size().to_string())

    practice, race = score_a_2026(clean)
    score_b_2026(practice, race)
    print(f"\nWrote {TRAIN_A_2026}")
    print(f"Wrote {RACE_A_2026}")
    print(f"Wrote {STINT_B_2026}")
    print("2025 training files were not touched. A and B were not refit.")
