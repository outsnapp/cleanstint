"""Clean laps for modelling. Does not change data/processed/laps.csv."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "processed" / "laps.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "laps_model.csv"

DROP_COLS = ["deleted", "lap_time_fuel_corr_s"]
STINT_KEYS = ["year", "event", "session", "driver", "stint"]
FLYING_PAD_S = 5.0


def load_raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}. Run ingest.py first.")
    return pd.read_csv(RAW_CSV)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["lap_time_s"].notna()]
    out = out[~_as_bool(out["is_pit_lap"])]
    out = out[_as_bool(out["is_green"])]
    if "deleted" in out.columns:
        out = out[~_as_bool(out["deleted"])]
    if "is_accurate" in out.columns:
        out = out[_as_bool(out["is_accurate"])]

    med = out.groupby(STINT_KEYS)["lap_time_s"].transform("median")
    out = out[out["lap_time_s"] <= med + FLYING_PAD_S]

    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns])
    out["fuel_kg_burned"] = pd.to_numeric(out["fuel_kg_burned"], errors="coerce").fillna(0)

    session_dummies = pd.get_dummies(out["session"], prefix="sess")
    out = pd.concat([out, session_dummies], axis=1)
    return out.reset_index(drop=True)


def write_model_csv(df: pd.DataFrame) -> Path:
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    return CLEAN_CSV


if __name__ == "__main__":
    raw = load_raw()
    clean = clean_laps(raw)
    path = write_model_csv(clean)
    print(f"Raw rows:   {len(raw)}  ({RAW_CSV.name} unchanged)")
    print(f"Model rows: {len(clean)}  -> {path}")
    print(clean.groupby(["event", "session"]).size().to_string())
    extra = [c for c in clean.columns if c.startswith("sess_")]
    print("Session dummies:", extra)
