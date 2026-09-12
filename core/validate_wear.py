"""2025 race wear check: stint-demeaned slope of time vs tyre_life.

Same test as the circuit x compound table (laps, max_age, wear_s_per_lap,
cliff). Reads laps_model.csv. Does not touch laps.csv. Does not train Model B.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor

from config import TIME_GAIN_S_PER_KG

ROOT = Path(__file__).resolve().parents[1]
MODEL_CSV = ROOT / "data" / "processed" / "laps_model.csv"
RACE_A_CSV = ROOT / "data" / "processed" / "model_a_race.csv"
OUT_CSV = ROOT / "data" / "processed" / "wear_validation.csv"

STINT_KEYS = ["event", "driver", "stint", "compound"]
DRY = ("HARD", "MEDIUM", "SOFT")
COMPARE_EVENTS = ("Bahrain", "Imola", "Monza")
MIN_LAPS = 20
# Cliff = tyre laps to lose ~1s at the fitted linear rate. NaN if no real fade.
CLIFF_SECONDS = 1.0
MIN_WEAR_FOR_CLIFF = 0.02


def load_race() -> pd.DataFrame:
    if not MODEL_CSV.exists():
        raise FileNotFoundError(f"Missing {MODEL_CSV}. Run preprocess.py first.")
    df = pd.read_csv(MODEL_CSV)
    race = df[df["session"] == "R"].copy()
    race["fuel_corr_s"] = race["lap_time_s"] + TIME_GAIN_S_PER_KG * race["fuel_kg_burned"]
    if RACE_A_CSV.exists():
        a = pd.read_csv(RACE_A_CSV)
        cols = ["year", "event", "session", "driver", "lap_number", "stint", "residual_s"]
        have = [c for c in cols if c in a.columns]
        race = race.merge(a[have], on=[c for c in have if c != "residual_s"], how="left")
    return race.reset_index(drop=True)


def _stint_demean(series: pd.Series) -> pd.Series:
    return series - series.mean()


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    model = TheilSenRegressor(random_state=0, max_iter=500)
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0])


def _cliff(wear: float) -> float:
    if wear < MIN_WEAR_FOR_CLIFF:
        return float("nan")
    return round(CLIFF_SECONDS / wear, 1)


def wear_table(race: pd.DataFrame, y_col: str) -> pd.DataFrame:
    d = race[race[y_col].notna() & race["tyre_life"].notna()].copy()
    d["y_dm"] = d.groupby(STINT_KEYS, dropna=False)[y_col].transform(_stint_demean)
    rows = []
    for (event, compound), g in d.groupby(["event", "compound"], dropna=False):
        if compound not in DRY:
            continue
        if len(g) < MIN_LAPS:
            continue
        wear = _theil_sen(g["tyre_life"].to_numpy(float), g["y_dm"].to_numpy(float))
        rows.append(
            {
                "circuit": event,
                "compound": compound,
                "laps": int(len(g)),
                "max_age": float(g["tyre_life"].max()),
                "wear_s_per_lap": wear,
                "wear_clipped": max(wear, 0.0),
                "cliff_lap": _cliff(max(wear, 0.0)),
            }
        )
    out = pd.DataFrame(rows)
    order = {e: i for i, e in enumerate(["Bahrain", "Imola", "Monza", "Silverstone"])}
    comp = {c: i for i, c in enumerate(DRY)}
    return out.sort_values(
        by=["compound", "circuit"],
        key=lambda s: s.map(comp) if s.name == "compound" else s.map(order),
    ).reset_index(drop=True)


def mean_wear(table: pd.DataFrame, events: tuple[str, ...] = COMPARE_EVENTS) -> pd.Series:
    subset = table[table["circuit"].isin(events)]
    return subset.groupby("compound")["wear_clipped"].mean().reindex(list(DRY))


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    show = table.copy()
    show["wear_s_per_lap"] = show["wear_clipped"].map(lambda v: f"{v:.4f}")
    show["cliff_lap"] = show["cliff_lap"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    print(show[["circuit", "compound", "laps", "max_age", "wear_s_per_lap", "cliff_lap"]].to_string(index=False))
    print("\nUnclipped wear (negatives = fuel/traffic beating the tyre):")
    raw = table.copy()
    raw["wear_raw"] = raw["wear_s_per_lap"].map(lambda v: f"{v:.4f}")
    print(raw[["circuit", "compound", "laps", "wear_raw"]].to_string(index=False))


if __name__ == "__main__":
    race = load_race()
    print(f"Race laps: {len(race)} from {MODEL_CSV.name}")
    print("Method: stint-demean Y, Theil-Sen vs tyre_life, clip wear at 0 for the summary table.")
    print("Cliff: tyre laps to lose ~1.0s at that linear rate; blank if wear < 0.02 s/lap.")

    targets = [("lap_time_s", "2025 races, raw lap_time_s")]
    targets.append(("fuel_corr_s", f"2025 races, fuel-corrected (+{TIME_GAIN_S_PER_KG} s/kg burned)"))
    if "residual_s" in race.columns and race["residual_s"].notna().any():
        targets.append(("residual_s", "2025 races, Model A residual_s"))

    frames = []
    for y_col, title in targets:
        table = wear_table(race, y_col)
        table.insert(0, "y", y_col)
        _print_table(title, table)
        means = mean_wear(table)
        print(f"\nMean wear by compound (Bahrain/Imola/Monza, clipped):")
        print(means.map(lambda v: f"{v:.4f}" if pd.notna(v) else "").to_string())
        frames.append(table)

    out = pd.concat(frames, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    print("laps_model.csv was not overwritten. laps.csv was not touched.")
