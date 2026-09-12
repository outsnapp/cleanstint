"""Load FastF1 sessions and write one Track 3 laps CSV (all events in one file)."""

from pathlib import Path

import fastf1
import pandas as pd

from config import (
    CHECKPOINT_DRIVER,
    EVENTS,
    FUEL_BURN_KG_PER_LAP,
    HARVEST_MJ_BY_EVENT,
    HARVEST_MJ_DEFAULT,
    SESSIONS,
    TIME_GAIN_S_PER_KG,
    YEAR,
)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
PROCESSED_DIR = ROOT / "data" / "processed"
LAPS_CSV = PROCESSED_DIR / "laps.csv"


def enable_cache() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return CACHE_DIR


def load_session(year: int, event: str, session_name: str):
    session = fastf1.get_session(year, event, session_name)
    # Plumbing: laps only. Telemetry comes later.
    session.load(telemetry=False, weather=True, messages=False)
    if session.laps is None or session.laps.empty:
        raise RuntimeError(f"No laps loaded for {year} {event} {session_name}")
    return session


def download_weekend(year: int, event: str, sessions: list[str]) -> dict:
    loaded = {}
    for session_name in sessions:
        print(f"Loading {year} {event} {session_name}...")
        session = load_session(year, event, session_name)
        n_laps = len(session.laps)
        print(f"  OK - {n_laps} laps")
        loaded[session_name] = session
    return loaded


def print_race_checkpoint(race_session, driver: str, year: int) -> pd.DataFrame:
    laps = race_session.laps.pick_drivers(driver)
    if laps.empty:
        available = sorted(race_session.laps["Driver"].dropna().unique())
        raise RuntimeError(
            f"No race laps for {driver}. Available: {', '.join(available)}"
        )
    cols = [c for c in ("Driver", "LapNumber", "LapTime") if c in laps.columns]
    table = laps[cols].copy()
    print(f"\nCheckpoint: {driver} race lap times ({year} race)")
    print(table.to_string(index=False))
    return table


def _td_seconds(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    return pd.NA


def session_to_rows(session, year: int, event: str, session_name: str) -> pd.DataFrame:
    laps = session.laps.copy()
    out = pd.DataFrame(
        {
            "year": year,
            "event": event,
            "session": session_name,
            "driver": laps["Driver"] if "Driver" in laps else pd.NA,
            "driver_number": laps["DriverNumber"] if "DriverNumber" in laps else pd.NA,
            "team": laps["Team"] if "Team" in laps else pd.NA,
            "lap_number": laps["LapNumber"] if "LapNumber" in laps else pd.NA,
            "stint": laps["Stint"] if "Stint" in laps else pd.NA,
            "compound": laps["Compound"] if "Compound" in laps else pd.NA,
            "tyre_life": laps["TyreLife"] if "TyreLife" in laps else pd.NA,
            "fresh_tyre": laps["FreshTyre"] if "FreshTyre" in laps else pd.NA,
            "lap_time_s": laps["LapTime"].map(_td_seconds) if "LapTime" in laps else pd.NA,
            "sector1_s": laps["Sector1Time"].map(_td_seconds) if "Sector1Time" in laps else pd.NA,
            "sector2_s": laps["Sector2Time"].map(_td_seconds) if "Sector2Time" in laps else pd.NA,
            "sector3_s": laps["Sector3Time"].map(_td_seconds) if "Sector3Time" in laps else pd.NA,
            "position": laps["Position"] if "Position" in laps else pd.NA,
            "track_status": laps["TrackStatus"] if "TrackStatus" in laps else pd.NA,
            "deleted": laps["Deleted"] if "Deleted" in laps else pd.NA,
            "is_accurate": laps["IsAccurate"] if "IsAccurate" in laps else pd.NA,
        }
    )
    pit_in = laps["PitInTime"] if "PitInTime" in laps else None
    pit_out = laps["PitOutTime"] if "PitOutTime" in laps else None
    if pit_in is not None or pit_out is not None:
        in_flag = pit_in.notna() if pit_in is not None else False
        out_flag = pit_out.notna() if pit_out is not None else False
        out["is_pit_lap"] = in_flag | out_flag
    else:
        out["is_pit_lap"] = False

    status = out["track_status"].astype(str)
    out["is_green"] = status.str.fullmatch(r"1+") | (status == "1")

    lap_n = pd.to_numeric(out["lap_number"], errors="coerce")
    if session_name == "R":
        kg_burned = (lap_n.fillna(1) - 1).clip(lower=0) * FUEL_BURN_KG_PER_LAP
        out["fuel_kg_burned"] = kg_burned
        out["lap_time_fuel_corr_s"] = out["lap_time_s"] + kg_burned * TIME_GAIN_S_PER_KG
    else:
        out["fuel_kg_burned"] = pd.NA
        out["lap_time_fuel_corr_s"] = pd.NA

    out["assumed_harvest_mj"] = HARVEST_MJ_BY_EVENT.get(event, HARVEST_MJ_DEFAULT)

    weather = getattr(session, "weather_data", None)
    if weather is not None and not weather.empty:
        out["air_temp_c"] = float(weather["AirTemp"].mean()) if "AirTemp" in weather else pd.NA
        out["track_temp_c"] = float(weather["TrackTemp"].mean()) if "TrackTemp" in weather else pd.NA
        if "Rainfall" in weather:
            rain = weather["Rainfall"]
            out["rainfall"] = bool(rain.fillna(False).astype(bool).any())
        else:
            out["rainfall"] = pd.NA
    else:
        out["air_temp_c"] = pd.NA
        out["track_temp_c"] = pd.NA
        out["rainfall"] = pd.NA

    return out


def build_laps_csv(year: int, events: list[str], sessions: list[str]) -> Path:
    """One CSV for every event/session. Later GPs append into this same file."""
    enable_cache()
    frames = []
    for event in events:
        loaded = download_weekend(year, event, sessions)
        for session_name, session in loaded.items():
            frames.append(session_to_rows(session, year, event, session_name))
    table = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(LAPS_CSV, index=False)
    print(f"\nWrote {len(table)} rows -> {LAPS_CSV}")
    print(table.groupby(["year", "event", "session"]).size().to_string())
    return LAPS_CSV


if __name__ == "__main__":
    enable_cache()
    print(f"FastF1 cache: {CACHE_DIR}")
    build_laps_csv(YEAR, EVENTS, SESSIONS)
