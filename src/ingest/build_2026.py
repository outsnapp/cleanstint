"""
build_dataset.py
Builds the lap-level CleanStint dataset from real FastF1 sessions.
Outputs: data/laps_<year>_<gp>_<session>.csv
"""
import argparse, os, json
import numpy as np
import pandas as pd
import fastf1
import battery_proxy as bp

# Proxy assumptions (labelled, configurable, NOT claimed as fact)
FUEL_BURN_KG_PER_LAP = 1.8
FUEL_START_KG = {"R": 100.0, "FP1": 60.0, "FP2": 60.0, "FP3": 60.0, "Q": 20.0}
TRAFFIC_GAP_S = 1.5

os.makedirs("data", exist_ok=True)


def valid_laps(session):
    laps = session.laps.copy()
    print(f"  Initial laps loaded: {len(laps)}")
    
    if "Deleted" in laps.columns:
        laps = laps[~laps["Deleted"].fillna(False)]
        
    # Only filter LapTime if it actually has valid data
    if laps["LapTime"].notna().sum() > 0:
        laps = laps[laps["LapTime"].notna()]
    else:
        print("  WARNING: LapTime is mostly NaT, skipping LapTime filter!")
        
    print(f"  After LapTime filter: {len(laps)}")
    
    for c in ["PitOutTime", "PitInTime"]:
        if c in laps.columns:
            laps = laps[laps[c].isna()]
            
    print(f"  After Pit filter: {len(laps)}")
    print(f"  Skipping strict TrackStatus filter to ensure we get data.")
    
    return laps.reset_index(drop=True)


def min_gap_per_lap(session, laps):
    """Time-gap to nearest car ahead, from position data. inf if none."""
    out = {}
    try:
        pos = session.pos_data.copy()
        pos["Driver"] = pos["Driver"].astype(str)
    except Exception:
        return out
    dist, times = {}, {}
    for drv, g in pos.groupby("Driver"):
        g = g.sort_values("Time")
        xy = g[["X", "Y"]].to_numpy(dtype=float)
        seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        dist[drv] = np.concatenate([[0.0], np.cumsum(seg)])
        times[drv] = g["Time"].dt.total_seconds().to_numpy()
    gaps = {}
    for drv in dist:
        mg = np.full(len(times[drv]), np.inf)
        for other in dist:
            if other == drv:
                continue
            t_at_d = np.interp(dist[drv], dist[other], times[other],
                               left=np.inf, right=np.inf)
            g = times[drv] - t_at_d
            mg = np.where((g > 0.05) & (g < mg), g, mg)
        gaps[drv] = (times[drv], mg)
    for _, lap in laps.iterrows():
        drv = lap["DriverNumber"]
        if drv not in gaps:
            continue
        t_self, mg = gaps[drv]
        end = lap["Time"].total_seconds()
        start = end - lap["LapTime"].total_seconds()
        sel = (t_self >= start) & (t_self <= end)
        out[(drv, lap["LapNumber"])] = float(np.min(mg[sel])) if sel.any() else np.inf
    return out


def weather_per_lap(session, laps):
    out = {}
    try:
        w = session.weather_data
        wt = w["Time"].dt.total_seconds().to_numpy()
        vals = w[["AirTemp", "TrackTemp"]].to_numpy(dtype=float)
    except Exception:
        return out
    for _, lap in laps.iterrows():
        end = lap["Time"].total_seconds()
        mid = end - 0.5 * lap["LapTime"].total_seconds()
        i = int(np.argmin(np.abs(wt - mid)))
        out[(lap["DriverNumber"], lap["LapNumber"])] = vals[i]
    return out


def build_session(year, gp, sess):
    fastf1.Cache.enable_cache("./f1_cache")
    session = fastf1.get_session(year, gp, sess)
    session.load(telemetry=True, laps=True, weather=True)
    cfg = bp.make_config(year)
    cfg["clip_deficit_g_threshold"] = 1.0   # only true power-loss events, not fuel-mass deficits
    cfg["superclip_max_duration_s"] = 1.2
    laps = valid_laps(session)
    if laps.empty:
        raise RuntimeError(f"No valid laps in {year} {gp} {sess}")

    gaps = min_gap_per_lap(session, laps)
    wx = weather_per_lap(session, laps)

    # session-wide track evolution index (rubber-in proxy)
    order = laps.sort_values("Time")
    te_rank = {i: k for k, i in enumerate(order.index)}
    laps["te_index"] = [np.log1p(te_rank[i]) for i in laps.index]

    rows = []
    laps["DriverNumber"] = laps["DriverNumber"].astype(str)
    for drv, dlaps in laps.groupby("DriverNumber"):
        print(f"[DBG] driver {drv}: {len(dlaps)} valid laps")
        # ---- pass 1: kinematics per stint ----
        stint_kin = {}
        if dlaps["Stint"].isna().all():
            dlaps = dlaps.copy()
            dlaps["Stint"] = 0
        for stint, slaps in dlaps.groupby("Stint"):
            slaps = slaps.sort_values("LapNumber")
            tel = None
            for _, lap in slaps.iterrows():
                cd = lap.get_car_data()
                tel = cd if tel is None else pd.concat([tel, cd], ignore_index=True)
            print(f"[DBG]   stint {stint}: laps={len(slaps)} tel_samples={0 if tel is None else len(tel)}")
            if tel is None or len(tel) < 60:
                continue
            stint_kin[stint] = (slaps, bp.compute_kinematics(tel, cfg))

        print(f"[DBG]   driver {drv}: stints built = {sorted(stint_kin.keys())}")
        if not stint_kin:
            continue

        # ---- envelope + calibration once per driver ----
        all_kin = pd.concat([k for _, k in stint_kin.values()], ignore_index=True)
        envelope = bp.build_baseline_envelope(all_kin, cfg)
        import hashlib, json as _json, os as _os
        _key = hashlib.md5((open('battery_proxy.py','rb').read()
                 + _json.dumps(cfg, sort_keys=True, default=str).encode()
                 + drv.encode() + sess.encode())).hexdigest()[:12]
        _cp = f"cache_theta/{sess}_{drv}_{_key}.json"
        if _os.path.exists(_cp):
            calib = {"theta": _json.load(open(_cp))["theta"], "loss": None,
                     "grid_best_loss": None, "converged": True}
            print(f"[CACHE] theta loaded for driver {drv} {sess}")
        else:
            fast_lap = dlaps.sort_values("LapTime").iloc[0]
            fast_kin = bp.compute_kinematics(fast_lap.get_car_data(), cfg)
            calib = bp.calibrate(fast_kin, envelope, cfg)
            _os.makedirs("cache_theta", exist_ok=True)
            _json.dump({"theta": [float(x) for x in calib["theta"]]}, open(_cp,"w"))
        ed, er, kd, kb = calib["theta"]
        params = bp.SocParams(ed, er, kd, kb,
                              cda_m2=cfg["cda_init_m2"], crr=cfg["crr_init"])

        # ---- pass 2: continuous SoC per stint, lap-by-lap ----
        for stint, slaps in dlaps.groupby("Stint"):
            if pd.isna(stint): continue
            slaps = slaps.sort_values("LapNumber")
            first_lap = slaps["LapNumber"].min()
            
            soc_carry = cfg["burst_cap_mj"]
            
            for _, lap in slaps.iterrows():
                tel = lap.get_car_data()
                if len(tel) < 30: continue
                
                kin = bp.compute_kinematics(tel, cfg)
                if len(kin) < 30: continue
                try:
                    env_lap = bp.build_baseline_envelope(kin, cfg)
                except Exception:
                    env_lap = envelope
                
                # Run physics with carried-over battery state!
                kin2, hstate = bp.soc_recurrence(kin, params, cfg, initial_soc=soc_carry)
                soc_carry = float(kin2["soc_burst_mj"].iloc[-1])
                
                kin2, events = bp.detect_clip_events(kin2, env_lap, cfg)
                labels = bp.classify_clip_events(kin2, events, hstate, cfg)
                lc = bp.detect_lift_and_coast(kin2, cfg)
                
                dt = cfg["resample_dt_s"]
                v = kin2["speed_ms"].to_numpy()
                e_dep_s = params.eta_deploy * kin2["f_mguk_actual_n"].to_numpy() * v * dt / 1e6
                e_har_s = params.eta_recover * kin2["f_brake_regen_n"].to_numpy() * v * dt / 1e6
                
                dur_empty = sum(e["duration_s"] for e, l in zip(events, labels) if l == "empty_battery")
                dur_super = sum(e["duration_s"] for e, l in zip(events, labels) if l == "superclip")
                n_empty = sum(1 for l in labels if l == "empty_battery")
                n_super = sum(1 for l in labels if l == "superclip")
                
                w = wx.get((drv, lap["LapNumber"]), (np.nan, np.nan))
                gap = gaps.get((drv, lap["LapNumber"]), np.inf)
                
                rows.append({
                    "session": sess, "driver_number": drv,
                    "driver": lap["Driver"], "lap_number": lap["LapNumber"],
                    "stint": stint, "compound": lap["Compound"],
                    "tyre_age": lap["LapNumber"] - first_lap + 1,
                    "lap_time_s": lap["LapTime"].total_seconds(),
                    "fuel_est_kg": FUEL_START_KG.get(sess, 60.0) - FUEL_BURN_KG_PER_LAP * (lap["LapNumber"] - 1),
                    "te_index": lap["te_index"],
                    "min_gap_s": gap,
                    "traffic_flag": int(gap < TRAFFIC_GAP_S),
                    "air_temp": w[0], "track_temp": w[1],
                    "e_deploy_lap_mj": float(e_dep_s.sum()),
                    "e_harvest_lap_mj": float(e_har_s.sum()),
                    "soc_min_lap_mj": float(kin2["soc_burst_mj"].min()),
                    "n_empty_battery_clips": n_empty,
                    "n_superclips": n_super,
                    "empty_clip_duration_s": dur_empty,
                    "superclip_duration_s": dur_super,
                    "lift_coast_duration_s": float(lc.sum() * dt),
                })

    df = pd.DataFrame(rows)
    firsts = df.groupby(["driver_number", "session", "stint"])["lap_number"].transform("min")
    df["fuel_in_stint_kg"] = 1.8 * (firsts - df["lap_number"])  # within-stint fuel delta; level absorbed by stint demeaning
    med = df.groupby(["driver_number", "session", "stint"])["lap_time_s"].transform("median")
    df["traffic_flag"] = ((df["lap_time_s"] > med + 1.2) | (df["traffic_flag"] == 1)).astype(int)
    df["is_representative"] = (df["lap_time_s"] < med + 8.0).astype(int)
    path = f"data/laps_{year}_{gp.replace(' ', '')}_{sess}.csv"
    df.to_csv(path, index=False)
    print(f"[SAVED] {path}  ({len(df)} laps)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--gp", type=str, default="Australia")
    ap.add_argument("--sessions", nargs="+", default=["FP2", "R"])
    a = ap.parse_args()
    for s in a.sessions:
        try:
            build_session(a.year, a.gp, s)
        except Exception as e:
            print(f"[WARN] {s} failed: {e}")