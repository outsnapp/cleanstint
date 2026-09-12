"""
battery_proxy.py
================

CleanStint -- Battery State-of-Charge (SoC) Proxy for 2026 F1 Power Units
--------------------------------------------------------------------------

PURPOSE
-------
F1 does not publish MGU-K battery telemetry. This module reconstructs a
*proxy* for battery State-of-Charge -- specifically, position within the
regulated 4 MJ per-burst deployment swing -- from public telemetry alone
(Speed, Throttle, Brake, nGear, RPM), via FastF1.

The proxy exists to produce causal-ML control features for CleanStint's
tyre-degradation model:

    E_deploy_lap_mj
    E_harvest_lap_mj
    SoC_min_lap_mj
    empty_clip_duration_s
    superclip_duration_s
    lift_coast_duration_s
    q_tyre_lap_total  # optional convenience proxy

IMPORTANT
---------
This is an engineering PROXY, not ground truth.

True battery capacity, the ICE torque curve, the aero map, the brake-by-wire
split, and the internal energy-management control logic are proprietary and
NOT present in public telemetry.

Every constant that is not drawn from published regulations is a free
parameter calibrated from data. Treat outputs as relative, session-internal
signals for causal ML -- never as absolute truth.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import savgol_filter

try:
    import fastf1
except ImportError:
    fastf1 = None
    warnings.warn(
        "fastf1 is not installed. Install with: pip install fastf1",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    # --- 2026 FIA regulation reference values ---
    "mass_kg": 768.0,
    "mguk_max_power_w": 350_000.0,
    "mguk_max_crank_torque_nm": 500.0,
    "burst_cap_mj": 4.0,
    "harvest_limit_mj_per_lap": 8.5,
    "harvest_limit_bounds_mj": (5.0, 9.0),
    "overtake_bonus_mj": 0.5,
    "overtake_taper_leading_kph": 290.0,
    "overtake_taper_chasing_kph": 337.0,

    # Extra 2026 regulation context, not directly used in this MVP
    "torque_reduction_rate_limit_kw_s": 100.0,
    "ice_power_floor_kw": 450.0,

    # --- Physical constants ---
    "air_density_kg_m3": 1.225,
    "g_ms2": 9.81,

    # --- Free/calibrated parameter initial guesses ---
    "eta_deploy_init": 0.92,
    "eta_recover_init": 0.75,
    "k_deploy_init_n": 6000.0,
    "k_brake_init": 0.45,
    "cda_init_m2": 1.10,
    "crr_init": 0.012,

    # --- Detection thresholds ---
    "wot_throttle_pct": 98.0,
    "clip_deficit_g_threshold": 0.03,
    "heavy_brake_decel_ms2": 15.0,
    "superclip_post_brake_window_s": 2.5,
    "superclip_max_duration_s": 1.0,
    "lift_coast_min_duration_s": 0.5,
    "resample_dt_s": 0.10,
}


def make_config(year: int = 2026) -> dict:
    """
    Returns regulation-aware config.

    For 2026:
        Uses 2026 PU regulation reference values.

    For <=2025:
        Uses approximate older PU values only so the pipeline can be tested
        on historical data without mixing wrong rules into the main model.
    """
    cfg = BASE_CONFIG.copy()

    if year <= 2025:
        cfg.update(
            mass_kg=798.0,
            mguk_max_power_w=120_000.0,
            mguk_max_crank_torque_nm=250.0,
            burst_cap_mj=4.0,
            harvest_limit_mj_per_lap=2.0,
            harvest_limit_bounds_mj=(2.0, 2.0),
            overtake_bonus_mj=0.0,
            overtake_taper_leading_kph=999.0,
            overtake_taper_chasing_kph=999.0,
        )

    return cfg


# Default config for imported functions
CONFIG = make_config(2026)


# ---------------------------------------------------------------------------
# Basic physics helpers
# ---------------------------------------------------------------------------

def kph_to_ms(v_kph):
    """Convert km/h -> m/s."""
    return np.asarray(v_kph, dtype=float) / 3.6


def drag_force(v_ms, cda_m2, rho=CONFIG["air_density_kg_m3"]):
    """Aerodynamic drag force [N]: F = 0.5 * rho * CdA * v^2."""
    v_ms = np.asarray(v_ms, dtype=float)
    return 0.5 * rho * cda_m2 * np.square(v_ms)


def rolling_resistance_force(mass_kg, crr, g=CONFIG["g_ms2"]):
    """Rolling resistance [N]: Crr * m * g."""
    return crr * mass_kg * g


def mguk_force_limit(v_ms, k_deploy_n, config, overtake_mode="normal"):
    """
    Max MGU-K forward force [N] achievable at speed v_ms, ignoring SoC
    availability.

    - Torque-limited below crossover.
    - Power-limited above crossover.
    - Tapered above regulation taper speed.
    """
    v_ms = np.asarray(v_ms, dtype=float)
    p_max = config["mguk_max_power_w"]

    f_power_limited = p_max / np.maximum(v_ms, 1e-3)
    f_torque_power_limited = np.minimum(k_deploy_n, f_power_limited)

    taper_start_kph = (
        config["overtake_taper_chasing_kph"]
        if overtake_mode == "overtake_chasing"
        else config["overtake_taper_leading_kph"]
    )

    # Assumed taper width. The regulations give taper start speed,
    # but not exact taper width.
    taper_width_kph = 15.0
    taper_end_kph = taper_start_kph + taper_width_kph

    taper_start_ms = taper_start_kph / 3.6
    taper_end_ms = taper_end_kph / 3.6

    taper_factor = np.clip(
        (taper_end_ms - v_ms) / max(taper_end_ms - taper_start_ms, 1e-6),
        0.0,
        1.0,
    )
    taper_factor = np.where(v_ms <= taper_start_ms, 1.0, taper_factor)

    return f_torque_power_limited * taper_factor


def soc_availability_factor(soc_mj, ramp_width_mj=0.2):
    """
    Smooth ramp-down of deployable force as SoC nears floor.

    This is more realistic than a hard cutoff and gives the optimizer
    a smoother loss surface.
    """
    return float(np.clip(soc_mj / ramp_width_mj, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def compute_kinematics(df: pd.DataFrame, config: dict = CONFIG) -> pd.DataFrame:
    """
    Builds uniformly-sampled kinematics DataFrame from raw FastF1 telemetry.

    Outputs:
        t_s
        speed_ms
        accel_ms2
        Throttle
        Brake
        nGear
        RPM
    """
    if "Time" not in df.columns or "Speed" not in df.columns:
        raise ValueError("df must contain 'Time' and 'Speed' columns")

    df = df.copy()

    if np.issubdtype(df["Time"].dtype, np.timedelta64):
        df["t_s"] = df["Time"].dt.total_seconds()
    else:
        df["t_s"] = pd.to_numeric(df["Time"], errors="coerce")

    df = df.drop_duplicates(subset="t_s").sort_values("t_s")

    df["Speed"] = pd.to_numeric(df["Speed"], errors="coerce").interpolate(
        limit=5,
        limit_direction="both",
    )

    df = df.dropna(subset=["Speed", "t_s"]).reset_index(drop=True)

    if len(df) < 11:
        raise ValueError("Not enough valid samples after cleaning.")

    dt = config["resample_dt_s"]
    t_uniform = np.arange(df["t_s"].iloc[0], df["t_s"].iloc[-1], dt)

    if len(t_uniform) < 11:
        raise ValueError("Segment too short after resampling.")

    v_kph_uniform = np.interp(t_uniform, df["t_s"], df["Speed"])
    v_ms = kph_to_ms(v_kph_uniform)

    max_window = len(v_ms) - 1
    if max_window < 5:
        raise ValueError("Not enough samples for smoothing.")

    window = min(11, max_window)
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)

    v_ms_smooth = savgol_filter(
        v_ms,
        window_length=window,
        polyorder=2,
        mode="interp",
    )

    a_ms2 = savgol_filter(
        v_ms,
        window_length=window,
        polyorder=2,
        deriv=1,
        delta=dt,
        mode="interp",
    )

    out = pd.DataFrame(
        {
            "t_s": t_uniform,
            "speed_ms": v_ms_smooth,
            "accel_ms2": a_ms2,
        }
    )

    for col in ["Throttle", "Brake", "nGear", "RPM"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
            out[col] = np.interp(t_uniform, df["t_s"], vals)
        else:
            out[col] = np.nan

    return out


# ---------------------------------------------------------------------------
# Empirical no-clip baseline envelope
# ---------------------------------------------------------------------------

def build_baseline_envelope(
    kin_df: pd.DataFrame,
    config: dict = CONFIG,
    speed_bin_width_ms: float = 2.0,
):
    """
    Empirically estimates the no-clip full-throttle acceleration envelope.

    We do not model ICE torque parametrically because public data does not
    contain the ICE torque curve or gear ratios.

    Instead:
        For WOT samples, bin by speed and take 95th percentile acceleration.
    """
    wot = kin_df[kin_df["Throttle"] >= config["wot_throttle_pct"]].dropna(
        subset=["speed_ms", "accel_ms2"]
    ).copy()

    if wot.empty:
        raise ValueError("No full-throttle samples found.")

    lo = wot["speed_ms"].min()
    hi = wot["speed_ms"].max()

    if hi - lo < speed_bin_width_ms:
        q = wot["accel_ms2"].quantile(0.95)
        return lambda v: np.full_like(np.asarray(v, dtype=float), q)

    bins = np.arange(lo, hi + speed_bin_width_ms, speed_bin_width_ms)
    wot["speed_bin"] = pd.cut(wot["speed_ms"], bins)

    tbl = (
        wot.groupby("speed_bin", observed=True)["accel_ms2"]
        .quantile(0.95)
        .dropna()
    )

    bin_centers = np.array([iv.mid for iv in tbl.index])
    vals = tbl.to_numpy()

    order = np.argsort(bin_centers)
    bin_centers = bin_centers[order]
    vals = vals[order]

    def envelope(v_ms):
        return np.interp(
            v_ms,
            bin_centers,
            vals,
            left=vals[0],
            right=vals[-1],
        )

    return envelope


# ---------------------------------------------------------------------------
# Clip detection
# ---------------------------------------------------------------------------

def detect_clip_events(kin_df: pd.DataFrame, envelope, config: dict = CONFIG):
    """
    Flags samples where car is WOT but underperforms the empirical envelope.

    Groups consecutive flagged samples into discrete clip events.
    """
    kin_df = kin_df.copy()

    kin_df["expected_accel"] = envelope(kin_df["speed_ms"].to_numpy())
    kin_df["deficit"] = kin_df["expected_accel"] - kin_df["accel_ms2"]

    is_wot = kin_df["Throttle"] >= config["wot_throttle_pct"]
    is_deficit = kin_df["deficit"] > config["clip_deficit_g_threshold"]

    kin_df["is_clip_sample"] = (is_wot & is_deficit).fillna(False)

    if kin_df["is_clip_sample"].sum() == 0:
        return kin_df, []

    kin_df["clip_group"] = (
        kin_df["is_clip_sample"] != kin_df["is_clip_sample"].shift()
    ).cumsum()

    events = []

    for _, g in kin_df[kin_df["is_clip_sample"]].groupby("clip_group"):
        events.append(
            {
                "t_start": float(g["t_s"].iloc[0]),
                "t_end": float(g["t_s"].iloc[-1]),
                "duration_s": float(
                    g["t_s"].iloc[-1]
                    - g["t_s"].iloc[0]
                    + config["resample_dt_s"]
                ),
                "speed_onset_ms": float(g["speed_ms"].iloc[0]),
                "mean_deficit_ms2": float(g["deficit"].mean()),
                "indices": g.index.to_numpy(),
            }
        )

    events = [e for e in events if e["duration_s"] >= 0.3]
    return kin_df, events


def time_since_last_heavy_brake(kin_df: pd.DataFrame, config: dict = CONFIG) -> np.ndarray:
    """
    Seconds elapsed since most recent heavy braking sample.
    """
    heavy_brake = kin_df["accel_ms2"].to_numpy() < -config["heavy_brake_decel_ms2"]
    t_s = kin_df["t_s"].to_numpy()

    last_brake_t = np.where(heavy_brake, t_s, np.nan)
    last_brake_t = pd.Series(last_brake_t).ffill().to_numpy()

    delta = t_s - last_brake_t
    delta[np.isnan(last_brake_t)] = np.inf

    return delta


# ---------------------------------------------------------------------------
# Superclip vs Empty-Battery-Clip classification
# ---------------------------------------------------------------------------

def classify_clip_events(
    kin_df: pd.DataFrame,
    events: list,
    harvest_state: dict,
    config: dict = CONFIG,
) -> list:
    """
    Implements Superclip vs Empty-Battery-Clip heuristic.

    Rule order:
        1. Burst deployment already near 4 MJ cap -> empty_battery
        2. Harvest near lap cap shortly after heavy braking -> superclip
        3. Short event shortly after heavy braking -> superclip
        4. Default -> empty_battery
    """
    if not events:
        return []

    time_since_brake = time_since_last_heavy_brake(kin_df, config)
    labels = []

    for ev in events:
        i0 = int(ev["indices"][0])

        deployed_this_burst = harvest_state["cum_deployed_mj"][i0]
        harvested_this_lap = harvest_state["cum_harvested_mj"][i0]
        harvest_cap = harvest_state["harvest_limit_mj"]
        t_since_brake = time_since_brake[i0]

        soc_at_onset = kin_df["soc_burst_mj"].to_numpy()[i0]
        if soc_at_onset < 0.4:
            label = "empty_battery"
        elif (soc_at_onset > 1.5
              and t_since_brake < config["superclip_post_brake_window_s"]):
            label = "superclip"
        elif deployed_this_burst >= config["burst_cap_mj"] - 0.05:
            label = "empty_battery"

        elif (
            harvested_this_lap >= harvest_cap - 0.3
            and t_since_brake < config["superclip_post_brake_window_s"]
        ):
            label = "superclip"

        elif (
            ev["duration_s"] < config["superclip_max_duration_s"]
            and t_since_brake < config["superclip_post_brake_window_s"]
        ):
            label = "superclip"

        else:
            label = "empty_battery"

        labels.append(label)

    return labels


# ---------------------------------------------------------------------------
# Lift-and-coast detection
# ---------------------------------------------------------------------------

def detect_lift_and_coast(kin_df: pd.DataFrame, config: dict = CONFIG) -> np.ndarray:
    """
    Detects lift-and-coast events:
        Throttle < 10%
        Brake < 5%
        Speed decreasing
        sustained for >= lift_coast_min_duration_s
    """
    throttle = kin_df["Throttle"].fillna(0).to_numpy()
    brake = kin_df["Brake"].fillna(0).to_numpy()
    accel = kin_df["accel_ms2"].to_numpy()

    raw_flag = (throttle < 10.0) & (brake < 5.0) & (accel < 0.0)

    window = max(
        3,
        int(config["lift_coast_min_duration_s"] / config["resample_dt_s"])
    )

    flag_series = pd.Series(raw_flag)

    sustained = (
        flag_series.rolling(window=window, center=True, min_periods=window)
        .sum()
        .fillna(0)
        .to_numpy()
        >= window
    )

    return sustained


# ---------------------------------------------------------------------------
# SoC recurrence
# ---------------------------------------------------------------------------

@dataclass
class SocParams:
    eta_deploy: float
    eta_recover: float
    k_deploy_n: float
    k_brake: float
    cda_m2: float
    crr: float


def soc_recurrence(
    kin_df: pd.DataFrame,
    params: SocParams,
    config: dict = CONFIG,
    harvest_limit_mj: Optional[float] = None,
    overtake_mask: Optional[np.ndarray] = None,
    initial_soc: Optional[float] = None,
):
    """
    Runs per-sample SoC recurrence.

    Returns:
        kin_df with SoC columns
        harvest_state dict
    """
    dt = config["resample_dt_s"]
    n = len(kin_df)

    v_ms = kin_df["speed_ms"].to_numpy()
    a_ms2 = kin_df["accel_ms2"].to_numpy()
    throttle = kin_df["Throttle"].fillna(0).to_numpy()

    if "Brake" in kin_df.columns:
        brake_raw = kin_df["Brake"].fillna(0).to_numpy()
    else:
        brake_raw = np.zeros(n)

    harvest_limit_mj = harvest_limit_mj or config["harvest_limit_mj_per_lap"]

    if overtake_mask is None:
        overtake_mask = np.zeros(n, dtype=bool)

    f_drag = drag_force(v_ms, params.cda_m2, config["air_density_kg_m3"])
    f_roll_scalar = rolling_resistance_force(
        config["mass_kg"],
        params.crr,
        config["g_ms2"],
    )
    f_roll = np.full_like(v_ms, float(f_roll_scalar), dtype=float)

    is_wot = throttle >= config["wot_throttle_pct"]
    is_braking = brake_raw > 0

    if not is_braking.any():
        is_braking = a_ms2 < -1.0

    soc_raw = np.zeros(n)
    soc_burst = np.zeros(n)
    cum_deployed_mj = np.zeros(n)
    cum_harvested_mj = np.zeros(n)
    f_mguk_actual = np.zeros(n)
    f_brake_regen = np.zeros(n)

    soc = config["burst_cap_mj"]
    dep_acc = 0.0
    harv_acc = 0.0

    for i in range(n):
        v = max(v_ms[i], 0.1)
        mode = "overtake_chasing" if overtake_mask[i] else "normal"

        if is_wot[i]:
            f_limit = float(
                mguk_force_limit(
                    np.array([v]),
                    params.k_deploy_n,
                    config,
                    mode,
                )[0]
            )
            avail = soc_availability_factor(soc)
            f_dep = f_limit * avail
            e_dep = params.eta_deploy * f_dep * v * dt / 1e6
        else:
            f_dep = 0.0
            e_dep = 0.0

        if is_braking[i]:
            f_total_brake = max(
                config["mass_kg"] * abs(a_ms2[i]) - f_drag[i] - f_roll[i],
                0.0,
            )
            f_regen = params.k_brake * f_total_brake
            e_harv = params.eta_recover * f_regen * v * dt / 1e6
            if harv_acc >= harvest_limit_mj:
                e_harv = 0.0  # FIA per-lap harvest cap: excess is dumped, not stored
        else:
            f_regen = 0.0
            e_harv = 0.0

        soc = soc - e_dep + e_harv
        soc_clamped = float(np.clip(soc, 0.0, config["burst_cap_mj"]))

        if soc_clamped >= config["burst_cap_mj"] - 1e-6:
            dep_acc = 0.0
        else:
            dep_acc += e_dep

        harv_acc += e_harv

        soc_raw[i] = soc
        soc_burst[i] = soc_clamped
        cum_deployed_mj[i] = dep_acc
        cum_harvested_mj[i] = harv_acc
        f_mguk_actual[i] = f_dep
        f_brake_regen[i] = f_regen

        soc = soc_clamped

    out = kin_df.copy()
    out["soc_raw_mj"] = soc_raw
    out["soc_burst_mj"] = soc_burst
    out["f_mguk_actual_n"] = f_mguk_actual
    out["f_brake_regen_n"] = f_brake_regen

    harvest_state = {
        "cum_deployed_mj": cum_deployed_mj,
        "cum_harvested_mj": cum_harvested_mj,
        "harvest_limit_mj": harvest_limit_mj,
    }

    return out, harvest_state


def model_predicted_accel(
    kin_df,
    params: SocParams,
    config=CONFIG,
    harvest_limit_mj=None,
    overtake_mask=None,
):
    """
    Model-predicted acceleration at WOT samples using SoC-limited MGU-K force.

    This is used only for calibration loss construction.
    """
    kin2, _ = soc_recurrence(
        kin_df,
        params,
        config,
        harvest_limit_mj,
        overtake_mask,
    )

    v_ms = kin2["speed_ms"].to_numpy()

    f_drag = drag_force(v_ms, params.cda_m2, config["air_density_kg_m3"])
    f_roll_scalar = rolling_resistance_force(
        config["mass_kg"],
        params.crr,
        config["g_ms2"],
    )
    f_roll = np.full_like(v_ms, float(f_roll_scalar), dtype=float)

    f_mguk = kin2["f_mguk_actual_n"].to_numpy()

    a_pred = (f_mguk - f_drag - f_roll) / config["mass_kg"]

    return a_pred, kin2


# ---------------------------------------------------------------------------
# Calibration loss + optimizer
# ---------------------------------------------------------------------------

def loss_function(
    theta,
    kin_df,
    envelope,
    config=CONFIG,
    lam=1.0,
    harvest_limit_mj=None,
    overtake_mask=None,
):
    """
    Calibration loss.

    L_floor:
        Empty-battery clip events should occur near SoC floor.

    L_fit:
        Model-predicted acceleration deficit should match observed deficit,
        excluding superclip events.

    penalty:
        Keeps parameters physically plausible.
    """
    eta_deploy, eta_recover, k_deploy_n, k_brake = theta[:4]

    params = SocParams(
        eta_deploy=eta_deploy,
        eta_recover=eta_recover,
        k_deploy_n=k_deploy_n,
        k_brake=k_brake,
        cda_m2=config["cda_init_m2"],
        crr=config["crr_init"],
    )

    kin2, harvest_state = soc_recurrence(
        kin_df,
        params,
        config,
        harvest_limit_mj,
        overtake_mask,
    )

    kin2, events = detect_clip_events(kin2, envelope, config)
    labels = classify_clip_events(kin2, events, harvest_state, config)

    empty_idx = np.array([], dtype=int)
    superclip_idx = set()

    if any(lab == "empty_battery" for lab in labels):
        empty_idx = np.concatenate(
            [
                ev["indices"]
                for ev, lab in zip(events, labels)
                if lab == "empty_battery"
            ]
        )

    if any(lab == "superclip" for lab in labels):
        superclip_idx = set(
            np.concatenate(
                [
                    ev["indices"]
                    for ev, lab in zip(events, labels)
                    if lab == "superclip"
                ]
            ).tolist()
        )

    if len(empty_idx):
        l_floor = float(np.mean(kin2["soc_raw_mj"].to_numpy()[empty_idx] ** 2))
    else:
        l_floor = 0.0

    is_wot = kin2["Throttle"] >= config["wot_throttle_pct"]
    not_superclip = ~kin2.index.to_series().isin(superclip_idx)
    fit_mask = (is_wot & not_superclip).to_numpy()

    if fit_mask.any():
        expected = envelope(kin2.loc[fit_mask, "speed_ms"].to_numpy())

        a_pred, _ = model_predicted_accel(
            kin_df,
            params,
            config,
            harvest_limit_mj,
            overtake_mask,
        )

        deficit_actual = expected - kin2.loc[fit_mask, "accel_ms2"].to_numpy()
        deficit_model = expected - a_pred[fit_mask]

        l_fit = float(np.mean((deficit_actual - deficit_model) ** 2))
    else:
        l_fit = 0.0

    penalty = 0.0

    for val, lo, hi in [
        (eta_deploy, 0.5, 1.0),
        (eta_recover, 0.3, 1.0),
        (k_brake, 0.0, 1.0),
    ]:
        penalty += 1e3 * (min(val - lo, 0.0) ** 2 + max(val - hi, 0.0) ** 2)

    if k_deploy_n <= 0:
        penalty += 1e6

    total = l_floor + lam * l_fit + penalty

    if not np.isfinite(total):
        return 1e12

    return total


def calibrate(
    kin_df,
    envelope,
    config=CONFIG,
    lam=1.0,
    harvest_limit_mj=None,
    overtake_mask=None,
):
    """
    Two-stage optimizer:
        1. Coarse grid search
        2. Nelder-Mead polish
    """
    grid_eta_deploy = [0.88, 0.92, 0.96]
    grid_eta_recover = [0.65, 0.75, 0.85]
    grid_k_deploy = [4000.0, 6000.0, 8000.0]
    grid_k_brake = [0.35, 0.45, 0.55]

    best_loss = np.inf
    best_theta = None

    for ed in grid_eta_deploy:
        for er in grid_eta_recover:
            for kd in grid_k_deploy:
                for kb in grid_k_brake:
                    theta = [ed, er, kd, kb]

                    loss = loss_function(
                        theta,
                        kin_df,
                        envelope,
                        config,
                        lam,
                        harvest_limit_mj,
                        overtake_mask,
                    )

                    if np.isfinite(loss) and loss < best_loss:
                        best_loss = loss
                        best_theta = theta

    if best_theta is None:
        best_theta = [
            config["eta_deploy_init"],
            config["eta_recover_init"],
            config["k_deploy_init_n"],
            config["k_brake_init"],
        ]

    bounds = [
        (0.5, 1.0),
        (0.3, 1.0),
        (500.0, 15000.0),
        (0.0, 1.0),
    ]

    try:
        result = minimize(
            loss_function,
            x0=best_theta,
            args=(
                kin_df,
                envelope,
                config,
                lam,
                harvest_limit_mj,
                overtake_mask,
            ),
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "xatol": 1e-3,
                "fatol": 1e-3,
                "maxiter": 120,
            },
        )
    except Exception:
        result = minimize(
            loss_function,
            x0=best_theta,
            args=(
                kin_df,
                envelope,
                config,
                lam,
                harvest_limit_mj,
                overtake_mask,
            ),
            method="Nelder-Mead",
            options={
                "xatol": 1e-3,
                "fatol": 1e-3,
                "maxiter": 120,
            },
        )

    return {
        "theta": result.x,
        "loss": float(result.fun),
        "grid_best_loss": float(best_loss),
        "converged": bool(result.success),
        "raw_result": result,
    }


# ---------------------------------------------------------------------------
# Tyre coupling proxy + lap aggregation
# ---------------------------------------------------------------------------

def compute_tyre_coupling(
    kin_df,
    params: SocParams,
    config=CONFIG,
    k_slip_rear=0.02,
    k_regen_balance=0.01,
    mu_nominal=1.6,
):
    """
    Optional electro-thermal tyre-load proxy.

    IMPORTANT:
        MGU-K acts on the rear axle only.
        Harvesting also acts through the rear MGU-K, but changes brake bias.

        The final causal model should learn separate weights from:
            E_deploy_lap_mj
            E_harvest_lap_mj
            Clip_duration_lap
            LiftCoast_duration_lap

        q_tyre_lap_total is only a convenience proxy, not physical truth.
    """
    f_mguk = kin_df["f_mguk_actual_n"].to_numpy()
    f_regen = kin_df["f_brake_regen_n"].to_numpy()

    f_grip_limit = max(
        mu_nominal * config["mass_kg"] * config["g_ms2"],
        1.0,
    )

    slip_rear_deploy = np.clip(f_mguk / f_grip_limit, 0.0, 1.0)
    slip_rear_regen = np.clip(f_regen / f_grip_limit, 0.0, 1.0)

    out = kin_df.copy()

    out["q_tyre_w"] = (
        k_slip_rear * f_mguk * slip_rear_deploy
        + k_regen_balance * f_regen * slip_rear_regen
    )

    return out


def lap_aggregate_features(
    kin_df,
    harvest_state,
    events,
    labels,
    config=CONFIG,
):
    """
    Per-lap aggregation features for causal ML model.
    """
    dt = config["resample_dt_s"]

    n_empty = sum(1 for l in labels if l == "empty_battery")
    n_super = sum(1 for l in labels if l == "superclip")

    dur_empty = sum(
        ev["duration_s"]
        for ev, l in zip(events, labels)
        if l == "empty_battery"
    )

    dur_super = sum(
        ev["duration_s"]
        for ev, l in zip(events, labels)
        if l == "superclip"
    )

    lift_coast_mask = detect_lift_and_coast(kin_df, config)
    lift_coast_duration = float(lift_coast_mask.sum() * dt)

    if len(harvest_state["cum_deployed_mj"]):
        e_deploy = float(harvest_state["cum_deployed_mj"][-1])
    else:
        e_deploy = 0.0

    if len(harvest_state["cum_harvested_mj"]):
        e_harvest = float(harvest_state["cum_harvested_mj"][-1])
    else:
        e_harvest = 0.0

    return {
        "e_deploy_lap_mj": e_deploy,
        "e_harvest_lap_mj": e_harvest,
        "soc_min_lap_mj": float(kin_df["soc_burst_mj"].min()),
        "soc_at_lap_end_mj": float(kin_df["soc_burst_mj"].iloc[-1]),
        "n_empty_battery_clips": n_empty,
        "n_superclips": n_super,
        "empty_clip_duration_s": dur_empty,
        "superclip_duration_s": dur_super,
        "lift_coast_duration_s": lift_coast_duration,
        "q_tyre_lap_total": float(kin_df["q_tyre_w"].sum() * dt)
        if "q_tyre_w" in kin_df.columns
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# FastF1 loading
# ---------------------------------------------------------------------------

def load_lap_telemetry(
    year: int,
    gp: str,
    session_type: str,
    driver: Optional[str] = None,
    lap_number: Optional[int] = None,
) -> pd.DataFrame:
    """
    Loads one lap's car telemetry via FastF1.

    If driver is None, uses fastest lap from the session.
    """
    if fastf1 is None:
        raise ImportError("fastf1 is required. Install with: pip install fastf1")

    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)

    laps = session.laps

    if driver is not None:
        laps = laps.pick_driver(driver)

    if lap_number is not None:
        lap = laps[laps["LapNumber"] == lap_number].iloc[0]
    else:
        lap = laps.pick_fastest()

    car_data = lap.get_car_data()

    try:
        car_data = car_data.add_distance()
    except Exception:
        pass

    return car_data


def load_first_available_lap(events: list):
    """
    Tries multiple events until one loads successfully.

    Returns:
        telemetry_df, config, source_name
    """
    errors = []

    for year, gp, session_type in events:
        try:
            print(f"Trying: {year} {gp} {session_type} ...")
            tel = load_lap_telemetry(year, gp, session_type)
            cfg = make_config(year)
            source = f"{year} {gp} {session_type}"
            return tel, cfg, source
        except Exception as e:
            errors.append(f"{year} {gp} {session_type}: {e}")
            continue

    raise RuntimeError(
        "Could not load any FastF1 session.\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_telemetry(dt: float = 0.1) -> pd.DataFrame:
    """
    Generates a synthetic telemetry lap.

    This is ONLY for testing the pipeline when FastF1 is blocked or no
    real session can be downloaded.

    It is not real F1 data and must not be used for final validation.
    """
    t = np.arange(0.0, 90.0, dt)

    speed = []
    throttle = []
    brake = []
    gear = []
    rpm = []

    for ti in t:

        # First straight
        if ti < 18:
            phase = ti / 18.0
            v = 80.0 + 240.0 * phase
            thr = 100.0
            br = 0.0

        # Heavy braking zone
        elif ti < 23:
            phase = (ti - 18.0) / 5.0
            v = 320.0 - 230.0 * phase
            thr = 0.0
            br = 100.0

        # Corner exit
        elif ti < 28:
            phase = (ti - 23.0) / 5.0
            v = 90.0 + 50.0 * phase
            thr = 45.0
            br = 0.0

        # Second long straight
        elif ti < 48:
            phase = (ti - 28.0) / 20.0
            v = 140.0 + 180.0 * phase
            thr = 100.0
            br = 0.0

        # Flat full-throttle segment, mimics clipping / superclip-like dip
        elif ti < 50:
            v = 320.0
            thr = 100.0
            br = 0.0

        # Heavy braking zone 2
        elif ti < 58:
            phase = (ti - 50.0) / 8.0
            v = 320.0 - 220.0 * phase
            thr = 0.0
            br = 100.0

        # Final corner exit
        else:
            phase = (ti - 58.0) / (90.0 - 58.0)
            v = 100.0 + 80.0 * phase
            thr = 55.0
            br = 0.0

        speed.append(max(v, 0.0))
        gear.append(min(8, max(1, int(v // 40.0) + 1)))
        rpm.append(9000.0 + min(3000.0, v * 10.0))

    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(t, unit="s"),
            "Speed": speed,
            "Throttle": throttle,
            "Brake": brake,
            "nGear": gear,
            "RPM": rpm,
        }
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    tel_df: pd.DataFrame,
    config: dict = CONFIG,
    harvest_limit_mj: Optional[float] = None,
    overtake_mask: Optional[np.ndarray] = None,
    initial_soc: Optional[float] = None,
):
    """
    End-to-end:

        raw telemetry
        -> kinematics
        -> baseline envelope
        -> clip detection/classification
        -> calibration
        -> SoC trace
        -> tyre coupling proxy
        -> lap features
    """
    kin = compute_kinematics(tel_df, config)
    envelope = build_baseline_envelope(kin, config)

    calib = calibrate(
        kin,
        envelope,
        config,
        harvest_limit_mj=harvest_limit_mj,
        overtake_mask=overtake_mask,
    )

    eta_deploy, eta_recover, k_deploy_n, k_brake = calib["theta"]

    params = SocParams(
        eta_deploy=eta_deploy,
        eta_recover=eta_recover,
        k_deploy_n=k_deploy_n,
        k_brake=k_brake,
        cda_m2=config["cda_init_m2"],
        crr=config["crr_init"],
    )

    kin2, harvest_state = soc_recurrence(
        kin,
        params,
        config,
        harvest_limit_mj,
        overtake_mask,
    )

    kin2, events = detect_clip_events(kin2, envelope, config)
    labels = classify_clip_events(kin2, events, harvest_state, config)

    kin2 = compute_tyre_coupling(kin2, params, config)

    lap_features = lap_aggregate_features(
        kin2,
        harvest_state,
        events,
        labels,
        config,
    )

    lap_features["calibrated_theta"] = {
        "eta_deploy": float(eta_deploy),
        "eta_recover": float(eta_recover),
        "k_deploy_n": float(k_deploy_n),
        "k_brake": float(k_brake),
    }

    return kin2, calib, lap_features


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # REAL 2026 DATA STRATEGY
    # -----------------------------------------------------------------------
    # Since 2026 races have already happened, FastF1 can provide real 2026
    # speed/throttle/brake traces.
    #
    # However, public FastF1 telemetry still does NOT include real battery
    # State-of-Charge. That is still team-private.
    #
    # So we use real 2026 kinematic telemetry, then apply the 2026
    # regulation-capped physics proxy to estimate battery behavior.
    #
    # If FastF1 is blocked or unavailable, the script falls back to a
    # synthetic demo lap so the pipeline can still be tested.
    # -----------------------------------------------------------------------

    TARGET_EVENTS = [
        # Prefer real 2026 practice sessions first
        (2026, "Bahrain", "FP2"),
        (2026, "Saudi Arabia", "FP2"),
        (2026, "Australia", "FP2"),
        (2026, "Japan", "FP2"),
        (2026, "Miami", "FP2"),

        # If practice fails, try race sessions
        (2026, "Bahrain", "R"),
        (2026, "Saudi Arabia", "R"),
        (2026, "Australia", "R"),

        # Fallback historical data for pipeline testing only.
        # The code will automatically use older-regulation config for these.
        (2025, "Bahrain", "R"),
        (2024, "Bahrain", "R"),
    ]

    if fastf1 is not None:
        fastf1.Cache.enable_cache("./f1_cache")

        try:
            tel, cfg, source = load_first_available_lap(TARGET_EVENTS)
            print(f"\nLoaded real telemetry source: {source}\n")

        except Exception as e:
            print("\nFastF1 load failed:")
            print(e)
            print("\nFalling back to synthetic demo lap for pipeline testing.\n")

            tel = generate_synthetic_telemetry()
            cfg = make_config(2026)
            source = "SYNTHETIC DEMO LAP - NOT REAL DATA"

    else:
        print("\nfastf1 not installed. Using synthetic demo lap.\n")
        tel = generate_synthetic_telemetry()
        cfg = make_config(2026)
        source = "SYNTHETIC DEMO LAP - NOT REAL DATA"

    print("=" * 70)
    print("CLEANSTINT BATTERY PROXY PIPELINE")
    print("=" * 70)
    print("Source:", source)
    print("Regulation config year:")
    if "2026" in source:
        print("2026 PU regulations")
    elif "2025" in source:
        print("2025 PU approximation for historical testing")
    elif "2024" in source:
        print("2024 PU approximation for historical testing")
    else:
        print("2026 PU regulations applied to synthetic demo")

    print("\nRunning pipeline... This may take a little while on CPU.\n")

    kin_out, calib_out, features = run_pipeline(tel, config=cfg)

    print("=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print("Theta:", calib_out["theta"])
    print("Loss:", calib_out["loss"])
    print("Converged:", calib_out["converged"])

    print("\n" + "=" * 70)
    print("LAP FEATURES FOR CAUSAL MODEL")
    print("=" * 70)

    for k, v in features.items():
        if k == "calibrated_theta":
            continue

        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nCalibrated parameters:")
    for k, v in features["calibrated_theta"].items():
        print(f"{k}: {v:.4f}")

    print("\nDone.")