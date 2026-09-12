"""
CleanStint — single-screen decision-first dashboard (final version)
"""
import glob, json, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CLEANSTINT", layout="wide", initial_sidebar_state="collapsed")

DATA_DIR = "data"
METRICS_PATH = "metrics.json"
PROJECTION_LAPS = 10
PIT_LANE_LOSS_S = 21.0
APP_VERSION = "CLEANSTINT v3.0"
ACCENT = "#ffb000"
RISK_COLORS = {"LOW": "#2ecc71", "MEDIUM": "#ffb000", "HIGH": "#ff4d4d", "—": "#555555"}

@st.cache_data(show_spinner=False)
def load_metrics():
    try:
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_laps():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "laps_*.csv")))
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

metrics = load_metrics()
laps = load_laps()

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def fmt(value, suffix="", digits=2):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "—"
        return f"{value:.{digits}f}{suffix}"
    except Exception:
        return "—"

def curve_value(compound, age):
    curve = safe_get(metrics, "clean_curves", compound)
    if not curve or not curve.get("ages") or not curve.get("vals"):
        return None
    ages = np.array(curve["ages"], dtype=float)
    vals = np.array(curve["vals"], dtype=float)
    if len(ages) == 0:
        return None
    return float(np.interp(age, ages, vals))

def cliff_for(compound):
    cliff = safe_get(metrics, "cliff_windows", compound)
    if isinstance(cliff, (list, tuple)) and len(cliff) == 2:
        try:
            return int(cliff[0]), int(cliff[1])
        except Exception:
            return None
    return None

def cliff_missing(compound):
    not_observed = metrics.get("cliff_not_observed_in_session") or []
    return compound is None or compound in not_observed or cliff_for(compound) is None

def wear_rate_for(compound):
    wr = safe_get(metrics, "wear_rate_s_per_lap_by_compound", compound)
    if wr is None:
        wr = metrics.get("causal_wear_rate_s_per_lap")
    return wr

def project_total(compound, start_age, box_at_age, n_laps):
    if compound is None:
        return None
    total, have_any, age, boxed = 0.0, False, start_age, False
    for _ in range(n_laps):
        age += 1
        if (not boxed) and (box_at_age is not None) and age >= box_at_age:
            total += PIT_LANE_LOSS_S
            boxed = True
            age = 1
        v = curve_value(compound, age)
        if v is not None:
            total += v
            have_any = True
    return total if have_any else None

def risk_chip_html(level):
    level = level if level in RISK_COLORS else "—"
    color = RISK_COLORS[level]
    return f'<span class="risk-chip" style="background:{color}22;color:{color};border:1px solid {color}66;">{level}</span>'

def build_chart(df_stint, compound, cliff):
    fig = go.Figure()
    has_raw = (
        df_stint is not None
        and not df_stint.empty
        and "tyre_age" in df_stint.columns
        and "lap_time_s" in df_stint.columns
    )
    y_lo, y_hi = None, None
    if has_raw:
        fig.add_trace(
            go.Scatter(
                x=df_stint["tyre_age"], y=df_stint["lap_time_s"], mode="markers",
                marker=dict(color="#5a5a5a", size=6),
                hovertemplate="age %{x}<br>%{y:.3f}s<extra></extra>", showlegend=False, 
            )
        )
        y_lo, y_hi = df_stint["lap_time_s"].min(), df_stint["lap_time_s"].max()

    curve = safe_get(metrics, "clean_curves", compound)
    if curve and curve.get("ages") and curve.get("vals"):
        ages = np.array(curve["ages"], dtype=float)
        vals = np.array(curve["vals"], dtype=float)
        offset = 0.0
        if has_raw and len(df_stint):
            nearest_idx = (df_stint["tyre_age"] - ages[0]).abs().idxmin()
            offset = df_stint.loc[nearest_idx, "lap_time_s"] - vals[0]
        y_curve = vals + offset
        fig.add_trace(
            go.Scatter(
                x=ages, y=y_curve, mode="lines", line=dict(color=ACCENT, width=2.5),
                hovertemplate="age %{x}<br>%{y:.3f}s<extra></extra>", showlegend=False, 
            )
        )
        y_lo = y_curve.min() if y_lo is None else min(y_lo, y_curve.min())
        y_hi = y_curve.max() if y_hi is None else max(y_hi, y_curve.max())

        if cliff:
            a, b = cliff
            pad = 0.5 if y_lo is not None else 0
            fig.add_shape(
                type="rect", x0=a, x1=b,
                y0=(y_lo - pad) if y_lo is not None else 0,
                y1=(y_hi + pad) if y_hi is not None else 1,
                fillcolor=ACCENT, opacity=0.12, line_width=0, layer="below",
            )

    fig.update_layout(
        margin=dict(l=44, r=10, t=6, b=34),
        paper_bgcolor="#0b0e11", plot_bgcolor="#0b0e11",
        font=dict(color="#8a8a8a", size=10),
        xaxis=dict(title="TYRE AGE (LAPS)", showgrid=False, zeroline=False, color="#6a6a6a"),
        yaxis=dict(title="LAP TIME (S)", showgrid=True, gridcolor="#16191c", zeroline=False, color="#6a6a6a"),
        showlegend=False,
        height=330,
    )
    return fig

st.markdown(
    """
<style>
#MainMenu, header, footer {visibility:hidden;}
[data-testid="stHeader"], .stDeployButton {display:none !important;}
html, body, .stApp {background-color:#0b0e11 !important; color:#e6e6e6 !important; overflow:hidden !important; height:100vh;}
.block-container {padding:0.5rem 1.1rem 0.3rem 1.1rem !important; max-width:100% !important;}
[data-testid="stVerticalBlock"] {gap:0.3rem !important;}
[data-testid="stHorizontalBlock"] {gap:0.6rem !important; align-items:center;}
* {font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;} 
div[data-baseweb="select"] > div {min-height:28px !important; background-color:#14181c !important; border:1px solid #222 !important; border-radius:4px !important; font-size:12px !important; color:#e6e6e6 !important;}
ul[role="listbox"] {background-color:#14181c !important;}
.wordmark {font-size:13px; font-weight:700; letter-spacing:.12em; color:#e6e6e6;}
.tag {font-size:11px; color:#8a8a8a; text-align:right; letter-spacing:.02em;}
.decision-line {display:flex; align-items:baseline; gap:14px; margin-top:4px;}
.decision-text {font-size:48px; font-weight:600; line-height:1; font-variant-numeric:tabular-nums;}
.risk-chip {font-size:10px; font-weight:700; letter-spacing:.06em; padding:3px 8px; border-radius:4px; text-transform:uppercase;}
.reason-line {font-size:12px; color:#8a8a8a; margin-top:3px;}
.radio-line {font-size:13px; color:#cfcfcf; font-family:"SFMono-Regular",Consolas,Menlo,monospace; margin-top:4px;}
.option-block {border-top:1px solid #222; border-left:2px solid transparent; padding:8px 10px 4px 10px; height:110px; display:flex; flex-direction:column; gap:4px;}
.option-block.recommended {border-left:2px solid #ffb000;}
.option-title {font-size:10px; letter-spacing:.08em; color:#8a8a8a; text-transform:uppercase;}
.option-delta {font-size:22px; font-weight:600; font-variant-numeric:tabular-nums;}
.option-consequence {font-size:11px; color:#9a9a9a; margin-top:auto;}
.key-numbers {display:flex; flex-direction:column; justify-content:space-between; height:330px; padding-left:10px; border-left:1px solid #222;}
.key-label {font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:#7a7a7a;}
.key-value {font-size:26px; font-weight:600; font-variant-numeric:tabular-nums;}
.footer {font-size:10px; color:#6a6a6a; border-top:1px solid #222; padding-top:4px; margin-top:2px;}
</style>
""",
    unsafe_allow_html=True,
)

top = st.columns([1.1, 1, 1, 1, 1.7])
with top[0]:
    st.markdown('<div class="wordmark">CLEANSTINT</div>', unsafe_allow_html=True)

sessions = sorted(laps["session"].dropna().unique().tolist()) if "session" in laps.columns else []
with top[1]:
    session_sel = st.selectbox("session", sessions, label_visibility="collapsed") if sessions else None

df_session = laps[laps["session"] == session_sel] if session_sel is not None else laps.iloc[0:0]

drivers = sorted(df_session["driver"].dropna().unique().tolist()) if "driver" in df_session.columns else []
with top[2]:
    driver_sel = st.selectbox("driver", drivers, label_visibility="collapsed") if drivers else None

df_driver = df_session[df_session["driver"] == driver_sel] if driver_sel is not None else df_session.iloc[0:0]

compounds = sorted(df_driver["compound"].dropna().unique().tolist()) if "compound" in df_driver.columns else []
with top[3]:
    compound_sel = st.selectbox("compound", compounds, label_visibility="collapsed") if compounds else None

with top[4]:
    st.markdown(f'<div class="tag">{session_sel or "—"} · frozen model</div>', unsafe_allow_html=True)

df_stint = df_driver[df_driver["compound"] == compound_sel] if compound_sel is not None else df_driver.iloc[0:0]
df_plot = df_stint[df_stint["is_representative"] == 1] if "is_representative" in df_stint.columns else df_stint

current_age = (
    int(df_stint["tyre_age"].max())
    if not df_stint.empty and "tyre_age" in df_stint.columns and df_stint["tyre_age"].notna().any()
    else 0
)
is_race = session_sel is not None and str(session_sel).strip().upper() in ("R", "RACE") 

no_cliff = cliff_missing(compound_sel)
cliff = None if no_cliff else cliff_for(compound_sel)
wr = wear_rate_for(compound_sel)

if no_cliff:
    decision_key = "RUN TO TARGET"
    decision_text = "NO CLIFF IN SESSION WINDOW — RUN TO TARGET"
    reason = f"cliff not observed this session · wear {fmt(wr, ' s/lap', 3)}"
    radio_text = 'RADIO: "Stay out, manage the pace, run to target."'
else:
    a, b = cliff
    if current_age >= a - 2:
        decision_key, decision_text = "PIT NOW", "PIT NOW"
        radio_text = 'RADIO: "Box, box — pit now."'
    elif a - 5 <= current_age < a - 2:
        decision_key, decision_text = "EXTEND", f"EXTEND — BOX LAP {a}"
        radio_text = f'RADIO: "Stay out, stay out — box lap {a}."'
    else:
        decision_key, decision_text = "RUN TO TARGET", "RUN TO TARGET"
        radio_text = 'RADIO: "Stay out, manage the pace, run to target."'
    reason = f"cliff window {a}–{b} · wear {wr:+.3f} s/lap after {a}" if wr is not None else f"cliff window {a}–{b}"

if is_race:
    reason = "retrospective (race complete) · " + reason
    radio_text = 'RADIO: session complete — review only.'

def option_risk(name):
    if name == "PIT NOW":
        return "LOW"
    if name == "EXTEND":
        if no_cliff:
            return "HIGH"
        return "HIGH" if (cliff[1] - cliff[0]) <= 2 else "MEDIUM"
    if no_cliff:
        return "LOW"
    a = cliff[0]
    if current_age + PROJECTION_LAPS >= a:
        return "HIGH"
    if current_age + PROJECTION_LAPS >= a - 3:
        return "MEDIUM"
    return "LOW"

def decision_risk():
    level = option_risk(decision_key)
    if wr is not None and wr > 0.12 and level != "HIGH":
        level = "MEDIUM" if level == "LOW" else "HIGH"
    return level

st.markdown(
    f"""
<div class="decision-line"><span class="decision-text">{decision_text}</span>{risk_chip_html(decision_risk())}</div>
<div class="reason-line">{reason}</div>
<div class="radio-line">{radio_text}</div>
""",
    unsafe_allow_html=True,
)

box_targets = {"PIT NOW": current_age, "EXTEND": None if no_cliff else cliff[0], "RUN TO TARGET": None}
totals = {k: project_total(compound_sel, current_age, v, PROJECTION_LAPS) for k, v in box_targets.items()}
ref = totals.get(decision_key)
deltas = {
    k: (ref - totals[k] if totals[k] is not None and ref is not None else None)
    for k in totals
}

consequences = {
    "PIT NOW": "Fresh tyre immediately, fixed pit-lane cost, resets the wear clock.",
    "EXTEND": (
        f"Runs {max(cliff[0] - current_age, 0)} more laps before boxing at the model cliff."
        if not no_cliff
        else "No cliff observed — extend point is not data-backed."
    ),
    "RUN TO TARGET": (
        "Holds the tyre through the projection window without pitting."
        if no_cliff or current_age + PROJECTION_LAPS < cliff[0]
        else "Pace risk rises — projection window crosses the cliff."
    ),
}

cols = st.columns(3)
for col, name in zip(cols, ["PIT NOW", "EXTEND", "RUN TO TARGET"]):
    d = deltas.get(name)
    delta_str = "PLAN" if name == decision_key else (f"{d:+.2f}s" if d is not None else "—")
    rec_class = "recommended" if name == decision_key else ""
    with col:
        st.markdown(
            f"""
<div class="option-block {rec_class}">
  <div class="option-title">{name}</div>
  <div class="option-delta">{delta_str}</div>
  {risk_chip_html(option_risk(name))}
  <div class="option-consequence">{consequences[name]}</div>
</div>
""",
            unsafe_allow_html=True,
        )

left, right = st.columns([0.6, 0.4])
with left:
    st.plotly_chart(build_chart(df_plot, compound_sel, cliff), width="stretch", config={"displayModeBar": False})

with right:
    cliff_txt = f"{cliff[0]}–{cliff[1]} laps" if not no_cliff else ("not observed" if compound_sel else "—")
    c_now = curve_value(compound_sel, current_age) if compound_sel else None
    c_then = curve_value(compound_sel, current_age + PROJECTION_LAPS) if compound_sel else None
    pace10 = (c_then - c_now) if (c_now is not None and c_then is not None) else None
    mae = metrics.get("cleanstint_MAE_s_per_lap")
    base_mae = metrics.get("baseline_MAE_s_per_lap")
    pace10_str = f"{pace10:+.2f}s (±{mae:.2f})" if pace10 is not None and mae is not None else fmt(pace10, "s", 2)
    val_str = f"{mae:.3f} vs {base_mae:.3f} s/lap" if mae is not None and base_mae is not None else "—"
    
    net_bias = None
    if not df_stint.empty and {"e_deploy_lap_mj", "e_harvest_lap_mj"} <= set(df_stint.columns):
        _nb = (df_stint["e_deploy_lap_mj"] - df_stint["e_harvest_lap_mj"]).dropna()
        if len(_nb):
            net_bias = float(_nb.mean())
    net_bias_str = fmt(net_bias, " MJ", 2)

    st.markdown(
        f"""
<div class="key-numbers">
  <div><div class="key-label">WEAR RATE (S/LAP)</div><div class="key-value">{fmt(wr, "", 3)}</div></div>
  <div><div class="key-label">CLIFF WINDOW</div><div class="key-value">{cliff_txt}</div></div>
  <div><div class="key-label">PACE IN +10 LAPS</div><div class="key-value">{pace10_str}</div></div>
  <div><div class="key-label">VALIDATION (MAE VS BASELINE)</div><div class="key-value">{val_str}</div></div>
  <div><div class="key-label">NET ENERGY BIAS (DEPLOY−HARVEST)</div><div class="key-value">{net_bias_str}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="footer">Limits: wet sessions, SC/VSC laps, short FP stints not modeled · '
    f'battery = regulation-capped SoC proxy, not raw telemetry · option deltas vs plan, + = faster · {APP_VERSION}</div>',
    unsafe_allow_html=True,
)
