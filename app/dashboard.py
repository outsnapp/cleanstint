# CleanStint UI — reference design, verbatim (5 repo adaptations only)
from pathlib import Path
import json, re, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CleanStint", layout="wide", initial_sidebar_state="collapsed")

BASE = Path(__file__).resolve().parents[1]
METRICS_PATH = BASE / "data" / "processed" / "metrics_2026.json"
FORECAST_PATH = BASE / "data" / "processed" / "forecast_2026.json"
DATA_DIR = BASE / "data" / "processed"

ACCENT = "#e10600"; BG = "#0b0e11"; TEXT = "#e6e6e6"; MUTED = "#858b92"
DIM = "#555b61"; LINE = "#22272c"; RAW = "#6e747b"; PANEL = "#0d1115"; WHITE = "#f1f1f1"
RISK_COLORS = {"LOW": "#6f777e", "MEDIUM": "#aeb3b8", "HIGH": "#e10600"}

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists(): return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

@st.cache_data(show_spinner=False)
def load_forecast():
    if not FORECAST_PATH.exists(): return {}
    try:
        with open(FORECAST_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

@st.cache_data(show_spinner=False)
def load_csv(path_str):
    try: return pd.read_csv(path_str)
    except Exception: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def discover_csvs():
    return sorted(DATA_DIR.glob("laps_2026_*.csv"))

metrics = load_metrics(); forecast = load_forecast(); csv_files = discover_csvs()

def safe_float(value, default=None):
    try:
        if value is None or pd.isna(value): return default
        value = float(value); return value if math.isfinite(value) else default
    except Exception: return default

def fmt_num(value, decimals=3, unit=""):
    value = safe_float(value)
    if value is None: return "—"
    return f"{value:.{decimals}f}{unit}"

def clean_name(path): return path.stem.replace("laps_", "")

def infer_session(path):
    name = clean_name(path); parts = re.split(r"[_\-\s]+", name)
    session = "SESSION"
    for candidate in ("race", "qualifying", "quali", "practice", "fp1", "fp2", "fp3", "sprint"):
        if candidate in name.lower():
            session = candidate.upper().replace("QUALIFYING", "QUALI"); break
    if session == "SESSION" and parts:
        session = {"r": "RACE", "q": "QUALI"}.get(parts[-1].lower(), "SESSION")
    return session

def infer_year(path):
    match = re.search(r"(20\d{2})", path.name); return match.group(1) if match else "—"

def infer_gp(path):
    name = clean_name(path)
    match = re.search(r"20\d{2}[_\-\s]+(.+?)(?:[_\-\s]+(?:race|qualifying|quali|practice|fp1|fp2|fp3|sprint)|$)", name, re.IGNORECASE)
    if match: return match.group(1).replace("_", " ").title()
    parts = re.split(r"[_\-\s]+", name)
    if len(parts) >= 2 and re.fullmatch(r"20\d{2}", parts[0]):
        return " ".join(parts[1:-1]).title() if len(parts) > 2 else parts[1].title()
    return "—"

def get_compounds(df):
    if "compound" not in df.columns: return []
    return sorted(df["compound"].dropna().astype(str).str.strip().str.upper().unique().tolist())

def get_drivers(df):
    if "driver" not in df.columns: return []
    return sorted(df["driver"].dropna().astype(str).str.strip().unique().tolist())

def get_clean_curve(metrics, compound):
    curves = metrics.get("clean_curves", {})
    if not isinstance(curves, dict): return None, None
    item = curves.get(compound) or curves.get(compound.upper()) or curves.get(compound.lower())
    if not isinstance(item, dict): return None, None
    ages, vals = item.get("ages"), item.get("vals")
    if not isinstance(ages, list) or not isinstance(vals, list): return None, None
    n = min(len(ages), len(vals))
    ages = [safe_float(x) for x in ages[:n]]; vals = [safe_float(x) for x in vals[:n]]
    pairs = sorted((a, v) for a, v in zip(ages, vals) if a is not None and v is not None)
    if not pairs: return None, None
    return [x[0] for x in pairs], [x[1] for x in pairs]

def get_cliff(metrics, compound):
    windows = metrics.get("cliff_windows", {})
    if not isinstance(windows, dict): return None
    value = windows.get(compound) or windows.get(compound.upper()) or windows.get(compound.lower())
    if not isinstance(value, (list, tuple)) or len(value) < 2: return None
    a, b = safe_float(value[0]), safe_float(value[1])
    if a is None or b is None: return None
    return int(a), int(b)

def cliff_not_observed(metrics, compound):
    values = metrics.get("cliff_not_observed_in_session", [])
    if not isinstance(values, list): return False
    return compound.upper() in {str(x).upper() for x in values}

def get_wear_rate(metrics, compound):
    by = metrics.get("wear_rate_s_per_lap_by_compound", {})
    if isinstance(by, dict):
        v = by.get(compound) or by.get(compound.upper()) or by.get(compound.lower())
        if v is not None: return safe_float(v)
    return safe_float(metrics.get("causal_wear_rate_s_per_lap"))

def calculate_recommendation(age, cliff, no_cliff):
    if no_cliff or cliff is None: return "RUN TO TARGET", None
    start, end = cliff
    if age >= start - 2: return "PIT NOW", start
    if start - 2 > age >= start - 5: return "EXTEND", start
    return "RUN TO TARGET", start

def calculate_risk(compound, cliff, age, no_cliff):
    if no_cliff or cliff is None: return "LOW"
    start, end = cliff; width = max(0, end - start + 1)
    if age >= start: return "HIGH"
    if age >= start - 2: return "HIGH"
    if width <= 2: return "MEDIUM"
    return "LOW"

def expected_pace_change(curve_ages, curve_vals, age, laps=10):
    if not curve_ages or not curve_vals: return None, None, None
    series = pd.Series(curve_vals, index=curve_ages).sort_index()
    current = future = None
    for a in series.index:
        if a <= age: current = series.loc[a]
        if a >= age + laps and future is None: future = series.loc[a]
    if current is None: current = series.iloc[0]
    if future is None: future = series.iloc[-1]
    delta = safe_float(future) - safe_float(current)
    slopes = series.diff().dropna(); slope = safe_float(slopes.median(), 0.0)
    band = abs(slope) * 2.5 * laps
    return delta, max(0, delta - band), delta + band

def raw_laps(df, driver, compound):
    if df.empty: return pd.DataFrame()
    out = df.copy()
    if "driver" in out.columns: out = out[out["driver"].astype(str).str.strip() == str(driver).strip()]
    if "compound" in out.columns: out = out[out["compound"].astype(str).str.strip().str.upper() == compound.upper()]
    if not {"tyre_age", "lap_time_s"}.issubset(out.columns): return pd.DataFrame()
    out["tyre_age"] = pd.to_numeric(out["tyre_age"], errors="coerce")
    out["lap_time_s"] = pd.to_numeric(out["lap_time_s"], errors="coerce")
    out = out.dropna(subset=["tyre_age", "lap_time_s"])
    if "is_representative" in out.columns:
        rep = out["is_representative"].astype(str).str.lower().isin(["true", "1", "yes"])
        if rep.any(): out = out[rep]
    return out

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{ background: {BG} !important; color: {TEXT} !important; overflow: hidden !important; }}
[data-testid="stAppViewContainer"] {{ min-height: 100vh !important; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], #MainMenu, footer, header {{ display: none !important; visibility: hidden !important; }}
.block-container {{ padding: 0 !important; margin: 0 !important; max-width: none !important; }}
section.main > div {{ padding: 0 !important; }}
* {{ box-sizing: border-box; }}
body {{ font-family: Arial, Helvetica, sans-serif !important; }}
.cleanstint-shell {{ height: 100vh; max-height: 100vh; padding: 0 30px; overflow: hidden; background: {BG}; }}
.topbar-rule {{ border-bottom: 1px solid {LINE}; height: 0; margin: 4px 0 0 0; }}
.wordmark {{ font-size: 18px; line-height: 1; font-weight: 800; letter-spacing: .04em; color: {WHITE}; }}
.wordmark span {{ color: {ACCENT}; }}
.top-meta {{ text-align: right; color: {MUTED}; font-size: 10px; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; padding-top: 8px; }}
.selector-label {{ color: {MUTED}; font-size: 9px; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 2px; }}
.decision {{ height: 245px; border-bottom: 1px solid {LINE}; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; }}
.decision-label {{ color: {MUTED}; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; margin-bottom: 7px; }}
.decision-line {{ display: flex; align-items: center; justify-content: center; gap: 16px; flex-wrap: nowrap; }}
.decision-main {{ color: {WHITE}; font-size: clamp(38px, 4vw, 54px); line-height: 1; font-weight: 650; letter-spacing: -.035em; font-variant-numeric: tabular-nums; }}
.decision-main .accent {{ color: {ACCENT}; }}
.risk {{ display: inline-block; padding: 5px 8px; border: 1px solid {DIM}; border-radius: 2px; font-size: 9px; font-weight: 700; letter-spacing: .08em; line-height: 1; }}
.risk-high {{ border-color: {ACCENT}; color: {ACCENT}; }}
.risk-medium {{ border-color: #777d83; color: #d0d3d6; }}
.risk-low {{ border-color: #42484d; color: #9da2a7; }}
.reason {{ color: #b3b7bb; font-size: 12px; margin-top: 9px; font-variant-numeric: tabular-nums; }}
.radio {{ margin-top: 11px; color: {TEXT}; font-family: "SFMono-Regular", Consolas, monospace; font-size: 11px; letter-spacing: .01em; }}
.options {{ height: 184px; padding: 12px 0; border-bottom: 1px solid {LINE}; }}
.option {{ height: 160px; border: 1px solid {LINE}; background: {PANEL}; padding: 17px 20px; position: relative; }}
.option.recommended {{ border-left: 2px solid {ACCENT}; }}
.option-title {{ color: {TEXT}; font-size: 11px; font-weight: 700; letter-spacing: .09em; }}
.option-value {{ color: {WHITE}; font-size: 31px; font-weight: 650; margin-top: 10px; font-variant-numeric: tabular-nums; }}
.option-value span {{ color: {MUTED}; font-size: 11px; font-weight: 400; margin-left: 4px; }}
.option-risk {{ position: absolute; top: 17px; right: 18px; }}
.consequence {{ position: absolute; bottom: 16px; left: 20px; right: 18px; color: #a5aaaf; font-size: 11px; line-height: 1.35; }}
.split {{ height: 340px; display: flex; border-bottom: 1px solid {LINE}; }}
.chart {{ width: 67%; border-right: 1px solid {LINE}; padding: 7px 12px 0 0; }}
.stats {{ width: 33%; display: flex; flex-direction: column; }}
.stat {{ flex: 1; padding: 13px 0 10px 22px; border-bottom: 1px solid {LINE}; }}
.stat:last-child {{ border-bottom: 0; }}
.stat-label {{ color: {MUTED}; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; }}
.stat-number {{ margin-top: 6px; color: {WHITE}; font-size: 25px; line-height: 1; font-weight: 650; font-variant-numeric: tabular-nums; }}
.stat-number small {{ color: {TEXT}; font-size: 12px; font-weight: 400; }}
.stat-sub {{ color: {MUTED}; font-size: 9px; margin-top: 6px; }}
.footer {{ height: 28px; display: flex; align-items: center; justify-content: space-between; color: #666c72; font-size: 9px; letter-spacing: .01em; }}
.stSelectbox {{ margin: 0 !important; }}
div[data-baseweb="select"] > div {{ min-height: 27px !important; height: 27px !important; background: transparent !important; border: 1px solid #30363b !important; border-radius: 2px !important; color: {TEXT} !important; box-shadow: none !important; }}
div[data-baseweb="select"] span {{ font-size: 10px !important; }}
[data-testid="column"] {{ padding-left: 5px !important; padding-right: 5px !important; }}
[data-testid="stVerticalBlock"] {{ gap: 0 !important; }}
div[data-testid="stMarkdownContainer"] p {{ margin: 0 !important; }}
.stats {{ border-left: 1px solid #22272c; }}
[data-testid="stPlotlyChart"] {{ border-right: 1px solid #22272c; padding-right: 12px; }}
</style>
""", unsafe_allow_html=True)

if not csv_files:
    st.error("No laps_2026_*.csv files found in data/processed/."); st.stop()

file_names = [p.name for p in csv_files]

hdr = st.columns([2.3, 1.7, 1.2, 1.2, 2.3])
with hdr[0]:
    st.markdown('<div class="wordmark">CLEAN<span>STINT</span></div>', unsafe_allow_html=True)
with hdr[1]:
    st.markdown('<div class="selector-label">SESSION FILE</div>', unsafe_allow_html=True)
    selected_file_name = st.selectbox("Session file", file_names, label_visibility="collapsed")
selected_path = next(p for p in csv_files if p.name == selected_file_name)
df = load_csv(str(selected_path))
session_name = infer_session(selected_path); year = infer_year(selected_path); gp = infer_gp(selected_path)
drivers = get_drivers(df) or ["—"]; compounds = get_compounds(df) or ["—"]
with hdr[2]:
    st.markdown('<div class="selector-label">DRIVER</div>', unsafe_allow_html=True)
    driver = st.selectbox("Driver", drivers, label_visibility="collapsed")
with hdr[3]:
    st.markdown('<div class="selector-label">COMPOUND</div>', unsafe_allow_html=True)
    compound = st.selectbox("Compound", compounds, index=compounds.index("SOFT") if "SOFT" in compounds else 0, label_visibility="collapsed")
with hdr[4]:
    st.markdown(f'<div class="top-meta">{year} {gp} {session_name} · frozen model</div>', unsafe_allow_html=True)
st.markdown('<div class="topbar-rule"></div>', unsafe_allow_html=True)

filtered = raw_laps(df, driver, compound)
current_age = safe_float(filtered["tyre_age"].max(), 0) if (not filtered.empty and "tyre_age" in filtered.columns) else 0
current_age_int = int(current_age or 0)
cliff = get_cliff(metrics, compound)
no_cliff = cliff_not_observed(metrics, compound)
decision, target_lap = calculate_recommendation(current_age, cliff, no_cliff)
risk = calculate_risk(compound, cliff, current_age, no_cliff)
wear_rate = get_wear_rate(metrics, compound)
curve_ages, curve_vals = get_clean_curve(metrics, compound)
pace_delta, pace_low, pace_high = expected_pace_change(curve_ages, curve_vals, current_age, 10)
_ent = forecast.get(f"{compound}@{current_age_int}") or forecast.get(f"{compound}@{min(max(current_age_int,5),15)}") if isinstance(forecast, dict) else None
if isinstance(_ent, dict):
    _v = _ent.get("+10") or _ent.get("10")
    if isinstance(_v, (list, tuple)) and len(_v) == 3:
        pace_delta, pace_low, pace_high = (safe_float(x) for x in _v)

if no_cliff:
    reason = f"no observed cliff · current tyre age {current_age_int} laps"
    radio = 'RADIO: "No cliff observed — run to target."'
elif cliff:
    start, end = cliff
    wear_text = f"wear +{wear_rate:.3f} s/lap after {start}" if wear_rate is not None else "wear rate unavailable"
    reason = f"cliff window {start}–{end} · {wear_text}"
    if decision == "PIT NOW": radio = f'RADIO: "Box this lap — tyre is entering the cliff window."'
    elif decision == "EXTEND": radio = f'RADIO: "Stay out, stay out — box lap {start}."'
    else: radio = 'RADIO: "Stay out — run to target."'
else:
    reason = f"current tyre age {current_age_int} laps · cliff unavailable"
    radio = 'RADIO: "Run to target — cliff data unavailable."'

PIT_LANE_LOSS_S = 21.0
def _cat(x):
    return float(np.interp(x, curve_ages, curve_vals))
def _total(box_at):
    if curve_ages is None: return None
    t, ag, boxed = 0.0, current_age, False
    for _ in range(10):
        ag += 1
        if (not boxed) and (box_at is not None) and ag >= box_at:
            t += PIT_LANE_LOSS_S; boxed = True; ag = 1
        t += _cat(ag)
    return t
def option_delta(option):
    if cliff is None or curve_ages is None: return None
    targets = {"PIT NOW": current_age, "EXTEND": cliff[0], "RUN TO TARGET": None}
    ref = _total(targets.get(decision))
    val = _total(targets.get(option))
    if ref is None or val is None: return None
    return val - ref

option_data = {
    "PIT NOW": {"delta": option_delta("PIT NOW"), "risk": "HIGH" if cliff and current_age >= cliff[0] - 2 else "MEDIUM",
                "consequence": "Track position cost increases; degradation exposure falls."},
    "EXTEND": {"delta": option_delta("EXTEND"), "risk": "MEDIUM",
               "consequence": (f"Box at lap {cliff[0]} before the steepest degradation." if cliff else "No observed cliff; target lap remains undefined.")},
    "RUN TO TARGET": {"delta": option_delta("RUN TO TARGET"), "risk": "HIGH" if cliff and current_age >= cliff[0] else "LOW",
                      "consequence": (f"Carries tyre beyond the {cliff[0]}–{cliff[1]} cliff window." if cliff else "No observed cliff in the available session window.")},
}

def format_delta(value):
    if value is None: return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f} s"

# shell wrapper removed: Streamlit fragments auto-close divs

decision_html = decision
if decision in ("PIT NOW", "EXTEND"):
    first, *rest = decision.split(" ", 1)
    decision_html = f'<span class="accent">{first}</span>' + ((" " + rest[0]) if rest else "")
if target_lap is not None and decision != "RUN TO TARGET":
    decision_html += f" — BOX LAP {target_lap}"

st.markdown(f"""
<div class="decision">
    <div class="decision-label">RECOMMENDATION</div>
    <div class="decision-line">
        <div class="decision-main">{decision_html}</div>
        <div class="risk risk-{risk.lower()}">{risk}</div>
    </div>
    <div class="reason">{reason}</div>
    <div class="radio">{radio}</div>
</div>
""", unsafe_allow_html=True)

option_cols = st.columns(3)
for col, name in zip(option_cols, ["PIT NOW", "EXTEND", "RUN TO TARGET"]):
    data = option_data[name]; recommended = name == decision
    with col:
        st.markdown(f"""
<div class="option {'recommended' if recommended else ''}">
    <div class="option-title">{name}</div>
    <div class="option-value">{format_delta(data["delta"])}<span>vs plan</span></div>
    <div class="option-risk risk risk-{data["risk"].lower()}">{data["risk"]}</div>
    <div class="consequence">{data["consequence"]}</div>
</div>
""", unsafe_allow_html=True)

chart_df = filtered.copy()
fig = go.Figure()
if not chart_df.empty:
    fig.add_trace(go.Scatter(x=chart_df["tyre_age"], y=chart_df["lap_time_s"], mode="markers",
                marker=dict(size=4, color=RAW, opacity=0.65),
                hovertemplate="tyre age %{x:.0f} laps<br>lap time %{y:.3f} s<extra></extra>", showlegend=False))
if curve_ages and curve_vals:
    off = float(chart_df["lap_time_s"].median()) - float(np.median(curve_vals)) if not chart_df.empty else 0.0
    y_curve = [v + off for v in curve_vals]
    fig.add_trace(go.Scatter(x=curve_ages, y=y_curve, mode="lines", line=dict(color=ACCENT, width=2.5),
                hovertemplate="tyre age %{x:.0f} laps<br>clean curve %{y:.3f} s<extra></extra>", showlegend=False))
    if cliff:
        start, end = cliff
        if any(start <= a <= end for a in curve_ages):
            y0, y1 = min(y_curve), max(y_curve)
            if y0 == y1: y0 -= 1; y1 += 1
            fig.add_shape(type="rect", x0=start, x1=end, y0=y0, y1=y1, fillcolor=ACCENT, opacity=0.10, line=dict(width=0), layer="below")
            fig.add_annotation(x=(start + end) / 2, y=y1, text=f"CLIFF WINDOW<br>{start}–{end}", showarrow=False, yshift=-7, font=dict(size=9, color=ACCENT))
fig.update_layout(height=326, margin=dict(l=45, r=8, t=18, b=30), paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(family="Arial, Helvetica, sans-serif", color=TEXT, size=10),
    xaxis=dict(title=dict(text="TYRE AGE (LAPS)", font=dict(size=9, color=MUTED)), tickfont=dict(size=9, color=MUTED), showgrid=False, zeroline=False, linecolor="#30353a", ticks="outside", tickcolor="#30353a"),
    yaxis=dict(title=dict(text="LAP TIME (S)", font=dict(size=9, color=MUTED)), tickfont=dict(size=9, color=MUTED), showgrid=False, zeroline=False, linecolor="#30353a", ticks="outside", tickcolor="#30353a"),
    hoverlabel=dict(bgcolor="#11161a", bordercolor="#33383d", font=dict(color=TEXT, size=10)))

chart_cols = st.columns([2.0, 1.0])
with chart_cols[0]:
    st.markdown('<div style="color:#858b92;font-size:9px;letter-spacing:.1em;padding:7px 0 0 0;">LAP TIME VS TYRE AGE</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "responsive": True})

validation_clean = safe_float(metrics.get("cleanstint_MAE_s_per_lap"))
validation_base = safe_float(metrics.get("baseline_MAE_s_per_lap"))
cliff_text = f"{cliff[0]}–{cliff[1]} laps" if cliff else "—"
if pace_delta is not None:
    pace_number = f"{pace_delta:+.2f}"; pace_band = f"({pace_low:+.2f} to {pace_high:+.2f}) s/lap"
else:
    pace_number = "—"; pace_band = ""
validation_text = (f'{validation_clean:.3f} <small>vs</small> {validation_base:.3f} <small>s/lap</small>') if (validation_clean is not None and validation_base is not None) else "—"

with chart_cols[1]:
    st.markdown(f"""
<div class="stats">
    <div class="stat"><div class="stat-label">WEAR RATE (S/LAP)</div><div class="stat-number">{fmt_num(wear_rate, 3)} <small>s/lap</small></div><div class="stat-sub">causal wear rate</div></div>
    <div class="stat"><div class="stat-label">CLIFF WINDOW</div><div class="stat-number">{cliff_text}</div><div class="stat-sub">tyre age</div></div>
    <div class="stat"><div class="stat-label">PACE IN +10 LAPS</div><div class="stat-number">{pace_number} <small>s/lap</small></div><div class="stat-sub">{pace_band}</div></div>
    <div class="stat"><div class="stat-label">VALIDATION</div><div class="stat-number">{validation_text}</div><div class="stat-sub">CleanStint MAE vs baseline MAE</div></div>
</div>
""", unsafe_allow_html=True)

n_anomaly = metrics.get("n_anomaly_laps_excluded_SC_VSC")
version = "v0.1.0"
footer_text = ("LIMITS: wet conditions · SC/VSC laps excluded · short FP stints · battery = regulation-capped proxy, not telemetry")
if n_anomaly is not None:
    footer_text += f" · {n_anomaly} anomaly laps excluded"
st.markdown('<div class="topbar-rule"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="footer">
    <span>{footer_text}</span>
    <span>CleanStint {version}</span>
</div>
""", unsafe_allow_html=True)

