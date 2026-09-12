"""Track 3: tyre degradation (practice -> race).

2026 PU is a tyre confounder, not the product.
350 kW on the rear axle cooks the rears. We keep e_deploy_lap_mj so
this is not a 2025 tyre model that ignores the power unit.
"""

# --- 2026 car (tyre thermal / slip load) ---
CAR_MASS_KG = 768
MGU_K_KW = 350
MGU_K_TORQUE_NM = 500
ES_SOC_SWING_MJ = 4.0
HARVEST_MJ_DEFAULT = 8.5
HARVEST_MJ_BY_EVENT = {
    "Bahrain": 8.5,
    "Silverstone": 8.5,
    "Imola": 8.5,
    "Monza": 5.0,
}
OVERTAKE_GAP_S = 1.0
OVERTAKE_BONUS_MJ = 0.5
LEAD_TAPER_KMH = 290
CHASE_HOLD_KMH = 337

# Fuel still confounds lap time
FUEL_BURN_KG_PER_LAP = 1.8
TIME_GAIN_S_PER_KG = 0.03

# Track 3 outputs
PREDICT_LAPS_AHEAD = (5, 10, 15)

# Timing data we actually have. Bahrain 2026 (Sakhir) was cancelled;
# the 2026 Bahrain GP is Sepang, 2-4 Oct 2026 — do not request it until it has run.
# Imola is not on the 2026 calendar.
YEAR = 2025
EVENTS = ["Bahrain", "Silverstone", "Imola", "Monza"]
YEAR_TEST = 2026
EVENTS_2026 = ["Silverstone", "Monza"]
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]
# Model A: practice is the fit set. Q and R stay frozen until A is scored.
FIT_SESSIONS = ["FP1", "FP2", "FP3"]
HOLD_SESSIONS = ["Q", "R"]
CHECKPOINT_DRIVER = "VER"
