import fastf1
import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('cache_theta')
HAAS = ['31', '87']

# ==========================================
# METHOD 1: FP2 Long-Run Z-Score (Prediction)
# ==========================================
def method1_zscore(session, event_name):
    rows = []
    for drv in session.drivers:
        laps = session.laps.pick_driver(drv)
        for st, g in laps.groupby('Stint'):
            if len(g) < 5: continue
            g = g.copy()
            # Fuel correction
            g['fuel_kg'] = (g['LapNumber'] - g['LapNumber'].min()) * 1.8
            g['lap_time_corr'] = g['LapTime'].dt.total_seconds() + (g['fuel_kg'] * 0.03)
            x = g['TyreLife'].values
            y = g['lap_time_corr'].values
            try:
                slope, _, _, _ = theilslopes(y, x)
                team = 'Haas' if str(drv) in HAAS else 'Field'
                rows.append({'driver': drv, 'team': team, 'compound': str(g['Compound'].iloc[0]),
                             'slope_ms': slope * 1000, 'n_laps': len(g)})
            except: continue
    df = pd.DataFrame(rows)
    if df.empty: return
    
    print(f"\n--- METHOD 1: {event_name} FP2 Z-Score (Pre-Race Prediction) ---")
    for comp in ['MEDIUM', 'HARD']:
        c_df = df[df['compound'] == comp]
        if c_df.empty: continue
        field = c_df[c_df['team'] == 'Field']['slope_ms'].values
        if len(field) < 3: continue
        med = np.median(field)
        mad = np.median(np.abs(field - med)) * 1.4826
        if mad == 0: mad = 1e-6
        for _, r in c_df[c_df['team'] == 'Haas'].iterrows():
            z = (r['slope_ms'] - med) / mad
            flag = "🚨 HIGH RISK" if z > 1.5 else "✅ NORMAL"
            print(f"{comp:6} | {r['driver']} | Z: {z:+.2f} | Slope: {r['slope_ms']:+.1f} ms/lap | {flag}")

# ==========================================
# METHOD 2: Aero Deficit Index (Mechanism)
# ==========================================
def method2_aero(session, event_name):
    print(f"\n--- METHOD 2: {event_name} Aero Deficit (High-Speed Lat G) ---")
    field_g, haas_g = [], []
    for drv in session.drivers:
        fl = session.laps.pick_driver(drv).pick_fastest()
        if fl is None or pd.isna(fl['LapTime']): continue
        try:
            # lap.get_car_data() safely returns a DataFrame, avoiding the session dict crash
            car = fl.get_car_data().add_distance()
            pos = fl.get_pos_data()
            if car.empty or pos.empty: continue
            
            m = pd.merge_asof(car.sort_values('Time'), pos.sort_values('Time'), on='Time', direction='nearest')
            m = m.dropna(subset=['Speed', 'X', 'Y'])
            if len(m) < 20: continue
            
            # Savitzky-Golay smoothing for GPS noise
            win = min(21, len(m) - (1 - len(m) % 2))
            if win < 5: continue
            x_s = savgol_filter(m['X'].values, win, 3)
            y_s = savgol_filter(m['Y'].values, win, 3)
            
            dx, dy = np.gradient(x_s), np.gradient(y_s)
            ddx, ddy = np.gradient(dx), np.gradient(dy)
            denom = (dx**2 + dy**2) ** 1.5
            denom[denom < 1e-6] = np.nan
            kappa = (dx * ddy - dy * ddx) / denom
            
            v_ms = m['Speed'].values / 3.6
            m['a_lat_g'] = np.abs(v_ms**2 * kappa) / 9.81
            
            # Isolate high-speed corners (v > 180 km/h, Lat G > 2.0)
            high_speed = m[(m['Speed'] > 180) & (m['a_lat_g'] > 2.0)]
            if high_speed.empty: continue
            
            mean_g = high_speed['a_lat_g'].mean()
            if str(drv) in HAAS: haas_g.append(mean_g)
            else: field_g.append(mean_g)
        except: continue
        
    if haas_g and field_g:
        h_med, f_med = np.median(haas_g), np.median(field_g)
        deficit = h_med - f_med
        print(f"Haas High-Speed Lat G: {h_med:.3f} g | Field: {f_med:.3f} g | Deficit: {deficit:+.3f} g")
        if deficit < -0.1: print("🚨 AERO FLAGGED: Missing downforce confirmed via telemetry")
        else: print("✅ NORMAL")
    else: print("Insufficient high-speed telemetry")

# ==========================================
# MAIN EXECUTION
# ==========================================
for ev in ['Australia', 'Silverstone', 'Monza', 'Austria']:
    try:
        s = fastf1.get_session(2026, ev, 'FP2')
        s.load(telemetry=True, laps=True)
        method1_zscore(s, ev)
        method2_aero(s, ev)
    except Exception as e:
        print(f"[Skip {ev}] {type(e).__name__}: {e}")
