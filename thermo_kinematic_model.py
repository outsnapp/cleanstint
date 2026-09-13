import fastf1
import pandas as pd
import numpy as np
from scipy.stats import theilslopes
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('cache_theta') # Use your existing cache

HAAS_IDS = {'31', '87', 'OCO', 'BEA'}

def team_of(session, drv):
    """2026 driver-info lacks 'Team'; derive from laps, Haas by verified ids."""
    laps = session.laps.pick_driver(drv)
    if 'Team' in laps.columns:
        tt = laps['Team'].dropna()
        if len(tt):
            return str(tt.mode().iloc[0])
    abb = ''
    if 'Driver' in laps.columns:
        aa = laps['Driver'].dropna()
        if len(aa):
            abb = str(aa.mode().iloc[0]).upper()
    if str(drv) in HAAS_IDS or abb in HAAS_IDS:
        return 'Haas F1 Team'
    return 'Field Team'

def calculate_tsi_and_predict(year, event, session_prac='FP2', session_race='R'):
    print(f"Loading {event} {year} Telemetry...")
    prac = fastf1.get_session(year, event, session_prac)
    prac.load(telemetry=True, laps=True, weather=True)
    
    race = fastf1.get_session(year, event, session_race)
    race.load(telemetry=True, laps=True, weather=True)
    
    # 1. CALCULATE LEADING INDICATOR (FP2 Traction Slip Index)
    tsi_data = []
    for drv in prac.drivers:
        drv_laps = prac.laps.pick_driver(drv)
        if drv_laps.empty: continue
        
        # Get car telemetry (Speed, Throttle)
        tel = drv_laps.get_telemetry()
        if tel.empty or 'Throttle' not in tel.columns: continue
        
        # Filter for full throttle zones (traction zones)
        full_throttle = tel[tel['Throttle'] > 95]
        if len(full_throttle) < 50: continue
        
        # Calculate acceleration (m/s^2)
        full_throttle = full_throttle.copy()
        full_throttle['dt'] = full_throttle['Time'].dt.total_seconds().diff()
        full_throttle['dv'] = full_throttle['Speed'].diff() / 3.6 # km/h to m/s
        full_throttle['accel'] = full_throttle['dv'] / full_throttle['dt']
        
        # TSI = Expected Accel (Field Median) - Actual Accel. 
        # Higher TSI = More wheelspin = More heat = More Deg
        median_accel = full_throttle['accel'].median()
        tsi_data.append({'Driver': drv, 'Team': team_of(prac, drv), 'FP2_TSI': -median_accel}) # Negative because lower accel = worse
        
    tsi_df = pd.DataFrame(tsi_data)
    
    # 2. CALCULATE LAGGING INDICATOR (Race Degradation Slope)
    race_slopes = []
    for drv in race.drivers:
        drv_laps = race.laps.pick_driver(drv).pick_quicklaps()
        if len(drv_laps) < 10: continue
        
        # Fuel correct the race laps roughly
        drv_laps['FuelCorr'] = drv_laps['LapTime'].dt.total_seconds() + (drv_laps['LapNumber'] * 0.03)
        
        x = drv_laps['LapNumber'].values
        y = drv_laps['FuelCorr'].values
        slope, _, _, _ = theilslopes(y, x)
        race_slopes.append({'Driver': drv, 'Team': team_of(race, drv), 'Race_Deg_Slope': slope * 1000}) # ms/lap
        
    race_df = pd.DataFrame(race_slopes)
    
    # 3. MERGE AND PREDICT
    merged = pd.merge(tsi_df, race_df, on=['Driver', 'Team'])
    merged = merged.dropna()
    
    print("\n" + "="*60)
    print(f"THERMO-KINEMATIC ANALYSIS: {event} {year}")
    print("="*60)
    
    # Field baseline
    field_tsi = merged[~merged.Team.str.contains('Haas', case=False, na=False)]['FP2_TSI'].median()
    field_deg = merged[~merged.Team.str.contains('Haas', case=False, na=False)]['Race_Deg_Slope'].median()
    
    haas = merged[merged.Team.str.contains('Haas', case=False, na=False)]
    
    for _, row in haas.iterrows():
        tsi_delta = row['FP2_TSI'] - field_tsi
        deg_delta = row['Race_Deg_Slope'] - field_deg
        
        print(f"\nDriver: {row['Driver']} ({row['Team']})")
        print(f"FP2 Traction Slip (vs Field): {tsi_delta:+.3f} m/s^2 (Positive = More Wheelspin/Sliding)")
        print(f"Race Degradation  (vs Field): {deg_delta:+.1f} ms/lap (Positive = Faster Deg)")
        
        if tsi_delta > 0.1 and deg_delta > 10:
            print("🚨 PREDICTION CONFIRMED: High FP2 Slip Energy successfully predicted Race Thermal Degradation!")
        elif tsi_delta > 0.1 and deg_delta <= 10:
            print("⚠️ WARNING: High Slip Energy detected in FP2, but Race Deg didn't materialize (Setup changed or Track Temp dropped).")
        elif deg_delta > 10:
            print("🟠 DEG-ONLY: race degradation worse than field, but FP2 slip proxy silent (leading indicator failed here).")
        else:
            print("✅ NORMAL: slip energy and degradation both match field baseline.")

if __name__ == "__main__":
    for ev in ['Australia', 'Silverstone', 'Monza']:
        try:
            calculate_tsi_and_predict(2026, ev)
        except Exception as e:
            print(f"[skip] {ev}: {type(e).__name__}: {e}")
