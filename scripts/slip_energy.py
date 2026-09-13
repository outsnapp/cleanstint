import fastf1
import pandas as pd
import numpy as np
from scipy.stats import theilslopes
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('cache_theta')
HAAS_IDS = {'31', '87'}

def get_team(drv):
    return 'Haas' if str(drv) in HAAS_IDS else 'Field'

def calc_tsi(session):
    rows = []
    # session.car_data is a dict keyed by driver number string
    for drv, tel in session.car_data.items():
        if tel is None or tel.empty or 'Throttle' not in tel.columns:
            continue
        
        # Filter full throttle traction zones
        ft = tel[tel['Throttle'] > 95].copy()
        if len(ft) < 100: continue
            
        ft['dt'] = ft['Time'].dt.total_seconds().diff()
        ft['dv'] = ft['Speed'].diff() / 3.6 # km/h to m/s
        ft['accel'] = ft['dv'] / ft['dt']
        
        # Low speed traction zones (10-120 km/h) are where wheelspin happens
        low_speed = ft[(ft['Speed'] > 10) & (ft['Speed'] < 120)]
        if len(low_speed) < 50: continue
            
        accel_clean = low_speed['accel'].dropna()
        accel_clean = accel_clean[(accel_clean > -2) & (accel_clean < 15)]
        if len(accel_clean) < 50: continue
            
        rows.append({'Driver': str(drv), 'Team': get_team(drv), 'FP2_Accel': accel_clean.median()})
    return pd.DataFrame(rows)

def calc_race_deg(session):
    rows = []
    for drv in session.drivers:
        laps = session.laps.pick_driver(drv)
        if len(laps) < 10: continue
        
        laps = laps.copy()
        laps['LapNum'] = np.arange(1, len(laps) + 1)
        # Rough fuel correction
        laps['FuelCorr'] = laps['LapTime'].dt.total_seconds() + (laps['LapNum'] * 0.03)
        
        # Filter pits and SC
        mask = (laps['PitInTime'].isna()) & (laps['PitOutTime'].isna()) & (laps['LapTime'].dt.total_seconds() < 120)
        x, y = laps.loc[mask, 'LapNum'].values, laps.loc[mask, 'FuelCorr'].values
        if len(x) < 10: continue
            
        slope, _, _, _ = theilslopes(y, x)
        rows.append({'Driver': str(drv), 'Team': get_team(drv), 'Race_Deg_ms': slope * 1000})
    return pd.DataFrame(rows)

def main():
    for ev in ['Australia', 'Silverstone', 'Monza']:
        print(f"\n=== {ev} 2026 ===")
        try:
            prac = fastf1.get_session(2026, ev, 'FP2')
            prac.load(telemetry=True, laps=True, weather=True)
            tsi_df = calc_tsi(prac)
            
            race = fastf1.get_session(2026, ev, 'R')
            race.load(telemetry=True, laps=True, weather=True)
            deg_df = calc_race_deg(race)
            
            merged = pd.merge(tsi_df, deg_df, on=['Driver', 'Team'])
            if merged.empty:
                print("No overlapping data (maybe no FP2).")
                continue
                
            field_tsi = merged[merged.Team == 'Field']['FP2_Accel'].median()
            field_deg = merged[merged.Team == 'Field']['Race_Deg_ms'].median()
            
            print(f"{'Drv':<4} | {'Team':<6} | {'FP2 Accel':<10} | {'vs Field':<10} | {'Race Deg':<10} | {'vs Field':<10}")
            print("-" * 70)
            for _, r in merged.sort_values('Team').iterrows():
                d_tsi = r['FP2_Accel'] - field_tsi
                d_deg = r['Race_Deg_ms'] - field_deg
                flag = "🚨" if r['Team'] == 'Haas' else "  "
                print(f"{flag}{r['Driver']:<3} | {r['Team']:<6} | {r['FP2_Accel']:>7.3f}    | {d_tsi:>+7.3f}  | {r['Race_Deg_ms']:>7.1f}    | {d_deg:>+7.1f}")
                
            haas = merged[merged.Team == 'Haas']
            if not haas.empty:
                print(f"\nHaas FP2 traction deficit: {-haas['FP2_Accel'].mean() + field_tsi:+.3f} m/s2 (Negative = more wheelspin/slip)")
                print(f"Haas Race deg penalty: {haas['Race_Deg_ms'].mean() - field_deg:+.1f} ms/lap")
        except Exception as e:
            print(f"Skip {ev}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
