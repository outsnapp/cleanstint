import pandas as pd
import numpy as np
from scipy.stats import theilslopes
from pathlib import Path
import glob
import re

# 1. Find all 2026 Race files automatically
files = glob.glob('data/**/laps_2026_*_R.csv', recursive=True)
print(f"Found {len(files)} race files: {[Path(f).name for f in files]}")

HAAS_DRIVERS = ['OCO', 'BEA', '31', '87'] # Ocon, Bearman

results = []

for f in files:
    df = pd.read_csv(f)
    
    # Extract Event Name if missing
    if 'event' not in df.columns:
        m = re.search(r'2026_(.+?)_R', Path(f).name)
        df['event'] = m.group(1) if m else 'Unknown'
        
    # Normalise age column (tyre_age vs tyre_life)
    age_col = 'tyre_age' if 'tyre_age' in df.columns else 'tyre_life'
    df = df.rename(columns={age_col: 'tyre_age'})
    # Filter Out-laps & Clean Data
    df = df[df['tyre_age'] >= 2].copy()
    df = df.dropna(subset=['lap_time_s', 'tyre_age'])
    if 'is_green' in df.columns:
        df = df[df['is_green'] != False]
    # CRITICAL FIX: Fuel Correction (keep old behaviour where column exists)
    if 'fuel_in_stint_kg' in df.columns:
        df['fuel_in_stint_kg'] = df['fuel_in_stint_kg'].fillna(0)
        df['lap_time_corr'] = df['lap_time_s'] + (df['fuel_in_stint_kg'] * 0.03)
    elif 'lap_time_fuel_corr_s' in df.columns and df['lap_time_fuel_corr_s'].notna().any():
        df['lap_time_corr'] = df['lap_time_fuel_corr_s']
    else:
        df['lap_time_corr'] = df['lap_time_s']
        
    # Calculate Wear Slopes
    for (event, driver, stint, comp), g in df.groupby(['event', 'driver', 'stint', 'compound']):
        if len(g) < 5: continue 
        
        x = g['tyre_age'].values
        y = g['lap_time_corr'].values
        
        try:
            slope, _, _, _ = theilslopes(y, x)
            is_haas = driver in HAAS_DRIVERS
            
            results.append({
                'event': event,
                'compound': comp,
                'driver': driver,
                'is_haas': is_haas,
                'wear_ms_lap': slope * 1000
            })
        except:
            continue

res_df = pd.DataFrame(results)

# 2. Print Report
print("\n" + "="*70)
print("HAAS DEGRADATION CHECK (Fuel-Corrected)")
print("="*70)

for event in sorted(res_df['event'].unique()):
    print(f"\n--- {event} ---")
    ev_df = res_df[res_df['event'] == event]
    
    for comp in ['SOFT', 'MEDIUM', 'HARD']:
        comp_df = ev_df[ev_df['compound'] == comp]
        if len(comp_df) == 0: continue
        
        haas = comp_df[comp_df['is_haas']]['wear_ms_lap']
        field = comp_df[~comp_df['is_haas']]['wear_ms_lap']
        
        if len(haas) > 0 and len(field) > 0:
            h_med = haas.median()
            f_med = field.median()
            delta = h_med - f_med
            
            if delta > 10: verdict = "🔴 HAAS WORSE (Matches Quotes)"
            elif delta < -10: verdict = "🟢 HAAS BETTER"
            else: verdict = "⚪ NEUTRAL"
            
            print(f"{comp:6} | Haas: {h_med:+6.1f} | Field: {f_med:+6.1f} | Delta: {delta:+6.1f} ms/lap  {verdict}")
