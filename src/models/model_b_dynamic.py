"""
Model B Dynamic: Compound-aware calibration windows
- HARD/MEDIUM: 8-lap calibration (stable)
- SOFT: 4-lap calibration (fast degradation)
- Recalculate wear every lap after minimum history
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'core'))

from config import TIME_GAIN_S_PER_KG

def _theil_sen(x, y):
    if len(x) < 3:
        return np.nan
    model = TheilSenRegressor()
    model.fit(x.reshape(-1, 1), y)
    return float(model.coef_[0])

def dynamic_wear_slope(laps_df, compound):
    """Calculate wear slope with compound-specific calibration window"""
    # Compound-specific minimum calibration laps
    min_cal_laps = {'SOFT': 4, 'MEDIUM': 8, 'HARD': 8}.get(compound.upper(), 8)
    
    laps_df = laps_df.copy()
    for c in ('lap_time_s', 'fuel_kg_burned', 'tyre_life', 'lap_time_fuel_corr_s'):
        if c in laps_df.columns:
            laps_df[c] = pd.to_numeric(laps_df[c], errors='coerce')
    for c in ('is_pit_lap', 'is_green'):
        if c in laps_df.columns:
            laps_df[c] = laps_df[c].astype(str).str.lower().isin(['true', '1', 'yes'])
    if 'is_pit_lap' in laps_df.columns:
        laps_df = laps_df[~laps_df['is_pit_lap']]
    if 'is_green' in laps_df.columns:
        laps_df = laps_df[laps_df['is_green']]
    if 'lap_time_fuel_corr_s' in laps_df.columns:
        laps_df['lap_time_corr'] = laps_df['lap_time_fuel_corr_s']
    else:
        laps_df['lap_time_corr'] = laps_df['lap_time_s'] + (laps_df['fuel_kg_burned'] * TIME_GAIN_S_PER_KG)
    laps_df = laps_df.dropna(subset=['lap_time_corr', 'tyre_life'])
    med = laps_df['lap_time_corr'].median()
    laps_df = laps_df[laps_df['lap_time_corr'] <= med + 8.0]
    laps_df = laps_df.sort_values('tyre_life')
    
    # Use all available laps after minimum calibration
    if len(laps_df) < min_cal_laps:
        return np.nan
    
    # For Softs: use all laps (they degrade fast, need full picture)
    # For Hard/Medium: use first 8 laps for stability, then update
    if compound.upper() == 'SOFT':
        x = laps_df['tyre_life'].values
        y = laps_df['lap_time_corr'].values
    else:
        # Use up to 12 laps for Hard/Medium (balance stability vs coverage)
        use_laps = laps_df.head(min(12, len(laps_df)))
        x = use_laps['tyre_life'].values
        y = use_laps['lap_time_corr'].values
    
    return _theil_sen(x, y)

def score_monza_2026():
    """Score Monza 2026 with dynamic compound windows"""
    df = pd.read_csv('data/his/processed/laps_2026.csv')
    monza = df[(df['event'] == 'Monza') & (df['session'].astype(str).str.upper() == 'R')].copy()
    
    results = []
    for (driver, compound), group in monza.groupby(['driver', 'compound']):
        # Group by stint
        for stint, stint_laps in group.groupby('stint'):
            if len(stint_laps) < 3:
                continue
            
            slope = dynamic_wear_slope(stint_laps, compound)
            if not np.isnan(slope):
                results.append({
                    'driver': driver,
                    'compound': compound,
                    'stint': stint,
                    'n_laps': len(stint_laps),
                    'wear_ms_per_lap': slope * 1000,  # Convert to ms
                    'team': stint_laps['team'].iloc[0] if 'team' in stint_laps.columns else 'Unknown'
                })
    
    results_df = pd.DataFrame(results)
    print('stints scored by compound:', results_df['compound'].value_counts().to_dict() if len(results_df) else {})
    
    print("\n=== MONZA 2026 DYNAMIC MODEL B RESULTS ===")
    print(results_df.to_string())
    
    # Compare Haas vs Field
    if 'team' in results_df.columns:
        haas = results_df[results_df['team'].str.contains('Haas', case=False, na=False)]
        field = results_df[~results_df['team'].str.contains('Haas', case=False, na=False)]
        
        print("\n=== HAAS vs FIELD COMPARISON ===")
        print(f"Haas stints: {len(haas)}, Mean wear: {haas['wear_ms_per_lap'].mean():.1f} ms/lap")
        print(f"Field stints: {len(field)}, Mean wear: {field['wear_ms_per_lap'].mean():.1f} ms/lap")
        print(f"Haas deficit: {haas['wear_ms_per_lap'].mean() - field['wear_ms_per_lap'].mean():.1f} ms/lap")
        
        # By compound
        for comp in ['HARD', 'MEDIUM', 'SOFT']:
            h = haas[haas['compound'] == comp]['wear_ms_per_lap'].mean()
            f = field[field['compound'] == comp]['wear_ms_per_lap'].mean()
            if not np.isnan(h) and not np.isnan(f):
                print(f"{comp}: Haas {h:.1f} vs Field {f:.1f} ms/lap (delta: {h-f:.1f})")
    
    return results_df

if __name__ == '__main__':
    score_monza_2026()
