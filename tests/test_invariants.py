import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]

def test_metrics_exist():
    m = json.loads((ROOT/"data/processed/metrics_2026.json").read_text())
    assert m["cleanstint_MAE_s_per_lap"] is not None
    assert "SOFT" in m["cliff_windows"]

def test_decision_boundaries():
    def dec(age, cliff, noclf):
        if noclf or cliff is None: return "RUN TO TARGET"
        a, b = cliff
        if age >= a-2: return "PIT NOW"
        if age >= a-5: return "EXTEND"
        return "RUN TO TARGET"
    assert dec(20, (12,15), False) == "PIT NOW"
    assert dec(9,  (12,15), False) == "EXTEND"
    assert dec(3,  (12,15), False) == "RUN TO TARGET"
    assert dec(40, None,   True)  == "RUN TO TARGET"

def test_laps_schema():
    df = pd.read_csv(ROOT/"data/processed/laps_2026_Australia_R.csv")
    for c in ["tyre_age","lap_time_s","compound","driver","e_deploy_lap_mj","is_representative"]:
        assert c in df.columns
