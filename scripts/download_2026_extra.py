import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, 'core')
from ingest import load_session, session_to_rows, enable_cache
enable_cache()
for ev in ["Austria", "Spain"]:
    try:
        s = load_session(2026, ev, "R")
        rows = session_to_rows(s, 2026, ev, "R")
        f = Path('data') / f"laps_2026_{ev}_R.csv"
        rows.to_csv(f, index=False)
        print(f"[saved] {f} rows={len(rows)}")
    except Exception as e:
        print(f"[skip] {ev}: {type(e).__name__}: {e}")
print("done")
