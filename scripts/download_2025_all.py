"""Download remaining 2025 events (timing+weather only) -> laps_2025_all.csv.
Never calls build_laps_csv (it overwrites LAPS_CSV). Never touches core/ files.
Resumable: skips events already present in the output checkpoint."""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import ingest

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "data" / "processed" / "laps_2025_all.csv"
HIS  = ROOT / "data" / "his" / "processed" / "laps.csv"
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R"]
NEW_EVENTS = ["Australia", "China", "Japan", "Saudi Arabia", "Miami", "Monaco",
              "Spain", "Canada", "Austria", "Belgium", "Hungary", "Netherlands",
              "Azerbaijan", "Singapore", "USA", "Mexico", "Brazil",
              "Las Vegas", "Qatar", "Abu Dhabi"]

def main():
    ingest.enable_cache()
    frames, done = [], set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        frames.append(prev); done = set(prev.event.unique())
        print(f"resume: {len(done)} events already in {OUT.name}", flush=True)
    his = pd.read_csv(HIS)
    for ev in NEW_EVENTS:
        if ev in done: continue
        rows = []
        for s in SESSIONS:
            try:
                sess = ingest.load_session(2025, ev, s)
                rows.append(ingest.session_to_rows(sess, 2025, ev, s))
                print(f"  ok {ev} {s}: {len(sess.laps)} laps", flush=True)
            except Exception as e:
                print(f"  skip {ev} {s}: {type(e).__name__}", flush=True)
        if rows:
            frames.append(pd.concat(rows, ignore_index=True))
            pd.concat(frames, ignore_index=True).to_csv(OUT, index=False)
            print(f"[saved] {ev} -> total {sum(len(f) for f in frames)} rows", flush=True)
    final = pd.concat(frames, ignore_index=True)
    print("\nNEW file events:", sorted(final.event.unique()))
    print("HIS file events (concat at model time):", sorted(his.event.unique()))

if __name__ == "__main__":
    t0 = time.time(); main(); print(f"done in {(time.time()-t0)/60:.1f} min")
