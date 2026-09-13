import sys; sys.path.insert(0, 'core')
import fastf1, ingest
sched = fastf1.get_event_schedule(2025)
for _, ev in sched.iterrows():
    try:
        ingest.build_laps_csv(2025, [ev['EventName']], ["FP1","FP2","FP3","Q","R"])
        print("OK", ev['EventName'], flush=True)
    except Exception as e:
        print("SKIP", ev['EventName'], e, flush=True)
