import pandas as pd
df = pd.read_csv("data/laps_2026_Australia_FP2.csv")
d = df[df.is_representative == 1]
print("representative laps:", len(d), "of", len(df))
print("empty clips/lap  : mean %.1f  max %d" % (d.n_empty_battery_clips.mean(), d.n_empty_battery_clips.max()))
print("empty clip s/lap : mean %.1f  max %.1f" % (d.empty_clip_duration_s.mean(), d.empty_clip_duration_s.max()))
print("superclips total :", int(d.n_superclips.sum()))
print("INCONSISTENT laps (clips>0 but soc_min>0.5):", int(((d.n_empty_battery_clips > 0) & (d.soc_min_lap_mj > 0.5)).sum()))
print("traffic flagged  :", int(df.traffic_flag.sum()), "of", len(df))
print("deploy MJ/lap    : mean %.1f  max %.1f" % (d.e_deploy_lap_mj.mean(), d.e_deploy_lap_mj.max()))
print("harvest MJ/lap   : mean %.1f  max %.1f" % (d.e_harvest_lap_mj.mean(), d.e_harvest_lap_mj.max()))
