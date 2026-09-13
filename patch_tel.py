with open('thermo_kinematic_model.py', 'r') as f:
    t = f.read()
t = t.replace("tel = prac.get_telemetry(drv_laps)", "tel = drv_laps.get_telemetry()")
with open('thermo_kinematic_model.py', 'w') as f:
    f.write(t)
print("✅ Patched telemetry call")
