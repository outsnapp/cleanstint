with open('scripts/build_dashboard.py') as f: t = f.read()
ok = lambda n,c: print(("✅ " if c else "❌ MISS ")+n)

old = "#stats{border:1px solid var(--line);background:var(--panel);display:grid;grid-template-rows:repeat(4,1fr);}"
new = "#stats{border:1px solid var(--line);background:var(--panel);display:grid;grid-template-rows:repeat(6,1fr);}"
ok("stats 6 rows", old in t); t = t.replace(old, new)

old = ".stat{padding:14px 20px;"
new = ".stat{padding:9px 20px;"
ok("stat padding", old in t); t = t.replace(old, new)

old = 'const DATA=__DATA__;'
new = 'const DATA=__DATA__;const FROZEN=__FROZEN__;'
ok("FROZEN js", old in t); t = t.replace(old, new)

old = 'html = TEMPLATE.replace("__DATA__", json.dumps(OUTDATA)).replace("__LIMITS__", lim)'
new = 'html = TEMPLATE.replace("__DATA__", json.dumps(OUTDATA)).replace("__LIMITS__", lim).replace("__FROZEN__", json.dumps(FROZEN))'
ok("FROZEN inject", old in t); t = t.replace(old, new)

old = 'metrics = json.loads((DATA / "metrics_2026.json").read_text())'
new = old + '\nFROZEN = {"mae": 0.0482, "rmse": 0.0690, "n": 73, "src": "core/test_2026: frozen 2025 model scored on 2026 races, not refit"}'
ok("FROZEN const", old in t); t = t.replace(old, new)

old = '    return {"age": age, "decision": dec,'
new = '''    p5 = (interp(age + 5, xs, ys) - interp(age, xs, ys)) if xs else None
    p15 = (interp(age + 15, xs, ys) - interp(age, xs, ys)) if xs else None
    nb = float((d["e_deploy_lap_mj"] - d["e_harvest_lap_mj"]).mean()) if (len(d) and {"e_deploy_lap_mj", "e_harvest_lap_mj"} <= set(d.columns)) else None
    return {"age": age, "decision": dec,'''
ok("pace5/15/nb compute", old in t); t = t.replace(old, new)

old = '"pace": pace, "band": band,'
new = '"pace": pace, "pace5": p5, "pace15": p15, "band": band, "nb": nb,'
ok("combo keys", old in t); t = t.replace(old, new)

old = """  '<div class="stat"><div class="l">Wear rate</div><div><div class="val">'+wear+' <small>s/lap</small></div><div class="sub">Causal wear rate (session)</div></div></div>'+
  '<div class="stat"><div class="l">Cliff window</div><div><div class="val">'+cliff+'</div><div class="sub">Tyre age (laps)</div></div></div>'+
  '<div class="stat"><div class="l">Pace in +10 laps</div><div><div class="val">'+pace+' <small>s/lap</small>'+band+'</div><div class="sub">Expected vs. now</div></div></div>'+
  '<div class="stat"><div class="l">Validation</div><div><div class="val">'+mae+' <small>vs</small> '+base+' <small>s/lap</small></div><div class="sub">CleanStint MAE vs baseline MAE</div></div></div>';"""
new = """  '<div class="stat"><div class="l">Frozen-core validation</div><div><div class="val">'+FROZEN.mae.toFixed(3)+' <small>s/lap</small></div><div class="sub">wear-slope MAE, '+FROZEN.n+' dry 2026 stints (2025 model, not refit)</div></div></div>'+
  '<div class="stat"><div class="l">Wear rate</div><div><div class="val">'+wear+' <small>s/lap</small></div><div class="sub">Causal wear rate (session)</div></div></div>'+
  '<div class="stat"><div class="l">Cliff window</div><div><div class="val">'+cliff+'</div><div class="sub">Tyre age (laps)</div></div></div>'+
  '<div class="stat"><div class="l">Pace in +5 / +10 / +15</div><div><div class="val">'+sgn(c.pace5)+' / '+pace+' / '+sgn(c.pace15)+' <small>s/lap</small>'+band+'</div><div class="sub">Expected vs. now · band on +10</div></div></div>'+
  '<div class="stat"><div class="l">Net energy bias</div><div><div class="val">'+((c.nb===null||c.nb===undefined)?'—':c.nb.toFixed(2)+' <small>MJ</small>')+'</div><div class="sub">deploy − harvest (2026 proxy)</div></div></div>'+
  '<div class="stat"><div class="l">Shape MAE (Australia)</div><div><div class="val">'+mae+' <small>vs</small> '+base+' <small>s/lap</small></div><div class="sub">causal vs naive within-stint · naive wins on short FP stints — reported honestly</div></div></div>';"""
ok("6-row stats", old in t); t = t.replace(old, new)

with open('scripts/build_dashboard.py','w') as f: f.write(t)
print("🎉 patch_stats complete")
