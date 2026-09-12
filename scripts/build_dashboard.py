"""Build pixel-faithful static pit-wall dashboard (no Streamlit)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "app" / "dashboard.html"
PIT_S, HORIZON = 21.0, 10

metrics = json.loads((DATA / "metrics_2026.json").read_text())
FROZEN = {"mae": 0.0482, "rmse": 0.0690, "n": 73, "src": "core/test_2026: frozen 2025 model scored on 2026 races, not refit"}
try: forecast = json.loads((DATA / "forecast_2026.json").read_text())
except Exception: forecast = {}

def curve_for(c):
    it = (metrics.get("clean_curves") or {}).get(c) or {}
    return [float(x) for x in it.get("ages", [])], [float(x) for x in it.get("vals", [])]
def cliff_for(c):
    v = (metrics.get("cliff_windows") or {}).get(c)
    return (int(v[0]), int(v[1])) if isinstance(v, (list, tuple)) and len(v) == 2 else None
def noclf_for(c):
    return c.upper() in {str(x).upper() for x in metrics.get("cliff_not_observed_in_session") or []} or cliff_for(c) is None
def wear_for(c):
    v = (metrics.get("wear_rate_s_per_lap_by_compound") or {}).get(c)
    return float(v) if v is not None else None
def interp(x, xs, ys):
    return float(np.interp(x, xs, ys)) if xs else None

def build_combo(df, comp):
    xs, ys = curve_for(comp); cliff = cliff_for(comp); noclf = noclf_for(comp); wear = wear_for(comp)
    d = df.copy()
    d["tyre_age"] = pd.to_numeric(d["tyre_age"], errors="coerce")
    d["lap_time_s"] = pd.to_numeric(d["lap_time_s"], errors="coerce")
    d = d.dropna(subset=["tyre_age", "lap_time_s"])
    if "is_representative" in d.columns:
        rep = d["is_representative"].astype(str).str.lower().isin(["true", "1", "yes"])
        if rep.any(): d = d[rep]
    if d.empty: return None
    age = int(d["tyre_age"].max())
    off = float(d["lap_time_s"].median() - np.median(ys)) if ys else 0.0
    dots = [[float(a), float(t)] for a, t in zip(d["tyre_age"], d["lap_time_s"])]
    curve = [[a, v + off] for a, v in zip(xs, ys)]
    a, b = cliff if cliff else (None, None)
    if noclf or cliff is None: dec, target = "RUN TO TARGET", None
    elif age >= a - 2: dec, target = "PIT NOW", a
    elif age >= a - 5: dec, target = "EXTEND", a
    else: dec, target = "RUN TO TARGET", a
    width = (b - a + 1) if cliff else 0
    risks = {"PIT NOW": "HIGH" if (cliff and age >= a - 2) else "MEDIUM",
             "EXTEND": "HIGH" if (cliff and width <= 2) else "MEDIUM",
             "RUN TO TARGET": "HIGH" if (cliff and age + HORIZON >= a) else "LOW"}
    def total(box):
        if not xs: return None
        t, ag, boxed = 0.0, age, False
        for _ in range(HORIZON):
            ag += 1
            if (not boxed) and box is not None and ag >= box:
                t += PIT_S; boxed = True; ag = 1
            t += interp(ag, xs, ys) or 0.0
        return t
    tg = {"PIT NOW": age, "EXTEND": a, "RUN TO TARGET": None}
    ref = total(tg[dec])
    deltas = {k: ((total(v) - ref) if (total(v) is not None and ref is not None) else None) for k, v in tg.items()}
    cons = {"PIT NOW": "Lose track position. Undercut unlikely to pay off.",
            "EXTEND": (f"Stay out and box lap {a}. Optimal window before cliff." if cliff else "No observed cliff; target lap remains undefined."),
            "RUN TO TARGET": (f"Risk high degradation after lap {a}. Pace drop expected." if (cliff and age + HORIZON >= a) else "Holds the tyre through the projection window without pitting.")}
    if noclf or cliff is None:
        reason = f"no observed cliff · current tyre age {age} laps"
        radio = 'RADIO: "No cliff observed — run to target."'
    else:
        wtxt = f"wear +{wear:.3f} s/lap after {a}" if wear is not None else "wear rate unavailable"
        reason = f"cliff window {a}–{b} · {wtxt}"
        radio = {"PIT NOW": 'RADIO: "Box this lap — tyre is entering the cliff window."',
                 "EXTEND": f'RADIO: "Stay out, stay out — box lap {a}."',
                 "RUN TO TARGET": 'RADIO: "Stay out — run to target."'}[dec]
    cur, fut = interp(age, xs, ys), interp(age + HORIZON, xs, ys)
    pace = (fut - cur) if (cur is not None and fut is not None) else None
    band = None
    ent = forecast.get(f"{comp}@{age}") or forecast.get(f"{comp}@{min(max(age,5),15)}") if isinstance(forecast, dict) else None
    if isinstance(ent, dict):
        v = ent.get("+10") or ent.get("10")
        if isinstance(v, (list, tuple)) and len(v) == 3: pace, band = float(v[0]), [float(v[1]), float(v[2])]
    if pace is not None and band is None:
        h = max(0.15, 0.25 * abs(pace)); band = [pace - h, pace + h]
    p5 = (interp(age + 5, xs, ys) - interp(age, xs, ys)) if xs else None
    p15 = (interp(age + 15, xs, ys) - interp(age, xs, ys)) if xs else None
    nb = float((d["e_deploy_lap_mj"] - d["e_harvest_lap_mj"]).mean()) if (len(d) and {"e_deploy_lap_mj", "e_harvest_lap_mj"} <= set(d.columns)) else None
    return {"age": age, "decision": dec, "target": target, "risk": risks[dec], "risks": risks,
            "deltas": deltas, "cons": cons, "reason": reason, "radio": radio, "wear": wear,
            "cliff": list(cliff) if cliff else None, "no_cliff": noclf, "pace": pace, "pace5": p5, "pace15": p15, "band": band, "nb": nb,
            "dots": dots, "curve": curve, "mae": metrics.get("cleanstint_MAE_s_per_lap"),
            "base": metrics.get("baseline_MAE_s_per_lap")}

OUTDATA = {}
for p in sorted(DATA.glob("laps_2026_*.csv")):
    df = pd.read_csv(p); drivers = {}
    for drv, g in df.groupby("driver"):
        comps = {}
        for comp, gg in g.groupby("compound"):
            combo = build_combo(gg, str(comp).upper())
            if combo: comps[str(comp).upper()] = combo
        if comps: drivers[str(drv)] = comps
    if drivers: OUTDATA[p.name] = {"label": p.stem.replace("laps_", "").replace("_", " ").upper(), "drivers": drivers}

lim = "Limits: dry conditions only · excludes SC/VSC laps · short practice stints less reliable · battery features are regulation-capped proxies, not raw telemetry."
n = metrics.get("n_anomaly_laps_excluded_SC_VSC")
if n is not None: lim += f" · {n} anomaly laps excluded"

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>CleanStint</title>
<style>
:root{--bg:#0b0e11;--panel:#0d1115;--line:#22272c;--text:#e6e6e6;--white:#f1f1f1;--muted:#858b92;--acc:#e10600;--curve:#ff6a3d;--salmon:#f43f4e;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:Arial,Helvetica,sans-serif;overflow:hidden;}
#shell{height:100vh;display:grid;grid-template-rows:auto auto auto 1fr auto;padding:0 30px;}
#topbar{display:grid;grid-template-columns:250px 1px 1.5fr 1.05fr 1.05fr 1.5fr;gap:26px;align-items:center;height:96px;border-bottom:1px solid var(--line);}
.divider{background:var(--line);width:1px;height:56px;}
.wordmark{font-size:26px;font-weight:800;letter-spacing:.04em;color:var(--white);line-height:1;}
.wordmark span{color:var(--acc);font-style:italic;}
.tagline{font-size:10px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;margin-top:6px;}
.sel label{display:block;font-size:9px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:5px;}
.sel select{width:100%;height:34px;background:transparent;border:1px solid #30363b;border-radius:2px;color:var(--text);font-size:12px;padding:0 8px;outline:none;}
.sel select option{background:var(--panel);}
#brandright{text-align:right;}
.logos{display:flex;justify-content:flex-end;align-items:center;gap:14px;}
.lg-haas{color:var(--acc);text-align:left;line-height:1.1;}
.lg-haas .mg{font-size:11px;font-style:italic;font-weight:700;}
.lg-haas .hf{font-size:13px;font-weight:800;}
.lg-sep{width:1px;height:34px;background:var(--line);}
.lg-mph{text-align:left;line-height:1.1;}
.lg-mph .m1{font-size:15px;font-weight:800;color:#63c1e8;}
.lg-mph .m2{font-size:8px;color:var(--muted);letter-spacing:.08em;}
#meta{margin-top:8px;font-family:"SFMono-Regular",Consolas,monospace;font-style:italic;font-size:11px;color:var(--muted);}
#decision{position:relative;padding:26px 0 22px;border-bottom:1px solid var(--line);text-align:center;}
#dec-label{position:absolute;left:0;top:26px;font-size:10px;letter-spacing:.12em;color:var(--muted);text-transform:uppercase;}
#dec-line{display:flex;justify-content:center;align-items:center;gap:18px;}
#dec-main{font-size:clamp(40px,4.6vw,64px);font-weight:650;letter-spacing:-.035em;color:var(--white);line-height:1;font-variant-numeric:tabular-nums;}
#dec-main .acc{color:var(--salmon);}
.chip{display:inline-block;padding:7px 12px;border-radius:2px;font-size:10px;font-weight:700;letter-spacing:.08em;}
.chip.red{background:#2b0b0d;color:var(--salmon);}
.chip.grey{background:#23272b;color:#aeb3b8;}
#dec-reason{margin-top:12px;font-size:15px;color:#b3b7bb;}
#dec-radio{margin-top:12px;font-family:"SFMono-Regular",Consolas,monospace;font-size:12px;color:var(--text);}
#options{display:grid;grid-template-columns:1fr 1fr 1fr;gap:26px;padding:22px 0;}
.card{position:relative;background:var(--panel);border:1px solid var(--line);padding:20px 22px;height:150px;}
.card.recommended{border:2px solid var(--acc);}
.card .t{font-size:12px;font-weight:700;letter-spacing:.09em;color:var(--text);}
.card .v{margin-top:14px;font-size:34px;font-weight:650;color:var(--white);font-variant-numeric:tabular-nums;}
.card .v small{font-size:12px;color:var(--muted);font-weight:400;margin-left:6px;}
.card .chip{position:absolute;top:18px;right:20px;}
.card .c{position:absolute;left:22px;right:20px;bottom:18px;font-size:12px;color:#a5aaaf;line-height:1.4;}
#split{display:grid;grid-template-columns:1fr 380px;gap:26px;min-height:0;padding-bottom:18px;}
#chartbox{border:1px solid var(--line);background:var(--panel);padding:14px 16px;}
#chart-title{font-size:10px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;}
#chartsvg{width:100%;height:calc(100% - 24px);display:block;}
#stats{border:1px solid var(--line);background:var(--panel);display:grid;grid-template-rows:repeat(6,1fr);}
.stat{padding:9px 20px;display:grid;grid-template-columns:118px 1fr;align-items:start;border-bottom:1px solid var(--line);}
.stat:last-child{border-bottom:0;}
.stat .l{font-size:9px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;padding-top:7px;}
.stat .val{font-size:26px;font-weight:650;color:var(--white);font-variant-numeric:tabular-nums;line-height:1.1;}
.stat .val small{font-size:12px;color:var(--text);font-weight:400;}
.stat .band{font-size:12px;color:var(--muted);font-weight:400;}
.stat .sub{font-size:9px;color:var(--muted);margin-top:6px;}
#footer{display:flex;justify-content:space-between;align-items:center;height:34px;border-top:1px solid var(--line);font-size:10px;color:#666c72;}
</style></head><body>
<div id="shell">
 <div id="topbar">
  <div><div class="wordmark">CLEAN<span>STINT</span></div><div class="tagline">Tyre intelligence. Clearer calls.</div></div>
  <div class="divider"></div>
  <div class="sel"><label>Session file</label><select id="s-sess"></select></div>
  <div class="sel"><label>Driver</label><select id="s-drv"></select></div>
  <div class="sel"><label>Compound</label><select id="s-comp"></select></div>
  <div id="brandright"><div class="logos">
    <div class="lg-haas"><div class="mg">MoneyGram</div><div class="hf">HAAS F1 TEAM</div></div>
    <div class="lg-sep"></div>
    <div class="lg-mph"><div class="m1">Mphasis</div><div class="m2">THE NEXT APPLIED</div></div>
  </div><div id="meta"></div></div>
 </div>
 <div id="decision"><div id="dec-label">Recommendation</div>
  <div id="dec-line"><div id="dec-main"></div><span id="dec-chip" class="chip red"></span></div>
  <div id="dec-reason"></div><div id="dec-radio"></div></div>
 <div id="options"></div>
 <div id="split">
  <div id="chartbox"><div id="chart-title">Lap time vs tyre age</div><svg id="chartsvg"></svg></div>
  <div id="stats"></div>
 </div>
 <div id="footer"><span>__LIMITS__</span><span>CleanStint v1.0</span></div>
</div>
<script>
const DATA=__DATA__;const FROZEN=__FROZEN__;
const $=id=>document.getElementById(id);
const sessKeys=Object.keys(DATA);
function fill(sel,items,keep){sel.innerHTML='';items.forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o);});if(keep&&items.includes(keep))sel.value=keep;}
function cur(){return DATA[$('s-sess').value].drivers[$('s-drv').value][$('s-comp').value];}
function fmtD(d){if(d===null||d===undefined)return '—';return (d>=0?'+':'')+d.toFixed(1)+' s';}
function sgn(v){return (v>=0?'+':'')+v.toFixed(2);}
function render(){
 const S=DATA[$('s-sess').value];$('meta').textContent=S.label+' · frozen model';
 const c=cur();
 $('dec-main').innerHTML = c.decision==='PIT NOW' ? '<span class="acc">PIT NOW</span>'
   : c.decision==='EXTEND' ? '<span class="acc">EXTEND</span> — BOX LAP '+c.target : 'RUN TO TARGET';
 const dc=$('dec-chip');dc.textContent=c.risk+' RISK';dc.className=(c.decision==='RUN TO TARGET')?'chip grey':'chip red';
 $('dec-reason').textContent=c.reason;$('dec-radio').textContent=c.radio;
 $('options').innerHTML=['PIT NOW','EXTEND','RUN TO TARGET'].map(n=>{
   const rec=n===c.decision;
   return '<div class="card'+(rec?' recommended':'')+'"><div class="t">'+n+'</div>'+
     '<div class="v">'+(rec?'+0.0 s':fmtD(c.deltas[n]))+'<small>vs plan</small></div>'+
     '<span class="'+(n==='RUN TO TARGET'?'chip grey':'chip red')+'">'+c.risks[n]+' RISK</span>'+
     '<div class="c">'+c.cons[n]+'</div></div>';}).join('');
 const wear=c.wear===null?'—':c.wear.toFixed(3);
 const cliff=c.cliff?(c.cliff[0]+' – '+c.cliff[1]):'not observed';
 const pace=c.pace===null?'—':sgn(c.pace);
 const band=c.band?' <span class="band">('+sgn(c.band[0])+' to '+sgn(c.band[1])+')</span>':'';
 const mae=c.mae===null?'—':c.mae.toFixed(3), base=c.base===null?'—':c.base.toFixed(3);
 $('stats').innerHTML=
  '<div class="stat"><div class="l">Frozen-core validation</div><div><div class="val">'+FROZEN.mae.toFixed(3)+' <small>s/lap</small></div><div class="sub">wear-slope MAE, '+FROZEN.n+' dry 2026 stints (2025 model, not refit)</div></div></div>'+
  '<div class="stat"><div class="l">Wear rate</div><div><div class="val">'+wear+' <small>s/lap</small></div><div class="sub">Causal wear rate (session)</div></div></div>'+
  '<div class="stat"><div class="l">Cliff window</div><div><div class="val">'+cliff+'</div><div class="sub">Tyre age (laps)</div></div></div>'+
  '<div class="stat"><div class="l">Pace in +5 / +10 / +15</div><div><div class="val">'+sgn(c.pace5)+' / '+pace+' / '+sgn(c.pace15)+' <small>s/lap</small>'+band+'</div><div class="sub">Expected vs. now · band on +10</div></div></div>'+
  '<div class="stat"><div class="l">Net energy bias</div><div><div class="val">'+((c.nb===null||c.nb===undefined)?'—':c.nb.toFixed(2)+' <small>MJ</small>')+'</div><div class="sub">deploy − harvest (2026 proxy)</div></div></div>'+
  '<div class="stat"><div class="l">Shape MAE (Australia)</div><div><div class="val">'+mae+' <small>vs</small> '+base+' <small>s/lap</small></div><div class="sub">causal vs naive within-stint · naive wins on short FP stints — reported honestly</div></div></div>';
 drawChart(c);
}
function drawChart(c){
 const svg=$('chartsvg');const W=svg.clientWidth||900,H=svg.clientHeight||380;
 const m={l:46,r:16,t:40,b:44};const pts=c.dots,cv=c.curve;
 const xs=pts.map(p=>p[0]).concat(cv.map(p=>p[0])),ysv=pts.map(p=>p[1]).concat(cv.map(p=>p[1]));
 if(!xs.length){svg.innerHTML='';return;}
 const x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs)+1,y0=Math.min.apply(null,ysv),y1=Math.max.apply(null,ysv);
 const px=v=>m.l+(W-m.l-m.r)*((v-x0)/((x1-x0)||1));
 const py=v=>m.t+(H-m.t-m.b)*(1-(v-y0)/((y1-y0)||1));
 let g='';const step=Math.max(2,Math.round((x1-x0)/7/2)*2);
 for(let v=Math.ceil(x0/step)*step;v<=x1;v+=step){g+='<line x1="'+px(v)+'" y1="'+m.t+'" x2="'+px(v)+'" y2="'+(H-m.b)+'" stroke="#161b20"/><text x="'+px(v)+'" y="'+(H-m.b+16)+'" fill="#858b92" font-size="10" text-anchor="middle">'+v+'</text>';}
 const ystep=Math.max(0.5,Math.round((y1-y0)/5*2)/2);
 for(let v=Math.ceil(y0/ystep)*ystep;v<=y1+1e-9;v+=ystep){g+='<line x1="'+m.l+'" y1="'+py(v)+'" x2="'+(W-m.r)+'" y2="'+py(v)+'" stroke="#161b20"/><text x="'+(m.l-8)+'" y="'+(py(v)+3)+'" fill="#858b92" font-size="10" text-anchor="end">'+(v%1===0?v:v.toFixed(1))+'</text>';}
 g+='<line x1="'+m.l+'" y1="'+m.t+'" x2="'+m.l+'" y2="'+(H-m.b)+'" stroke="#30353a"/><line x1="'+m.l+'" y1="'+(H-m.b)+'" x2="'+(W-m.r)+'" y2="'+(H-m.b)+'" stroke="#30353a"/>';
 if(c.cliff&&!c.no_cliff){const a=px(c.cliff[0]),b=px(c.cliff[1]);
  g+='<rect x="'+a+'" y="'+m.t+'" width="'+Math.max(0,b-a)+'" height="'+(H-m.t-m.b)+'" fill="rgba(225,6,0,0.12)"/>';
  g+='<line x1="'+a+'" y1="'+m.t+'" x2="'+a+'" y2="'+(H-m.b)+'" stroke="#e10600" stroke-dasharray="4 3"/><line x1="'+b+'" y1="'+m.t+'" x2="'+b+'" y2="'+(H-m.b)+'" stroke="#e10600" stroke-dasharray="4 3"/>';
  g+='<text x="'+((a+b)/2)+'" y="'+(m.t-18)+'" fill="#e10600" font-size="9" text-anchor="middle" style="letter-spacing:1px">CLIFF WINDOW</text>';
  g+='<text x="'+((a+b)/2)+'" y="'+(m.t-7)+'" fill="#e10600" font-size="9" text-anchor="middle">'+c.cliff[0]+' – '+c.cliff[1]+'</text>';}
 pts.forEach(p=>{g+='<circle cx="'+px(p[0])+'" cy="'+py(p[1])+'" r="2.2" fill="#6e747b" opacity="0.85"/>';});
 if(cv.length)g+='<polyline points="'+cv.map(p=>px(p[0]).toFixed(1)+','+py(p[1]).toFixed(1)).join(' ')+'" fill="none" stroke="#ff6a3d" stroke-width="2.5"/>';
 const lx=W-m.r-210,ly=H-m.b-12;
 g+='<circle cx="'+lx+'" cy="'+ly+'" r="2.2" fill="#6e747b"/><text x="'+(lx+8)+'" y="'+(ly+3)+'" fill="#858b92" font-size="10">Raw lap (session)</text>';
 g+='<line x1="'+(lx+118)+'" y1="'+ly+'" x2="'+(lx+138)+'" y2="'+ly+'" stroke="#ff6a3d" stroke-width="2.5"/><text x="'+(lx+144)+'" y="'+(ly+3)+'" fill="#858b92" font-size="10">CleanStint curve</text>';
 svg.innerHTML=g;
}
function onDrv(){const S=DATA[$('s-sess').value];const comps=Object.keys(S.drivers[$('s-drv').value]);fill($('s-comp'),comps,comps.includes('SOFT')?'SOFT':comps[0]);render();}
function onSess(){const S=DATA[$('s-sess').value];const drvs=Object.keys(S.drivers);fill($('s-drv'),drvs);onDrv();}
$('s-sess').addEventListener('change',onSess);
$('s-drv').addEventListener('change',onDrv);
$('s-comp').addEventListener('change',render);
window.addEventListener('resize',render);
fill($('s-sess'),sessKeys,'laps_2026_Australia_FP2.csv');onSess();
</script></body></html>"""

html = TEMPLATE.replace("__DATA__", json.dumps(OUTDATA)).replace("__LIMITS__", lim).replace("__FROZEN__", json.dumps(FROZEN))
OUT.write_text(html)
print("wrote", OUT, "| sessions:", list(OUTDATA.keys()))
