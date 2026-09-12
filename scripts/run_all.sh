#!/usr/bin/env bash
set -e; cd "$(dirname "$0")/.."
echo "== 1/4 2026 features (battery proxy) =="; time python src/ingest/build_2026.py --year 2026 --gp Australia --sessions FP2 R || true
cp data/laps_2026_Australia_*.csv data/processed/ 2>/dev/null || true
echo "== 2/4 causal curve/cliff/metrics =="; time python src/models/curve_cliff.py --gp Australia || true
cp metrics.json data/processed/metrics_2026.json; cp forecast.json data/processed/forecast_2026.json
echo "== 3/4 frozen-core transfer + battery ablation =="; time PYTHONPATH=core python scripts/merge_experiment.py
echo "== 4/4 static dashboard =="; python scripts/build_dashboard.py
echo "DONE -> open app/dashboard.html"
