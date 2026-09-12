# CleanStint Architecture
DAG: FastF1 telemetry -> src/ingest (2026 features + battery proxy) -> src/models (causal curve + cliff) -> core/ (frozen 2025 Model A/B + 2026 transfer exam) -> scripts/merge_experiment (ablation gate) -> scripts/build_dashboard (static HTML/SVG pit wall).
Ownership: core/ = teammate's frozen statistical core (never edited, provenance-tracked); src/ = 2026 physics layer; scripts/ = adapters + validation; app/ = presentation.
Key decisions: (1) battery proxy is a Track-3 confounder/input, never a deployment advisor; (2) merge experiment pre-registered rule fired OUT -> battery removed from wear path, kept as strategy-layer feature; (3) UI is static HTML for pixel fidelity + offline demo reliability.
Known limits: wet/SC excluded; short FP stints -> cliffs "not observed"; clip detector deliberately sensitive (upper bound, not displayed); stage-2 R² negative by construction (residuals = noise + wear) — wear signal validated externally via frozen cross-year transfer (0.048 s/lap, 73 stints).
Reproduce: ./scripts/run_all.sh (prints per-stage runtimes; laptop CPU only, public data, no paid APIs).
