# Project: California Rates

## Overview
Analysis of California electricity rates and billing across utilities.

## Key Code Files
- `corrected_bill_calc.py` — bill calculation logic
- `calculate_TOU_rates.ipynb` — time-of-use rate calculations
- `rate_builder.ipynb` — builds rate scenarios
- `RASS_Analysis.ipynb` — RASS survey analysis
- `OpenEI.ipynb` — OpenEI data processing
- `SDGE.ipynb` — SDG&E specific analysis

## Key Data Files (in git)
- `rate_scenarios.csv`, `rate_scenarios_all_corrected.csv` — rate scenario definitions
- `baseline_bills_*.csv` — computed baseline bills
- `retail_rates_data_*.xlsx` — retail rate input data
- `sdge_rates.csv`, `tou_weights_sdge.csv` — SDG&E rate data
- `puma_utility_data.csv`, `puma-zipcode.csv` — geographic/utility mappings
- `Final19_SW_CleanedSurvey.csv` (62 MB) — cleaned survey data
- `CA_Baseline_metadata_rescaled.parquet` — metadata (whitelisted)
- `CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet` — metadata (whitelisted)
- `CA_baseline_tmy_metadata_and_annual_results.parquet` — metadata (whitelisted)

## Large Data Files (local only, NOT in git)
- All other `*.parquet` files (~21GB total)
- Parquet files in `Baseline_SDGE/` and `Upgrade11_SDGE/`

## Folders
- `Baseline_SDGE/` — SDG&E baseline scenario data
- `Upgrade11_SDGE/` — SDG&E upgrade scenario data

## Server (REAM Lab @ UCSD)
- Servers: shasta-db1.ream.ucsd.edu, shasta-db2.ream.ucsd.edu
- SSH: `ssh YOURNAME@shasta-db1.ream.ucsd.edu`
- Long jobs: use tmux (`tmux new -s [name]`, detach: ctrl+b then d)
- Solvers: CPLEX 20.10, Gurobi (per-user install)
- Python venvs: `python3.12 -m venv ~/venvs/myenv`

---

## SCE Pipeline — Active Work Checkpoint

**Branch:** `claude/sce-pipeline-checkpoints-RCmTl`
**Last updated:** 2026-03-28

### Pipeline Stages (modeled after PGE/SDGE pipelines)

| # | Stage | Status | Files | Notes |
|---|-------|--------|-------|-------|
| 0 | TOU weights calculation | DONE | `calculate_tou_weights_sce.py` → `tou_weights_sce.csv` | 5 periods (summer peak/offpeak, winter peak/midpeak/offpeak) |
| 1 | Rate scenario design | DONE | `rate_designer_sce.py`, `rate_builder_sce.py` → `rate_scenarios_sce.csv` | 20 scenarios, revenue-neutral, TOU-D-4-9 structure |
| 1b | Building ID list | DONE | `sce_building_ids.txt` | ~5000 building IDs from PUMA-based filtering |
| 2 | Baseline bill calculation | **TODO** | Need: hourly bill calc from parquets for all SCE buildings | Reference: `run_pge_pipeline.py` Stage 2 |
| 3 | Tech assignments | **TODO** | Need: `tech_assignments_sce.csv` (PV, battery, EV, heat pump) | Reference: `run_pge_pipeline.py` Stage 3 |
| 4 | Solar profiles | **TODO** | Need: pvlib generation for SCE climate zones | Reference: `run_pge_pipeline.py` Stage 4 |
| 5 | Battery dispatch | **TODO** | Need: LP or heuristic optimization | Reference: `run_pge_pipeline.py` Stage 5 (integrated in 6) |
| 6 | Post-adoption bills | **TODO** | Need: `post_adoption_bills_sce.csv` with net billing | Reference: `run_pge_pipeline.py` Stage 6 |
| 7 | Summary & distributional analysis | **TODO** | Need: payback analysis, equity metrics | Reference: `run_pge_pipeline.py` Stage 7 |
| E2E | Pipeline orchestration | **TODO** | Need: `run_sce_pipeline.py` | Reference: `run_pge_pipeline.py` (orchestrates all stages) |

### Key SCE-Specific Details
- **Tariff:** TOU-D-4-9 with weekday/weekend split for summer peak
- **Summer weekday peak:** $0.49/kWh, **weekend:** $0.38/kWh
- **Baseline credit:** $0.09514/kWh
- **TOU weights:** summer_peak 15.69%, summer_offpeak 34.05%, winter_peak 13.59%, winter_midpeak 20.29%, winter_offpeak 16.38%
- **Data folders:** `Baseline_SCE/` (parquets, local only)
- **`corrected_bill_calc.py`** already supports SCE (has EEC rates, net billing) but is NOT yet integrated into pipeline

### Instructions for Next Session
<!-- UPDATE THIS SECTION each session with what was just completed and what to do next -->
**What was just completed:** Checkpoint system created. Stages 0-1 were done in prior sessions.
**Next step:** Build Stage 2 (baseline bill calculation) — start by reading `run_pge_pipeline.py` Stage 2 and adapting for SCE.
**User preferences:** (add any noted preferences here)
**Known issues:** (add any blockers here)
