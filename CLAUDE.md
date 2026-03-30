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
**Last updated:** 2026-03-30

### Pipeline Architecture (modular, split into separate files)

| # | Stage | Status | Files | Notes |
|---|-------|--------|-------|-------|
| 0 | TOU weights calculation | DONE | `calculate_tou_weights_sce.py` → `tou_weights_sce.csv` | 5 periods |
| 1 | Rate scenario design | DONE | `rate_designer_sce.py`, `rate_builder_sce.py` → `rate_scenarios_sce.csv` | 20 scenarios |
| 1b | Building ID list | DONE | `sce_building_ids.txt` | ~5000 building IDs |
| 2 | Baseline bill calculation | **CODE WRITTEN** | `sce_baseline_bills.py` → `baseline_bills_sce.csv` | Native demand, weekday/weekend actual tariff |
| 3 | Tech assignments | **CODE WRITTEN** | `sce_tech_assign.py` → `tech_assignments_sce.csv` | Survey-based or simplified |
| 4 | Solar profiles | **CODE WRITTEN** | `sce_solar.py` | Per CZ (9 zones), pvlib or synthetic |
| 5 | Battery dispatch | **CODE WRITTEN** | `sce_battery_lp.py` | LP only, native demand, no heuristic |
| 6 | Post-adoption bills | **CODE WRITTEN** | `sce_post_adoption.py` → `post_adoption_bills_sce.csv` | 4 scenarios: EV, PV+stor, PV+EV+stor, fully_elec |
| 7 | Summary & analysis | **CODE WRITTEN** | `sce_summary.py` → `pipeline_summary_sce.csv` | 5 customer types, CZ, exports, self-sufficiency |
| cfg | Shared config | **CODE WRITTEN** | `sce_config.py` | Constants, utility data, TOU helpers |
| E2E | Pipeline orchestration | **CODE WRITTEN** | `run_sce_pipeline.py` | `--test`, `--stage N`, `--skip-tech` |

### Key Design Decisions (SCE vs PGE/SDGE)
- **Native demand**: RASS scaling factor stored but NOT applied to load profiles; used only for population extrapolation
- **PV sizing**: 90% offset of native annual demand (not 80%, not RASS-scaled)
- **Battery**: LP only via scipy linprog/HiGHS — no heuristic fallback
- **No Upgrade 11**: "Fully electrified" = buildings with HP already in ResStock baseline + PV + EV + battery
- **Designed scenarios**: blended weekday/weekend rates (constant across week)
- **Actual tariff**: weekday/weekend distinction for TOU-D-4-9 summer peak ($0.627 wd / $0.507 we)
- **5 customer types**: non-adopter, EV only, PV+storage, PV+EV+storage, fully electrified
- **Enhanced metrics**: bill Δ$ and Δ%, by CZ, grid demand changes, exports (EEC), self-sufficiency ratio

### Key SCE-Specific Details
- **Tariff:** TOU-D-4-9 with weekday/weekend split for summer peak
- **Rates:** summer wd peak $0.627, we $0.507, offpeak $0.387; winter peak $0.557, midpeak $0.417, offpeak $0.377
- **Baseline credit:** $0.10/kWh, **CARE discount:** 32.5%
- **TOU weights:** summer_peak 15.69%, summer_offpeak 34.05%, winter_peak 13.59%, winter_midpeak 20.29%, winter_offpeak 16.38%
- **CEC climate zones:** 5, 6, 8, 9, 10, 13, 14, 15, 16
- **Data folders:** `Baseline_SCE/` (parquets, local only)
- **Utility data:** from `utility_data_inputs.tex` — revenue $7.75B, customers 4.59M, rate base $41.43B

### All-Utility Modularization Status

| Utility | Config | Baseline Bills | Tech Assign | Solar | Battery | Post-Adoption | Summary | Orchestrator | Status |
|---------|--------|---------------|-------------|-------|---------|--------------|---------|-------------|--------|
| **SCE** | `sce_config.py` | `sce_baseline_bills.py` | `sce_tech_assign.py` | `sce_solar.py` | `sce_battery_lp.py` | `sce_post_adoption.py` | `sce_summary.py` | `run_sce_pipeline.py` | **DONE** |
| **PGE** | `pge_config.py` | `pge_baseline_bills.py` | `pge_tech_assign.py` | `pge_solar.py` | `pge_battery_lp.py` | `pge_post_adoption.py` | `pge_summary.py` | `run_pge_pipeline.py` | **DONE** |
| **SDGE** | `sdge_config.py` | `sdge_baseline_bills.py` | `sdge_tech_assign.py` | `sdge_solar.py` | `sdge_battery_lp.py` | `sdge_post_adoption.py` | `sdge_summary.py` | `run_sdge_pipeline.py` | **DONE** |

**Unified runner:** `run_all_pipelines.py` — runs PGE/SCE/SDGE with `--utility pge sce sdge`, `--test`, `--stage N`

### Rate Scenarios (all 3 utilities)
8 total: 2 actual tariff + 6 designed (incl. F0_WF0_ROE1.0 = ROE-only reduction)

### Instructions for Next Session
**What was just completed:** All 3 utilities fully modularized. SCE (8 files), PGE (7 files), SDGE (7 files). Unified runner `run_all_pipelines.py` created. NOT yet tested/run.
**Next step:** Test all three with `python run_all_pipelines.py --test`. Need scipy for LP. Debug any import issues.
**User preferences:** No heuristic battery for SCE (LP only). Native demand for SCE. PGE/SDGE keep RASS-scaled demand + both LP/heuristic + Upgrade 11.
**Known issues:** Need Baseline_*/Upgrade11_* parquets (local only). scipy needed for LP. EEC file may lack sce_total column.
**User preferences:** No heuristic battery for SCE (LP only). Native demand for SCE. PGE/SDGE keep RASS-scaled demand + both LP/heuristic + Upgrade 11.
**Known issues:** Need Baseline_*/Upgrade11_* parquets (local only). scipy needed for LP. EEC file may lack sce_total column.
