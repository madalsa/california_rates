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
