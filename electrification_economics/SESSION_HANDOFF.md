# Session handoff — May 2026

## Where this repo came from

Extracted from `madalsa/california_rates` subfolder `electrification_economics/`
via `git subtree split --prefix=electrification_economics`. Original branch
was `claude/electrification-economics-repo-MYqD0`. Full prior history is
preserved in this repo's `main` branch.

The parent repo (`california_rates`) still has the source data and
pipelines. This repo depends on it being available at a path configurable
via `config.CR_ROOT`.

## Current state (commit by commit at extraction time)

1. Scaffold + paper outline
2. Assumptions grounded in 2026 CA policy (OBBB sunsets, HEAR waitlist,
   CC4A by air district, full HP subsidy stack)
3. `payback_npv.py` — NPV / payback / capex stack with rebate accounting
4. `representative_buildings.py` — 2,541 medoids from ResStock CA, IOU
   owner-occupied non-EBD, weighted to 3.37M households
5. `rate_designer_extended.py` — 47 rate scenarios per utility (40 designed
   + 2 demand-charge + 1 EV-TOU + 4 export overlays)
6. `vmt_sensitivity.py` — EV economics sweep
7. `sizing_optimizer.py` — v1 TOU-aggregate PV+battery sizing (runs
   anywhere)
8. `sizing_optimizer_hourly.py` — v2 hourly LP refinement (requires
   parent's `Baseline_<u>/` parquets, runs on user's machine only)
9. `upgrade11_economics.py` — heat pump / HPWH / induction / panel
   bundle economics
10. `run_economics.py` — orchestrator chains stages 0-4
11. Safety guards: `config.assert_safe_out_dir()` prevents writes
    outside `data/`. Tests verify.

Pipeline ran end-to-end on full 2,541 medoids x 47 rates in ~28 min.
Output: ~144 MB of parquets locally (sizing_results, ev_sensitivity,
upgrade11), tracked summaries:
  - `data/representative_buildings.parquet` (288 KB)
  - `data/rate_scenarios_extended_<u>.csv`
  - `data/ev_sensitivity_summary.csv`
  - `data/population_excluded_summary.csv`

## Headline v1 findings (preliminary)

- Whole-home electrification (HP + HPWH + induction + panel): median
  NPV negative across all 3 IOUs (-$15K to -$18K) under 2026 post-OBBB
  incentive stack. Counterfactual `INCENTIVES_2024` set restores 30% ITC
  + 25C credits — useful for quantifying federal policy reversal.
- Battery storage: not NPV-positive at General Market SGIP tier
  ($200/kWh); flips positive at Equity tier ($850/kWh) - equity story.
- EV economics: payback 1.6-33 yrs across VMT x gas-price grid;
  CC4A (income-eligible scrap-and-replace) flips NPV positive everywhere.

But these are medians. The paper rests on the **distribution** of NPV
across the weighted population, not the median.

## Immediate next steps

### (a) Lens analysis - top priority
Build `analysis/lens.py` that takes any of the parquet outputs and
produces population-weighted P10/P50/P90 + breakeven-share statistics
grouped by any axis (utility, CZ, AMI bin, rate, etc.). Then write small
notebooks for each "tension" identified in the conversation:

  1. Fixed vs volumetric (rate design)
  2. Opex vs capex (CA gas/gasoline tailwind vs install-cost headwind)
  3. Self-cons vs export (NBT vs NEM2 vs counterfactual flat rates)
  4. Income tier vs rebate stack (equity gradient)
  5. Climate zone vs HP suitability (COP x baseline therms)
  6. VMT vs charging cost (when fuel savings beat marginal kWh)
  7. Bundle interactions (bundled NPV vs sum-of-parts)
  8. Time horizon (10/20/30 yr breakevens)

Each → one figure for the paper.

### (b) Hourly LP refinement on user's machine
For sizing surface figure (Fig 3), need hourly LP on ~10-20 archetype
buildings. Run `sizing_optimizer_hourly.py` against parent
`Baseline_<u>/` parquets. Schema matches v1 so figures can toggle v1/v2.

### (c) Path config for standalone repo
`config.CR_ROOT` currently points at `parents[2]` (parent repo path when
EE was a subfolder). After extraction, this needs to be an env var like
`CALIFORNIA_RATES_ROOT` so the EE repo can find parent data files at any
location. Small change, do this before further runs.

## Open questions to confirm with user

- Whitelist `sizing_optimal_*.parquet` (3.2 MB) in git for paper figure
  inputs? Skip the heavy 100 MB `sizing_results_*.parquet`.
- Population scope: keep dropping renters? Currently excluded ~50% of
  households (15,711 owner-occupied of 30,185 IOU rows).
- HOMES eligibility flag: how strictly to model "available but limited"
  status? Currently treated as available; could toggle to off.

## Files to read first for orientation

1. `README.md` - repo overview, read/write contract
2. `paper/outline.md` - paper thesis, 5 RQs, 7 planned figures
3. `paper/assumptions_sources.md` - every numeric assumption with citation
4. `src/config.py` - constants + safety guard
5. `src/payback_npv.py` - financial helpers
6. `src/run_economics.py` - orchestrator entry point
