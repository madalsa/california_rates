# Electrification Economics

Personal economics of residential electrification in California: how rate
design, system sizing, VMT, and incentives change payback and NPV for
PV + storage + EV + heat pump adoption.

Built on top of the `california_rates` rate-design and bill-calculation
pipelines (PGE / SCE / SDGE).

## Research questions

1. Under which rate designs is electrification (any combination of EV,
   PV+storage, heat pump) economically attractive for which customers?
2. What is the *optimal* PV+storage size by customer × rate × climate zone,
   and how sensitive is NPV to deviation from optimum?
3. How do annual VMT and gasoline price change EV + whole-home payback?
4. Which lever — rate design, sizing, incentives, fuel price, or load shape
   — moves payback the most?

## Scope

- **Utilities:** PGE, SCE, SDGE (reuse pipeline outputs).
- **Rate designs:** ~15-20 (existing 8 + fixed-charge variants, demand
  charges, EV-only TOU, NEM 2.0 / NBT / full-retail export, dynamic-pricing
  proxy).
- **Tech bundles:** EV, PV+storage, PV+EV+storage, heat pump (Upgrade 11),
  full electrification.
- **Sensitivities:** PV kW, battery kWh, VMT, gasoline price, gas price
  trajectory, IRA/SGIP/TECH incentive levels, discount rate, financing.
- **Buildings:** representative sample (~500-1000 medoids from stratified
  k-means clustering) with population weights; full-population validation
  on a subset.

## Layout

```
electrification_economics/
  src/
    config.py                  # paths, constants, links to california_rates outputs
    representative_buildings.py# stratified sampling + clustering -> medoids + weights
    rate_designer_extended.py  # expand from 8 to ~15-20 rate designs
    sizing_optimizer.py        # PV+battery NPV optimization (grid / LP)
    vmt_sensitivity.py         # VMT and gasoline-price sweep for EV economics
    upgrade11_economics.py     # heat pump bundle: Upgrade 11 load delta + capex + rebates
    payback_npv.py             # capex, incentives, financing, NPV, payback
    run_economics.py           # orchestrator
  data/                        # local outputs (gitignored when large)
  paper/
    outline.md                 # paper outline + figure list
  notebooks/                   # exploratory analysis
  tests/                       # unit tests for NPV / payback / sizing
```

## Status

Scaffold only. See each `src/*.py` for design notes and TODOs.

## Moving to a standalone repo later

When ready:

```bash
# In this repo:
git subtree split --prefix=electrification_economics -b ee-extract
# Push that branch to a new GitHub repo as `main`.
```
