# Paper outline: Personal economics of residential electrification in California

## Working title
Rate design, sizing, and driving: what determines the personal economics of
residential electrification in California?

## Thesis
Personal payback for electrification (EV, PV+storage, heat pump bundles)
hinges on a small number of policy- and household-controllable levers. We
quantify which levers matter most, by how much, and where they bind.

## Research questions
1. Which rate designs make electrification economically attractive, for which
   customers, in which climate zones?
2. What is the optimal PV+storage sizing under each rate, and how steep is
   the NPV penalty for sizing away from optimum?
3. How do annual VMT and gasoline price determine EV adoption economics?
4. Does adding a heat pump (Upgrade 11) change the optimal PV+storage
   solution and the bundle-level payback?
5. Tornado: rank levers (rate, sizing, VMT, gas price, incentives, fuel
   price trajectory, discount rate) by impact on payback.

## Methods (1 paragraph each in paper)
- **Buildings:** ResStock CA stock, stratified by CZ × heating fuel ×
  dwelling type × income tier × vintage; medoid representatives (~500-1000)
  with population weights.
- **Rates:** 8 existing + ~10 new (fixed-charge, demand-charge, EV-only TOU,
  export-rate variants, RTP proxy), all revenue-neutral by utility.
- **Sizing:** outer grid search over (PV kW, battery kWh) × inner LP
  battery dispatch.
- **EV economics:** VMT × gas-price × charging-profile sweep; rate-effective
  $/kWh for charging; avoided fuel.
- **Heat pumps:** ResStock Upgrade 11 load + gas deltas; whole-home capex
  with IRA / TECH / SGIP incentives.
- **NPV / payback:** 20-year, real discount 6%, bill escalator 2%/yr real.

## Key figures (planned)
1. Map of CA: payback for EV-only adoption by CZ under current tariffs.
2. Heatmap: customer type × rate -> payback for PV+storage.
3. Sizing surface: NPV(PV kW, battery kWh) for one archetype; iso-NPV
   contours.
4. Tornado: lever -> $-impact on 20-yr NPV (one per archetype).
5. VMT × gas-price contour: EV breakeven year.
6. Heat-pump bundle payback under each rate, with/without PV+storage.
7. Optimal-rate-per-customer chart: which rate maximizes electrification
   adoption rate at zero NPV?

## Policy framing
- Rate-design choices the CPUC controls vs. household choices (sizing,
  VMT, charging).
- Implication for IGFC / NBT / Net Billing Tariff debate.
- Equity: which customers get left behind by which rate designs?

## Datasets
- ResStock CA baseline + Upgrade 11 (PGE / SCE / SDGE)
- RASS 2019 cleaned survey
- Utility tariffs, EEC export prices, CAISO LMP
- Capex from NREL ATB; incentives from IRA / DOE / CA programs
