"""Expand the rate-design space from 8 to ~15-20 scenarios.

Carry forward the 8 scenarios from `california_rates/rate_scenarios_*.csv`
(2 actual tariffs + 6 designed including F0_WF0_ROE1.0) and add:

  Fixed-charge variants (IGFC-style)
    - FIX_10:  $10/mo fixed, volumetric reduced to keep revenue neutral
    - FIX_30:  $30/mo fixed
    - FIX_50:  $50/mo fixed (CA AB 205 ceiling-style)

  Demand-charge variants
    - DC_LOW:  $5/kW-mo on monthly peak, volumetric reduced
    - DC_HIGH: $15/kW-mo

  EV-only TOU
    - EV_TOU:  super off-peak overnight ($0.18) + on-peak peak ($0.55),
               applies to EV submetered load only

  Export-rate variants (applied on top of any import tariff)
    - EXP_NEM2:    full retail credit
    - EXP_NBT:     avoided-cost (NEM 3.0) hourly export rates
    - EXP_FLAT:    flat $0.05/kWh export

  Dynamic / RTP proxy
    - RTP:     hourly LMP + delivery adder; use CAISO 2024 SP15 day-ahead
               as a stand-in until we get utility-specific RTP filings

Each scenario must specify:
  - import price schedule (TOU/flat/dynamic, with weekday/weekend split)
  - fixed monthly charge ($)
  - demand charge ($/kW-mo, on what peak window)
  - export price schedule
  - revenue-neutrality target (calibrate volumetric to match utility revenue)

Output: rate_scenarios_extended_<utility>.csv with the same schema as
`rate_scenarios_<utility>.csv` so downstream stages don't need changes.
"""

# TODO(impl): build extended scenario table per utility, calibrate revenue
# neutrality against existing baseline_bills, write CSV.
