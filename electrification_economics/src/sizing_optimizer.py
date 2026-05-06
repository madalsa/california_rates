"""Optimize PV kW and battery kWh for NPV per (building, rate, utility).

Two-level optimization:

  Outer loop: search over (pv_kw, batt_kwh) on PV_KW_GRID x BATT_KWH_GRID
              (config.py). Optionally refine top candidate with a small
              local search.

  Inner loop: for each (pv_kw, batt_kwh) call into the existing
              `<utility>_battery_lp.py` dispatcher to get hourly net load,
              imports, and exports. Compute annual bill under the rate
              tariff using `<utility>_post_adoption.py` logic.

NPV =  - capex_after_incentives
       + sum_{t=1..N} (bill_savings_t + fuel_savings_t) / (1+r)^t
       - O&M and inverter replacement at year 12

Outputs per row:
  building_id, utility, rate_id, pv_kw_opt, batt_kwh_opt, npv_opt,
  payback_years, npv_at_default_size, sensitivity_to_size

The `sensitivity_to_size` column captures NPV loss when sized at +/- 25%
of optimal, which feeds into the paper's "how much does sizing matter?"
question.
"""

# TODO(impl): wire to california_rates battery LP + post-adoption modules,
# run grid search, write parquet of optimal sizes per (building, rate).
