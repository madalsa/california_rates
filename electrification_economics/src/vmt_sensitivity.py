"""VMT and gasoline-price sensitivity for EV economics.

EV economics are dominated by:
  - annual VMT
  - $/gallon gasoline price (avoided fuel)
  - ICE MPG and EV mi/kWh
  - EV charging profile -> determines the rate-effective $/kWh
  - export-rate environment (when paired with PV+storage)

Sweep:
  VMT in {5_000, 8_000, 12_000, 15_000, 20_000, 25_000}
  gas_price in {3.50, 4.50, 5.50, 6.50}
  charging_profile in {overnight_offpeak, opportunistic, smart_TOU}

For each sweep cell × rate × representative building:
  1. Add EV load profile to building load (charging shaped per profile).
  2. Compute annual bill delta vs no-EV under that rate.
  3. Compute fuel savings = VMT * (gas_price/MPG  -  rate_kWh_per_mile/EV_eff).
  4. Net annual savings; payback against EV premium net of incentives.

Output: ev_sensitivity_<utility>.parquet
"""

# TODO(impl): build EV load profile generator, run sweep, write parquet.
