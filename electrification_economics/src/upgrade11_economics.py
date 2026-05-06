"""Heat pump (and full whole-home electrification) economics via ResStock Upgrade 11.

We do NOT re-simulate heat pumps — we use ResStock's Upgrade 11 results
already in the parent repo (`Upgrade11_PGE/`, `Upgrade11_SDGE/`; SCE TBD).

Per building:
  load_delta_kWh(t) = upgrade11_load(t) - baseline_load(t)
  gas_displaced_therms = baseline_gas - upgrade11_gas

Annual cost change under each rate:
  delta_electric_bill = bill(baseline + load_delta) - bill(baseline)
  gas_savings = gas_displaced_therms * NG_THERM_PRICE
  net_annual_opex = delta_electric_bill - gas_savings  (negative = savings)

Capex (whole-home electrification bundle):
  HP space + HP water + induction range + panel upgrade
Less incentives: IRA HEEHRA / 25C, TECH for low-income, utility on-bill.

Payback / NPV vs business-as-usual gas appliances at end-of-life.

Optionally pair with PV+storage by passing the post-upgrade load profile
into sizing_optimizer.py — quantifies whether HP changes optimal PV/battery
size and overall payback.

Output: upgrade11_economics_<utility>.parquet
"""

# TODO(impl): join baseline and upgrade11 parquets on bldg_id, compute
# load + gas deltas, run rate calc per scenario, NPV.
