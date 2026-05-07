"""VMT and gasoline-price sensitivity for EV economics.

For each (representative building x rate x VMT x gas price x charging
profile x EV scenario), compute:
    - effective $/kWh paid for EV charging (TOU-shape weighted)
    - annual EV electric cost  = (VMT / EV_eff) * eff_kwh_price
    - annual ICE fuel cost     = (VMT / MPG) * gas_price
    - annual fuel savings      = ICE - EV
    - net annual cashflow      = fuel savings - any standing-charge delta
                                 from rate switch (zero if same rate)
    - net premium              = EV_premium - rebate (per scenario+district)
    - simple + discounted payback against net premium
    - 20-year NPV

Charging profiles (when each kWh is consumed, used to weight TOU prices):
    overnight_offpeak  - 95% midnight-7am, 5% other (idealized)
    opportunistic       - flat across day (no smart scheduling)
    smart_TOU           - mostly off-peak windows of host tariff

Output: data/ev_sensitivity_<utility>.parquet (one row per cell).
        data/ev_sensitivity_summary.csv (aggregated by district + scenario).
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, payback_npv as p


# Hourly weights per charging profile (sum = 1.0).
CHARGING_PROFILES = {
    "overnight_offpeak": np.zeros(24),
    "opportunistic":     np.full(24, 1 / 24),
    "smart_tou":         np.zeros(24),
}
# overnight: 0..7 split 95% evenly; remaining 17 hours split 5% evenly
CHARGING_PROFILES["overnight_offpeak"][0:7] = 0.95 / 7
CHARGING_PROFILES["overnight_offpeak"][7:24] = 0.05 / 17
# smart_tou: assumes user shifts to lowest-price hours of host tariff;
# proxy: same as overnight for now, refined when rate hourly schedule is
# joined in the bill simulator.
CHARGING_PROFILES["smart_tou"] = CHARGING_PROFILES["overnight_offpeak"].copy()


def hourly_to_tou_weights(hourly_weights: np.ndarray, utility: str
                          ) -> dict[str, float]:
    """Map a 24-hr weight vector into TOU period shares.

    Period definitions per utility (existing pipeline conventions):
      PGE summer:  peak 16-21, offpeak rest
      PGE winter:  peak 16-21, offpeak rest
      SCE summer:  peak 16-21 (wd) - simplified to all-week here, offpeak rest
      SCE winter:  peak 16-21, midpeak 8-16, offpeak rest
      SDGE summer: peak 16-21, midpeak 6-16, offpeak rest
      SDGE winter: peak 16-21, midpeak 6-16, offpeak rest

    For VMT charging cost we use the average over summer + winter (since
    EV charging happens year-round). Returned dict has keys matching the
    rate scenario columns ("summer_peak", ...).
    """
    if utility == "pge":
        s_peak = hourly_weights[16:21].sum()
        s_off = 1 - s_peak
        return {"summer_peak": s_peak / 2, "summer_offpeak": s_off / 2,
                "summer_midpeak": 0,
                "winter_peak": s_peak / 2, "winter_offpeak": s_off / 2,
                "winter_midpeak": 0}
    if utility == "sce":
        peak = hourly_weights[16:21].sum()
        mid = hourly_weights[8:16].sum()
        off = 1 - peak - mid
        return {"summer_peak": peak / 2, "summer_offpeak": (mid + off) / 2,
                "summer_midpeak": 0,
                "winter_peak": peak / 2, "winter_midpeak": mid / 2,
                "winter_offpeak": off / 2}
    if utility == "sdge":
        peak = hourly_weights[16:21].sum()
        mid = hourly_weights[6:16].sum()
        off = 1 - peak - mid
        return {"summer_peak": peak / 2, "summer_midpeak": mid / 2,
                "summer_offpeak": off / 2,
                "winter_peak": peak / 2, "winter_midpeak": mid / 2,
                "winter_offpeak": off / 2}
    raise ValueError(utility)


def effective_kwh_price(rate_row: pd.Series, profile: str, utility: str
                        ) -> float:
    """Charging-profile-weighted average $/kWh for a TOU rate row."""
    h = CHARGING_PROFILES[profile]
    weights = hourly_to_tou_weights(h, utility)
    total = 0.0
    for col, w in weights.items():
        if w == 0:
            continue
        v = rate_row.get(col)
        if v is None or pd.isna(v):
            # Some utilities lack a midpeak; skip
            continue
        total += w * v
    # Normalize in case some periods missing
    used = sum(w for col, w in weights.items() if not pd.isna(rate_row.get(col)))
    return total / used if used > 0 else np.nan


def add_ev_load_to_household(annual_load_kwh: float, vmt: float,
                             ev_eff: float) -> tuple[float, float]:
    """Return (new_annual_kwh, ev_annual_kwh)."""
    ev_kwh = vmt / ev_eff
    return annual_load_kwh + ev_kwh, ev_kwh


def build_sweep(utility: str,
                rates: pd.DataFrame,
                buildings: pd.DataFrame,
                ev_scenarios: tuple[str, ...] = ("new_new",
                                                 "new_ev_dcap",
                                                 "scrap_replace_cc4a"),
                vehicle_class: str = "crossover",
                ) -> pd.DataFrame:
    """Compute per-(building, rate, VMT, gas, profile, ev_scenario) row."""
    rate_rows = rates[rates["rate_type"].isin(
        ("designed_tou", "demand_charge", "ev_submetered_tou"))]

    eff_eff = config.EV_EFFICIENCY[vehicle_class]
    ice_mpg = config.ICE_MPG[vehicle_class]
    profiles = list(CHARGING_PROFILES.keys())
    vmt_grid = config.VMT_GRID
    gas_grid = [3.50, 4.50, 5.50, 6.50]

    air_district_default = {"pge": "BAAQMD", "sce": "SCAQMD",
                            "sdge": "SDAPCD"}[utility]

    bldgs = buildings[buildings["utility"].str.lower() == utility].copy()
    if "annual_kwh" not in bldgs.columns:
        bldgs["annual_kwh"] = bldgs[
            "out.electricity.total.energy_consumption.kwh"].astype(float)

    rows = []
    for _, b in bldgs.iterrows():
        for _, r in rate_rows.iterrows():
            for profile, vmt, gas, scen in product(
                    profiles, vmt_grid, gas_grid, ev_scenarios):
                eff_kwh = effective_kwh_price(r, profile, utility)
                if np.isnan(eff_kwh):
                    continue
                fuel_savings = p.ev_annual_fuel_savings(
                    vmt=vmt, gas_price=gas, ice_mpg=ice_mpg,
                    ev_eff_mi_per_kwh=eff_eff,
                    rate_effective_per_kwh=eff_kwh)
                net_premium = p.ev_net_premium(
                    scen, air_district=air_district_default)
                # 20-yr cashflow with bill escalator for grid electricity
                # (gas component escalator is implicit in fuel savings)
                cashflows = p.annual_cashflow_series(
                    fuel_savings, escalator_real=config.BILL_ESCALATOR_REAL)
                npv = p.npv(cashflows, capex=max(net_premium, 0))
                payback = p.simple_payback(max(net_premium, 0), fuel_savings)
                rows.append({
                    "utility": utility,
                    "bldg_id": b.get("bldg_id"),
                    "cec_cz":  b.get("cec_cz"),
                    "rate_id": r["scenario_id"],
                    "rate_type": r["rate_type"],
                    "profile": profile,
                    "vmt": vmt,
                    "gas_price": gas,
                    "ev_scenario": scen,
                    "eff_charge_per_kwh": eff_kwh,
                    "annual_fuel_savings": fuel_savings,
                    "ev_net_premium": net_premium,
                    "npv": npv,
                    "simple_payback_yrs": payback,
                    "cluster_weight": b.get("cluster_weight", 1.0),
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--vehicle", default="crossover",
                    choices=list(config.EV_EFFICIENCY.keys()))
    ap.add_argument("--limit-buildings", type=int, default=0,
                    help="If >0, sample this many buildings per utility "
                         "(useful for smoke tests).")
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = config.assert_safe_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bldgs = pd.read_parquet(out_dir / "representative_buildings.parquet")
    print(f"Loaded {len(bldgs)} representatives")

    summaries = []
    for u in args.utilities:
        rates = pd.read_csv(out_dir / f"rate_scenarios_extended_{u}.csv")
        u_b = bldgs[bldgs["utility"].str.lower() == u]
        if args.limit_buildings:
            u_b = u_b.sample(min(args.limit_buildings, len(u_b)),
                             random_state=42)
        print(f"\n{u}: {len(u_b)} buildings x {len(rates)} rates ...")
        df = build_sweep(u, rates, u_b, vehicle_class=args.vehicle)
        path = out_dir / f"ev_sensitivity_{u}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {len(df):,} cells -> {path}")

        s = (df.groupby(["ev_scenario", "vmt", "gas_price"])
               .agg(median_payback=("simple_payback_yrs", "median"),
                    median_npv=("npv", "median"),
                    n_cells=("npv", "size")).reset_index())
        s["utility"] = u
        summaries.append(s)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "ev_sensitivity_summary.csv", index=False)
    print(f"\nWrote summary -> {out_dir / 'ev_sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
