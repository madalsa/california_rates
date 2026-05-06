"""Optimize PV kW and battery kWh for NPV per (building, rate, utility).

APPROXIMATE TOU-AGGREGATE MODEL (no hourly dispatch required).

For a high-fidelity check on representative archetypes, the parent repo
modules (`<utility>_battery_lp.py`, `<utility>_post_adoption.py`) can be
run on a machine that has the Baseline_<utility>/ parquets. This module
is for the sensitivity-sweep portion of the paper that needs to cover all
~2,500 medoids x 8+ rates x ~10 size candidates without 21GB of hourly
data.

Model:
  Load (kWh/yr by TOU period) = ResStock annual total kWh x utility TOU
    weights (tou_weights_<u>.csv).
  PV generation (kWh/yr) = pv_kw x PV_YIELD_PER_KW_YR (~1700 kWh/kW/yr in
    CA), split into TOU periods using PV_GEN_TOU_SHARE.
  Self-consumption per period = min(load_period, gen_period).
  Export per period = max(0, gen_period - load_period), valued at the
    utility's annual-average EEC for hourly NBT (config.EEC_ANNUAL_AVG).
  Battery arbitrage = min(batt_kwh x 365 x roundtrip_eff,
                          peak_load_per_yr,
                          offpeak_to_peak_capacity)
    valued at peak - offpeak price spread.
  Net annual bill change = (load - self_cons - batt_shifted) x rates
                            - export_credits - rebates_recurring
                            - existing_fixed - new_DC_charge.

This captures the FIRST-ORDER mechanism: PV serves daytime / offpeak load,
battery shifts excess into peak hours. Underestimates LP-optimal value by
roughly 10-20% (LP captures within-period arbitrage); overestimates if
load shape is unusual (large daytime AC peak in CZ 14/15).

Outputs:
  data/sizing_results_<utility>.parquet: one row per (bldg, rate, scenario_combo)
    with columns: pv_kw, batt_kwh, npv, simple_payback, capex_after_inc, ...
  data/sizing_optimal_<utility>.parquet: best (pv_kw, batt_kwh) per (bldg, rate).
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


# CA residential PV yield. NREL PVWatts central CA tilted south, no shading.
PV_YIELD_PER_KW_YR = 1700.0  # kWh/kW DC/yr

# How PV generation distributes across TOU periods. Approximate: most
# generation lands in offpeak/midpeak (mid-day); little in 4-9pm peak;
# biased toward summer (~60/40 summer/winter).
# Validate against hourly PVWatts on the server when refining.
PV_GEN_TOU_SHARE = {
    "summer_peak":     0.05,
    "summer_midpeak":  0.20,
    "summer_offpeak":  0.35,
    "winter_peak":     0.05,
    "winter_midpeak":  0.15,
    "winter_offpeak":  0.20,
}

# Battery params
BATTERY_ROUNDTRIP_EFF = 0.88
BATTERY_DAILY_CYCLES = 1.0   # one full discharge per day average


def load_tou_weights(utility: str) -> dict[str, float]:
    df = pd.read_csv(config.CR_ROOT / f"tou_weights_{utility}.csv")
    return dict(zip(df["period"], df["weight"]))


def split_annual_kwh_by_tou(annual_kwh: float, weights: dict[str, float]
                            ) -> dict[str, float]:
    """Split annual load by TOU weights. Missing periods get 0."""
    s = sum(weights.values())
    return {k: annual_kwh * (w / s) for k, w in weights.items()}


def split_pv_by_tou(pv_kwh: float, periods: list[str]) -> dict[str, float]:
    """Split annual PV gen across periods present for the utility."""
    shares = {k: PV_GEN_TOU_SHARE.get(k, 0.0) for k in periods}
    s = sum(shares.values())
    if s == 0:
        return {k: 0.0 for k in periods}
    return {k: pv_kwh * (v / s) for k, v in shares.items()}


def get_period_prices(rate_row: pd.Series, periods: list[str]
                      ) -> dict[str, float]:
    out = {}
    for p_ in periods:
        v = rate_row.get(p_)
        if v is None or pd.isna(v):
            continue
        out[p_] = float(v)
    return out


def battery_arbitrage_kwh(batt_kwh: float, load_peak_kwh: float,
                          offpeak_to_peak_kwh: float) -> float:
    """Annual kWh shifted via battery (capacity-bounded)."""
    capacity = batt_kwh * 365 * BATTERY_DAILY_CYCLES * BATTERY_ROUNDTRIP_EFF
    return min(capacity, load_peak_kwh, offpeak_to_peak_kwh)


def annual_bill(
    load_by_period: dict[str, float],
    prices: dict[str, float],
    fixed_monthly: float,
    demand_charge_per_kw_mo: float = 0.0,
    avg_peak_kw: float = 0.0,
) -> float:
    energy = sum(load_by_period.get(p, 0) * prices.get(p, 0) for p in prices)
    fixed = fixed_monthly * 12
    dc = demand_charge_per_kw_mo * avg_peak_kw * 12
    return energy + fixed + dc


def evaluate_size(
    pv_kw: float, batt_kwh: float,
    load_by_period: dict[str, float],
    prices: dict[str, float],
    eec: float,
    fixed_monthly: float,
    demand_charge_per_kw_mo: float,
    avg_peak_kw: float,
) -> tuple[float, float]:
    """Return (annual_bill_change_$, annual_export_credit_$)."""
    pv_kwh = pv_kw * PV_YIELD_PER_KW_YR
    pv_by_period = split_pv_by_tou(pv_kwh, list(prices.keys()))

    self_cons = {pp: min(load_by_period.get(pp, 0), pv_by_period.get(pp, 0))
                 for pp in prices}
    export_kwh = sum(max(0, pv_by_period.get(pp, 0) - load_by_period.get(pp, 0))
                     for pp in prices)
    load_after_pv = {pp: load_by_period.get(pp, 0) - self_cons[pp]
                     for pp in prices}

    # Battery: shift load_after_pv from peak periods to (cheaper) offpeak
    peak_periods = [pp for pp in prices if "peak" in pp and "off" not in pp
                    and "mid" not in pp]
    off_periods = [pp for pp in prices if "offpeak" in pp]
    peak_load_kwh = sum(load_after_pv[pp] for pp in peak_periods)
    off_capacity_kwh = sum(load_after_pv[pp] for pp in off_periods)
    shifted = battery_arbitrage_kwh(batt_kwh, peak_load_kwh, off_capacity_kwh)
    if peak_load_kwh > 0:
        peak_reduction_share = shifted / peak_load_kwh
    else:
        peak_reduction_share = 0
    load_final = dict(load_after_pv)
    for pp in peak_periods:
        load_final[pp] = load_after_pv[pp] * (1 - peak_reduction_share)
    # Add shifted load to off periods (pay offpeak price for the same kWh)
    if off_periods:
        per_off = shifted / len(off_periods)
        for pp in off_periods:
            load_final[pp] = load_after_pv[pp] + per_off

    bill_after = annual_bill(load_final, prices, fixed_monthly,
                             demand_charge_per_kw_mo, avg_peak_kw)
    bill_before = annual_bill(load_by_period, prices, fixed_monthly,
                              demand_charge_per_kw_mo, avg_peak_kw)
    bill_change = bill_after - bill_before
    export_credit = export_kwh * eec
    return bill_change, export_credit


def build_sizing_table(utility: str,
                       buildings: pd.DataFrame,
                       rates: pd.DataFrame,
                       pv_grid: list[float] = None,
                       batt_grid: list[float] = None,
                       ) -> pd.DataFrame:
    pv_grid = pv_grid or config.PV_KW_GRID
    batt_grid = batt_grid or config.BATT_KWH_GRID
    weights = load_tou_weights(utility)
    eec = config.EEC_ANNUAL_AVG[utility]

    rate_rows = rates[rates["rate_type"].isin(("designed_tou", "demand_charge"))]

    bldgs = buildings[buildings["utility"].str.lower() == utility].copy()
    bldgs["annual_kwh"] = bldgs[
        "out.electricity.total.energy_consumption.kwh"].astype(float)

    rows = []
    for _, b in bldgs.iterrows():
        load_by_period = split_annual_kwh_by_tou(b["annual_kwh"], weights)
        avg_peak_kw = float(b.get("summer_peak_kw") or 5.0)
        for _, r in rate_rows.iterrows():
            prices = get_period_prices(r, list(weights.keys()))
            if not prices:
                continue
            fixed_monthly = float(r.get("fixed_monthly_dollars") or 0.0)
            dc = float(r.get("demand_charge_per_kw_mo") or 0.0)
            for pv_kw, batt_kwh in product(pv_grid, batt_grid):
                if pv_kw == 0 and batt_kwh == 0:
                    continue
                bill_chg, export_credit = evaluate_size(
                    pv_kw, batt_kwh, load_by_period, prices, eec,
                    fixed_monthly, dc, avg_peak_kw)
                annual_savings = -bill_chg + export_credit  # bill_chg<0 = save
                # Capex with stacked rebates
                capex = p.CapexBreakdown(pv_kw=pv_kw, battery_kwh=batt_kwh)
                ctx = p.IncentiveContext()
                net_capex, _ = p.apply_capex_stack(capex, ctx)
                cashflows = p.annual_cashflow_series(
                    annual_savings,
                    midlife_replacement_year=config.INVERTER_REPLACEMENT_YEAR,
                    midlife_replacement_cost=(
                        config.INVERTER_REPLACEMENT_COST if pv_kw > 0 else 0))
                npv = p.npv(cashflows, capex=net_capex)
                payback = p.simple_payback(net_capex,
                                           max(annual_savings, 0))
                rows.append({
                    "utility": utility,
                    "bldg_id": b.get("bldg_id"),
                    "cec_cz":  b.get("cec_cz"),
                    "rate_id": r["scenario_id"],
                    "pv_kw":   pv_kw,
                    "batt_kwh": batt_kwh,
                    "annual_savings": annual_savings,
                    "export_credit": export_credit,
                    "net_capex": net_capex,
                    "npv": npv,
                    "simple_payback_yrs": payback,
                    "cluster_weight": b.get("cluster_weight", 1.0),
                })
    return pd.DataFrame(rows)


def find_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """Per (bldg, rate) pick the (pv, batt) that maximizes NPV."""
    idx = df.groupby(["bldg_id", "rate_id"])["npv"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--limit-buildings", type=int, default=0)
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    bldgs = pd.read_parquet(out_dir / "representative_buildings.parquet")

    for u in args.utilities:
        rates = pd.read_csv(out_dir / f"rate_scenarios_extended_{u}.csv")
        u_b = bldgs[bldgs["utility"].str.lower() == u]
        if args.limit_buildings:
            u_b = u_b.sample(min(args.limit_buildings, len(u_b)),
                             random_state=42)
        print(f"{u}: {len(u_b)} buildings ...")
        df = build_sizing_table(u, u_b, rates)
        df.to_parquet(out_dir / f"sizing_results_{u}.parquet", index=False)
        opt = find_optimal(df)
        opt.to_parquet(out_dir / f"sizing_optimal_{u}.parquet", index=False)
        print(f"  wrote {len(df):,} sizing cells, {len(opt):,} optima")


if __name__ == "__main__":
    main()
