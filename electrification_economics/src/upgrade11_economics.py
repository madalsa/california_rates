"""Heat pump (whole-home electrification) economics via ResStock Upgrade 11.

Two implementation paths:

  PATH A (preferred, when available): use parent-pipeline outputs from
    Upgrade11_<utility>/ hourly parquets. Joins on bldg_id and computes
    per-period load/gas deltas. Run on the server when those exist.

  PATH B (this file's default): annual-aggregate approximation using
    ResStock baseline metadata + COP/UEF assumptions. Sufficient for
    sensitivity-paper sweeps; does not capture hour-of-day load shape.

Approximation (PATH B):
  HP space heating delta_kWh = baseline_gas_heating_therms x 29.3 / COP_space
  HPWH delta_kWh             = baseline_gas_hot_water_therms x 29.3 / UEF_HPWH
  Induction range delta_kWh  = baseline_gas_range_therms x 29.3 x 0.85
                               (small; mostly comparable efficiency)
  Total delta_kWh = HP + HPWH + induction
  Gas displaced  = sum of those baseline gas therms

  Annual electric bill change = delta_kWh evaluated at the rate's
    weighted-average $/kWh for HVAC-heavy load (winter-dominated for
    heating, year-round for HPWH).
  Annual gas savings = gas_displaced_therms x NG_THERM_PRICE[utility].

  Capex stack: HP + HPWH + induction + panel via payback_npv.apply_capex_stack.
  NPV / payback computed from those.

Output: data/upgrade11_economics_<utility>.parquet, one row per
        (bldg, rate, scenario_combo).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, payback_npv as p


# COP / UEF assumptions per CEC climate zone. Cold zones (1, 16) get lower
# space heating COP because of lower outdoor temps. Refine vs. ResStock
# Upgrade 11 actuals when available.
COP_SPACE_BY_CZ = {
    1: 2.4, 2: 2.7, 3: 3.0, 4: 3.0, 5: 3.0,
    6: 3.2, 7: 3.2, 8: 3.2, 9: 3.1, 10: 3.0,
    11: 2.9, 12: 2.9, 13: 2.9, 14: 2.7, 15: 2.6, 16: 2.3,
}
UEF_HPWH = 3.0   # typical CA-rated HPWH integrated efficiency
INDUCTION_FRACTION = 0.85  # induction range vs gas: similar primary energy

KWH_PER_THERM = 29.3001


def project_upgrade11_annual(buildings: pd.DataFrame) -> pd.DataFrame:
    """Add columns for projected post-upgrade load delta and gas displaced."""
    df = buildings.copy()
    cz_int = df["cec_cz"].astype(int)
    df["cop_space"] = cz_int.map(COP_SPACE_BY_CZ).fillna(3.0)

    # Gas usage by end-use (kWh in source -> therms)
    gas_heat_kwh = df.get(
        "out.natural_gas.heating.energy_consumption.kwh",
        pd.Series(0, index=df.index)).astype(float)
    gas_hot_water_kwh = df.get(
        "out.natural_gas.hot_water.energy_consumption.kwh",
        pd.Series(0, index=df.index)).astype(float)
    gas_range_kwh = df.get(
        "out.natural_gas.range_oven.energy_consumption.kwh",
        pd.Series(0, index=df.index)).astype(float)

    df["baseline_therms_heat"] = gas_heat_kwh / KWH_PER_THERM
    df["baseline_therms_hpwh"] = gas_hot_water_kwh / KWH_PER_THERM
    df["baseline_therms_range"] = gas_range_kwh / KWH_PER_THERM

    df["delta_kwh_hp_space"] = gas_heat_kwh / df["cop_space"]
    df["delta_kwh_hpwh"] = gas_hot_water_kwh / UEF_HPWH
    df["delta_kwh_induction"] = gas_range_kwh * INDUCTION_FRACTION

    df["total_delta_kwh"] = (df["delta_kwh_hp_space"]
                             + df["delta_kwh_hpwh"]
                             + df["delta_kwh_induction"])
    df["total_therms_displaced"] = (df["baseline_therms_heat"]
                                    + df["baseline_therms_hpwh"]
                                    + df["baseline_therms_range"])
    return df


def _avg_winter_price(rate_row: pd.Series) -> float:
    """Heat is winter-loaded; weight winter periods only."""
    cols = [c for c in ("winter_peak", "winter_midpeak", "winter_offpeak")
            if c in rate_row.index]
    vals = [float(rate_row[c]) for c in cols if not pd.isna(rate_row[c])]
    return np.mean(vals) if vals else np.nan


def _avg_yearround_price(rate_row: pd.Series) -> float:
    """HPWH is roughly year-round; average all TOU periods present."""
    cols = ["summer_peak", "summer_midpeak", "summer_offpeak",
            "winter_peak", "winter_midpeak", "winter_offpeak"]
    vals = [float(rate_row[c]) for c in cols
            if c in rate_row.index and not pd.isna(rate_row[c])]
    return np.mean(vals) if vals else np.nan


def evaluate_per_rate(
    df_proj: pd.DataFrame,
    rates: pd.DataFrame,
    utility: str,
) -> pd.DataFrame:
    """For each (building, rate) compute Upgrade 11 net annual cost."""
    rate_rows = rates[rates["rate_type"] == "designed_tou"]
    gas_price_per_therm = config.gas_price(utility)

    # Capex (gross + after stacked rebates) is constant across buildings of
    # similar income tier. We'll compute per row to allow income variation.
    rows = []
    for _, b in df_proj.iterrows():
        ami_frac = b.get("ami_frac")
        if ami_frac is None or pd.isna(ami_frac):
            ami_frac = 1.0
        capex = p.CapexBreakdown(
            heat_pump_space=True,
            heat_pump_water=True,
            induction_range=True,
            panel_upgrade=True)
        ctx = p.IncentiveContext(income_pct_ami=float(ami_frac))
        net_capex, items = p.apply_capex_stack(capex, ctx)

        for _, r in rate_rows.iterrows():
            heat_price = _avg_winter_price(r)
            yearround_price = _avg_yearround_price(r)
            if np.isnan(heat_price) or np.isnan(yearround_price):
                continue
            elec_cost_increase = (
                b["delta_kwh_hp_space"] * heat_price
                + b["delta_kwh_hpwh"] * yearround_price
                + b["delta_kwh_induction"] * yearround_price)
            gas_savings = b["total_therms_displaced"] * gas_price_per_therm
            net_annual_savings = gas_savings - elec_cost_increase

            cashflows = p.annual_cashflow_series(net_annual_savings)
            npv = p.npv(cashflows, capex=net_capex)
            payback = p.simple_payback(net_capex,
                                       max(net_annual_savings, 0))
            rows.append({
                "utility": utility,
                "bldg_id": b.get("bldg_id"),
                "cec_cz":  b.get("cec_cz"),
                "rate_id": r["scenario_id"],
                "delta_kwh_total": b["total_delta_kwh"],
                "therms_displaced": b["total_therms_displaced"],
                "elec_cost_increase": elec_cost_increase,
                "gas_savings": gas_savings,
                "net_annual_savings": net_annual_savings,
                "net_capex": net_capex,
                "rebates": net_capex - capex.gross_capex(),  # negative
                "npv": npv,
                "simple_payback_yrs": payback,
                "cluster_weight": b.get("cluster_weight", 1.0),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utilities", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--limit-buildings", type=int, default=0)
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    bldgs = pd.read_parquet(out_dir / "representative_buildings.parquet")

    # Annual gas figures live in metadata - re-merge if not present
    needed = "out.natural_gas.heating.energy_consumption.kwh"
    if needed not in bldgs.columns:
        meta = pd.read_parquet(
            config.CR_ROOT / "CA_baseline_tmy_metadata_and_annual_results.parquet")
        gas_cols = [c for c in meta.columns
                    if c.startswith("out.natural_gas.")]
        meta_subset = meta[gas_cols].copy()
        meta_subset = meta_subset.reset_index(drop=False).rename(
            columns={"index": "bldg_id"})
        # representative_buildings.py uses metadata index as bldg_id; align
        meta_subset["bldg_id"] = meta_subset["bldg_id"].astype("int64")
        bldgs = bldgs.reset_index(drop=True).merge(
            meta_subset, on="bldg_id", how="left")

    bldgs = project_upgrade11_annual(bldgs)

    for u in args.utilities:
        rates = pd.read_csv(out_dir / f"rate_scenarios_extended_{u}.csv")
        u_b = bldgs[bldgs["utility"].str.lower() == u]
        if args.limit_buildings:
            u_b = u_b.sample(min(args.limit_buildings, len(u_b)),
                             random_state=42)
        print(f"{u}: {len(u_b)} buildings ...")
        df = evaluate_per_rate(u_b, rates, u)
        path = out_dir / f"upgrade11_economics_{u}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {len(df):,} rows -> {path}")
        print(f"  median NPV: ${df['npv'].median():,.0f}, "
              f"median payback: {df['simple_payback_yrs'].median():.1f} yrs")


if __name__ == "__main__":
    main()
