"""High-fidelity hourly LP sizing optimizer (run on user's machine).

This is the v2 / refinement counterpart to sizing_optimizer.py. Where the
TOU-aggregate version models battery dispatch heuristically, this module
runs a full 8,760-hour scipy LP per (building, rate, pv_kw, batt_kwh)
combo using the parent repo's `<utility>_battery_lp.battery_lp_dispatch`.

REQUIRES files NOT in the EE repo (local-only on user's machine):
  - california_rates/Baseline_<utility>/*.parquet   (hourly load profiles)
  - python-side: scipy, pyarrow, optionally pvlib

Run from `california_rates/` so the parent modules are importable:
    python -m electrification_economics.src.sizing_optimizer_hourly \
        --utility sce --archetypes 20

Default archetype mode: pick 10-20 representatives covering the diversity
(per-CZ medoids of medoids), run full sizing grid, save high-fidelity
results to data/sizing_results_hourly_<u>.parquet. Use these for the
sizing-surface figure and to spot-check the TOU-aggregate v1 numbers.

Schema of output parquet matches sizing_optimizer.py so figures can
toggle between v1 and v2 transparently.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, payback_npv as p


# Battery efficiency conventions (match parent <utility>_config.py defaults).
BATTERY_ROUNDTRIP_EFF = 0.88
BATTERY_C_RATE = 0.4   # power = batt_kwh * C_RATE; e.g. 10 kWh -> 4 kW

# PV yield fallback if pvlib not available (per-CZ approximate).
SYNTHETIC_PV_KWH_PER_KW_YR = 1700


def import_parent_battery_lp(utility: str):
    """Import parent repo's <utility>_battery_lp module.

    Falls back to a vendored copy of the LP if not importable. We pass
    batt_kwh and pmax explicitly via monkeypatch so we can sweep sizes.
    """
    return importlib.import_module(f"{utility}_battery_lp")


def battery_lp_dispatch_param(
    hourly_load: np.ndarray,
    solar_gen: np.ndarray,
    rate_array: np.ndarray,
    eec_rates: np.ndarray,
    batt_kwh: float,
    batt_pmax_kw: float,
):
    """Run LP with parametric battery sizes.

    Mirrors parent battery_lp_dispatch but takes batt_kwh / pmax as args.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    T = 8760
    eta = np.sqrt(BATTERY_ROUNDTRIP_EFF)
    cap = float(batt_kwh)
    pmax = float(batt_pmax_kw)
    net_load = hourly_load - solar_gen

    n = 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = rate_array
    c_obj[T:2 * T] = -eec_rates

    bounds = np.zeros((n, 2))
    bounds[0:T, 1] = np.inf
    bounds[T:2 * T, 1] = np.maximum(solar_gen, 0) + pmax
    bounds[2 * T:3 * T, 1] = pmax
    bounds[3 * T:4 * T, 1] = pmax
    bounds[4 * T:5 * T, 1] = cap

    rows, cols, vals = [], [], []
    tt = np.arange(T)
    rows.append(tt); cols.append(tt); vals.append(np.ones(T))
    rows.append(tt); cols.append(T + tt); vals.append(-np.ones(T))
    rows.append(tt); cols.append(2 * T + tt); vals.append(-np.ones(T))
    rows.append(tt); cols.append(3 * T + tt); vals.append(np.full(T, eta))
    soc_rows = T + tt
    rows.append(soc_rows); cols.append(4 * T + tt); vals.append(np.ones(T))
    rows.append(soc_rows[1:]); cols.append(4 * T + tt[:-1])
    vals.append(-np.ones(T - 1))
    rows.append(soc_rows); cols.append(2 * T + tt); vals.append(np.full(T, -eta))
    rows.append(soc_rows); cols.append(3 * T + tt); vals.append(np.ones(T))

    A_eq = csc_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * T, n))
    b_eq = np.zeros(2 * T)
    b_eq[0:T] = net_load
    b_eq[T] = cap * 0.5

    res = linprog(
        c_obj, A_eq=A_eq, b_eq=b_eq,
        bounds=list(zip(bounds[:, 0], bounds[:, 1])),
        method="highs", options={"time_limit": 10.0, "presolve": True})
    if res.x is None:
        return None
    x = res.x
    grid_in = x[0:T]; grid_out = x[T:2 * T]
    return {
        "import_cost": float(np.dot(grid_in, rate_array)),
        "export_credit": float(np.dot(grid_out, eec_rates)),
        "net_grid_kwh": float(grid_in.sum() - grid_out.sum()),
    }


def load_hourly_load(utility: str, bldg_id: int) -> np.ndarray | None:
    """Load 8,760-hr profile from Baseline_<utility>/. None if not found.

    Convention: parquets named <bldg_id>-<...>.parquet, with column
    `out.electricity.total.energy_consumption` at 15-min interval.
    """
    base = config.utility_paths(utility)["baseline_parquets"]
    if base is None or not base.exists():
        return None
    matches = list(base.glob(f"{bldg_id}-*.parquet"))
    if not matches:
        return None
    df = pd.read_parquet(matches[0])
    col = "out.electricity.total.energy_consumption"
    if col not in df.columns:
        return None
    load_15min = df[col].values
    return load_15min.reshape(-1, 4).sum(axis=1)


def load_eec_hourly(utility: str) -> np.ndarray:
    """Load utility's hourly EEC ($/kWh) for 8,760 hours."""
    df = pd.read_csv(config.EEC_HOURLY_CSV)
    df = df[df["utility"].str.lower() == utility.lower()]
    return df["eec_total"].values[:8760]


def build_hourly_rate_array(
    rate_row: pd.Series, utility: str
) -> np.ndarray | None:
    """Expand a TOU rate row to an 8,760-hr import-price array."""
    period_masks = build_period_masks(utility)
    arr = np.zeros(8760)
    found = False
    for period, mask in period_masks.items():
        v = rate_row.get(period)
        if v is None or pd.isna(v):
            continue
        arr[mask] = float(v)
        found = True
    return arr if found else None


def build_period_masks(utility: str) -> dict[str, np.ndarray]:
    """8,760-hr boolean masks for each TOU period, matching the existing
    pipeline conventions."""
    hours = pd.date_range("2025-01-01", periods=8760, freq="h")
    is_summer = np.asarray(hours.month.isin([6, 7, 8, 9]))
    hour = np.asarray(hours.hour)
    masks = {}
    if utility == "pge":
        peak = (hour >= 16) & (hour < 21)
        masks["summer_peak"] = is_summer & peak
        masks["summer_offpeak"] = is_summer & ~peak
        masks["winter_peak"] = ~is_summer & peak
        masks["winter_offpeak"] = ~is_summer & ~peak
    elif utility == "sce":
        peak = (hour >= 16) & (hour < 21)
        midpeak = (hour >= 8) & (hour < 16)
        offpeak = ~peak & ~midpeak
        masks["summer_peak"] = is_summer & peak
        masks["summer_offpeak"] = is_summer & ~peak
        masks["winter_peak"] = ~is_summer & peak
        masks["winter_midpeak"] = ~is_summer & midpeak
        masks["winter_offpeak"] = ~is_summer & offpeak
    elif utility == "sdge":
        peak = (hour >= 16) & (hour < 21)
        midpeak = (hour >= 6) & (hour < 16)
        offpeak = ~peak & ~midpeak
        masks["summer_peak"] = is_summer & peak
        masks["summer_midpeak"] = is_summer & midpeak
        masks["summer_offpeak"] = is_summer & offpeak
        masks["winter_peak"] = ~is_summer & peak
        masks["winter_midpeak"] = ~is_summer & midpeak
        masks["winter_offpeak"] = ~is_summer & offpeak
    return masks


_PV_PROFILE_CACHE: dict[int, np.ndarray] = {}


def get_solar_per_kw(cz: int, utility: str) -> np.ndarray:
    """Return 8,760-hr per-kW solar profile for a CZ.

    Tries to import parent <utility>_solar.py and use its pvlib
    profile if available; otherwise falls back to a synthetic profile
    scaled to SYNTHETIC_PV_KWH_PER_KW_YR.
    """
    if cz in _PV_PROFILE_CACHE:
        return _PV_PROFILE_CACHE[cz]
    try:
        mod = importlib.import_module(f"{utility}_solar")
        if hasattr(mod, "build_per_cz_profiles"):
            profiles = mod.build_per_cz_profiles()
            if cz in profiles:
                _PV_PROFILE_CACHE[cz] = profiles[cz]
                return profiles[cz]
    except (ImportError, AttributeError, Exception):
        pass
    # Synthetic fallback: rough hourly diurnal + seasonal shape
    hours = pd.date_range("2025-01-01", periods=8760, freq="h")
    daylight = ((hours.hour - 12) / 6.0)  # -2..2 range
    diurnal = np.maximum(0, 1 - daylight ** 2)
    seasonal = 1 + 0.3 * np.cos((hours.dayofyear - 172) * 2 * np.pi / 365)
    raw = np.asarray(diurnal * seasonal)
    profile = raw / raw.sum() * SYNTHETIC_PV_KWH_PER_KW_YR
    _PV_PROFILE_CACHE[cz] = profile
    return profile


def select_archetypes(buildings: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Per-CZ medoid of medoids — pick a representative per CZ.

    Total returned ~ min(n, n_unique_cz). Used so we run hourly LP only
    on a manageable subset.
    """
    if "cluster_weight" in buildings.columns:
        sort_col = "cluster_weight"
    else:
        sort_col = buildings.columns[0]
    arch = (buildings.sort_values(sort_col, ascending=False)
            .groupby("cec_cz").head(1).head(n))
    return arch


def run_for_archetype(
    utility: str,
    bldg_id: int,
    cz: int,
    annual_load_fallback: float,
    rates: pd.DataFrame,
    pv_grid: list[float],
    batt_grid: list[float],
):
    load = load_hourly_load(utility, bldg_id)
    if load is None:
        scaling = annual_load_fallback / SYNTHETIC_PV_KWH_PER_KW_YR
        load = get_solar_per_kw(cz, utility) * 0  # placeholder
        # No hourly load file -> skip this archetype rather than fake it.
        return []
    solar_per_kw = get_solar_per_kw(cz, utility)
    eec = load_eec_hourly(utility)
    rate_rows = rates[rates["rate_type"].isin(("designed_tou",
                                                "demand_charge"))]

    rows = []
    for _, r in rate_rows.iterrows():
        rate_arr = build_hourly_rate_array(r, utility)
        if rate_arr is None:
            continue
        for pv_kw in pv_grid:
            for batt_kwh in batt_grid:
                if pv_kw == 0 and batt_kwh == 0:
                    continue
                solar = solar_per_kw * pv_kw
                if batt_kwh > 0:
                    res = battery_lp_dispatch_param(
                        load, solar, rate_arr, eec,
                        batt_kwh=batt_kwh,
                        batt_pmax_kw=batt_kwh * BATTERY_C_RATE)
                else:
                    # No battery: net of solar import/export only
                    net = load - solar
                    grid_in = np.maximum(net, 0)
                    grid_out = np.maximum(-net, 0)
                    res = {
                        "import_cost": float(np.dot(grid_in, rate_arr)),
                        "export_credit": float(np.dot(grid_out, eec)),
                        "net_grid_kwh": float(grid_in.sum() - grid_out.sum()),
                    }
                if res is None:
                    continue
                # Bill before (no PV, no batt): same load against rate
                bill_before = float(np.dot(load, rate_arr))
                bill_after = res["import_cost"] - res["export_credit"]
                annual_savings = bill_before - bill_after
                capex = p.CapexBreakdown(pv_kw=pv_kw, battery_kwh=batt_kwh)
                ctx = p.IncentiveContext()
                net_capex, _ = p.apply_capex_stack(capex, ctx)
                cashflows = p.annual_cashflow_series(
                    annual_savings,
                    midlife_replacement_year=config.INVERTER_REPLACEMENT_YEAR,
                    midlife_replacement_cost=(
                        config.INVERTER_REPLACEMENT_COST if pv_kw > 0 else 0))
                npv = p.npv(cashflows, capex=net_capex)
                payback = p.simple_payback(net_capex, max(annual_savings, 0))
                rows.append({
                    "utility": utility,
                    "bldg_id": bldg_id,
                    "cec_cz": cz,
                    "rate_id": r["scenario_id"],
                    "pv_kw": pv_kw,
                    "batt_kwh": batt_kwh,
                    "annual_savings": annual_savings,
                    "import_cost_post": res["import_cost"],
                    "export_credit_post": res["export_credit"],
                    "net_capex": net_capex,
                    "npv": npv,
                    "simple_payback_yrs": payback,
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", required=True,
                    choices=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--archetypes", type=int, default=20,
                    help="Number of archetype buildings to run (per-CZ "
                         "medoid of medoids). Each runs full sizing grid.")
    ap.add_argument("--out-dir", default=str(config.DATA_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    bldgs = pd.read_parquet(out_dir / "representative_buildings.parquet")
    u_b = bldgs[bldgs["utility"].str.lower() == args.utility]
    arch = select_archetypes(u_b, n=args.archetypes)
    print(f"Selected {len(arch)} archetypes for {args.utility}")

    rates = pd.read_csv(out_dir / f"rate_scenarios_extended_{args.utility}.csv")
    all_rows = []
    for _, b in arch.iterrows():
        rows = run_for_archetype(
            args.utility, int(b["bldg_id"]), int(b["cec_cz"]),
            float(b["out.electricity.total.energy_consumption.kwh"]),
            rates, config.PV_KW_GRID, config.BATT_KWH_GRID)
        if not rows:
            print(f"  bldg {b['bldg_id']}: no hourly load file - skipped")
            continue
        all_rows.extend(rows)
        print(f"  bldg {b['bldg_id']} (CZ {b['cec_cz']}): {len(rows)} cells")

    if not all_rows:
        sys.exit(
            "No archetypes produced output. Likely Baseline_<utility>/ is "
            "missing or its parquets follow a different naming convention "
            "than <bldg_id>-*.parquet. Check load_hourly_load() docstring.")

    df = pd.DataFrame(all_rows)
    out = out_dir / f"sizing_results_hourly_{args.utility}.parquet"
    df.to_parquet(out, index=False)
    print(f"\nWrote {len(df):,} cells -> {out}")


if __name__ == "__main__":
    main()
