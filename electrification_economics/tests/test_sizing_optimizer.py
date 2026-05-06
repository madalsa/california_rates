"""Smoke tests for sizing_optimizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd

from src import config, sizing_optimizer as so


def test_pv_tou_split_sums_to_one_when_all_periods_present():
    pv_kwh = 1000
    periods = list(so.PV_GEN_TOU_SHARE.keys())
    out = so.split_pv_by_tou(pv_kwh, periods)
    assert math.isclose(sum(out.values()), 1000, abs_tol=0.01)


def test_pv_tou_split_normalizes_to_subset():
    """If utility omits midpeak, PV should redistribute to remaining periods."""
    periods = ["summer_peak", "summer_offpeak", "winter_peak", "winter_offpeak"]
    out = so.split_pv_by_tou(1000, periods)
    assert math.isclose(sum(out.values()), 1000, abs_tol=0.01)


def test_load_split_sums_to_annual():
    weights = so.load_tou_weights("sce")
    out = so.split_annual_kwh_by_tou(10000, weights)
    assert math.isclose(sum(out.values()), 10000, abs_tol=0.1)


def test_battery_arbitrage_capacity_bounded():
    # 10 kWh battery x 365 cycles x 0.88 RT = 3,212 kWh max
    cap = so.battery_arbitrage_kwh(
        batt_kwh=10, load_peak_kwh=10000, offpeak_to_peak_kwh=10000)
    assert cap < 3300
    # If load is small, that limits
    cap2 = so.battery_arbitrage_kwh(
        batt_kwh=10, load_peak_kwh=500, offpeak_to_peak_kwh=10000)
    assert cap2 == 500


def test_zero_pv_zero_batt_skipped_in_grid():
    """The (0, 0) combo is skipped (it's the do-nothing baseline)."""
    bldgs = pd.read_parquet(
        config.DATA_DIR / "representative_buildings.parquet").head(1)
    rates = pd.read_csv(config.DATA_DIR / "rate_scenarios_extended_pge.csv")
    rates = rates[rates["scenario_id"] == "F0_WF0_ROE0"]
    df = so.build_sizing_table("pge", bldgs, rates)
    assert not ((df["pv_kw"] == 0) & (df["batt_kwh"] == 0)).any()


if __name__ == "__main__":
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}  {e}")
                failures += 1
            except Exception as e:
                print(f"  ERR   {name}  {type(e).__name__}: {e}")
                failures += 1
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
