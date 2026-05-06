"""Smoke tests for vmt_sensitivity."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np
import pandas as pd

from src import config, vmt_sensitivity as vs


def test_charging_profile_weights_sum_to_one():
    for name, w in vs.CHARGING_PROFILES.items():
        assert math.isclose(w.sum(), 1.0, abs_tol=1e-9), name


def test_overnight_profile_concentrates_offhours():
    w = vs.CHARGING_PROFILES["overnight_offpeak"]
    assert w[0:7].sum() > 0.9


def test_hourly_to_tou_weights_sums_to_one_pge():
    w = vs.CHARGING_PROFILES["overnight_offpeak"]
    weights = vs.hourly_to_tou_weights(w, "pge")
    assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9)


def test_effective_kwh_lower_for_overnight_charging():
    """Overnight charging should hit lower TOU rates than opportunistic."""
    df = pd.read_csv(config.DATA_DIR / "rate_scenarios_extended_pge.csv")
    rate = df[df["scenario_id"] == "F0_WF0_ROE0"].iloc[0]
    overnight = vs.effective_kwh_price(rate, "overnight_offpeak", "pge")
    opportunistic = vs.effective_kwh_price(rate, "opportunistic", "pge")
    assert overnight < opportunistic


def test_higher_vmt_higher_savings():
    """Higher VMT must produce higher annual fuel savings (positive case)."""
    df = pd.read_csv(config.DATA_DIR / "rate_scenarios_extended_sce.csv")
    rate = df[df["scenario_id"] == "F0_WF0_ROE0"].iloc[0]
    eff = vs.effective_kwh_price(rate, "overnight_offpeak", "sce")
    from src import payback_npv as p
    s_low = p.ev_annual_fuel_savings(
        vmt=5000, gas_price=4.50, ev_eff_mi_per_kwh=3.3,
        rate_effective_per_kwh=eff)
    s_hi = p.ev_annual_fuel_savings(
        vmt=20000, gas_price=4.50, ev_eff_mi_per_kwh=3.3,
        rate_effective_per_kwh=eff)
    assert s_hi > s_low > 0


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
