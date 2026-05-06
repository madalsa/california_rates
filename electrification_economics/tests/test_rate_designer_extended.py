"""Smoke tests for rate_designer_extended."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import rate_designer_extended as rde, config


def test_build_extended_pge_has_all_categories():
    df = rde.build_extended("pge")
    types = set(df["rate_type"].unique())
    assert "designed_tou" in types
    assert "demand_charge" in types
    assert "ev_submetered_tou" in types
    assert "export_overlay" in types


def test_designed_count_matches_source():
    """Re-emitted designed rows should match the 40 in the source."""
    src = pd.read_csv(config.CR_ROOT / "rate_scenarios_pge_fresh.csv")
    df = rde.build_extended("pge")
    n_designed = (df["rate_type"] == "designed_tou").sum()
    assert n_designed == len(src)


def test_dc_volumetric_lower_than_base():
    """DC scenarios fund part of revenue via demand charge, so volumetric
    must be lower than the F0_WF0_ROE0 base."""
    df = rde.build_extended("sce")
    base = pd.read_csv(config.CR_ROOT / "rate_scenarios_sce_fresh.csv")
    base_summer_peak = base[base["Scenario"] == "F0_WF0_ROE0"]["summer_peak"].iloc[0]
    for dc_id in ("DC_5", "DC_15"):
        dc_row = df[df["scenario_id"] == dc_id].iloc[0]
        assert dc_row["summer_peak"] < base_summer_peak


def test_dc_revenue_lowers_with_higher_dc():
    df = rde.build_extended("pge")
    dc5 = df[df["scenario_id"] == "DC_5"].iloc[0]
    dc15 = df[df["scenario_id"] == "DC_15"].iloc[0]
    assert dc15["summer_peak"] < dc5["summer_peak"]


def test_ev_tou_super_offpeak_lower_than_peak():
    df = rde.build_extended("pge")
    ev = df[df["scenario_id"] == "EV_TOU"].iloc[0]
    assert ev["ev_super_offpeak"] < ev["ev_on_peak"]
    assert ev["ev_super_offpeak"] == 0.18
    assert ev["ev_on_peak"] == 0.55


def test_export_regimes_present():
    df = rde.build_extended("sdge")
    regimes = set(
        df[df["rate_type"] == "export_overlay"]["export_regime"])
    assert {"nbt_hourly", "nem2_retail", "flat_5c", "flat_15c"} <= regimes


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
