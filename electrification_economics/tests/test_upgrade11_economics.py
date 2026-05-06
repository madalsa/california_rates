"""Smoke tests for upgrade11_economics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd

from src import config, upgrade11_economics as u11


def test_cop_table_covers_all_cz():
    for cz in range(1, 17):
        assert cz in u11.COP_SPACE_BY_CZ
        assert 2.0 < u11.COP_SPACE_BY_CZ[cz] < 4.0


def test_cold_zones_lower_cop_than_mild():
    assert u11.COP_SPACE_BY_CZ[16] < u11.COP_SPACE_BY_CZ[6]
    assert u11.COP_SPACE_BY_CZ[1] < u11.COP_SPACE_BY_CZ[7]


def test_project_upgrade11_zero_gas_means_zero_delta():
    df = pd.DataFrame({
        "cec_cz": [6, 12],
        "out.natural_gas.heating.energy_consumption.kwh": [0, 0],
        "out.natural_gas.hot_water.energy_consumption.kwh": [0, 0],
        "out.natural_gas.range_oven.energy_consumption.kwh": [0, 0],
    })
    out = u11.project_upgrade11_annual(df)
    assert (out["total_delta_kwh"] == 0).all()
    assert (out["total_therms_displaced"] == 0).all()


def test_project_upgrade11_delta_uses_cop():
    """For 1000 kWh of gas heating, HP electric load = 1000/COP / kwh_per_therm."""
    df = pd.DataFrame({
        "cec_cz": [6],  # COP_SPACE 3.2
        "out.natural_gas.heating.energy_consumption.kwh": [3200.0],
        "out.natural_gas.hot_water.energy_consumption.kwh": [0],
        "out.natural_gas.range_oven.energy_consumption.kwh": [0],
    })
    out = u11.project_upgrade11_annual(df)
    # 3200 / 3.2 = 1000 kWh electric
    assert math.isclose(out["delta_kwh_hp_space"].iloc[0], 1000.0, rel_tol=0.01)
    # Therms displaced = 3200 / 29.3001 = ~109.2
    assert math.isclose(out["baseline_therms_heat"].iloc[0], 109.2, abs_tol=0.5)


def test_artifacts_runnable_when_present():
    """If a smoke run wrote outputs, NPVs should be finite (non-NaN)."""
    p = config.DATA_DIR / "upgrade11_economics_sdge.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p)
    nan_share = df["npv"].isna().mean()
    assert nan_share < 0.05, f"too many NaN NPVs: {nan_share:.0%}"


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
