"""Smoke tests for representative_buildings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pandas as pd

from src import config, representative_buildings as rb


def test_ami_bin_mapping():
    assert rb.AMI_BIN_TO_FRAC["0-30%"] == 0.15
    assert rb.AMI_BIN_TO_FRAC["60-80%"] == 0.70
    assert rb.AMI_BIN_TO_FRAC["150%+"] == 2.00
    assert math.isnan(rb.AMI_BIN_TO_FRAC["Not Available"])


def test_ebd_threshold_consistent():
    """Bins <=80% AMI midpoints should all flag <=EBD threshold."""
    for bin_name in ("0-30%", "30-60%", "60-80%"):
        assert rb.AMI_BIN_TO_FRAC[bin_name] <= config.EBD_AMI_THRESHOLD
    for bin_name in ("80-100%", "100-120%", "120-150%", "150%+"):
        assert rb.AMI_BIN_TO_FRAC[bin_name] > config.EBD_AMI_THRESHOLD


def test_vintage_decade_covers_all_known():
    expected = {"<1940", "1940s", "1950s", "1960s", "1970s",
                "1980s", "1990s", "2000s", "2010s"}
    assert expected.issubset(set(rb.VINTAGE_DECADE.keys()))


def test_output_artifact_present_after_run():
    """If user has run the script, artifact should exist; else skip."""
    p = config.DATA_DIR / "representative_buildings.parquet"
    if not p.exists():
        return  # not run yet; this is a smoke test, not a precondition
    df = pd.read_parquet(p)
    assert len(df) > 0
    # All medoids must be in scope utilities
    assert set(df["utility"].str.lower()).issubset(
        set(config.INCLUDED_UTILITIES))
    # No EBD-eligible should remain
    df["cec_cz_int"] = df["cec_cz"].astype(int)
    df["ami_frac"] = df["ami_bin"].map(rb.AMI_BIN_TO_FRAC)
    is_ebd = (df["ami_frac"] <= config.EBD_AMI_THRESHOLD) & (
        df["cec_cz_int"].isin(config.EBD_PRIORITY_CEC_CZS))
    assert not is_ebd.any(), "EBD-eligible rows leaked into representatives"


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
