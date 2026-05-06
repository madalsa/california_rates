"""Orchestrator for the personal-economics pipeline.

Stages:
  0. representative_buildings -> medoids + weights
  1. rate_designer_extended    -> rate_scenarios_extended_<u>.csv
  2. sizing_optimizer          -> sizing_results / sizing_optimal_<u>.parquet
  3. vmt_sensitivity           -> ev_sensitivity_<u>.parquet
  4. upgrade11_economics       -> upgrade11_economics_<u>.parquet

Usage:
    python -m electrification_economics.src.run_economics \
        --utility pge sce sdge --stage all
    python -m electrification_economics.src.run_economics --stage 2 --test

--test runs each stage on a 50-building subset for fast smoke validation.

Stage 0 reads ResStock metadata directly; stages 1-4 read the artifacts
from prior stages (data/*.parquet|csv). Each stage writes its outputs to
data/ and is independently re-runnable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


STAGES = {
    "0": ("representative_buildings", "Representative buildings"),
    "1": ("rate_designer_extended",   "Extended rate scenarios"),
    "2": ("sizing_optimizer",          "PV/battery sizing"),
    "3": ("vmt_sensitivity",           "VMT / gas-price sweep"),
    "4": ("upgrade11_economics",       "Upgrade 11 (heat pump)"),
}


def run_stage(stage_id: str, utilities: list[str], test: bool) -> None:
    mod, label = STAGES[stage_id]
    cmd = [sys.executable, "-m", f"electrification_economics.src.{mod}"]
    if mod != "representative_buildings":
        cmd += ["--utilities"] + utilities
    if test and mod != "rate_designer_extended" and mod != "representative_buildings":
        cmd += ["--limit-buildings", "50"]
    elif test and mod == "representative_buildings":
        cmd += ["--target", "50"]
    print(f"\n=== Stage {stage_id}: {label} ===")
    print("  $ " + " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(config.CR_ROOT))
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"Stage {stage_id} failed (rc={proc.returncode})")
    print(f"  done in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", nargs="+",
                    default=list(config.INCLUDED_UTILITIES))
    ap.add_argument("--stage", default="all",
                    help="0-4 or 'all'")
    ap.add_argument("--test", action="store_true",
                    help="Smoke run on a small subset.")
    args = ap.parse_args()

    if args.stage == "all":
        stages = list(STAGES.keys())
    else:
        stages = [args.stage]

    for s in stages:
        run_stage(s, args.utility, args.test)

    print("\n=== Done ===")
    print(f"Outputs in {config.DATA_DIR}/")


if __name__ == "__main__":
    main()
