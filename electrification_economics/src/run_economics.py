"""Orchestrator for the personal-economics pipeline.

Stages:
  0. representative_buildings -> medoids + weights
  1. rate_designer_extended   -> rate_scenarios_extended_<utility>.csv
  2. sizing_optimizer         -> optimal PV/batt per (bldg, rate)
  3. vmt_sensitivity          -> EV sweep
  4. upgrade11_economics      -> HP / whole-home bundle
  5. payback_npv aggregation  -> master results parquet
  6. tornado / sensitivity figures

Usage:
    python -m electrification_economics.src.run_economics \
        --utility sce --stage all --test
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--utility", nargs="+", default=["pge", "sce", "sdge"])
    p.add_argument("--stage", default="all",
                   help="0..6 or 'all'")
    p.add_argument("--test", action="store_true",
                   help="Small subset: 50 representative buildings, 3 rates.")
    args = p.parse_args()

    raise NotImplementedError("Stages not yet implemented; see src/*.py stubs.")


if __name__ == "__main__":
    main()
