"""
run_sce_pipeline.py — End-to-end SCE rate analysis pipeline

Stages:
  1. Rate scenarios already generated (rate_designer_sce.py → rate_scenarios_sce.csv)
  2. Compute baseline bills for all SCE buildings (from hourly parquets)
  3. Assign technology adoption (PV, battery, EV, heat pump)
  4. Generate solar profiles with pvlib (per CEC climate zone)
  5. LP battery dispatch (integrated in stage 6)
  6. Compute post-adoption bills (LP only, native demand, net billing)
  7. Enhanced distributional analysis (5 customer types, CZ, grid, exports)

Key SCE-specific design choices:
  - TOU-D-4-9 with 5 periods (summer peak/offpeak, winter peak/midpeak/offpeak)
  - Designed scenarios use blended weekday/weekend rates
  - PV sized to 90% of NATIVE demand (not RASS-scaled)
  - LP-only battery dispatch on NATIVE demand
  - No Upgrade 11 / fully electrified load — HP from ResStock baseline
  - 5 customer types: non-adopter, EV, PV+stor, PV+EV+stor, fully electrified

Usage:
  python run_sce_pipeline.py                   # full run
  python run_sce_pipeline.py --test            # test with 50 buildings
  python run_sce_pipeline.py --stage 2         # run from stage 2 onward
  python run_sce_pipeline.py --skip-tech       # skip tech adoption (stages 3-6)
  python run_sce_pipeline.py --n-buildings 200 # limit building count

Run unattended (Linux):
  nohup python run_sce_pipeline.py > pipeline_sce.log 2>&1 &
"""

import argparse
import time
import os
import pandas as pd

from sce_config import (
    RATE_SCENARIOS_OUT, BASELINE_BILLS_OUT, TECH_ASSIGNMENTS_OUT,
    POSTADOPT_BILLS_OUT, SCE_CZ_COORDINATES, SCE_ANNUAL_KWH_PER_KW,
)


def main():
    parser = argparse.ArgumentParser(description='SCE Rate Analysis Pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process 50 buildings only')
    parser.add_argument('--stage', type=int, default=2,
                        help='Start from this stage (2-7, stage 1 pre-computed)')
    parser.add_argument('--skip-tech', action='store_true',
                        help='Skip technology adoption stages (3-6)')
    parser.add_argument('--n-buildings', type=int, default=None,
                        help='Number of buildings to process')
    parser.add_argument('--tech-only', action='store_true',
                        help='Run only tech stages (3-6)')
    args = parser.parse_args()

    n_buildings = 50 if args.test else args.n_buildings

    if args.tech_only:
        args.stage = max(args.stage, 3)
        args.skip_tech = False

    print("=" * 80)
    print("SCE RATE ANALYSIS PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}"
          f"{' (tech-only)' if args.tech_only else ''}")
    print(f"Starting stage: {args.stage}")
    print(f"PV sizing: 90% native demand offset")
    print(f"Battery dispatch: LP only (native demand)")
    print("=" * 80)

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Rate scenarios (loaded if available, generated in Stage 2)
    # ------------------------------------------------------------------
    if os.path.exists(RATE_SCENARIOS_OUT):
        rate_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
        print(f"\nLoaded {len(rate_scenarios)} rate scenarios from {RATE_SCENARIOS_OUT}")
    else:
        rate_scenarios = None
        print(f"\n  Rate scenarios not found — will generate in Stage 2 after computing R_sample")

    # ------------------------------------------------------------------
    # Stage 2: Baseline bills
    # ------------------------------------------------------------------
    if args.stage <= 2:
        from sce_baseline_bills import stage2_compute_baseline_bills
        bills_df, rate_scenarios = stage2_compute_baseline_bills(
            rate_scenarios, n_buildings)
    else:
        bills_df = pd.read_csv(BASELINE_BILLS_OUT)
        print(f"\nLoaded baseline bills from {BASELINE_BILLS_OUT}")

    # ------------------------------------------------------------------
    # Skip tech if requested
    # ------------------------------------------------------------------
    if args.skip_tech:
        print("\n  Skipping technology adoption stages (--skip-tech)")
        from sce_summary import stage7_summary
        stage7_summary(bills_df, rate_scenarios)
    else:
        # --------------------------------------------------------------
        # Stage 3: Tech assignments
        # --------------------------------------------------------------
        if args.stage <= 3:
            from sce_tech_assign import stage3_tech_assignments
            tech_df = stage3_tech_assignments(bills_df)
        else:
            tech_df = pd.read_csv(TECH_ASSIGNMENTS_OUT)
            print(f"\nLoaded tech assignments from {TECH_ASSIGNMENTS_OUT}")

        # --------------------------------------------------------------
        # Stage 4: Solar profiles
        # --------------------------------------------------------------
        if args.stage <= 4:
            from sce_solar import stage4_solar_profiles
            solar_profiles, annual_kwh_per_kw_by_cz = stage4_solar_profiles(
                tech_df, bills_df)
        else:
            from sce_solar import _synthetic_solar_profile
            solar_profiles = {}
            annual_kwh_per_kw_by_cz = {}
            for cz, (lat, lon, alt, name) in SCE_CZ_COORDINATES.items():
                synth = _synthetic_solar_profile(lat)
                solar_profiles[cz] = synth
                annual_kwh_per_kw_by_cz[cz] = synth.sum()

        # --------------------------------------------------------------
        # Stage 5+6: LP battery dispatch + post-adoption bills
        # --------------------------------------------------------------
        if args.stage <= 6:
            from sce_post_adoption import stage6_post_adoption_bills
            final_df = stage6_post_adoption_bills(
                bills_df, tech_df, solar_profiles, rate_scenarios,
                annual_kwh_per_kw_by_cz=annual_kwh_per_kw_by_cz)
        else:
            final_df = pd.read_csv(POSTADOPT_BILLS_OUT)
            print(f"\nLoaded post-adoption bills from {POSTADOPT_BILLS_OUT}")

        # --------------------------------------------------------------
        # Stage 7: Summary
        # --------------------------------------------------------------
        if not args.tech_only:
            from sce_summary import stage7_summary
            stage7_summary(final_df, rate_scenarios)
        else:
            print("\n  --tech-only: skipping summary")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
