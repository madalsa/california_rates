"""
run_sdge_pipeline.py — End-to-end SDGE rate analysis pipeline

Stages:
  1. Generate fresh rate scenarios (rate_designer.py)
  2. Compute baseline bills for all SDGE buildings (from hourly parquets)
  3. Assign technology adoption (PV, battery, EV, heat pump)
  4. Generate solar profiles with pvlib for PV-adopted buildings
  5. Run battery LP dispatch for battery-adopted buildings
  6. Compute post-adoption bills (net billing for solar, LP-optimized for battery)
  7. Output distributional analysis

Usage:
  python run_sdge_pipeline.py                    # full run
  python run_sdge_pipeline.py --test             # test with 50 buildings
  python run_sdge_pipeline.py --stage 2          # run from stage 2 onward
  python run_sdge_pipeline.py --skip-tech        # skip tech adoption (stages 3-6)
  python run_sdge_pipeline.py --tech-only        # run only tech stages (3-6)

Run unattended (Linux):
  nohup python run_sdge_pipeline.py > pipeline_sdge.log 2>&1 &
"""

import argparse
import time
import os
import pandas as pd

from sdge_config import (
    RATE_SCENARIOS_OUT, BASELINE_BILLS_OUT, TECH_ASSIGNMENTS_OUT,
    POSTADOPT_BILLS_OUT, BUILDING_WEIGHT,
    SDGE_LATITUDE, SDGE_ANNUAL_KWH_PER_KW, DEFAULT_PV_SIZE_KW,
)


# ---------------------------------------------------------------------------
# Stage 1: Generate fresh rate scenarios
# ---------------------------------------------------------------------------

def stage1_generate_rate_scenarios(r_sample=None, sample_n_care=None,
                                    sample_n_noncare=None):
    """Generate fresh revenue-neutral rate scenarios."""
    import sys

    print("\n" + "=" * 80)
    print("STAGE 1: GENERATE FRESH RATE SCENARIOS")
    print("=" * 80)

    if r_sample is None:
        if os.path.exists(BASELINE_BILLS_OUT):
            bills_df = pd.read_csv(BASELINE_BILLS_OUT)
            r_sample = bills_df['tou_dr_bill'].dropna().sum() * BUILDING_WEIGHT
            sample_n_care = int((bills_df['is_care'] == True).sum() * BUILDING_WEIGHT)
            sample_n_noncare = int((bills_df['is_care'] == False).sum() * BUILDING_WEIGHT)
            print(f"  Loaded R_sample from {BASELINE_BILLS_OUT}: ${r_sample/1e9:.4f}B")
        else:
            print("  ERROR: R_sample required but no baseline bills available.")
            print("  Run stage 2 first (--stage 2) to compute TOU-DR bills.")
            sys.exit(1)

    from rate_designer import generate_all_scenarios
    df = generate_all_scenarios(
        output_csv=RATE_SCENARIOS_OUT,
        r_sample=r_sample,
        sample_n_care=sample_n_care,
        sample_n_noncare=sample_n_noncare,
    )
    print(f"\nSaved {len(df)} scenarios to {RATE_SCENARIOS_OUT}")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SDGE Rate Analysis Pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process 50 buildings only')
    parser.add_argument('--stage', type=int, default=1,
                        help='Start from this stage (1-7)')
    parser.add_argument('--skip-tech', action='store_true',
                        help='Skip technology adoption stages (3-6)')
    parser.add_argument('--use-lp', action='store_true',
                        help='Use LP for battery dispatch (slower but optimal)')
    parser.add_argument('--n-buildings', type=int, default=None,
                        help='Number of buildings to process')
    parser.add_argument('--tech-only', action='store_true',
                        help='Run only tech stages (3-6)')
    parser.add_argument('--skip-s3', action='store_true',
                        help='Skip S3 (PV+storage+EV) adoption scenario')
    args = parser.parse_args()

    n_buildings = 50 if args.test else args.n_buildings

    if args.tech_only:
        args.stage = max(args.stage, 3)
        args.skip_tech = False

    print("=" * 80)
    print("SDGE RATE ANALYSIS PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}"
          f"{' (tech-only)' if args.tech_only else ''}")
    print(f"Starting stage: {args.stage}")
    print("=" * 80)

    pipeline_start = time.time()

    # ------------------------------------------------------------------
    # Stage 2: Baseline bills
    # ------------------------------------------------------------------
    if args.stage <= 2:
        existing_scenarios = None
        if args.stage > 1 and os.path.exists(RATE_SCENARIOS_OUT):
            existing_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
            print(f"\nLoaded existing rate scenarios from {RATE_SCENARIOS_OUT}")

        from sdge_baseline_bills import stage2_compute_baseline_bills
        bills_df, rate_scenarios = stage2_compute_baseline_bills(
            existing_scenarios, n_buildings)
    else:
        bills_df = pd.read_csv(BASELINE_BILLS_OUT)
        rate_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
        print(f"\nLoaded existing bills from {BASELINE_BILLS_OUT}")
        print(f"Loaded existing rate scenarios from {RATE_SCENARIOS_OUT}")

    # ------------------------------------------------------------------
    # Skip tech if requested
    # ------------------------------------------------------------------
    if args.skip_tech:
        print("\n  Skipping technology adoption stages (--skip-tech)")
        from sdge_summary import stage7_summary
        stage7_summary(bills_df, rate_scenarios)
    else:
        # --------------------------------------------------------------
        # Stage 3: Tech assignments
        # --------------------------------------------------------------
        if args.stage <= 3:
            from sdge_tech_assign import stage3_tech_assignments
            tech_df = stage3_tech_assignments(bills_df)
        else:
            tech_df = pd.read_csv(TECH_ASSIGNMENTS_OUT)
            print(f"\nLoaded tech assignments from {TECH_ASSIGNMENTS_OUT}")

        # --------------------------------------------------------------
        # Stage 4: Solar profiles (single centroid for SDGE)
        # --------------------------------------------------------------
        if args.stage <= 4:
            from sdge_solar import stage4_solar_profiles
            solar_per_kw, annual_kwh_per_kw = stage4_solar_profiles(
                tech_df, bills_df)
        else:
            from sdge_solar import _synthetic_solar_profile
            synth = _synthetic_solar_profile()
            solar_per_kw = synth / DEFAULT_PV_SIZE_KW
            annual_kwh_per_kw = solar_per_kw.sum()

        # Stage 5 is integrated into Stage 6

        # --------------------------------------------------------------
        # Stage 6: Post-adoption bills
        # --------------------------------------------------------------
        if args.stage <= 6:
            from sdge_post_adoption import stage6_post_adoption_bills
            final_df = stage6_post_adoption_bills(
                bills_df, tech_df, solar_per_kw, rate_scenarios,
                use_lp=args.use_lp, annual_kwh_per_kw=annual_kwh_per_kw,
                skip_s3=args.skip_s3)
        else:
            final_df = pd.read_csv(POSTADOPT_BILLS_OUT)
            print(f"\nLoaded post-adoption bills from {POSTADOPT_BILLS_OUT}")

        # --------------------------------------------------------------
        # Stage 7: Summary
        # --------------------------------------------------------------
        if not args.tech_only:
            from sdge_summary import stage7_summary
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
