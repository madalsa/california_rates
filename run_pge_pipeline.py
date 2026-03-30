"""
run_pge_pipeline.py — End-to-end PGE rate analysis pipeline

Stages:
  1. Generate fresh rate scenarios (rate_designer_pge.py)
  2. Compute baseline bills for all PGE buildings (from hourly parquets)
  3. Assign technology adoption (PV, battery, EV, heat pump)
  4. Generate solar profiles with pvlib (per CEC climate zone)
  5. Run battery LP dispatch for battery-adopted buildings
  6. Compute post-adoption bills (net billing for solar, LP-optimized for battery)
  7. Output distributional analysis

Usage:
  python run_pge_pipeline.py                    # full run
  python run_pge_pipeline.py --test             # test with 50 buildings
  python run_pge_pipeline.py --stage 2          # run from stage 2 onward
  python run_pge_pipeline.py --skip-tech        # skip tech adoption (stages 3-6)
  python run_pge_pipeline.py --tech-only        # run only tech stages (3-6)

Run unattended (Mac):
  caffeinate -s python run_pge_pipeline.py > pipeline_pge.log 2>&1 &

Run unattended (Linux):
  nohup python run_pge_pipeline.py > pipeline_pge.log 2>&1 &
"""

import argparse
import time
import os
import pandas as pd

from pge_config import (
    RATE_SCENARIOS_OUT, BASELINE_BILLS_OUT, TECH_ASSIGNMENTS_OUT,
    POSTADOPT_BILLS_OUT, PGE_CZ_COORDINATES, BUILDING_WEIGHT,
)


# ---------------------------------------------------------------------------
# Stage 1: Generate fresh rate scenarios
# ---------------------------------------------------------------------------

def stage1_generate_rate_scenarios(r_sample=None, sample_n_care=None,
                                    sample_n_noncare=None):
    """
    Generate fresh revenue-neutral rate scenarios using rate_designer_pge.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: GENERATE FRESH RATE SCENARIOS")
    print("=" * 80)

    import sys

    if r_sample is None:
        if os.path.exists(BASELINE_BILLS_OUT):
            bills_df = pd.read_csv(BASELINE_BILLS_OUT)
            r_sample = bills_df['e_tou_c_bill'].dropna().sum() * BUILDING_WEIGHT
            sample_n_care = int((bills_df['is_care'] == True).sum() * BUILDING_WEIGHT)
            sample_n_noncare = int((bills_df['is_care'] == False).sum() * BUILDING_WEIGHT)
            print(f"  Loaded R_sample from {BASELINE_BILLS_OUT}: ${r_sample/1e9:.4f}B")
        else:
            print("  ERROR: R_sample required but no baseline bills available.")
            print("  Run stage 2 first (--stage 2) to compute E-TOU-C bills.")
            sys.exit(1)

    from rate_designer_pge import generate_all_scenarios
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
    parser = argparse.ArgumentParser(description='PGE Rate Analysis Pipeline')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: process 50 buildings only')
    parser.add_argument('--stage', type=int, default=1,
                        help='Start from this stage (1-7)')
    parser.add_argument('--skip-tech', action='store_true',
                        help='Skip technology adoption stages (3-6)')
    # Battery dispatch is heuristic-only for PGE (LP reserved for SCE)
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
    print("PGE RATE ANALYSIS PIPELINE")
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

        from pge_baseline_bills import stage2_compute_baseline_bills
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
        from pge_summary import stage7_summary
        stage7_summary(bills_df, rate_scenarios)
    else:
        # --------------------------------------------------------------
        # Stage 3: Tech assignments
        # --------------------------------------------------------------
        if args.stage <= 3:
            from pge_tech_assign import stage3_tech_assignments
            tech_df = stage3_tech_assignments(bills_df)
        else:
            tech_df = pd.read_csv(TECH_ASSIGNMENTS_OUT)
            print(f"\nLoaded tech assignments from {TECH_ASSIGNMENTS_OUT}")

        # --------------------------------------------------------------
        # Stage 4: Solar profiles
        # --------------------------------------------------------------
        if args.stage <= 4:
            from pge_solar import stage4_solar_profiles
            solar_profiles, annual_kwh_per_kw_by_cz = stage4_solar_profiles(
                tech_df, bills_df)
        else:
            # Fallback: synthetic profiles for all known PGE CZs
            from pge_solar import _synthetic_solar_profile
            solar_profiles = {}
            annual_kwh_per_kw_by_cz = {}
            for cz, (lat, lon, alt, name) in PGE_CZ_COORDINATES.items():
                synth = _synthetic_solar_profile(lat)
                solar_profiles[cz] = synth
                annual_kwh_per_kw_by_cz[cz] = synth.sum()

        # Stage 5 is integrated into Stage 6

        # --------------------------------------------------------------
        # Stage 6: Post-adoption bills
        # --------------------------------------------------------------
        if args.stage <= 6:
            from pge_post_adoption import stage6_post_adoption_bills
            final_df = stage6_post_adoption_bills(
                bills_df, tech_df, solar_profiles, rate_scenarios,
                annual_kwh_per_kw_by_cz=annual_kwh_per_kw_by_cz,
                skip_s3=args.skip_s3)
        else:
            final_df = pd.read_csv(POSTADOPT_BILLS_OUT)
            print(f"\nLoaded post-adoption bills from {POSTADOPT_BILLS_OUT}")

        # --------------------------------------------------------------
        # Stage 7: Summary
        # --------------------------------------------------------------
        if not getattr(args, 'tech_only', False):
            from pge_summary import stage7_summary
            stage7_summary(final_df, rate_scenarios)
        else:
            print("\n  --tech-only: skipping summary stage")

    total_time = time.time() - pipeline_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
