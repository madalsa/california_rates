"""
calculate_tou_weights_sce.py — Compute TOU consumption weights for SCE buildings.

Reads all Baseline_SCE parquets, classifies each hour into the SCE TOU-D-4-9
period structure, and computes the fraction of total residential consumption
in each period.

SCE TOU-D-4-9 periods:
  Summer (Jun-Oct):  Peak 4-9pm, Off-peak all other hours
  Winter (Nov-May):  Peak 4-9pm, Mid-peak 9pm-8am, Off-peak 8am-4pm

Usage:
  python calculate_tou_weights_sce.py
  python calculate_tou_weights_sce.py --baseline-dir ./Baseline_SCE

Output:
  tou_weights_sce.csv  (5 rows: period, weight)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASELINE_DIR = './Baseline_SCE'
METADATA_FILE = 'CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet'
PUMA_UTILITY_FILE = 'puma_utility_data.csv'
OUTPUT_FILE = 'tou_weights_sce.csv'


def get_sce_tou_period_array():
    """Build 8760-length array of TOU period labels for SCE TOU-D-4-9."""
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1  # 1-indexed

    hour_of_day = hours % 24
    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)
    # Winter midpeak: 9pm-8am (hours 21-23, 0-7)
    is_winter_midpeak = (hour_of_day >= 21) | (hour_of_day < 8)

    periods = np.empty(8760, dtype='U20')

    # Summer periods (no midpeak)
    periods[is_summer & is_peak] = 'summer_peak'
    periods[is_summer & ~is_peak] = 'summer_offpeak'

    # Winter periods
    periods[~is_summer & is_peak] = 'winter_peak'
    periods[~is_summer & ~is_peak & is_winter_midpeak] = 'winter_midpeak'
    periods[~is_summer & ~is_peak & ~is_winter_midpeak] = 'winter_offpeak'

    return periods


def main():
    parser = argparse.ArgumentParser(description='Calculate SCE TOU weights')
    parser.add_argument('--baseline-dir', default=BASELINE_DIR,
                        help='Path to Baseline_SCE directory')
    parser.add_argument('--output', default=OUTPUT_FILE,
                        help='Output CSV path')
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        print(f"ERROR: {baseline_dir} not found!")
        print("Run this script in the directory containing Baseline_SCE/")
        return

    # Load SCE building IDs from metadata
    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    sce_pumas = puma_util[puma_util['utility_acronym'] == 'SCE']['PUMA'].tolist()
    sce_meta = meta[meta['puma20'].isin(sce_pumas)]
    sce_building_ids = set(sce_meta['building_id'].astype(str))
    print(f"SCE buildings in metadata: {len(sce_building_ids)}")

    # Get scaling factors
    sf_lookup = dict(zip(
        sce_meta['building_id'].astype(str),
        sce_meta['scaling_factor'].fillna(1.0)
    ))

    # Build TOU period array
    period_array = get_sce_tou_period_array()
    period_names = ['summer_peak', 'summer_offpeak',
                    'winter_peak', 'winter_midpeak', 'winter_offpeak']

    # Accumulate consumption by TOU period across all buildings
    period_totals = {p: 0.0 for p in period_names}
    total_kwh = 0.0

    parquet_files = sorted(baseline_dir.glob('*-0.parquet'))
    print(f"Parquet files found: {len(parquet_files)}")

    processed = 0
    skipped = 0
    for pq_file in parquet_files:
        building_id = pq_file.stem.split('-')[0]
        if building_id not in sce_building_ids:
            skipped += 1
            continue

        try:
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)

            sf = sf_lookup.get(building_id, 1.0)
            hourly_load_scaled = hourly_load * sf

            for p in period_names:
                mask = period_array == p
                period_totals[p] += hourly_load_scaled[mask].sum()
            total_kwh += hourly_load_scaled.sum()

            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed} buildings...")

        except Exception as e:
            if processed < 5:
                print(f"  Error on {pq_file.name}: {e}")

    print(f"\nProcessed: {processed} buildings, skipped: {skipped}")
    print(f"Total consumption: {total_kwh/1e9:.3f} TWh")

    # Compute weights
    weights = {p: period_totals[p] / total_kwh for p in period_names}

    print(f"\nSCE TOU-D-4-9 consumption weights:")
    for p in period_names:
        print(f"  {p:20s}: {weights[p]*100:5.2f}%  ({period_totals[p]/1e6:.1f} GWh)")
    print(f"  {'TOTAL':20s}: {sum(weights.values())*100:5.2f}%")

    # Save
    out_df = pd.DataFrame([
        {'period': p, 'weight': weights[p]} for p in period_names
    ])
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
