"""
sce_baseline_bills.py — Stage 2: Compute baseline bills for all SCE buildings

Reads hourly parquets from Baseline_SCE/, computes bills under:
  1. Actual SCE tariff rates (TOU-D-4-9, TOU-D-4-9-F) with weekday/weekend
  2. Designed rate scenarios (blended rates from rate_scenarios_sce.csv)

Uses native (unscaled) demand — RASS scaling factor is stored but NOT
applied to load profiles. Scaling factor used only for population
extrapolation in summary.
"""

import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sce_config import (
    BASELINE_DIR, METADATA_FILE, PUMA_UTILITY_FILE, EXCEL_FILE,
    RATE_SCENARIOS_OUT, BASELINE_BILLS_OUT, BUILDING_IDS_FILE,
    ACTUAL_SCE_RATES, DESIGNED_SCENARIOS, TOU_PERIODS,
    BUILDING_WEIGHT, WEEKDAY_FRAC, WEEKEND_FRAC,
    build_time_arrays, build_sce_period_masks,
    build_tou_rate_array, build_actual_tariff_rate_array, safe_float,
)


def load_sce_metadata():
    """Load metadata and filter to SCE buildings."""
    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    sce_pumas = puma_util[puma_util['utility_acronym'] == 'SCE']['PUMA'].tolist()
    sce_meta = meta[meta['puma20'].isin(sce_pumas)].copy()
    print(f"  SCE buildings in metadata: {len(sce_meta)}")
    return sce_meta


def normalize_income(income_str):
    """Normalize income category to low/medium/high."""
    mapping = {'Low': 'low', 'Medium': 'medium', 'High': 'high',
               'low': 'low', 'medium': 'medium', 'high': 'high'}
    return mapping.get(str(income_str).strip(), 'medium')


def calculate_actual_sce_bill(hourly_load, rate_code, puma_str, income, is_care,
                              rates_df, baseline_df, ta):
    """
    Vectorized bill calculation for actual SCE tariff (TOU-D-4-9 or TOU-D-4-9-F).

    Handles weekday/weekend rate differences, baseline credits, CARE discounts,
    and income-graduated fixed charges.
    """
    rate_entries = rates_df[rates_df['rate_type'] == rate_code]
    wd = rate_entries[rate_entries['weekday'] == 'weekday'].iloc[0].to_dict()
    we = rate_entries[rate_entries['weekday'] == 'weekend'].iloc[0].to_dict()

    # Get baseline allowance
    bl_entry = baseline_df[baseline_df['puma'] == puma_str]
    if bl_entry.empty:
        return np.nan
    daily_summer_bl = bl_entry['summer_baseline_allowance'].values[0]
    daily_winter_bl = bl_entry['winter_baseline_allowance'].values[0]

    # Build weekday/weekend rate dicts
    weekday_rates = {
        'summer_peak': safe_float(wd.get('peak_rate_summer1', 0)),
        'summer_offpeak': safe_float(wd.get('offpeak_rate_summer1', 0)),
        'winter_peak': safe_float(wd.get('peak_rate_winter1', 0)),
        'winter_midpeak': safe_float(wd.get('midpeak_rate_winter1', 0)),
        'winter_offpeak': safe_float(wd.get('offpeak_rate_winter1', 0)),
    }
    weekend_rates = {
        'summer_peak': safe_float(we.get('peak_rate_summer1', 0)),
        'summer_offpeak': safe_float(we.get('offpeak_rate_summer1', 0)),
        'winter_peak': safe_float(we.get('peak_rate_winter1', 0)),
        'winter_midpeak': safe_float(we.get('midpeak_rate_winter1', 0)),
        'winter_offpeak': safe_float(we.get('offpeak_rate_winter1', 0)),
    }

    rate_array = build_actual_tariff_rate_array(weekday_rates, weekend_rates, ta)

    # Energy charges
    energy_charges = np.dot(hourly_load, rate_array)

    # Baseline credit
    baseline_credit_rate = safe_float(wd.get('baseline_credit', 0))
    care_discount = abs(safe_float(wd.get('care_discount', 0)))

    days_per_month = ta['days_per_month']
    month_boundaries = ta['month_boundaries']

    total_bl_credit = 0.0
    for m in range(12):
        s, e = month_boundaries[m], month_boundaries[m + 1]
        monthly_kwh = hourly_load[s:e].sum()
        if 6 <= (m + 1) <= 10:
            monthly_bl = daily_summer_bl * days_per_month[m]
        else:
            monthly_bl = daily_winter_bl * days_per_month[m]
        total_bl_credit += baseline_credit_rate * min(monthly_kwh, monthly_bl)

    energy_after_credit = energy_charges - total_bl_credit

    # CARE discount
    if is_care and care_discount > 0:
        energy_after_credit *= (1 - care_discount)

    # Fixed charges (base service charge excluded for consistency across IOUs)
    annual_base_fixed = 0.0

    monthly_fixed = 0.0
    has_fixed = wd.get('Fixed', '') == 'Yes'
    if has_fixed:
        if income == 'low':
            monthly_fixed = safe_float(wd.get('fixedcharge_low', 0))
        elif income == 'medium':
            monthly_fixed = safe_float(wd.get('fixedcharge_med', 0))
        else:
            monthly_fixed = safe_float(wd.get('fixedcharge_high', 0))
    annual_fixed = annual_base_fixed + monthly_fixed * 12

    # Minimum bill
    min_bill_daily = safe_float(wd.get('minimum_bill_per_day', 0))
    annual_min = min_bill_daily * 365

    total_bill = energy_after_credit + annual_fixed
    return total_bill


def stage2_compute_baseline_bills(rate_scenarios_df=None, n_buildings=None):
    """
    Compute baseline bills for all SCE buildings.

    Uses NATIVE (unscaled) hourly demand from parquets.
    RASS scaling factor stored for population extrapolation only.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: COMPUTE BASELINE BILLS")
    print("=" * 80)

    ta = build_time_arrays()
    period_masks = build_sce_period_masks(ta)

    # Load metadata
    sce_meta = load_sce_metadata()
    metadata = {}
    for _, row in sce_meta.iterrows():
        metadata[str(row['building_id'])] = {
            'puma': row['puma20'],
            'puma_str': row['puma20'],
            'income_category': normalize_income(row.get('income_category', 'medium')),
            'scaling_factor': row.get('scaling_factor', 1.0),
            'cec_cz': int(row['in.cec_climate_zone']) if pd.notna(row.get('in.cec_climate_zone')) else 9,
        }

    # Get parquet files
    baseline_dir = Path(BASELINE_DIR)
    if not baseline_dir.exists():
        print(f"\n  ERROR: {BASELINE_DIR} not found!")
        sys.exit(1)

    parquet_files = sorted(baseline_dir.glob('*-0.parquet'))
    print(f"  Parquet files found: {len(parquet_files)}")

    if n_buildings:
        parquet_files = parquet_files[:n_buildings]
        print(f"  TEST MODE: processing {n_buildings} buildings")

    # Load rate data
    from corrected_bill_calc import load_excel_data
    rates_df, baseline_df = load_excel_data(EXCEL_FILE)

    # Filter out buildings whose PUMA has no baseline allowance entry
    valid_pumas = set(baseline_df['puma'].unique())
    metadata = {k: v for k, v in metadata.items() if v['puma_str'] in valid_pumas}
    print(f"  Buildings with valid baseline PUMA: {len(metadata)}")

    print(f"  Actual SCE rates: {', '.join(ACTUAL_SCE_RATES.keys())}")

    results = []
    tou_consumption = {}
    monthly_consumption = {}
    start_time = time.time()
    errors = 0

    for i, pq_file in enumerate(parquet_files):
        building_id = pq_file.stem.split('-')[0]
        if building_id not in metadata:
            errors += 1
            continue

        try:
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)
            # NATIVE demand — no RASS scaling applied

            income = metadata[building_id]['income_category']
            is_care = (income == 'low')
            puma_str = metadata[building_id]['puma_str']
            sf = metadata[building_id]['scaling_factor']
            cz = metadata[building_id]['cec_cz']

            row = {
                'building_id': int(building_id),
                'puma': metadata[building_id]['puma'],
                'income': income,
                'is_care': is_care,
                'annual_kwh': hourly_load.sum(),
                'scaling_factor': sf,
                'cec_cz': cz,
            }

            # TOU consumption by period (native demand)
            bldg_tou = {}
            for period, mask in period_masks.items():
                bldg_tou[period] = hourly_load[mask].sum()
            tou_consumption[int(building_id)] = bldg_tou

            # Monthly consumption (for baseline credit calculation)
            bldg_monthly = []
            for m in range(12):
                s, e = ta['month_boundaries'][m], ta['month_boundaries'][m + 1]
                bldg_monthly.append(hourly_load[s:e].sum())
            monthly_consumption[int(building_id)] = bldg_monthly

            # Actual tariff bills
            for rate_code, col_prefix in ACTUAL_SCE_RATES.items():
                try:
                    bill = calculate_actual_sce_bill(
                        hourly_load, rate_code, puma_str,
                        income, is_care, rates_df, baseline_df, ta)
                    row[f'{col_prefix}_bill'] = bill
                except Exception as e:
                    row[f'{col_prefix}_bill'] = np.nan
                    if errors <= 3:
                        print(f"    Bill error ({rate_code}, bldg {building_id}): {e}")

            results.append(row)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {pq_file.name}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(parquet_files) - i - 1) / rate
            print(f"  {i+1}/{len(parquet_files)} | "
                  f"{elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    df_bills = pd.DataFrame(results)
    elapsed = time.time() - start_time
    print(f"\n  Completed: {len(results)} buildings in {elapsed:.1f}s")
    print(f"  Errors/skipped: {errors}")

    # --- Compute R_0 (sample-weighted TOU-D-4-9 revenue) ---
    # Restrict customer counts, R_gross_vol, and BL_total to buildings with
    # valid actual-tariff bills. Otherwise FC per customer is calibrated
    # against a larger sample than the one used for revenue evaluation,
    # leaving a fixed-charge shortfall proportional to Fixed_Pct_TD.
    V = df_bills['tou_d_4_9_bill'].values
    valid = ~np.isnan(V)
    df_valid = df_bills[valid].reset_index(drop=True)
    R_0 = np.nansum(V * BUILDING_WEIGHT)
    sample_n_care = int((df_valid['is_care'] == True).sum() * BUILDING_WEIGHT)
    sample_n_noncare = int((df_valid['is_care'] == False).sum() * BUILDING_WEIGHT)

    print(f"\n  R_sample (R_0) from TOU-D-4-9 bills:")
    print(f"    Valid bills: {valid.sum()}/{len(V)}")
    print(f"    Sample weighted baseline revenue (R_0): ${R_0/1e9:.4f}B")
    print(f"    Mean bill: ${np.nanmean(V):,.0f}/yr")

    # --- Compute R_gross_vol and BL_total ---
    from rate_designer_sce import BASELINE_TOU_RATES
    sce_wd = rates_df[
        (rates_df['rate_type'] == 'TOU-D-4-9') &
        (rates_df['weekday'] == 'weekday')
    ].iloc[0]
    baseline_care_discount = abs(safe_float(sce_wd.get('care_discount', 0)))
    baseline_credit_rate = safe_float(sce_wd.get('baseline_credit', 0))

    days_per_month = ta['days_per_month']
    month_boundaries = ta['month_boundaries']

    # Pre-compute per-building monthly consumption for baseline credit calc
    # We stored hourly loads during the main loop; re-read from parquets is
    # too expensive, so we compute monthly totals in the main loop above.
    # Instead, compute BL credits from the parquet re-read below.

    r_gross_vol = 0.0
    bl_total = 0.0
    for _, bldg_row in df_valid.iterrows():
        bid = bldg_row['building_id']
        if bid not in tou_consumption:
            continue
        bldg_tou = tou_consumption[bid]
        gross = sum(bldg_tou[p] * BASELINE_TOU_RATES[p] for p in TOU_PERIODS)
        care_factor = (1 - baseline_care_discount) if bldg_row['is_care'] else 1.0
        r_gross_vol += gross * care_factor

        # Baseline credit for this building (same logic as actual tariff)
        puma_str = bldg_row.get('puma', '')
        bl_entry = baseline_df[baseline_df['puma'] == puma_str]
        if not bl_entry.empty and bid in monthly_consumption:
            d_sum = bl_entry['summer_baseline_allowance'].values[0]
            d_win = bl_entry['winter_baseline_allowance'].values[0]
            bldg_bl = 0.0
            for m in range(12):
                mkwh = monthly_consumption[bid][m]
                bl_allow = (d_sum if 6 <= (m + 1) <= 10 else d_win) * days_per_month[m]
                bldg_bl += baseline_credit_rate * min(mkwh, bl_allow)
            # Apply CARE factor to baseline credit (same as actual tariff)
            bldg_bl *= care_factor
            bl_total += bldg_bl

    r_gross_vol *= BUILDING_WEIGHT
    bl_total *= BUILDING_WEIGHT
    print(f"    Gross volumetric revenue: ${r_gross_vol/1e9:.4f}B")
    print(f"    Aggregate baseline credits (BL_total): ${bl_total/1e9:.4f}B")

    # --- Generate rate scenarios if needed ---
    if rate_scenarios_df is None:
        from rate_designer_sce import generate_all_scenarios
        rate_scenarios_df = generate_all_scenarios(
            output_csv=RATE_SCENARIOS_OUT,
            r_sample=R_0,
            r_gross_vol=r_gross_vol,
            bl_total=bl_total,
            sample_n_care=sample_n_care,
            sample_n_noncare=sample_n_noncare,
        )

    # --- Designed scenario bills (blended rates) ---
    selected = rate_scenarios_df[
        rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)
    ]
    print(f"\n  Designed scenarios: {len(selected)} "
          f"({', '.join(selected['Scenario'].tolist())})")
    print(f"  CARE volumetric discount: {baseline_care_discount:.2%}")

    for _, scenario in selected.iterrows():
        sname = scenario['Scenario']
        # Blend weekday/weekend rates for designed scenarios
        blended = {}
        for p in TOU_PERIODS:
            wd_val = scenario.get(f'{p}_wd', scenario.get(p, 0))
            we_val = scenario.get(f'{p}_we', wd_val)
            blended[p] = wd_val * WEEKDAY_FRAC + we_val * WEEKEND_FRAC

        fc_care_annual = scenario['Fixed_CARE'] * 12
        fc_noncare_annual = scenario['Fixed_NonCARE'] * 12

        bills = []
        for _, bldg_row in df_bills.iterrows():
            bid = bldg_row['building_id']
            if bid not in tou_consumption:
                bills.append(np.nan)
                continue
            bldg_tou = tou_consumption[bid]
            vol_bill = sum(bldg_tou[p] * blended[p] for p in TOU_PERIODS)

            # Baseline credit (same structure as actual tariff)
            puma_str = bldg_row.get('puma', '')
            bl_entry = baseline_df[baseline_df['puma'] == puma_str]
            if not bl_entry.empty and bid in monthly_consumption:
                d_sum = bl_entry['summer_baseline_allowance'].values[0]
                d_win = bl_entry['winter_baseline_allowance'].values[0]
                for m in range(12):
                    mkwh = monthly_consumption[bid][m]
                    bl_allow = (d_sum if 6 <= (m + 1) <= 10 else d_win) * days_per_month[m]
                    vol_bill -= baseline_credit_rate * min(mkwh, bl_allow)

            if bldg_row['is_care'] and baseline_care_discount > 0:
                vol_bill *= (1 - baseline_care_discount)
            fixed = fc_care_annual if bldg_row['is_care'] else fc_noncare_annual
            bills.append(vol_bill + fixed)

        df_bills[f'{sname}_bill'] = bills
        print(f"    {sname}: mean bill=${np.nanmean(bills):,.0f}/yr")

    df_bills.to_csv(BASELINE_BILLS_OUT, index=False)
    print(f"\n  Saved to: {BASELINE_BILLS_OUT}")

    return df_bills, rate_scenarios_df
