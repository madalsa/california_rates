"""
sdge_baseline_bills.py — Stage 2: Compute baseline bills for all SDGE buildings

Reads hourly parquets from Baseline_SDGE/, computes bills under:
  1. Actual SDGE tariff rates (TOU-DR, TOU-DR-F) with baseline credits
  2. Designed rate scenarios (blended rates from rate_scenarios_sdge.csv)

Uses native demand — RASS scaling factor stored but NOT applied to load profiles.
Baseline credits ($0.11017/kWh) included in designed scenario bills.
SDGE has 6 TOU periods (with midpeak).
"""

import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sdge_config import (
    BASELINE_DIR, METADATA_FILE, PUMA_UTILITY_FILE, EXCEL_FILE,
    RATE_SCENARIOS_OUT, BASELINE_BILLS_OUT,
    ACTUAL_SDGE_RATES, DESIGNED_SCENARIOS,
    BUILDING_WEIGHT,
    build_time_arrays, build_sdge_period_masks, build_tou_rate_array,
    safe_float,
)


def load_sdge_metadata():
    """Load metadata and filter to SDGE buildings."""
    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    sdge_pumas = puma_util[puma_util['utility_acronym'] == 'SDGE']['PUMA'].tolist()
    sdge_meta = meta[meta['puma20'].isin(sdge_pumas)].copy()
    print(f"  SDGE buildings in metadata: {len(sdge_meta)}")
    return sdge_meta


def normalize_income(income_str):
    """Normalize income category to low/medium/high."""
    mapping = {'Low': 'low', 'Medium': 'medium', 'High': 'high',
               'low': 'low', 'medium': 'medium', 'high': 'high'}
    return mapping.get(str(income_str).strip(), 'medium')


def calculate_actual_sdge_bill_vectorized(hourly_load, rate_code, puma_str,
                                          income, is_care):
    """
    Vectorized bill calculation for actual SDGE tariff rates (TOU-DR, TOU-DR-F).

    Key insight: SDGE tier 1 and tier 2 volumetric rates are IDENTICAL.
    Tiering is implemented via a baseline_credit applied to within-baseline kWh.
    This allows full vectorization:
        bill = sum(load x TOU_rate)
               - baseline_credit x sum_over_months(min(monthly_kwh, monthly_baseline))
               + fixed_charges
               x (1 - care_discount if CARE)
    """
    # Load rate data (cached)
    from corrected_bill_calc import load_excel_data
    rates_df, baseline_df = load_excel_data(EXCEL_FILE)

    # Get rate structure
    rate_entries = rates_df[rates_df['rate_type'] == rate_code]
    weekday_rate = rate_entries[rate_entries['weekday'] == 'weekday'].iloc[0].to_dict()

    # Get baseline allowance for this PUMA (string format like 'G06005928')
    baseline_entry = baseline_df[baseline_df['puma'] == puma_str]
    if baseline_entry.empty:
        return np.nan
    daily_summer_baseline = baseline_entry['summer_baseline_allowance'].values[0]
    daily_winter_baseline = baseline_entry['winter_baseline_allowance'].values[0]

    # TOU rates (tier 1 = tier 2 for SDGE)
    tou_rates = {
        'summer_peak': safe_float(weekday_rate.get('peak_rate_summer1', 0)),
        'summer_midpeak': safe_float(weekday_rate.get('midpeak_rate_summer1', 0)),
        'summer_offpeak': safe_float(weekday_rate.get('offpeak_rate_summer1', 0)),
        'winter_peak': safe_float(weekday_rate.get('peak_rate_winter1', 0)),
        'winter_midpeak': safe_float(weekday_rate.get('midpeak_rate_winter1', 0)),
        'winter_offpeak': safe_float(weekday_rate.get('offpeak_rate_winter1', 0)),
    }

    baseline_credit = safe_float(weekday_rate.get('baseline_credit', 0))
    care_discount = abs(safe_float(weekday_rate.get('care_discount', 0)))

    # Build 8760 TOU rate array
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1  # 1-indexed
    hour_of_day = hours % 24

    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)
    is_midpeak = ((hour_of_day >= 6) & (hour_of_day < 16)) | \
                 ((hour_of_day >= 21) & (hour_of_day < 22))

    rate_array = np.where(
        is_summer,
        np.where(is_peak, tou_rates['summer_peak'],
                 np.where(is_midpeak, tou_rates['summer_midpeak'],
                          tou_rates['summer_offpeak'])),
        np.where(is_peak, tou_rates['winter_peak'],
                 np.where(is_midpeak, tou_rates['winter_midpeak'],
                          tou_rates['winter_offpeak']))
    )

    # Energy charges
    energy_charges = np.dot(hourly_load, rate_array)

    # Baseline credit: for each month, credit = baseline_credit x min(monthly_kwh, baseline)
    total_baseline_credit = 0.0
    for m in range(12):
        s, e = month_boundaries[m], month_boundaries[m + 1]
        monthly_kwh = hourly_load[s:e].sum()
        # Summer vs winter baseline
        if 6 <= (m + 1) <= 10:
            monthly_baseline = daily_summer_baseline * days_per_month[m]
        else:
            monthly_baseline = daily_winter_baseline * days_per_month[m]
        total_baseline_credit += baseline_credit * min(monthly_kwh, monthly_baseline)

    energy_after_credit = energy_charges - total_baseline_credit

    # CARE discount (applied to volumetric charges)
    if is_care and care_discount > 0:
        energy_after_credit *= (1 - care_discount)

    # Fixed charges
    fixed_charges = safe_float(weekday_rate.get('base_service_charge_per_day', 0))
    annual_base_fixed = fixed_charges * 365

    monthly_fixed = 0.0
    has_fixed = weekday_rate.get('Fixed', '') == 'Yes'
    if has_fixed:
        if income == 'low':
            monthly_fixed = safe_float(weekday_rate.get('fixedcharge_low', 0))
        elif income == 'medium':
            monthly_fixed = safe_float(weekday_rate.get('fixedcharge_med', 0))
        else:
            monthly_fixed = safe_float(weekday_rate.get('fixedcharge_high', 0))
    annual_fixed = annual_base_fixed + monthly_fixed * 12

    total_bill = energy_after_credit + annual_fixed
    return total_bill


def stage2_compute_baseline_bills(rate_scenarios_df=None, n_buildings=None):
    """
    Compute bills for all SDGE buildings under selected rate scenarios.

    Two types of billing:
    1. Actual SDGE tariff rates (TOU-DR, TOU-DR-F) via corrected_bill_calc.py
       - Handles tiering, baseline allowances, CARE discounts, income-graduated fixed
    2. Designed rate scenarios (F0_WF0_ROE0, etc.) via direct TOU bill computation
       - Computes R_0 from TOU-DR bills, generates rate scenarios, applies rates directly

    Reads each building's 15-min parquet from Baseline_SDGE/,
    aggregates to hourly, scales by RASS scaling factor.
    """
    print("\n" + "=" * 80)
    print("STAGE 2: COMPUTE BASELINE BILLS")
    print("=" * 80)

    # Load metadata
    sdge_meta = load_sdge_metadata()

    # Build metadata lookup
    metadata = {}
    for _, row in sdge_meta.iterrows():
        metadata[str(row['building_id'])] = {
            'puma': row['puma20'],
            'puma_str': row['puma20'],  # string PUMA like 'G06005928' for baseline lookup
            'income_category': normalize_income(row.get('income_category', 'medium')),
            'scaling_factor': row.get('scaling_factor', 1.0),
        }

    # Get parquet files
    baseline_dir = Path(BASELINE_DIR)
    if not baseline_dir.exists():
        print(f"\n  ERROR: {BASELINE_DIR} not found!")
        print("  Copy your Baseline_SDGE/ folder to this directory and re-run.")
        sys.exit(1)

    parquet_files = sorted(baseline_dir.glob('*-0.parquet'))
    print(f"  Parquet files found: {len(parquet_files)}")

    if n_buildings:
        parquet_files = parquet_files[:n_buildings]
        print(f"  TEST MODE: processing {n_buildings} buildings")

    # Filter out buildings whose PUMA has no baseline allowance entry
    from corrected_bill_calc import load_excel_data as _load_bl
    _, baseline_df_check = _load_bl(EXCEL_FILE)
    valid_pumas = set(baseline_df_check['puma'].unique())
    metadata = {k: v for k, v in metadata.items() if v['puma_str'] in valid_pumas}
    print(f"  Buildings with valid baseline PUMA: {len(metadata)}")

    print(f"  Actual SDGE rates: {', '.join(ACTUAL_SDGE_RATES.keys())}")

    # --- Pass 1: Compute actual tariff bills + TOU consumption per building ---
    results = []
    tou_consumption = {}  # building_id -> {period: kwh}
    start_time = time.time()
    errors = 0

    # TOU period classification arrays (precompute once)
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1  # 1-indexed
    hour_of_day = hours % 24
    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)
    is_midpeak = ((hour_of_day >= 6) & (hour_of_day < 16)) | \
                 ((hour_of_day >= 21) & (hour_of_day < 22))

    # Build period masks
    period_masks = {
        'summer_peak': is_summer & is_peak,
        'summer_midpeak': is_summer & is_midpeak,
        'summer_offpeak': is_summer & ~is_peak & ~is_midpeak,
        'winter_peak': ~is_summer & is_peak,
        'winter_midpeak': ~is_summer & is_midpeak,
        'winter_offpeak': ~is_summer & ~is_peak & ~is_midpeak,
    }

    for i, pq_file in enumerate(parquet_files):
        building_id = pq_file.stem.split('-')[0]

        if building_id not in metadata:
            errors += 1
            continue

        try:
            # Read 15-min data -> hourly
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)

            # Native demand — do NOT apply RASS scaling factor to load
            sf = metadata[building_id]['scaling_factor']

            income = metadata[building_id]['income_category']
            is_care = (income == 'low')
            puma_str = metadata[building_id]['puma_str']

            row = {
                'building_id': int(building_id),
                'puma': metadata[building_id]['puma'],
                'income': income,
                'is_care': is_care,
                'annual_kwh': hourly_load.sum(),
                'scaling_factor': sf,
            }

            # Store TOU consumption by period for direct bill computation
            bldg_tou = {}
            for period, mask in period_masks.items():
                bldg_tou[period] = hourly_load[mask].sum()
            tou_consumption[int(building_id)] = bldg_tou

            # --- Actual SDGE tariff rates (vectorized) ---
            for rate_code, col_prefix in ACTUAL_SDGE_RATES.items():
                try:
                    bill = calculate_actual_sdge_bill_vectorized(
                        hourly_load, rate_code, puma_str,
                        income, is_care
                    )
                    row[f'{col_prefix}_bill'] = bill
                except Exception as e:
                    row[f'{col_prefix}_bill'] = np.nan
                    if errors <= 3:
                        print(f"    Bill calc error ({rate_code}, bldg {building_id}): {e}")

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

    # --- Compute R_0 (sample-weighted TOU-DR revenue) ---
    # Restrict customer counts, R_gross_vol, and BL_total to buildings with
    # valid actual-tariff bills. Otherwise FC per customer is calibrated
    # against a larger sample than the one used for revenue evaluation,
    # leaving a fixed-charge shortfall proportional to Fixed_Pct_TD.
    V = df_bills['tou_dr_bill'].values
    valid = ~np.isnan(V)
    df_valid = df_bills[valid].reset_index(drop=True)
    R_0 = np.nansum(V * BUILDING_WEIGHT)
    sample_n_care = int((df_valid['is_care'] == True).sum() * BUILDING_WEIGHT)
    sample_n_noncare = int((df_valid['is_care'] == False).sum() * BUILDING_WEIGHT)

    print(f"\n  R_sample (R_0) from TOU-DR bills:")
    print(f"    Valid TOU-DR bills: {valid.sum()}/{len(V)}")
    print(f"    Sample weighted baseline revenue (R_0): ${R_0/1e9:.4f}B")
    print(f"    Mean TOU-DR bill: ${np.nanmean(V):,.0f}/yr")
    print(f"    Sample customers: {sample_n_care:,} CARE, {sample_n_noncare:,} non-CARE")

    # --- Compute R_gross_vol and BL_total ---
    # R_gross_vol: sum(load x TOU-DR_rate) x care_factor for all buildings,
    # WITHOUT baseline credits subtracted.
    # BL_total: aggregate baseline credits across all buildings (weighted).
    from rate_designer import BASELINE_TOU_RATES
    from corrected_bill_calc import load_excel_data as _load_xl
    _rates_df, _baseline_df = _load_xl(EXCEL_FILE)
    _tou_dr_entries = _rates_df[_rates_df['rate_type'] == 'TOU-DR']
    _tou_dr_wd = _tou_dr_entries[_tou_dr_entries['weekday'] == 'weekday'].iloc[0]
    baseline_care_discount = abs(float(_tou_dr_wd.get('care_discount', 0) or 0))
    baseline_credit_rate = abs(float(_tou_dr_wd.get('baseline_credit', 0) or 0))
    tou_periods = ['summer_peak', 'summer_midpeak', 'summer_offpeak',
                   'winter_peak', 'winter_midpeak', 'winter_offpeak']

    # We need monthly consumption per building for baseline credit computation.
    # Re-read parquets to get monthly totals (or compute from hourly_load stored above).
    # Store monthly consumption during Pass 1 — recompute here from TOU data and hourly.
    # Actually, we need hourly loads again for monthly sums. Store them in Pass 1.
    # For efficiency, let's do a second pass only for monthly baseline credit computation.

    # First compute gross vol and baseline credits per building.
    # Also store per-building baseline credit for designed scenario billing.
    print(f"\n  Computing BL_total and R_gross_vol (baseline credit rate: ${baseline_credit_rate:.5f}/kWh)...")
    r_gross_vol = 0.0
    bl_total_unweighted = 0.0  # sum of baseline credits across sample (before weighting)
    bldg_bl_credits = {}  # building_id -> baseline credit (before CARE discount)

    for _, bldg_row in df_valid.iterrows():
        bid = bldg_row['building_id']
        if bid not in tou_consumption:
            continue
        bldg_tou = tou_consumption[bid]

        # Gross volumetric (no baseline credit)
        gross = sum(bldg_tou[p] * BASELINE_TOU_RATES[p] for p in tou_periods)
        care_factor = (1 - baseline_care_discount) if bldg_row['is_care'] else 1.0
        r_gross_vol += gross * care_factor

        # Baseline credit: need monthly kWh for this building
        puma_str = bldg_row.get('puma', '')
        bl_entry = _baseline_df[_baseline_df['puma'] == puma_str]
        if bl_entry.empty:
            bldg_bl_credits[bid] = 0.0
            continue
        d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
        d_win = bl_entry.iloc[0]['winter_baseline_allowance']

        # Re-read hourly load for monthly breakdown
        pq_file = Path(BASELINE_DIR) / f"{bid}-0.parquet"
        if not pq_file.exists():
            bldg_bl_credits[bid] = 0.0
            continue
        df_pq = pd.read_parquet(pq_file)
        load_15min = df_pq['out.electricity.total.energy_consumption'].values
        hourly_load_bldg = load_15min.reshape(-1, 4).sum(axis=1)  # native demand

        bldg_bl_credit = 0.0
        for m in range(12):
            s, e = month_boundaries[m], month_boundaries[m + 1]
            monthly_kwh = hourly_load_bldg[s:e].sum()
            if 6 <= (m + 1) <= 10:
                monthly_baseline = d_sum * days_per_month[m]
            else:
                monthly_baseline = d_win * days_per_month[m]
            bldg_bl_credit += baseline_credit_rate * min(monthly_kwh, monthly_baseline)

        bldg_bl_credits[bid] = bldg_bl_credit  # before CARE discount
        # Apply CARE discount to baseline credit (same as actual tariff)
        bl_total_unweighted += bldg_bl_credit * care_factor

    r_gross_vol *= BUILDING_WEIGHT
    bl_total = bl_total_unweighted * BUILDING_WEIGHT
    print(f"    Gross volumetric revenue (with CARE, no BL credits): ${r_gross_vol/1e9:.4f}B")
    print(f"    BL_total (aggregate baseline credits): ${bl_total/1e9:.4f}B")
    print(f"    Baseline credit + fixed charge gap: ${(r_gross_vol - R_0)/1e9:.4f}B")

    # --- Generate rate scenarios using R_sample approach ---
    if rate_scenarios_df is None:
        from rate_designer import generate_all_scenarios
        rate_scenarios_df = generate_all_scenarios(
            output_csv=RATE_SCENARIOS_OUT,
            r_sample=R_0,
            r_gross_vol=r_gross_vol,
            bl_total=bl_total,
            sample_n_care=sample_n_care,
            sample_n_noncare=sample_n_noncare,
        )

    # Filter designed scenarios to our selection
    selected_designed = rate_scenarios_df[
        rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)
    ]
    print(f"\n  Designed rate scenarios: {len(selected_designed)} "
          f"({', '.join(selected_designed['Scenario'].tolist())})")

    # --- Direct bill computation for designed scenarios ---
    # bill = sum(consumption[period] x rate[period]) x care_factor + annual_fixed_charge

    print(f"\n  Computing bills directly from designed rates:")
    print(f"  CARE volumetric discount for designed scenarios: {baseline_care_discount:.2%}")

    for _, scenario in selected_designed.iterrows():
        scenario_name = scenario['Scenario']
        fixed_care_annual = scenario['Fixed_CARE'] * 12
        fixed_noncare_annual = scenario['Fixed_NonCARE'] * 12

        bills = []
        for _, bldg_row in df_bills.iterrows():
            bid = bldg_row['building_id']
            if bid not in tou_consumption:
                bills.append(np.nan)
                continue
            bldg_tou = tou_consumption[bid]
            vol_bill = sum(bldg_tou[p] * scenario[p] for p in tou_periods)
            # Subtract baseline credit (same as actual tariff)
            bl_credit = bldg_bl_credits.get(bid, 0.0)
            vol_bill -= bl_credit
            if bldg_row['is_care'] and baseline_care_discount > 0:
                vol_bill *= (1 - baseline_care_discount)
            fixed = fixed_care_annual if bldg_row['is_care'] else fixed_noncare_annual
            bills.append(vol_bill + fixed)

        df_bills[f'{scenario_name}_bill'] = bills
        mean_bill = np.nanmean(bills)
        print(f"    {scenario_name}: scaling={scenario['Scaling']:.4f}, "
              f"FC_nonCARE=${scenario['Fixed_NonCARE']:.2f}/mo, "
              f"mean bill=${mean_bill:,.0f}/yr")

    df_bills.to_csv(BASELINE_BILLS_OUT, index=False)
    print(f"\n  Saved to: {BASELINE_BILLS_OUT}")

    # Revenue check
    if len(results) > 0:
        print("\n  Revenue check (sample mean annual bill):")
        bill_cols = [c for c in df_bills.columns if c.endswith('_bill')]
        for col in bill_cols:
            mean_bill = df_bills[col].mean()
            print(f"    {col}: ${mean_bill:,.0f}/yr avg")

    return df_bills, rate_scenarios_df
