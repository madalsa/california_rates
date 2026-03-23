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

Run unattended (Mac):
  caffeinate -s python run_sdge_pipeline.py > pipeline.log 2>&1 &

Run unattended (Linux):
  nohup python run_sdge_pipeline.py > pipeline.log 2>&1 &
"""

import argparse
import time
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_DIR = './Baseline_SDGE'
UPGRADE11_DIR = './Upgrade11_SDGE'  # Full electrification profiles (upgrade=11)
METADATA_FILE = 'CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet'
PUMA_UTILITY_FILE = 'puma_utility_data.csv'
TOU_WEIGHTS_FILE = 'tou_weights_sdge.csv'
EEC_FILE = 'eec_hourly_2025_wide.csv'
EXCEL_FILE = 'retail_rates_data_SDGE.xlsx'

# Output files
RATE_SCENARIOS_OUT = 'rate_scenarios_sdge_fresh.csv'
BASELINE_BILLS_OUT = 'baseline_bills_sdge_fresh.csv'
TECH_ASSIGNMENTS_OUT = 'tech_assignments_sdge.csv'
POSTADOPT_BILLS_OUT = 'post_adoption_bills_sdge.csv'
SUMMARY_OUT = 'pipeline_summary_sdge.csv'

# ---------------------------------------------------------------------------
# Scenario selection
# ---------------------------------------------------------------------------

# Actual SDGE tariff rate codes (computed via corrected_bill_calc.py)
ACTUAL_SDGE_RATES = {
    'TOU-DR': 'tou_dr',       # Current TOU rate (no fixed charges)
    'TOU-DR-F': 'tou_dr_f',   # Current TOU rate with income-graduated fixed charges
}

# Designed rate scenarios (computed via vectorized TOU weights)
DESIGNED_SCENARIOS = [
    'F0_WF0_ROE0',       # Designed baseline (all volumetric)
    'F50_WF0_ROE0',      # 50% T&D costs to fixed charges
    'F100_WF0_ROE0',     # 100% T&D costs to fixed charges
    'F0_WF1_ROE0',       # Remove wildfire costs
    'F0_WF0_ROE1.0',     # 1% ROE reduction
    'F50_WF1_ROE1.0',    # Combined: 50% fixed + wildfire removal + 1% ROE cut
]

# SDGE TOU periods
SUMMER_MONTHS = set(range(6, 11))  # June–October (1-indexed)

# SDGE PUMA centroid for pvlib (San Diego area approximate)
SDGE_LATITUDE = 32.9
SDGE_LONGITUDE = -117.1
SDGE_ALTITUDE = 130  # meters

# Solar sizing parameters
DEFAULT_PV_SIZE_KW = 5.0          # Fallback if consumption data unavailable
PV_OFFSET_TARGET = 0.80           # Size PV to offset 80% of annual consumption
PV_MIN_SIZE_KW = 2.0              # Floor: smallest system worth installing
PV_MAX_SIZE_KW = 12.0             # Cap: typical residential roof limit
# San Diego ~1,700 kWh/kWp/yr (PVGIS TMY); computed from profile in stage 4
SDGE_ANNUAL_KWH_PER_KW = 1700.0  # Updated at runtime from actual pvlib profile

# Default battery parameters
BATTERY_CAPACITY_KWH = 13.5  # Tesla Powerwall equivalent
BATTERY_POWER_KW = 5.0       # max charge/discharge rate
BATTERY_EFFICIENCY = 0.90    # round-trip efficiency

# Default EV charging parameters
EV_MILES_PER_KWH = 3.0    # wall-to-wheels efficiency
EV_CHARGE_START_HOUR = 22  # 10 PM
EV_CHARGE_HOURS = 5        # 10 PM – 3 AM

# BEV daily VMT empirical CDF (from vehicletrends.us, BEV All)
# Columns: (DVMT miles, cumulative probability 0-1)
BEV_DVMT_CDF = np.array([
    [0,  0.00],
    [5,  0.03],
    [10, 0.12],
    [15, 0.25],
    [20, 0.42],
    [25, 0.58],
    [30, 0.70],
    [35, 0.80],
    [40, 0.87],
    [50, 0.95],
    [60, 0.98],
    [70, 1.00],
])

# Sample building weight and customer count
TOTAL_RESIDENTIAL_CUSTOMERS = 1_364_361  # EIA 861 bundled+unbundled
BUILDING_WEIGHT = 252.3  # uniform ResStock weight per sample building


# ---------------------------------------------------------------------------
# Stage 1: Generate fresh rate scenarios
# ---------------------------------------------------------------------------

def stage1_generate_rate_scenarios(r_sample=None, sample_n_care=None,
                                    sample_n_noncare=None):
    """
    Generate fresh revenue-neutral rate scenarios using rate_designer.

    Requires R_sample (weighted TOU-DR revenue from building sample).
    If not provided, will attempt to load from existing baseline bills.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: GENERATE FRESH RATE SCENARIOS")
    print("=" * 80)

    if r_sample is None:
        # Try loading from previous baseline bills
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
# Stage 2: Compute baseline bills for all SDGE buildings
# ---------------------------------------------------------------------------

def get_tou_period(hour_of_day, month):
    """Classify hour into SDGE TOU period."""
    is_summer = month in SUMMER_MONTHS
    if 16 <= hour_of_day < 21:
        period = 'peak'
    elif 6 <= hour_of_day < 16 or 21 <= hour_of_day < 22:
        period = 'midpeak'
    else:
        period = 'offpeak'
    season = 'summer' if is_summer else 'winter'
    return f'{season}_{period}'


def build_tou_rate_array(scenario):
    """Build 8760-length rate array from a scenario row."""
    hours = np.arange(8760)
    # Approximate month from hour index (non-leap year)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1  # 1-indexed

    hour_of_day = hours % 24
    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)
    is_midpeak = ((hour_of_day >= 6) & (hour_of_day < 16)) | \
                 ((hour_of_day >= 21) & (hour_of_day < 22))

    rates = np.where(
        is_summer,
        np.where(is_peak, scenario['summer_peak'],
                 np.where(is_midpeak, scenario['summer_midpeak'],
                          scenario['summer_offpeak'])),
        np.where(is_peak, scenario['winter_peak'],
                 np.where(is_midpeak, scenario['winter_midpeak'],
                          scenario['winter_offpeak']))
    )
    return rates


def build_tou_rate_array_from_dict(rate_dict):
    """Build 8760-length rate array from a dict with keys like 'summer_peak', etc."""
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1

    hour_of_day = hours % 24
    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)
    is_midpeak = ((hour_of_day >= 6) & (hour_of_day < 16)) | \
                 ((hour_of_day >= 21) & (hour_of_day < 22))

    rates = np.where(
        is_summer,
        np.where(is_peak, rate_dict['summer_peak'],
                 np.where(is_midpeak, rate_dict['summer_midpeak'],
                          rate_dict['summer_offpeak'])),
        np.where(is_peak, rate_dict['winter_peak'],
                 np.where(is_midpeak, rate_dict['winter_midpeak'],
                          rate_dict['winter_offpeak']))
    )
    return rates


def calculate_bill_vectorized(hourly_load, rate_array, fixed_annual):
    """Calculate bill from hourly load and rate array."""
    return np.dot(hourly_load, rate_array) + fixed_annual


def calculate_actual_sdge_bill_vectorized(hourly_load, rate_code, puma_str,
                                          income, is_care):
    """
    Vectorized bill calculation for actual SDGE tariff rates (TOU-DR, TOU-DR-F).

    Key insight: SDGE tier 1 and tier 2 volumetric rates are IDENTICAL.
    Tiering is implemented via a baseline_credit applied to within-baseline kWh.
    This allows full vectorization:
        bill = sum(load × TOU_rate)
               - baseline_credit × sum_over_months(min(monthly_kwh, monthly_baseline))
               + fixed_charges
               × (1 - care_discount if CARE)
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

    def _safe(val):
        """Convert NaN/None to 0."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)

    # TOU rates (tier 1 = tier 2 for SDGE)
    tou_rates = {
        'summer_peak': _safe(weekday_rate.get('peak_rate_summer1', 0)),
        'summer_midpeak': _safe(weekday_rate.get('midpeak_rate_summer1', 0)),
        'summer_offpeak': _safe(weekday_rate.get('offpeak_rate_summer1', 0)),
        'winter_peak': _safe(weekday_rate.get('peak_rate_winter1', 0)),
        'winter_midpeak': _safe(weekday_rate.get('midpeak_rate_winter1', 0)),
        'winter_offpeak': _safe(weekday_rate.get('offpeak_rate_winter1', 0)),
    }

    baseline_credit = _safe(weekday_rate.get('baseline_credit', 0))
    care_discount = abs(_safe(weekday_rate.get('care_discount', 0)))

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

    # Baseline credit: for each month, credit = baseline_credit × min(monthly_kwh, baseline)
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
    fixed_charges = _safe(weekday_rate.get('base_service_charge_per_day', 0))
    annual_base_fixed = fixed_charges * 365

    monthly_fixed = 0.0
    has_fixed = weekday_rate.get('Fixed', '') == 'Yes'
    if has_fixed:
        if income == 'low':
            monthly_fixed = _safe(weekday_rate.get('fixedcharge_low', 0))
        elif income == 'medium':
            monthly_fixed = _safe(weekday_rate.get('fixedcharge_med', 0))
        else:
            monthly_fixed = _safe(weekday_rate.get('fixedcharge_high', 0))
    annual_fixed = annual_base_fixed + monthly_fixed * 12

    total_bill = energy_after_credit + annual_fixed
    return total_bill


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

    print(f"  Actual SDGE rates: {', '.join(ACTUAL_SDGE_RATES.keys())}")

    # --- Pass 1: Compute actual tariff bills + TOU consumption per building ---
    results = []
    tou_consumption = {}  # building_id → {period: kwh}
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
            # Read 15-min data → hourly
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)

            # Scale by RASS factor
            sf = metadata[building_id]['scaling_factor']
            hourly_load_scaled = hourly_load * sf

            income = metadata[building_id]['income_category']
            is_care = (income == 'low')
            puma_str = metadata[building_id]['puma_str']

            row = {
                'building_id': int(building_id),
                'puma': metadata[building_id]['puma'],
                'income': income,
                'is_care': is_care,
                'annual_kwh': hourly_load_scaled.sum(),
                'scaling_factor': sf,
            }

            # Store TOU consumption by period for direct bill computation
            bldg_tou = {}
            for period, mask in period_masks.items():
                bldg_tou[period] = hourly_load_scaled[mask].sum()
            tou_consumption[int(building_id)] = bldg_tou

            # --- Actual SDGE tariff rates (vectorized) ---
            for rate_code, col_prefix in ACTUAL_SDGE_RATES.items():
                try:
                    bill = calculate_actual_sdge_bill_vectorized(
                        hourly_load_scaled, rate_code, puma_str,
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
    V = df_bills['tou_dr_bill'].values
    valid = ~np.isnan(V)
    R_0 = np.nansum(V * BUILDING_WEIGHT)
    sample_n_care = int((df_bills['is_care'] == True).sum() * BUILDING_WEIGHT)
    sample_n_noncare = int((df_bills['is_care'] == False).sum() * BUILDING_WEIGHT)

    print(f"\n  R_sample (R_0) from TOU-DR bills:")
    print(f"    Valid TOU-DR bills: {valid.sum()}/{len(V)}")
    print(f"    Sample weighted baseline revenue (R_0): ${R_0/1e9:.4f}B")
    print(f"    Mean TOU-DR bill: ${np.nanmean(V):,.0f}/yr")
    print(f"    Sample customers: {sample_n_care:,} CARE, {sample_n_noncare:,} non-CARE")

    # --- Compute R_gross_vol: gross volumetric revenue with CARE discount ---
    # This is sum(load × TOU-DR_rate) × care_factor for all buildings,
    # WITHOUT baseline credits subtracted. Needed for correct rate scaling:
    # designed scenario bills don't have baseline credits, so the scaling
    # denominator must be the gross volumetric revenue, not R_sample.
    from rate_designer import BASELINE_TOU_RATES
    from corrected_bill_calc import load_excel_data as _load_xl
    _rates_df, _ = _load_xl(EXCEL_FILE)
    _tou_dr_entries = _rates_df[_rates_df['rate_type'] == 'TOU-DR']
    _tou_dr_wd = _tou_dr_entries[_tou_dr_entries['weekday'] == 'weekday'].iloc[0]
    baseline_care_discount = abs(float(_tou_dr_wd.get('care_discount', 0) or 0))
    tou_periods = ['summer_peak', 'summer_midpeak', 'summer_offpeak',
                   'winter_peak', 'winter_midpeak', 'winter_offpeak']
    r_gross_vol = 0.0
    for _, bldg_row in df_bills.iterrows():
        bid = bldg_row['building_id']
        if bid not in tou_consumption:
            continue
        bldg_tou = tou_consumption[bid]
        gross = sum(bldg_tou[p] * BASELINE_TOU_RATES[p] for p in tou_periods)
        care_factor = (1 - baseline_care_discount) if bldg_row['is_care'] else 1.0
        r_gross_vol += gross * care_factor
    r_gross_vol *= BUILDING_WEIGHT
    print(f"    Gross volumetric revenue (with CARE, no BL credits): ${r_gross_vol/1e9:.4f}B")
    print(f"    Baseline credit + fixed charge gap: ${(r_gross_vol - R_0)/1e9:.4f}B")

    # --- Generate rate scenarios using R_sample approach ---
    if rate_scenarios_df is None:
        from rate_designer import generate_all_scenarios
        rate_scenarios_df = generate_all_scenarios(
            output_csv=RATE_SCENARIOS_OUT,
            r_sample=R_0,
            r_gross_vol=r_gross_vol,
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
    # bill = sum(consumption[period] × rate[period]) × care_factor + annual_fixed_charge

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


# ---------------------------------------------------------------------------
# Stage 3: Technology adoption assignments
# ---------------------------------------------------------------------------

def stage3_tech_assignments(bills_df):
    """
    Assign PV, battery, EV, heat pump adoption to SDGE buildings.

    Uses the assign_technologies module with survey-derived probabilities.
    Falls back to simplified assignment if survey data is unavailable.
    """
    print("\n" + "=" * 80)
    print("STAGE 3: TECHNOLOGY ADOPTION ASSIGNMENTS")
    print("=" * 80)

    sdge_building_ids = set(bills_df['building_id'].astype(str))

    # Load full metadata
    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    sdge_pumas = puma_util[puma_util['utility_acronym'] == 'SDGE']['PUMA'].tolist()
    sdge_meta = meta[meta['puma20'].isin(sdge_pumas)].copy()

    # Filter to buildings we have bills for
    sdge_meta = sdge_meta[sdge_meta['building_id'].astype(str).isin(sdge_building_ids)].copy()
    print(f"  Buildings with bills: {len(sdge_meta)}")

    # Try to use the full survey-based assignment
    survey_path = 'survey_responses.csv'
    try:
        from assign_technologies import (
            prepare_survey, compute_survey_adoption_rates,
            score_buildings, assign_technology,
            map_income_bracket, income_to_bin, resstock_home_type, resstock_tenure
        )

        # Prepare ResStock covariates
        sdge_meta['income_numeric'] = sdge_meta['in.income'].apply(map_income_bracket)
        sdge_meta['inc_bin'] = sdge_meta['income_numeric'].apply(income_to_bin)
        sdge_meta['home_type_bin'] = sdge_meta['in.geometry_building_type_acs'].apply(resstock_home_type)
        sdge_meta['own_bin'] = sdge_meta['in.tenure'].apply(resstock_tenure)
        sdge_meta['cz'] = sdge_meta['in.cec_climate_zone']

        # Get weights
        if 'weight' in sdge_meta.columns:
            weights = np.array(sdge_meta['weight'], dtype=float)
        elif 'in.units_represented' in sdge_meta.columns:
            weights = np.array(sdge_meta['in.units_represented'], dtype=float)
        else:
            weights = np.ones(len(sdge_meta))

        if os.path.exists(survey_path):
            print("  Using survey-based adoption probabilities")
            sv = prepare_survey(survey_path)

            # PV assignment
            pv_groupby = ['inc_bin', 'home_type_bin', 'cz']
            pv_rates = compute_survey_adoption_rates(sv, 'PV', pv_groupby)
            pv_rates_coarse = compute_survey_adoption_rates(sv, 'PV', ['inc_bin', 'home_type_bin'])
            pv_scores_fine = score_buildings(sdge_meta, pv_rates, pv_groupby)
            pv_scores_coarse = score_buildings(sdge_meta, pv_rates_coarse, ['inc_bin', 'home_type_bin'])
            pv_scores = np.where(np.isnan(pv_scores_fine) | (pv_scores_fine == 0),
                                 pv_scores_coarse, pv_scores_fine)
            pv_scores = np.sqrt(np.maximum(pv_scores, 0))
            renter_mask = sdge_meta['own_bin'] == 0
            pv_scores[renter_mask] *= 0.15
            large_mf = sdge_meta['in.geometry_building_type_acs'].isin(
                ['50 or more Unit', '20 to 49 Unit'])
            pv_scores[large_mf.values] *= 0.05

            sdge_meta['assigned_pv'] = assign_technology(pv_scores, weights, 0.17, seed=42)

            # EV assignment
            ev_groupby = ['inc_bin', 'home_type_bin', 'cz']
            ev_rates = compute_survey_adoption_rates(sv, 'EV', ev_groupby)
            ev_rates_coarse = compute_survey_adoption_rates(sv, 'EV', ['inc_bin', 'home_type_bin'])
            ev_scores_fine = score_buildings(sdge_meta, ev_rates, ev_groupby)
            ev_scores_coarse = score_buildings(sdge_meta, ev_rates_coarse, ['inc_bin', 'home_type_bin'])
            ev_scores = np.where(np.isnan(ev_scores_fine) | (ev_scores_fine == 0),
                                 ev_scores_coarse, ev_scores_fine)
            ev_scores = np.sqrt(np.maximum(ev_scores, 0))
            ev_scores[renter_mask] *= 0.5

            sdge_meta['assigned_ev'] = assign_technology(ev_scores, weights, 0.12, seed=43)
        else:
            print("  Survey data not found — using simplified adoption")
            _assign_simplified(sdge_meta, weights)

        # Battery = all PV homes
        sdge_meta['assigned_battery'] = sdge_meta['assigned_pv'].copy()

        # Heat pump = keep ResStock
        sdge_meta['assigned_hp'] = sdge_meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)

    except Exception as e:
        print(f"  Survey-based assignment failed: {e}")
        print("  Falling back to simplified assignment")
        weights = np.ones(len(sdge_meta))
        _assign_simplified(sdge_meta, weights)

    # Summary
    if 'weight' in sdge_meta.columns:
        w = np.array(sdge_meta['weight'], dtype=float)
    else:
        w = np.ones(len(sdge_meta))

    print(f"\n  Adoption rates (weighted):")
    for tech in ['assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']:
        if tech in sdge_meta.columns:
            rate = np.average(sdge_meta[tech], weights=w)
            count = sdge_meta[tech].sum()
            print(f"    {tech}: {rate*100:.1f}% ({count} buildings)")

    # Save assignments
    out_cols = ['building_id', 'puma20', 'income_category',
                'assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']
    out_cols = [c for c in out_cols if c in sdge_meta.columns]
    sdge_meta[out_cols].to_csv(TECH_ASSIGNMENTS_OUT, index=False)
    print(f"  Saved to: {TECH_ASSIGNMENTS_OUT}")

    return sdge_meta


def _assign_simplified(meta, weights):
    """Simplified assignment without survey data."""
    rng = np.random.RandomState(42)
    n = len(meta)

    # PV: ~17%, biased toward SF owners
    pv_scores = np.ones(n) * 0.17
    if 'in.geometry_building_type_acs' in meta.columns:
        sf_mask = meta['in.geometry_building_type_acs'].isin(
            ['Single-Family Detached', 'Single-Family Attached', 'Mobile Home'])
        pv_scores[~sf_mask.values] *= 0.1
    if 'in.tenure' in meta.columns:
        renter = meta['in.tenure'] == 'Renter'
        pv_scores[renter.values] *= 0.15

    pv_probs = pv_scores / np.average(pv_scores, weights=weights) * 0.17
    pv_probs = np.clip(pv_probs, 0, 1)
    meta['assigned_pv'] = (rng.random(n) < pv_probs).astype(int)

    # Battery = all PV
    meta['assigned_battery'] = meta['assigned_pv'].copy()

    # EV: ~12%
    ev_scores = np.ones(n) * 0.12
    if 'income_category' in meta.columns:
        high_inc = meta['income_category'] == 'High'
        ev_scores[high_inc.values] *= 2.0
    ev_probs = ev_scores / np.average(ev_scores, weights=weights) * 0.12
    ev_probs = np.clip(ev_probs, 0, 1)
    meta['assigned_ev'] = (rng.random(n) < ev_probs).astype(int)

    # HP = ResStock
    if 'in.hvac_heating_type' in meta.columns:
        meta['assigned_hp'] = meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)
    else:
        meta['assigned_hp'] = 0


# ---------------------------------------------------------------------------
# Stage 4: Generate solar profiles with pvlib
# ---------------------------------------------------------------------------

def size_pv_system(annual_kwh, annual_kwh_per_kw):
    """Size a PV system to offset PV_OFFSET_TARGET of annual consumption.

    Parameters
    ----------
    annual_kwh : float
        Building annual electricity consumption (kWh).
    annual_kwh_per_kw : float
        Annual generation per kW of installed PV (kWh/kW), from the
        pvlib profile or SDGE_ANNUAL_KWH_PER_KW default.

    Returns
    -------
    float
        PV system size in kW DC, clamped to [PV_MIN_SIZE_KW, PV_MAX_SIZE_KW].
    """
    if annual_kwh_per_kw <= 0:
        return DEFAULT_PV_SIZE_KW
    ideal_kw = (annual_kwh * PV_OFFSET_TARGET) / annual_kwh_per_kw
    return float(np.clip(ideal_kw, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW))


def stage4_solar_profiles(tech_df, bills_df):
    """
    Generate a *per-kW* 8760 hourly solar generation profile using pvlib.

    Returns
    -------
    solar_per_kw : np.ndarray, shape (8760,)
        Hourly generation in kWh per 1 kW DC installed.
    annual_kwh_per_kw : float
        Sum of solar_per_kw — used by stage 6 to size each building's PV.
    """
    global SDGE_ANNUAL_KWH_PER_KW

    print("\n" + "=" * 80)
    print("STAGE 4: GENERATE SOLAR PROFILES (pvlib)")
    print("=" * 80)

    pv_buildings = tech_df[tech_df['assigned_pv'] == 1]['building_id'].values
    print(f"  PV-adopted buildings: {len(pv_buildings)}")

    if len(pv_buildings) == 0:
        print("  No PV buildings — skipping")
        return np.zeros(8760), SDGE_ANNUAL_KWH_PER_KW

    try:
        import pvlib
        from pvlib.pvsystem import PVSystem
        from pvlib.location import Location
        from pvlib.modelchain import ModelChain
        from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    except ImportError:
        print("  pvlib not installed — using synthetic solar profile")
        profile = _synthetic_solar_profile()
        per_kw = profile / DEFAULT_PV_SIZE_KW
        annual = per_kw.sum()
        SDGE_ANNUAL_KWH_PER_KW = annual
        return per_kw, annual

    print(f"  Location: lat={SDGE_LATITUDE}, lon={SDGE_LONGITUDE}")
    print(f"  Solar sizing: {PV_OFFSET_TARGET*100:.0f}% offset target, "
          f"{PV_MIN_SIZE_KW}-{PV_MAX_SIZE_KW} kW range")

    # Create location and get solar data
    location = Location(SDGE_LATITUDE, SDGE_LONGITUDE, 'US/Pacific',
                        SDGE_ALTITUDE, 'San Diego')

    try:
        tmy_result = pvlib.iotools.get_pvgis_tmy(
            SDGE_LATITUDE, SDGE_LONGITUDE, map_variables=True)
        tmy_data, tmy_meta = tmy_result[0], tmy_result[1]
        print("  Retrieved TMY data from PVGIS")
    except Exception as e:
        print(f"  Could not fetch TMY data: {e}")
        print("  Using synthetic solar profile instead")
        profile = _synthetic_solar_profile()
        per_kw = profile / DEFAULT_PV_SIZE_KW
        annual = per_kw.sum()
        SDGE_ANNUAL_KWH_PER_KW = annual
        return per_kw, annual

    # Set up PV system
    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = cec_modules.iloc[:, 0]
    inverter = cec_inverters.iloc[:, 0]

    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    system = PVSystem(
        surface_tilt=SDGE_LATITUDE,
        surface_azimuth=180,
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temp_params,
    )

    mc = ModelChain(system, location, dc_model='cec', aoi_model='physical',
                    spectral_model='no_loss')

    # Prepare weather data (ensure 8760 hours)
    weather = tmy_data.copy()
    if len(weather) != 8760:
        weather = weather.resample('h').mean()
    weather = weather.iloc[:8760]

    mc.run_model(weather)

    ac_power = mc.results.ac.values
    ac_power = np.nan_to_num(ac_power, nan=0.0)
    ac_power = np.maximum(ac_power, 0)

    # Normalize: module STC rating → per kW
    # CEC modules use I_mp_ref/V_mp_ref; Sandia uses Impo/Vmpo
    try:
        stc_rating = module['I_mp_ref'] * module['V_mp_ref']
    except KeyError:
        stc_rating = module.get('Impo', 0) * module.get('Vmpo', 0)
    if stc_rating > 0:
        solar_per_kw = ac_power / stc_rating
    else:
        solar_per_kw = ac_power / 1000.0

    annual_kwh_per_kw = solar_per_kw.sum()
    SDGE_ANNUAL_KWH_PER_KW = annual_kwh_per_kw

    capacity_factor = annual_kwh_per_kw / 8760 * 100
    print(f"  Per-kW annual generation: {annual_kwh_per_kw:,.0f} kWh/kW "
          f"({capacity_factor:.1f}% CF)")
    print(f"  Example sizes at {PV_OFFSET_TARGET*100:.0f}% offset: "
          f"5000 kWh/yr → {size_pv_system(5000, annual_kwh_per_kw):.1f} kW, "
          f"10000 kWh/yr → {size_pv_system(10000, annual_kwh_per_kw):.1f} kW, "
          f"20000 kWh/yr → {size_pv_system(20000, annual_kwh_per_kw):.1f} kW")

    return solar_per_kw, annual_kwh_per_kw


def _synthetic_solar_profile():
    """Generate a synthetic solar profile for San Diego (per 1 kW DC).

    Returns total-system kWh array for backward compatibility when called
    directly; stage4 divides by DEFAULT_PV_SIZE_KW to get per-kW.
    """
    print("  Generating synthetic solar profile for San Diego")
    hours = np.arange(8760)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    declination = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
    hour_angle = (hour_of_day - 12) * 15

    lat_rad = np.radians(SDGE_LATITUDE)
    decl_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    sin_alt = (np.sin(lat_rad) * np.sin(decl_rad) +
               np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    sin_alt = np.maximum(sin_alt, 0)

    ghi = 1000 * sin_alt ** 1.2

    # Per-kW: 1 kW nameplate → panel_area = 1/0.18 m²
    panel_area_per_kw = 1.0 / 0.18
    hourly_gen = ghi * panel_area_per_kw * 0.18 * 0.85 / 1000  # kWh per kW

    # Scale to DEFAULT_PV_SIZE_KW for backward compatibility
    hourly_gen_total = hourly_gen * DEFAULT_PV_SIZE_KW

    annual = hourly_gen_total.sum()
    cf = annual / (DEFAULT_PV_SIZE_KW * 8760) * 100
    print(f"  Synthetic annual generation: {annual:,.0f} kWh ({cf:.1f}% CF)")

    return hourly_gen_total


# ---------------------------------------------------------------------------
# Stage 5: Battery LP dispatch
# ---------------------------------------------------------------------------

def stage5_battery_dispatch(hourly_load, solar_gen, rate_array, eec_rates=None):
    """
    Optimize battery dispatch to minimize net electricity cost via LP.

    Uses scipy.optimize.linprog with sparse matrices for fast setup and solve.

    Objective: minimize (import cost - export credit).
    The battery charges when rates are low (or from excess solar)
    and discharges when rates are high.

    Parameters
    ----------
    hourly_load : np.array (8760,)
        Hourly electricity consumption (kWh)
    solar_gen : np.array (8760,)
        Hourly solar generation (kWh)
    rate_array : np.array (8760,)
        Hourly electricity import rate ($/kWh)
    eec_rates : np.array (8760,) or None
        Hourly export compensation rate ($/kWh). If None, export credit = 0.

    Returns
    -------
    dict with optimized load profile, battery actions, and bill
    """
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    if eec_rates is None:
        eec_rates = np.zeros(8760)

    T = 8760
    eta = np.sqrt(BATTERY_EFFICIENCY)  # one-way efficiency
    cap = BATTERY_CAPACITY_KWH
    pmax = BATTERY_POWER_KW

    # Net load after solar
    net_load = hourly_load - solar_gen

    # Decision variables layout (5T total):
    #   g[0:T]       grid_import  >= 0
    #   e[T:2T]      grid_export  >= 0
    #   c[2T:3T]     charge       [0, pmax]
    #   d[3T:4T]     discharge    [0, pmax]
    #   s[4T:5T]     SOC          [0, cap]
    n = 5 * T

    # Objective: min sum(rate*g - eec*e)
    c_obj = np.zeros(n)
    c_obj[0:T] = rate_array        # import cost
    c_obj[T:2*T] = -eec_rates      # export credit (negative = benefit)

    # Bounds
    bounds = np.zeros((n, 2))
    bounds[0:T, 1] = np.inf        # g >= 0, no upper bound
    bounds[T:2*T, 1] = np.inf      # e >= 0, no upper bound
    bounds[2*T:3*T, 1] = pmax      # 0 <= c <= pmax
    bounds[3*T:4*T, 1] = pmax      # 0 <= d <= pmax
    bounds[4*T:5*T, 1] = cap       # 0 <= s <= cap

    # Equality constraints (2T equations):
    # Row 0..T-1: energy balance
    #   g[t] - e[t] - c[t] + d[t]*eta = net_load[t]
    # Row T..2T-1: SOC dynamics
    #   s[t] - s[t-1] - c[t]*eta + d[t] = 0  (for t>0)
    #   s[0] - c[0]*eta + d[0] = cap*0.5      (for t=0)

    # Build sparse A_eq in COO format
    rows = []
    cols = []
    vals = []

    tt = np.arange(T)

    # Energy balance: g[t] coeff = 1
    rows.append(tt); cols.append(tt); vals.append(np.ones(T))
    # Energy balance: e[t] coeff = -1
    rows.append(tt); cols.append(T + tt); vals.append(-np.ones(T))
    # Energy balance: c[t] coeff = -1
    rows.append(tt); cols.append(2*T + tt); vals.append(-np.ones(T))
    # Energy balance: d[t] coeff = eta
    rows.append(tt); cols.append(3*T + tt); vals.append(np.full(T, eta))

    # SOC dynamics row indices: T + tt
    soc_rows = T + tt
    # s[t] coeff = 1
    rows.append(soc_rows); cols.append(4*T + tt); vals.append(np.ones(T))
    # s[t-1] coeff = -1 (for t > 0)
    rows.append(soc_rows[1:]); cols.append(4*T + tt[:-1]); vals.append(-np.ones(T-1))
    # c[t] coeff = -eta
    rows.append(soc_rows); cols.append(2*T + tt); vals.append(np.full(T, -eta))
    # d[t] coeff = 1
    rows.append(soc_rows); cols.append(3*T + tt); vals.append(np.ones(T))

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    val_arr = np.concatenate(vals)

    A_eq = csc_matrix((val_arr, (row_idx, col_idx)), shape=(2*T, n))

    b_eq = np.zeros(2*T)
    b_eq[0:T] = net_load               # energy balance RHS
    b_eq[T] = cap * 0.5                # SOC(0) = cap*0.5 + c[0]*eta - d[0]
    # b_eq[T+1:2T] = 0                 # already zeros

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq,
                     bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                     method='highs', options={'time_limit': 300.0})

    if not result.success:
        return None

    x = result.x
    grid_import_arr = x[0:T]
    grid_export_arr = x[T:2*T]
    charge_arr = x[2*T:3*T]
    discharge_arr = x[3*T:4*T]
    soc_arr = x[4*T:5*T]

    import_cost = np.dot(grid_import_arr, rate_array)
    export_credit = np.dot(grid_export_arr, eec_rates)

    return {
        'grid_import': grid_import_arr,
        'grid_export': grid_export_arr,
        'charge': charge_arr,
        'discharge': discharge_arr,
        'soc': soc_arr,
        'bill_energy': import_cost,
        'export_credit': export_credit,
        'net_cost': import_cost - export_credit,
    }


def stage5_battery_dispatch_heuristic(hourly_load, solar_gen, rate_array, eec_rates=None):
    """
    Fast heuristic battery dispatch (no LP solver needed).

    Strategy: charge during cheapest net-rate hours, discharge during most expensive.
    Also captures excess solar.
    """
    if eec_rates is None:
        eec_rates = np.zeros(8760)

    T = 8760
    eta = np.sqrt(BATTERY_EFFICIENCY)
    cap = BATTERY_CAPACITY_KWH
    pmax = BATTERY_POWER_KW

    net_load = hourly_load - solar_gen

    # Use net rate (import rate - export rate) for dispatch decisions
    net_rate = rate_array - eec_rates

    grid_import = np.maximum(net_load, 0).copy()
    grid_export = np.maximum(-net_load, 0).copy()
    soc = np.zeros(T)
    charge_arr = np.zeros(T)
    discharge_arr = np.zeros(T)

    # Simple forward pass
    current_soc = cap * 0.5
    sorted_net_rate = np.sort(net_rate)

    for t in range(T):
        # If excess solar, charge battery (reduce export)
        if net_load[t] < 0:
            excess = -net_load[t]
            can_charge = min(excess, pmax, (cap - current_soc) / eta)
            charge_arr[t] = can_charge
            current_soc += can_charge * eta
            grid_import[t] = 0
            grid_export[t] = max(excess - can_charge, 0)
        else:
            # If rate is high and battery has charge, discharge
            rate_pctile = np.searchsorted(sorted_net_rate, net_rate[t]) / T
            if rate_pctile > 0.6 and current_soc > 0:
                can_discharge = min(pmax, current_soc, net_load[t])
                discharge_arr[t] = can_discharge
                current_soc -= can_discharge
                grid_import[t] = max(net_load[t] - can_discharge * eta, 0)
            elif rate_pctile < 0.3 and current_soc < cap:
                # Charge during cheap hours
                can_charge = min(pmax, (cap - current_soc) / eta)
                charge_arr[t] = can_charge
                current_soc += can_charge * eta
                grid_import[t] = net_load[t] + can_charge

        soc[t] = current_soc

    import_cost = np.dot(grid_import, rate_array)
    export_credit = np.dot(grid_export, eec_rates)

    return {
        'grid_import': grid_import,
        'grid_export': grid_export,
        'charge': charge_arr,
        'discharge': discharge_arr,
        'soc': soc,
        'bill_energy': import_cost,
        'export_credit': export_credit,
        'net_cost': import_cost - export_credit,
    }


# ---------------------------------------------------------------------------
# Stage 6: Post-adoption bills
# ---------------------------------------------------------------------------

def sample_ev_dvmt(n, seed=44):
    """Sample daily VMT for *n* EV buildings from the empirical BEV CDF.

    Uses inverse-CDF (linear interpolation) so each building gets a
    persistent daily driving distance drawn from the vehicletrends.us
    BEV-All distribution.

    Returns
    -------
    np.ndarray, shape (n,)
        Daily VMT for each building (miles).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    # Inverse CDF: given uniform quantile u, interpolate to DVMT
    dvmt = np.interp(u, BEV_DVMT_CDF[:, 1], BEV_DVMT_CDF[:, 0])
    return dvmt


def make_ev_profile(daily_miles=None):
    """Generate a Level 2 EV charging profile (8760 hours).

    Parameters
    ----------
    daily_miles : float or None
        Daily VMT for this building. If None, uses the distribution mean
        (~25 mi/day).  Converted to daily kWh via EV_MILES_PER_KWH.
    """
    if daily_miles is None:
        # Distribution mean (trapezoidal integration of the CDF)
        daily_miles = np.trapz(BEV_DVMT_CDF[:, 0],
                               BEV_DVMT_CDF[:, 1])  # ≈25 mi
    daily_kwh = daily_miles / EV_MILES_PER_KWH
    hourly_charge_rate = daily_kwh / EV_CHARGE_HOURS  # kW

    profile = np.zeros(8760)
    for day in range(365):
        for h in range(EV_CHARGE_HOURS):
            hour = day * 24 + (EV_CHARGE_START_HOUR + h) % 24
            if hour < 8760:
                profile[hour] = hourly_charge_rate

    return profile


def stage6_post_adoption_bills(bills_df, tech_df, solar_profile, rate_scenarios_df,
                               use_lp=True, annual_kwh_per_kw=None,
                               skip_s3=False):
    """
    Compute post-adoption bills under 4 technology adoption scenarios.

    Parameters
    ----------
    solar_profile : np.ndarray, shape (8760,)
        Per-kW hourly generation (kWh per kW DC installed).
    annual_kwh_per_kw : float or None
        Annual kWh per kW DC (for sizing). Falls back to SDGE_ANNUAL_KWH_PER_KW.

    For each tech-adopted building, we compute bills under 4 adoption scenarios:
      S1 (ev_only):     baseline load + EV charging, no PV/battery
      S2 (pv_storage):  baseline load + PV + battery (no EV, no electrification)
      S3 (full_pre_pv): Upgrade11 load + EV + PV (sized on baseline+EV) + battery
      S4 (full_post_pv): Upgrade11 load + EV + PV (sized on upgrade11+EV) + battery

    Upgrade 11 = ENERGY STAR ASHP + Light Touch Envelope + Full Appliance
    Electrification (HPWH, induction cooktop, electric dryer).
    """
    print("\n" + "=" * 80)
    print("STAGE 6: POST-ADOPTION BILLS")
    print("=" * 80)

    if skip_s3:
        print("  Skipping S3 (PV+storage+EV) — use --skip-s3 to re-enable")

    # Merge tech assignments with bills
    tech_cols = ['building_id', 'assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']
    tech_cols = [c for c in tech_cols if c in tech_df.columns]
    merged = bills_df.merge(tech_df[tech_cols], on='building_id', how='left')
    for col in ['assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)

    # Identify buildings that need post-adoption billing
    has_tech = merged[(merged['assigned_pv'] == 1) |
                      (merged['assigned_ev'] == 1)].copy()
    print(f"  Buildings with tech adoption: {len(has_tech)}")

    if len(has_tech) == 0:
        print("  No tech-adopted buildings — skipping")
        merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
        return merged

    # Load EEC rates for net billing
    try:
        eec_df = pd.read_csv(EEC_FILE, parse_dates=['datetime'])
        eec_rates = eec_df['sdge_total'].values[:8760]
        print(f"  Loaded EEC rates from {EEC_FILE}")
    except Exception as e:
        print(f"  Could not load EEC data: {e}")
        print("  Using 0 export credit (conservative)")
        eec_rates = np.zeros(8760)

    # Sample per-building daily VMT from BEV empirical CDF
    ev_buildings = has_tech[has_tech['assigned_ev'] == 1]
    ev_dvmt_map = {}
    if len(ev_buildings) > 0:
        dvmt_samples = sample_ev_dvmt(len(ev_buildings))
        for i, (_, ev_row) in enumerate(ev_buildings.iterrows()):
            ev_dvmt_map[int(ev_row['building_id'])] = dvmt_samples[i]
        print(f"  EV daily VMT distribution (sampled from BEV CDF):")
        print(f"    Mean: {dvmt_samples.mean():.1f} mi | "
              f"Median: {np.median(dvmt_samples):.1f} mi | "
              f"Range: {dvmt_samples.min():.1f}–{dvmt_samples.max():.1f} mi")
        print(f"    → Mean daily charging: "
              f"{dvmt_samples.mean() / EV_MILES_PER_KWH:.1f} kWh/day")

    # Resolve per-kW annual generation for PV sizing
    if annual_kwh_per_kw is None:
        annual_kwh_per_kw = SDGE_ANNUAL_KWH_PER_KW

    # For each tech-adopted building, re-read hourly load and compute new bill
    baseline_dir = Path(BASELINE_DIR)
    results_update = {}
    start_time = time.time()
    processed = 0
    lp_failures = 0
    pv_sizes = []  # Track PV sizes for summary

    # Filter to selected designed scenarios only
    selected_designed = rate_scenarios_df[
        rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)
    ]

    # Build rate arrays and fixed charges for each designed scenario
    designed_rate_arrays = {}
    scenario_fixed_charges = {}
    for _, scenario in selected_designed.iterrows():
        sname = scenario['Scenario']
        rate_dict = {
            'summer_peak': scenario['summer_peak'],
            'summer_midpeak': scenario['summer_midpeak'],
            'summer_offpeak': scenario['summer_offpeak'],
            'winter_peak': scenario['winter_peak'],
            'winter_midpeak': scenario['winter_midpeak'],
            'winter_offpeak': scenario['winter_offpeak'],
        }
        designed_rate_arrays[sname] = build_tou_rate_array_from_dict(rate_dict)
        fc_care = scenario['Fixed_CARE'] * 12
        fc_noncare = scenario['Fixed_NonCARE'] * 12
        scenario_fixed_charges[sname] = {'care': fc_care, 'noncare': fc_noncare}
    print(f"  Built rate arrays for {len(designed_rate_arrays)} designed scenarios")

    # Build actual TOU-DR and TOU-DR-F rate arrays for import cost calculation
    from corrected_bill_calc import load_excel_data
    rates_df_xl, baseline_df_xl = load_excel_data(EXCEL_FILE)

    def _safe(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)

    def _load_actual_rate(rate_code):
        """Extract rate dict, baseline credit, CARE discount, and fixed charge info."""
        rate_entries = rates_df_xl[rates_df_xl['rate_type'] == rate_code]
        wd = rate_entries[rate_entries['weekday'] == 'weekday'].iloc[0].to_dict()
        rd = {
            'summer_peak': _safe(wd.get('peak_rate_summer1', 0)),
            'summer_midpeak': _safe(wd.get('midpeak_rate_summer1', 0)),
            'summer_offpeak': _safe(wd.get('offpeak_rate_summer1', 0)),
            'winter_peak': _safe(wd.get('peak_rate_winter1', 0)),
            'winter_midpeak': _safe(wd.get('midpeak_rate_winter1', 0)),
            'winter_offpeak': _safe(wd.get('offpeak_rate_winter1', 0)),
        }
        bl_credit = _safe(wd.get('baseline_credit', 0))
        care_disc = abs(_safe(wd.get('care_discount', 0)))
        base_svc = _safe(wd.get('base_service_charge_per_day', 0))
        has_fixed = wd.get('Fixed', '') == 'Yes'
        fc_low = _safe(wd.get('fixedcharge_low', 0)) if has_fixed else 0.0
        fc_med = _safe(wd.get('fixedcharge_med', 0)) if has_fixed else 0.0
        fc_high = _safe(wd.get('fixedcharge_high', 0)) if has_fixed else 0.0
        return {
            'rate_dict': rd,
            'rate_arr': build_tou_rate_array_from_dict(rd),
            'baseline_credit': bl_credit,
            'care_discount': care_disc,
            'base_svc_daily': base_svc,
            'has_fixed': has_fixed,
            'fc_monthly': {'low': fc_low, 'medium': fc_med, 'high': fc_high},
        }

    actual_rates = {}
    for rc in ACTUAL_SDGE_RATES:
        actual_rates[rc] = _load_actual_rate(rc)
        print(f"  Loaded {rc} rates: baseline_credit={actual_rates[rc]['baseline_credit']:.4f}, "
              f"has_fixed={actual_rates[rc]['has_fixed']}")

    # CARE discount for designed scenarios: use TOU-DR's CARE discount
    # (designed scenarios are alternative rate structures for the same utility)
    designed_care_discount = actual_rates['TOU-DR']['care_discount']

    print(f"  Per-scenario LP dispatch (designed + actual tariffs):")
    print(f"  CARE discount for designed scenarios: {designed_care_discount:.2%}")
    for sname in designed_rate_arrays:
        fc = scenario_fixed_charges[sname]
        print(f"    {sname}: FC=${fc['noncare']/12:.2f}/mo")

    # Check for Upgrade 11 directory
    upgrade11_dir = Path(UPGRADE11_DIR)
    has_upgrade11 = upgrade11_dir.exists() and any(upgrade11_dir.glob('*.parquet'))
    if has_upgrade11:
        print(f"  Upgrade 11 data found in {UPGRADE11_DIR}")
    else:
        print(f"  WARNING: No Upgrade 11 data in {UPGRADE11_DIR} — S3/S4 scenarios will be skipped")

    # Adoption scenario suffixes
    # S1: EV only (baseline + EV, no PV/battery)
    # S2: PV + storage (baseline + PV + battery, no EV)
    # S3: PV + storage + EV (baseline demand, PV sized on baseline)
    # S4: Full electrification + PV + storage + EV (Upgrade11 demand, PV sized on Upgrade11)
    ADOPT_SCENARIOS = ['s1_ev', 's2_pv_stor', 's3_pv_stor_ev', 's4_full_elec']

    def _compute_bill_for_rate(load_profile, solar_gen, rate_arr, eec_rate_arr,
                               is_care, care_disc, bl_entry_row, bl_credit_rate,
                               use_battery=False):
        """Compute net bill for a single rate scenario with optional battery dispatch.

        Runs LP/heuristic battery dispatch with the given rate array and EEC rates.
        Returns (net_bill_energy, export_credit, batt_grid_import_or_None).
        """
        net = load_profile - solar_gen
        hourly_import = np.maximum(net, 0)

        days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month = days_per_month * 24
        month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))

        if use_battery:
            batt_result = stage5_battery_dispatch(load_profile, solar_gen, rate_arr, eec_rate_arr)
            if batt_result is not None:
                import_cost = batt_result['bill_energy']
                export_credit = batt_result['export_credit']
                batt_import = batt_result['grid_import']

                # Baseline credit on battery-optimized import
                if bl_entry_row is not None:
                    d_sum_bl = bl_entry_row['summer_baseline_allowance']
                    d_win_bl = bl_entry_row['winter_baseline_allowance']
                    total_bl_credit = 0.0
                    for m in range(12):
                        s, e = month_boundaries[m], month_boundaries[m + 1]
                        monthly_import = batt_import[s:e].sum()
                        if 6 <= (m + 1) <= 10:
                            monthly_bl = d_sum_bl * days_per_month[m]
                        else:
                            monthly_bl = d_win_bl * days_per_month[m]
                        total_bl_credit += bl_credit_rate * min(monthly_import, monthly_bl)
                    import_cost -= total_bl_credit

                if is_care and care_disc > 0:
                    import_cost *= (1 - care_disc)
                return import_cost, export_credit, batt_import
            # LP failed — fall through to no-battery path

        # No battery or LP failed
        import_cost = np.dot(hourly_import, rate_arr)
        hourly_export = np.maximum(-net, 0)
        export_credit = np.dot(hourly_export, eec_rate_arr)

        if bl_entry_row is not None:
            d_sum_bl = bl_entry_row['summer_baseline_allowance']
            d_win_bl = bl_entry_row['winter_baseline_allowance']
            total_bl_credit = 0.0
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                monthly_import = hourly_import[s:e].sum()
                if 6 <= (m + 1) <= 10:
                    monthly_bl = d_sum_bl * days_per_month[m]
                else:
                    monthly_bl = d_win_bl * days_per_month[m]
                total_bl_credit += bl_credit_rate * min(monthly_import, monthly_bl)
            import_cost -= total_bl_credit

        if is_care and care_disc > 0:
            import_cost *= (1 - care_disc)

        return import_cost, export_credit, None

    def _compute_all_scenario_bills(load_profile, solar_gen, is_care, income,
                                     bl_entry_row, use_battery, prefix):
        """Compute post-adoption bills for all rate scenarios (designed + actual).

        Optimization: all designed scenarios share the same TOU rate shape
        (just uniformly scaled), so LP dispatch is solved ONCE using TOU-DR
        rates and the resulting grid_import/export profile is reused to
        compute bills under each designed scenario's rates.

        Actual tariffs (TOU-DR, TOU-DR-F) have different rate structures,
        so they get separate LP solves.

        Returns dict of {col_name: bill_value} and lp_fail count.
        """
        result = {}
        lp_fail = 0

        days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month = days_per_month * 24
        month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))

        # --- Designed scenarios: solve LP once, reuse dispatch ---
        # Use TOU-DR rate array for dispatch (same TOU shape as all designed)
        ref_rate_name = list(designed_rate_arrays.keys())[0]
        ref_rate_arr = designed_rate_arrays[ref_rate_name]

        batt_dispatch = None
        if use_battery:
            batt_dispatch = stage5_battery_dispatch(
                load_profile, solar_gen, ref_rate_arr, eec_rates)
            if batt_dispatch is None:
                lp_fail = 1

        if batt_dispatch is not None:
            # Reuse battery dispatch for all designed scenarios
            grid_import_arr = batt_dispatch['grid_import']
            grid_export_arr = batt_dispatch['grid_export']

            for sname, rate_arr in designed_rate_arrays.items():
                fc = scenario_fixed_charges[sname]
                fixed = fc['care'] if is_care else fc['noncare']
                import_cost = np.dot(grid_import_arr, rate_arr)
                export_credit = np.dot(grid_export_arr, eec_rates)
                if is_care and designed_care_discount > 0:
                    import_cost *= (1 - designed_care_discount)
                bill = max(import_cost - export_credit, 0) + fixed
                result[f'{sname}_bill_{prefix}'] = bill
        else:
            # No battery — compute bills directly from net load
            net = load_profile - solar_gen
            hourly_import = np.maximum(net, 0)
            hourly_export = np.maximum(-net, 0)

            for sname, rate_arr in designed_rate_arrays.items():
                fc = scenario_fixed_charges[sname]
                fixed = fc['care'] if is_care else fc['noncare']
                import_cost = np.dot(hourly_import, rate_arr)
                export_credit = np.dot(hourly_export, eec_rates)
                if is_care and designed_care_discount > 0:
                    import_cost *= (1 - designed_care_discount)
                bill = max(import_cost - export_credit, 0) + fixed
                result[f'{sname}_bill_{prefix}'] = bill

        # --- Actual tariff bills: separate LP dispatch per tariff ---
        for rc, rc_info in actual_rates.items():
            col_prefix = ACTUAL_SDGE_RATES[rc]
            import_cost, export_credit, batt_import = _compute_bill_for_rate(
                load_profile, solar_gen, rc_info['rate_arr'], eec_rates,
                is_care, rc_info['care_discount'], bl_entry_row,
                rc_info['baseline_credit'], use_battery=use_battery)
            rc_fixed = rc_info['base_svc_daily'] * 365
            if rc_info['has_fixed']:
                rc_fixed += rc_info['fc_monthly'].get(income, 0.0) * 12
            bill = max(import_cost - export_credit, 0) + rc_fixed
            result[f'{col_prefix}_bill_{prefix}'] = bill
            if use_battery and batt_import is None:
                lp_fail += 1

        return result, lp_fail

    def _bill_volumetric_only(load_profile, is_care, income, puma_str,
                              bl_entry_row, prefix):
        """Compute bills for a load profile with no PV (EV-only scenario)."""
        result = {}
        days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month = days_per_month * 24
        month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))

        # Designed scenarios — direct rate array application
        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            vol_cost = np.dot(load_profile, rate_arr)
            if is_care and designed_care_discount > 0:
                vol_cost *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = vol_cost + fixed

        # Actual tariff bills
        for rc in ACTUAL_SDGE_RATES:
            col_prefix = ACTUAL_SDGE_RATES[rc]
            rc_bill = calculate_actual_sdge_bill_vectorized(
                load_profile, rc, puma_str, income, is_care)
            result[f'{col_prefix}_bill_{prefix}'] = rc_bill
        return result

    for idx, row in has_tech.iterrows():
        bid = int(row['building_id'])
        pq_file = baseline_dir / f"{bid}-0.parquet"

        if not pq_file.exists():
            continue

        try:
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)
            sf = row.get('scaling_factor', 1.0)
            hourly_load = hourly_load * sf

            # Natural gas consumption from baseline (kBtu → therms: 1 therm = 100 kBtu)
            gas_col = 'out.natural_gas.total.energy_consumption'
            baseline_gas_therms = 0.0
            if gas_col in df.columns:
                gas_15min_kbtu = df[gas_col].values
                baseline_gas_therms = gas_15min_kbtu.sum() / 100.0  # kBtu → therms

            income = row.get('income', 'medium')
            is_care = (income == 'low')
            puma_str = row.get('puma', '')

            # EV charging profile for this building
            bldg_dvmt = ev_dvmt_map.get(bid, 30.0) if row['assigned_ev'] == 1 else 0.0
            ev_profile = make_ev_profile(daily_miles=bldg_dvmt) if bldg_dvmt > 0 else np.zeros(8760)

            # Baseline allowance lookup
            bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
            bl_row = bl_entry.iloc[0].to_dict() if not bl_entry.empty else None

            # Load Upgrade 11 profile if available
            u11_file = upgrade11_dir / f"{bid}-11.parquet"
            u11_load = None
            if has_upgrade11 and u11_file.exists():
                u11_df = pd.read_parquet(u11_file)
                u11_15min = u11_df['out.electricity.total.energy_consumption'].values
                u11_load = u11_15min.reshape(-1, 4).sum(axis=1) * sf

            update_row = {'building_id': bid,
                          'ev_daily_miles': bldg_dvmt,
                          'baseline_gas_therms': baseline_gas_therms,
                          'gas_savings_therms': baseline_gas_therms}  # Upgrade11 = 0 gas

            # ── S1: EV only (baseline + EV, no PV/battery) ──
            if row['assigned_ev'] == 1:
                s1_load = hourly_load + ev_profile
                s1_bills = _bill_volumetric_only(
                    s1_load, is_care, income, puma_str, bl_row, 's1_ev')
                update_row.update(s1_bills)

            # ── S2: PV + storage (baseline + PV + battery, no EV) ──
            if row['assigned_pv'] == 1:
                bldg_annual_kwh = row.get('annual_kwh', hourly_load.sum())
                pv_size_s2 = size_pv_system(bldg_annual_kwh, annual_kwh_per_kw)
                bldg_solar_s2 = solar_profile * pv_size_s2
                update_row['pv_size_kw_s2'] = pv_size_s2
                pv_sizes.append(pv_size_s2)

                s2_bills, s2_lp_fail = _compute_all_scenario_bills(
                    hourly_load, bldg_solar_s2, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s2_pv_stor')
                update_row.update(s2_bills)
                lp_failures += s2_lp_fail

            # ── S3: PV + Storage + EV (baseline demand, PV sized on baseline) ──
            if row['assigned_pv'] == 1 and row['assigned_ev'] == 1 and not skip_s3:
                s3_load = hourly_load + ev_profile
                # PV sized on baseline demand (before adding EV)
                pv_size_s3 = size_pv_system(row.get('annual_kwh', hourly_load.sum()), annual_kwh_per_kw)
                bldg_solar_s3 = solar_profile * pv_size_s3
                update_row['pv_size_kw_s3'] = pv_size_s3

                s3_bills, s3_lp_fail = _compute_all_scenario_bills(
                    s3_load, bldg_solar_s3, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s3_pv_stor_ev')
                update_row.update(s3_bills)
                update_row['annual_kwh_s3'] = s3_load.sum()
                lp_failures += s3_lp_fail

            # ── S4: Full electrification + PV + Storage + EV (Upgrade11, PV sized on Upgrade11) ──
            if row['assigned_pv'] == 1 and u11_load is not None:
                s4_load = u11_load + ev_profile
                # PV sized on Upgrade11 + EV consumption (post-electrification)
                post_elec_kwh = s4_load.sum()
                pv_size_s4 = size_pv_system(post_elec_kwh, annual_kwh_per_kw)
                bldg_solar_s4 = solar_profile * pv_size_s4
                update_row['pv_size_kw_s4'] = pv_size_s4

                s4_bills, s4_lp_fail = _compute_all_scenario_bills(
                    s4_load, bldg_solar_s4, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s4_full_elec')
                update_row.update(s4_bills)
                update_row['annual_kwh_s4'] = s4_load.sum()
                lp_failures += s4_lp_fail

            results_update[bid] = update_row
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(has_tech)} | {elapsed:.0f}s")

        except Exception as e:
            if processed < 5:
                print(f"  Error on building {bid}: {e}")

    # Merge post-adoption bills back
    if results_update:
        post_df = pd.DataFrame(results_update.values())
        merged = merged.merge(post_df, on='building_id', how='left')

    elapsed = time.time() - start_time
    print(f"\n  Processed {processed} buildings in {elapsed:.1f}s")
    if pv_sizes:
        pv_arr = np.array(pv_sizes)
        print(f"  PV sizing (consumption-based, {PV_OFFSET_TARGET*100:.0f}% offset target):")
        print(f"    Mean: {pv_arr.mean():.1f} kW | Median: {np.median(pv_arr):.1f} kW")
        print(f"    Range: {pv_arr.min():.1f}–{pv_arr.max():.1f} kW | Std: {pv_arr.std():.1f} kW")
    if lp_failures > 0:
        print(f"  LP failures (fell back to no-battery): {lp_failures}")

    merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
    print(f"  Saved to: {POSTADOPT_BILLS_OUT}")

    return merged


# ---------------------------------------------------------------------------
# Stage 7: Summary & distributional analysis
# ---------------------------------------------------------------------------

def stage7_summary(final_df, rate_scenarios_df):
    """Generate distributional analysis tables."""
    print("\n" + "=" * 80)
    print("STAGE 7: DISTRIBUTIONAL ANALYSIS")
    print("=" * 80)

    # All bill columns
    all_bill_cols = [c for c in final_df.columns if c.endswith('_bill') and 'postadopt' not in c]

    # Revenue neutrality check: extrapolate to population-level revenue
    # and compare designed scenarios against actual TOU-DR
    from rate_designer import RESIDENTIAL_REVENUE, CUSTOMERS
    total_customers = CUSTOMERS['total']

    print("\n--- Revenue Neutrality Check (population-level) ---")
    print(f"  EIA/SDGE filed residential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"  Total residential customers: {total_customers:,}")

    tou_dr_col = 'tou_dr_bill'
    designed_cols = [f'{s}_bill' for s in DESIGNED_SCENARIOS if f'{s}_bill' in final_df.columns]

    # Use only buildings with valid TOU-DR bills for apples-to-apples comparison
    if tou_dr_col in final_df.columns and final_df[tou_dr_col].notna().sum() > 0:
        valid_mask = final_df[tou_dr_col].notna()
        n_valid = valid_mask.sum()

        # Population-level revenue = mean(bill) × total_customers
        tou_dr_mean = final_df.loc[valid_mask, tou_dr_col].mean()
        tou_dr_pop_rev = tou_dr_mean * total_customers

        print(f"\n  Sample size: {n_valid} buildings "
              f"(each represents ~{total_customers/n_valid:.0f} households)")
        print(f"\n  {'Rate':<25s} {'Mean Bill':>12s} {'Pop Revenue':>14s} "
              f"{'vs TOU-DR':>10s} {'vs Filed':>10s}")
        print(f"  {'-'*71}")

        # Actual TOU-DR
        pct_vs_filed = (tou_dr_pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
        print(f"  {'Actual TOU-DR':<25s} ${tou_dr_mean:>10,.0f} "
              f"${tou_dr_pop_rev/1e9:>12.4f}B {'—':>10s} "
              f"{pct_vs_filed:>+9.2f}%")

        # Actual TOU-DR-F if available
        tou_drf_col = 'tou_dr_f_bill'
        if tou_drf_col in final_df.columns:
            drf_valid = valid_mask & final_df[tou_drf_col].notna()
            if drf_valid.sum() > 0:
                drf_mean = final_df.loc[drf_valid, tou_drf_col].mean()
                drf_pop_rev = drf_mean * total_customers
                pct_vs_dr = (drf_pop_rev - tou_dr_pop_rev) / tou_dr_pop_rev * 100
                pct_vs_filed = (drf_pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
                print(f"  {'Actual TOU-DR-F':<25s} ${drf_mean:>10,.0f} "
                      f"${drf_pop_rev/1e9:>12.4f}B {pct_vs_dr:>+9.2f}% "
                      f"{pct_vs_filed:>+9.2f}%")

        # Designed scenarios
        for col in designed_cols:
            mean_bill = final_df.loc[valid_mask, col].mean()
            pop_rev = mean_bill * total_customers
            pct_vs_dr = (pop_rev - tou_dr_pop_rev) / tou_dr_pop_rev * 100
            pct_vs_filed = (pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
            label = col.replace('_bill', '')
            print(f"  {label:<25s} ${mean_bill:>10,.0f} "
                  f"${pop_rev/1e9:>12.4f}B {pct_vs_dr:>+9.2f}% "
                  f"{pct_vs_filed:>+9.2f}%")
    else:
        # Fallback: compare designed scenarios against each other
        print("\n  WARNING: No valid TOU-DR bills — comparing designed scenarios only")
        if designed_cols:
            for col in designed_cols:
                mean_bill = final_df[col].mean()
                pop_rev = mean_bill * total_customers
                pct_vs_filed = (pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
                label = col.replace('_bill', '')
                print(f"  {label:<25s} ${mean_bill:>10,.0f} "
                      f"${pop_rev/1e9:>12.4f}B {pct_vs_filed:>+9.2f}% vs filed")

    # Bill distribution by CARE status
    print("\n--- Mean Annual Bill by CARE Status ---")
    for col in all_bill_cols:
        print(f"\n  {col}:")
        for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
            subset = final_df[final_df['is_care'] == care_val]
            if len(subset) > 0:
                valid = subset[col].dropna()
                if len(valid) > 0:
                    print(f"    {care_label:>8s}: mean=${valid.mean():,.0f}  "
                          f"median=${valid.median():,.0f}  (n={len(valid)})")

    # Bill change from baseline (TOU-DR actual or F0_WF0_ROE0) by CARE status
    base_col = 'tou_dr_bill' if 'tou_dr_bill' in final_df.columns else 'F0_WF0_ROE0_bill'
    if base_col in final_df.columns:
        print(f"\n--- Bill Change from {base_col} by CARE Status ---")
        compare_cols = [c for c in all_bill_cols if c != base_col]
        for col in compare_cols:
            print(f"\n  {col} vs {base_col}:")
            for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
                subset = final_df[final_df['is_care'] == care_val]
                if len(subset) > 0:
                    valid_mask = subset[col].notna() & subset[base_col].notna()
                    if valid_mask.sum() > 0:
                        change = subset.loc[valid_mask, col] - subset.loc[valid_mask, base_col]
                        print(f"    {care_label:>8s}: mean=${change.mean():+,.0f}  "
                              f"median=${change.median():+,.0f}  "
                              f"winners={(change < 0).sum()}/{valid_mask.sum()}")

    # Tech adoption impact — 4 adoption scenarios
    ADOPT_SUFFIXES = {
        's1_ev': 'S1: EV only',
        's2_pv_stor': 'S2: PV + storage',
        's3_pv_stor_ev': 'S3: PV + storage + EV',
        's4_full_elec': 'S4: Full elec + PV + storage + EV',
    }
    if 'assigned_pv' in final_df.columns or 'assigned_ev' in final_df.columns:
        print("\n--- Tech Adoption Bill Impact by Scenario ---")
        for sfx, sfx_label in ADOPT_SUFFIXES.items():
            # Find all bill columns for this adoption scenario
            adopt_cols = [c for c in final_df.columns if c.endswith(f'_bill_{sfx}')]
            if not adopt_cols:
                continue
            # Get buildings that have bills for this scenario
            mask = final_df[adopt_cols[0]].notna()
            n_bldgs = mask.sum()
            if n_bldgs == 0:
                continue
            print(f"\n  {sfx_label} (n={n_bldgs}):")
            # Compare each rate scenario's adoption bill vs baseline bill
            for acol in adopt_cols:
                # Extract rate scenario name: e.g. 'F50_WF0_ROE0_bill_s1_ev' -> 'F50_WF0_ROE0'
                rate_name = acol.replace(f'_bill_{sfx}', '')
                base_col = f'{rate_name}_bill'
                if base_col not in final_df.columns:
                    continue
                valid = mask & final_df[base_col].notna()
                if valid.sum() == 0:
                    continue
                change = final_df.loc[valid, base_col] - final_df.loc[valid, acol]
                print(f"    {rate_name:<25s}: mean savings ${change.mean():+,.0f}/yr  "
                      f"median ${change.median():+,.0f}/yr")
            # By income
            for inc in ['low', 'medium', 'high']:
                inc_mask = mask & (final_df['income'] == inc)
                if inc_mask.sum() == 0:
                    continue
                # Use TOU-DR baseline if available, else first designed scenario
                ref_acol = adopt_cols[0]
                ref_rate = ref_acol.replace(f'_bill_{sfx}', '')
                ref_base = f'{ref_rate}_bill'
                if 'tou_dr_bill' in final_df.columns and f'tou_dr_bill_{sfx}' in final_df.columns:
                    ref_base = 'tou_dr_bill'
                    ref_acol = f'tou_dr_bill_{sfx}'
                valid_inc = inc_mask & final_df[ref_base].notna() & final_df[ref_acol].notna()
                if valid_inc.sum() > 0:
                    ch = final_df.loc[valid_inc, ref_base] - final_df.loc[valid_inc, ref_acol]
                    print(f"      {inc:>8s} (n={valid_inc.sum()}): "
                          f"mean savings ${ch.mean():+,.0f}/yr")

        # PV sizing summary across scenarios
        pv_size_cols = [c for c in final_df.columns if c.startswith('pv_size_kw_s')]
        if pv_size_cols:
            print("\n  PV System Sizing by Scenario:")
            for pc in pv_size_cols:
                valid = final_df[pc].dropna()
                if len(valid) > 0:
                    sfx_key = pc.replace('pv_size_kw_', '')
                    label = ADOPT_SUFFIXES.get(sfx_key, sfx_key)
                    print(f"    {label}: mean={valid.mean():.1f} kW  "
                          f"median={valid.median():.1f} kW  range=[{valid.min():.1f}, {valid.max():.1f}]")

    # Save summary
    summary_rows = []
    for col in all_bill_cols:
        sname = col.replace('_bill', '')
        for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
            subset = final_df[final_df['is_care'] == care_val]
            if len(subset) > 0:
                summary_rows.append({
                    'scenario': sname,
                    'care_status': care_label,
                    'n_buildings': len(subset),
                    'mean_bill': subset[col].mean(),
                    'median_bill': subset[col].median(),
                    'p10_bill': subset[col].quantile(0.1),
                    'p90_bill': subset[col].quantile(0.9),
                    'mean_kwh': subset['annual_kwh'].mean(),
                })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        print(f"\n  Summary saved to: {SUMMARY_OUT}")


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
                        help='Run only tech stages (3-6), skip rate design and summary')
    parser.add_argument('--skip-s3', action='store_true',
                        help='Skip S3 (PV+storage+EV) adoption scenario in stage 6')
    args = parser.parse_args()

    n_buildings = 50 if args.test else args.n_buildings

    # --tech-only: force stage >= 3 (load existing rate scenarios & bills)
    if args.tech_only:
        args.stage = max(args.stage, 3)
        args.skip_tech = False  # override conflicting flag

    print("=" * 80)
    print("SDGE RATE ANALYSIS PIPELINE")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'TEST' if args.test else 'FULL'}"
          f"{' (tech-only)' if args.tech_only else ''}")
    print(f"Starting stage: {args.stage}")
    print("=" * 80)

    pipeline_start = time.time()

    # Stage 2: Baseline bills + rate scenario generation
    # Rate scenarios are generated inside stage 2 after R_0 (sample TOU-DR
    # revenue) is computed, so the R_sample approach works correctly.
    if args.stage <= 2:
        # Pass existing rate scenarios if loading from file (--stage > 1)
        existing_scenarios = None
        if args.stage > 1 and os.path.exists(RATE_SCENARIOS_OUT):
            existing_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
            print(f"\nLoaded existing rate scenarios from {RATE_SCENARIOS_OUT}")
        bills_df, rate_scenarios = stage2_compute_baseline_bills(
            existing_scenarios, n_buildings)
    else:
        bills_df = pd.read_csv(BASELINE_BILLS_OUT)
        rate_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
        print(f"\nLoaded existing bills from {BASELINE_BILLS_OUT}")
        print(f"Loaded existing rate scenarios from {RATE_SCENARIOS_OUT}")

    if args.skip_tech:
        print("\n  Skipping technology adoption stages (--skip-tech)")
        stage7_summary(bills_df, rate_scenarios)
    else:
        # Stage 3: Tech assignments
        if args.stage <= 3:
            tech_df = stage3_tech_assignments(bills_df)
        else:
            tech_df = pd.read_csv(TECH_ASSIGNMENTS_OUT)
            print(f"\nLoaded tech assignments from {TECH_ASSIGNMENTS_OUT}")

        # Stage 4: Solar profiles (returns per-kW profile + annual kWh/kW)
        if args.stage <= 4:
            solar_per_kw, annual_kwh_per_kw = stage4_solar_profiles(tech_df, bills_df)
        else:
            synth = _synthetic_solar_profile()
            solar_per_kw = synth / DEFAULT_PV_SIZE_KW
            annual_kwh_per_kw = solar_per_kw.sum()

        # Stage 5 is integrated into Stage 6

        # Stage 6: Post-adoption bills
        if args.stage <= 6:
            final_df = stage6_post_adoption_bills(
                bills_df, tech_df, solar_per_kw, rate_scenarios,
                use_lp=args.use_lp, annual_kwh_per_kw=annual_kwh_per_kw,
                skip_s3=args.skip_s3)
        else:
            final_df = pd.read_csv(POSTADOPT_BILLS_OUT)
            print(f"\nLoaded post-adoption bills from {POSTADOPT_BILLS_OUT}")

        # Stage 7: Summary (skip if --tech-only)
        if not getattr(args, 'tech_only', False):
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
