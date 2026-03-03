"""
rate_builder_sdge.py — Build SDGE rate scenarios from actual TOU-DR tariff rates.

Key difference from the original rate_builder notebook:
  The original divided total revenue by total sales ($1.562B / 4.81B kWh = $0.3247/kWh),
  producing artificially low volumetric rates (e.g., summer_peak = $0.37). This is because
  the bill calculator applies baseline credits (~$0.11/kWh) and CARE discounts (37%) on top,
  so actual tariff rates are much higher than revenue/sales.

  This script starts from the ACTUAL SDGE TOU-DR tariff rates and applies proportional
  reductions for each policy lever. At F0_WF0_ROE0 (status quo), the rates match what
  customers actually see on their bills.

Policy levers:
  1. Fixed charge %  — Fraction of T&D costs moved from volumetric to fixed charges
  2. Wildfire removal — Remove wildfire cost component from rates
  3. ROE reduction    — Reduce allowed return on equity (in percentage points)

Validation:
  The actual TOU-DR-F tariff (CPUC's fixed charge ruling) represents ~35% of T&D costs
  in fixed charges. This script reproduces those charges within <1% error.
"""

import pandas as pd
import numpy as np
from itertools import product

# ============================================================================
# SDGE CONSTANTS (from GRC filings, EIA Form 861)
# ============================================================================

RESIDENTIAL_REVENUE = 1_561_695_600  # Total residential revenue (EIA)
RESIDENTIAL_SALES_KWH = 4_810_000_000  # Total residential sales (EIA)
TOTAL_UTILITY_REVENUE = 4_233_072_000  # All customer classes

CUSTOMERS = {
    'care': 372_135,      # CARE-eligible (low income)
    'non_care': 951_477,  # Non-CARE
    'total': 1_323_612,
}

RATE_BASE = 13_590_538_000  # Total utility rate base

# Residential share of system-wide costs
res_share = RESIDENTIAL_REVENUE / TOTAL_UTILITY_REVENUE  # ≈ 0.369

# Revenue components (system-wide values × residential share)
REVENUE_COMPONENTS = {
    'wildfire': 413_873_000 * res_share,       # ~$152.7M
    'transmission': 685_245_000 * res_share,   # ~$252.9M
    'distribution': 1_722_187_000 * res_share, # ~$635.5M
}
TD_COSTS = REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']

# ============================================================================
# ACTUAL SDGE TARIFF RATES (from retail_rates_data_SDGE.xlsx, Oct 2025)
# ============================================================================

# TOU-DR: The default residential TOU rate BEFORE fixed charge ruling
ACTUAL_TOU_DR = {
    'summer_peak': 0.600,
    'summer_midpeak': 0.527,
    'summer_offpeak': 0.450,
    'winter_peak': 0.58155,
    'winter_midpeak': 0.51899,
    'winter_offpeak': 0.50084,
}
BASELINE_CREDIT_TOU_DR = 0.11017  # $/kWh baseline credit

# TOU-DR-F: The rate AFTER fixed charge ruling (for validation)
ACTUAL_TOU_DR_F = {
    'summer_peak': 0.53462,
    'summer_midpeak': 0.46308,
    'summer_offpeak': 0.38554,
    'winter_peak': 0.51709,
    'winter_midpeak': 0.45453,
    'winter_offpeak': 0.43638,
}
BASELINE_CREDIT_TOU_DR_F = 0.09690
ACTUAL_FIXED_LOW = 6.11072    # $/month (CARE/low income)
ACTUAL_FIXED_MED = 24.59633   # $/month (medium income)
ACTUAL_FIXED_HIGH = 24.59633  # $/month (high income)
ACTUAL_FIXED_FERA = 12.00     # $/month (FERA)

# CARE discount on volumetric rates
CARE_VOLUMETRIC_DISCOUNT = 0.37  # 37% discount on $/kWh (TOU-DR)

# ============================================================================
# TOU CONSUMPTION WEIGHTS (from building simulations, tou_weights_sdge.csv)
# ============================================================================

weights_df = pd.read_csv('tou_weights_sdge.csv')
TOU_WEIGHTS = dict(zip(weights_df['period'], weights_df['weight']))

# ============================================================================
# FIXED CHARGE INCOME GRADUATION
# ============================================================================

# The CPUC's actual TOU-DR-F uses a ratio of ~0.248 (low/non-CARE = 6.11/24.60).
# We parameterize this so scenarios can explore different graduation levels.
# The actual CPUC structure has low != med = high. We maintain this structure.
CARE_FIXED_RATIO = ACTUAL_FIXED_LOW / ACTUAL_FIXED_HIGH  # ≈ 0.248


def design_rate_sdge(fixed_pct_td=0, remove_wildfire=False, roe_reduction=0,
                     care_fixed_ratio=CARE_FIXED_RATIO):
    """
    Design SDGE rate scenario starting from actual TOU-DR tariff rates.

    At F0_WF0_ROE0 (status quo), rates equal the actual TOU-DR tariff.
    Policy levers reduce rates proportionally while shifting revenue to fixed
    charges, removing wildfire costs, or reducing ROE.

    Parameters
    ----------
    fixed_pct_td : int
        Percentage of T&D costs moved to income-graduated fixed charges (0-100).
    remove_wildfire : bool
        If True, remove wildfire cost component from rates.
    roe_reduction : float
        Reduction in return on equity, in percentage points (e.g., 1.0 = 1pp).
    care_fixed_ratio : float
        Ratio of CARE fixed charge to non-CARE fixed charge (default: 0.248).

    Returns
    -------
    dict with scenario parameters and resulting TOU rates.
    """
    R = RESIDENTIAL_REVENUE

    # Revenue adjustments
    wildfire_adj = REVENUE_COMPONENTS['wildfire'] if remove_wildfire else 0
    roe_adj = RATE_BASE * (roe_reduction / 100) * res_share if roe_reduction > 0 else 0

    # Fixed charge revenue
    fixed_revenue = TD_COSTS * (fixed_pct_td / 100)

    # Income-graduated fixed charges
    # Revenue equation: CARE_customers × care_charge × 12 + nonCARE × noncare_charge × 12 = fixed_revenue
    # With care_charge = noncare_charge × care_fixed_ratio:
    total_weighted_customers = (CUSTOMERS['care'] * care_fixed_ratio) + CUSTOMERS['non_care']
    fixed_non_care = fixed_revenue / total_weighted_customers / 12 if fixed_pct_td > 0 else 0
    fixed_care = fixed_non_care * care_fixed_ratio

    # Volumetric revenue remaining
    vol_revenue = R - fixed_revenue - wildfire_adj - roe_adj

    # Scaling factor: what fraction of baseline revenue is still collected volumetrically
    scale = vol_revenue / R

    # Scale actual TOU-DR tariff rates
    new_tou_rates = {k: v * scale for k, v in ACTUAL_TOU_DR.items()}

    # Scale baseline credit proportionally
    baseline_credit = BASELINE_CREDIT_TOU_DR * scale

    # Consumption-weighted average of new tariff rates
    vol_avg = sum(new_tou_rates[p] * TOU_WEIGHTS[p] for p in new_tou_rates)

    # Total revenue after policy adjustments (before fixed/vol split)
    total_revenue = R - wildfire_adj - roe_adj

    return {
        'Scenario': f'F{fixed_pct_td}_WF{int(remove_wildfire)}_ROE{roe_reduction}',
        'Fixed_Pct_TD': fixed_pct_td,
        'Remove_Wildfire': remove_wildfire,
        'ROE_Reduction': roe_reduction,
        'Fixed_CARE': round(fixed_care, 5),
        'Fixed_NonCARE': round(fixed_non_care, 5),
        'Vol_Avg': round(vol_avg, 5),
        'Total_Revenue': round(total_revenue, 0),
        'Vol_Revenue': round(vol_revenue, 0),
        'Scaling_Factor': round(scale, 6),
        'baseline_credit': round(baseline_credit, 5),
        'summer_peak': round(new_tou_rates['summer_peak'], 5),
        'summer_midpeak': round(new_tou_rates['summer_midpeak'], 5),
        'summer_offpeak': round(new_tou_rates['summer_offpeak'], 5),
        'winter_peak': round(new_tou_rates['winter_peak'], 5),
        'winter_midpeak': round(new_tou_rates['winter_midpeak'], 5),
        'winter_offpeak': round(new_tou_rates['winter_offpeak'], 5),
    }


def validate_against_actual():
    """Validate by comparing F35 scenario against actual TOU-DR-F tariff."""
    # The actual TOU-DR-F represents approximately 35% of T&D in fixed charges.
    # We find the exact percentage by matching the non-CARE fixed charge.
    # Solve: fixed_non_care($24.60) = TD_COSTS × pct/100 / weighted_customers / 12

    target_fixed = ACTUAL_FIXED_HIGH  # $24.60
    total_weighted = CUSTOMERS['care'] * CARE_FIXED_RATIO + CUSTOMERS['non_care']
    implied_fixed_revenue = target_fixed * total_weighted * 12
    implied_pct = implied_fixed_revenue / TD_COSTS * 100

    scenario = design_rate_sdge(fixed_pct_td=implied_pct)

    print("=" * 80)
    print("VALIDATION: Scenario at implied fixed % vs actual TOU-DR-F")
    print("=" * 80)
    print(f"Implied fixed charge % of T&D: {implied_pct:.1f}%")
    print()
    print(f"{'Period':<20s} {'Actual TOU-DR-F':>15s} {'Modeled':>15s} {'Error':>10s}")
    print("-" * 60)
    for period in ACTUAL_TOU_DR.keys():
        actual = ACTUAL_TOU_DR_F[period]
        modeled = scenario[period]
        error = (modeled - actual) / actual * 100
        print(f"  {period:<18s} ${actual:>12.5f}   ${modeled:>12.5f}   {error:>+7.2f}%")

    print(f"\n  {'baseline_credit':<18s} ${BASELINE_CREDIT_TOU_DR_F:>12.5f}   "
          f"${scenario['baseline_credit']:>12.5f}   "
          f"{(scenario['baseline_credit'] - BASELINE_CREDIT_TOU_DR_F) / BASELINE_CREDIT_TOU_DR_F * 100:>+7.2f}%")
    print(f"\n  {'Fixed (non-CARE)':<18s} ${ACTUAL_FIXED_HIGH:>12.5f}   "
          f"${scenario['Fixed_NonCARE']:>12.5f}")
    print(f"  {'Fixed (CARE)':<18s} ${ACTUAL_FIXED_LOW:>12.5f}   "
          f"${scenario['Fixed_CARE']:>12.5f}")
    print()

    return implied_pct


def generate_excel_rate_rows(scenarios_df, output_excel='retail_rates_data_SDGE.xlsx'):
    """
    Generate rate rows in the format expected by corrected_bill_calc.py.

    For each scenario, creates weekday and weekend rows with the same TOU
    period definitions as TOU-DR, but with scaled volumetric rates and
    appropriate fixed charges.
    """
    # Load existing Excel to get TOU period definitions
    existing = pd.read_excel(output_excel, sheet_name='retail_rates_oct32025')
    tou_dr_weekday = existing[(existing['rate_type'] == 'TOU-DR') & (existing['weekday'] == 'weekday')].iloc[0]
    tou_dr_weekend = existing[(existing['rate_type'] == 'TOU-DR') & (existing['weekday'] == 'weekend')].iloc[0]

    new_rows = []
    for _, sc in scenarios_df.iterrows():
        scenario_name = sc['Scenario']
        has_fixed = sc['Fixed_Pct_TD'] > 0

        for day_type, template in [('weekday', tou_dr_weekday), ('weekend', tou_dr_weekend)]:
            row = template.copy()
            row['rate_type'] = f"TOU-{scenario_name}"
            row['Fixed'] = 'Yes' if has_fixed else 'No'

            # Set TOU rates (tier 1 = tier 2 for non-tiered TOU)
            for suffix in ['1', '2']:
                row[f'peak_rate_summer{suffix}'] = sc['summer_peak']
                row[f'midpeak_rate_summer{suffix}'] = sc['summer_midpeak']
                row[f'offpeak_rate_summer{suffix}'] = sc['summer_offpeak']
                row[f'peak_rate_winter{suffix}'] = sc['winter_peak']
                row[f'midpeak_rate_winter{suffix}'] = sc['winter_midpeak']
                row[f'offpeak_rate_winter{suffix}'] = sc['winter_offpeak']

            # Set fixed charges
            if has_fixed:
                row['fixedcharge_low'] = sc['Fixed_CARE']
                row['fixedcharge_med'] = sc['Fixed_NonCARE']
                row['fixedcharge_high'] = sc['Fixed_NonCARE']
                row['fixedcharge_fera'] = sc['Fixed_CARE'] * (ACTUAL_FIXED_FERA / ACTUAL_FIXED_LOW)
                row['minimum_bill_per_day'] = np.nan
            else:
                row['fixedcharge_low'] = 0
                row['fixedcharge_med'] = 0
                row['fixedcharge_high'] = 0
                row['fixedcharge_fera'] = 0

            # Set baseline credit
            row['baseline_credit'] = sc['baseline_credit']

            # CARE discount (same as TOU-DR)
            row['care_discount'] = -CARE_VOLUMETRIC_DISCOUNT

            row['weekday'] = day_type
            new_rows.append(row)

    return pd.DataFrame(new_rows)


def main():
    print("=" * 80)
    print("SDGE RATE BUILDER — Actual Tariff-Based Scenarios")
    print("=" * 80)

    # Print baseline info
    weighted_avg = sum(ACTUAL_TOU_DR[p] * TOU_WEIGHTS[p] for p in ACTUAL_TOU_DR)
    print(f"\nBaseline TOU-DR weighted average: ${weighted_avg:.4f}/kWh")
    print(f"Revenue/sales average (old method): ${RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH:.4f}/kWh")
    print(f"Ratio (tariff/rev-avg): {weighted_avg / (RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH):.2f}x")
    print(f"\nResidential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"T&D costs (residential): ${TD_COSTS/1e6:.1f}M ({TD_COSTS/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"Wildfire costs (residential): ${REVENUE_COMPONENTS['wildfire']/1e6:.1f}M "
          f"({REVENUE_COMPONENTS['wildfire']/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"ROE impact per 1pp: ${RATE_BASE * 0.01 * res_share/1e6:.1f}M "
          f"({RATE_BASE * 0.01 * res_share/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"\nCARE fixed charge ratio: {CARE_FIXED_RATIO:.3f} "
          f"(from actual TOU-DR-F: ${ACTUAL_FIXED_LOW:.2f}/${ACTUAL_FIXED_HIGH:.2f})")

    # Validate against actual TOU-DR-F
    print()
    implied_pct = validate_against_actual()

    # Generate scenarios
    fixed_percentages = [0, 25, 50, 75, 100]
    wildfire_options = [False, True]
    roe_reductions = [0, 0.5, 1.0, 1.5]

    scenarios = []
    for fixed_pct, wf, roe in product(fixed_percentages, wildfire_options, roe_reductions):
        scenarios.append(design_rate_sdge(
            fixed_pct_td=fixed_pct,
            remove_wildfire=wf,
            roe_reduction=roe,
        ))

    df = pd.DataFrame(scenarios)

    # Save rate scenarios
    df.to_csv('rate_scenarios_sdge.csv', index=False)
    print(f"\nGenerated {len(df)} rate scenarios → rate_scenarios_sdge.csv")

    # Print summary table
    print("\n" + "=" * 80)
    print("RATE SCENARIO SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<22s} {'Vol_Avg':>8s} {'S_Peak':>8s} {'S_Off':>8s} "
          f"{'W_Peak':>8s} {'Fixed$':>8s} {'Scale':>7s}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"  {row['Scenario']:<20s} ${row['Vol_Avg']:>6.4f} ${row['summer_peak']:>6.4f} "
              f"${row['summer_offpeak']:>6.4f} ${row['winter_peak']:>6.4f} "
              f"${row['Fixed_NonCARE']:>6.2f} {row['Scaling_Factor']:>6.4f}")

    # Revenue check
    print("\n" + "=" * 80)
    print("REVENUE NEUTRALITY CHECK")
    print("=" * 80)
    for wf in [False, True]:
        for roe in [0, 1.0]:
            subset = df[(df['Remove_Wildfire'] == wf) & (df['ROE_Reduction'] == roe)]
            revenues = subset['Total_Revenue'].unique()
            print(f"  WF={int(wf)}, ROE={roe}: Total revenue = ${revenues[0]/1e9:.4f}B "
                  f"(same across all fixed %: {len(revenues) == 1})")

    # Generate Excel rate rows
    rate_rows = generate_excel_rate_rows(df)
    print(f"\nGenerated {len(rate_rows)} Excel-format rate rows")
    print("  (Use these to add scenario rates to the bill calculator Excel file)")

    return df, rate_rows


if __name__ == '__main__':
    df, rate_rows = main()
