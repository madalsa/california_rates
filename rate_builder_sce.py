"""
rate_builder_sce.py — Build SCE rate scenarios from actual TOU-D-4-9 tariff rates.

Mirrors the SDGE rate builder (rate_builder_sdge.py) structure:
  Starts from ACTUAL SCE TOU-D-4-9 tariff rates and applies proportional
  reductions for each policy lever. At F0_WF0_ROE0 (status quo), the rates
  match what customers actually see on their bills.

Note: SCE TOU-D-4-9 has weekday/weekend rate differences for summer peak.
  - Summer weekday peak: $0.627/kWh
  - Summer weekend peak: $0.507/kWh
  We use a blended summer peak rate (5/7 weekday + 2/7 weekend) for the
  rate design, and split back into weekday/weekend rates proportionally.

Policy levers:
  1. Fixed charge %  — Fraction of T&D costs moved from volumetric to fixed charges
  2. Wildfire removal — Remove wildfire cost component from rates
  3. ROE reduction    — Reduce allowed return on equity (in percentage points)

Data sources:
  - Revenue/customer data: EIA Form 861 (bundled + unbundled)
  - Rate structure: retail_rates_data_SCE.xlsx
  - Rate base & capital structure: SCE GRC filings
  - Financial inputs: utility_data_inputs.tex
"""

import pandas as pd
import numpy as np
from itertools import product

# ============================================================================
# SCE CONSTANTS (from GRC filings, EIA Form 861, utility_data_inputs.tex)
# ============================================================================

RESIDENTIAL_REVENUE = 7_745_773_000  # Total residential revenue (EIA 861 bundled+unbundled)
RESIDENTIAL_SALES_KWH = 27_414_312_000  # Total residential sales (EIA 861 bundled+unbundled)
TOTAL_UTILITY_REVENUE = 17_530_066_000  # All customer classes (revenue requirement)

CUSTOMERS = {
    'care': 1_353_981,      # CARE-eligible (enrolled)
    'non_care': 3_240_434,  # Non-CARE
    'total': 4_594_415,     # EIA 861 bundled+unbundled
}

RATE_BASE = 41_427_528_000  # Total utility rate base
EQUITY_SHARE = 0.52         # Equity portion of capital structure

# Residential share of system-wide costs
res_share = RESIDENTIAL_REVENUE / TOTAL_UTILITY_REVENUE

# Revenue components (system-wide values × residential share)
REVENUE_COMPONENTS = {
    'wildfire': 2_930_405_000 * res_share,       # Wildfire fund recovery
    'transmission': 1_116_093_000 * res_share,   # Transmission costs
    'distribution': 8_937_477_000 * res_share,   # Distribution costs
}
TD_COSTS = REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']

# ============================================================================
# ACTUAL SCE TARIFF RATES (from retail_rates_data_SCE.xlsx)
# ============================================================================

# TOU-D-4-9: Peak (4-9pm) / Off-peak (all other hours) for summer
#             Peak (4-9pm) / Mid-peak (9pm-8am) / Off-peak (8am-4pm) for winter
# Summer: June-October, Winter: November-May
# Note: Summer peak differs weekday ($0.627) vs weekend ($0.507)

# Weekday rates
ACTUAL_TOU_WEEKDAY = {
    'summer_peak': 0.627,
    'summer_offpeak': 0.387,
    'winter_peak': 0.557,
    'winter_midpeak': 0.417,
    'winter_offpeak': 0.377,
}

# Weekend rates
ACTUAL_TOU_WEEKEND = {
    'summer_peak': 0.507,
    'summer_offpeak': 0.387,
    'winter_peak': 0.557,
    'winter_midpeak': 0.417,
    'winter_offpeak': 0.377,
}

BASELINE_CREDIT_TOU = 0.10  # $/kWh baseline credit

# TOU-D-4-9-F: With fixed charges (for validation)
ACTUAL_TOU_F_WEEKDAY = {
    'summer_peak': 0.580,
    'summer_offpeak': 0.340,
    'winter_peak': 0.510,
    'winter_midpeak': 0.370,
    'winter_offpeak': 0.330,
}
ACTUAL_TOU_F_WEEKEND = {
    'summer_peak': 0.460,
    'summer_offpeak': 0.340,
    'winter_peak': 0.510,
    'winter_midpeak': 0.370,
    'winter_offpeak': 0.330,
}

ACTUAL_FIXED_LOW = 6.00       # $/month (CARE/low income)
ACTUAL_FIXED_MED = 24.15      # $/month (medium income)
ACTUAL_FIXED_HIGH = 24.15     # $/month (high income)
ACTUAL_FIXED_FERA = 12.00     # $/month (FERA)

# CARE discount on volumetric rates
CARE_VOLUMETRIC_DISCOUNT = 0.325  # 32.5% discount

# ============================================================================
# TOU CONSUMPTION WEIGHTS
# ============================================================================

# TODO: Calculate from SCE ResStock building profiles once available
# SCE TOU-D-4-9 period structure:
#   Summer (Jun-Oct): Peak 4-9pm, Off-peak all other hours
#   Winter (Nov-May): Peak 4-9pm, Mid-peak 9pm-8am, Off-peak 8am-4pm
TOU_WEIGHTS = {
    'summer_peak': 0.090,
    'summer_offpeak': 0.327,
    'winter_peak': 0.108,
    'winter_midpeak': 0.190,
    'winter_offpeak': 0.285,
}

# Weekday/weekend fraction (for splitting blended rates)
WEEKDAY_FRAC = 5 / 7
WEEKEND_FRAC = 2 / 7

# ============================================================================
# FIXED CHARGE INCOME GRADUATION
# ============================================================================

CARE_FIXED_RATIO = ACTUAL_FIXED_LOW / ACTUAL_FIXED_HIGH  # ≈ 0.248


def design_rate_sce(fixed_pct_td=0, remove_wildfire=False, roe_reduction=0,
                    care_fixed_ratio=CARE_FIXED_RATIO):
    """
    Design SCE rate scenario starting from actual TOU-D-4-9 tariff rates.

    At F0_WF0_ROE0 (status quo), rates equal the actual TOU-D-4-9 tariff.
    Policy levers reduce rates proportionally while shifting revenue to fixed
    charges, removing wildfire costs, or reducing ROE.

    Returns both weekday and weekend rate variants (summer peak differs).
    """
    R = RESIDENTIAL_REVENUE

    # Revenue adjustments
    wildfire_adj = REVENUE_COMPONENTS['wildfire'] if remove_wildfire else 0
    roe_adj = RATE_BASE * (roe_reduction / 100) * EQUITY_SHARE * res_share if roe_reduction > 0 else 0

    # Fixed charge revenue
    fixed_revenue = TD_COSTS * (fixed_pct_td / 100)

    # Income-graduated fixed charges
    total_weighted_customers = (CUSTOMERS['care'] * care_fixed_ratio) + CUSTOMERS['non_care']
    fixed_non_care = fixed_revenue / total_weighted_customers / 12 if fixed_pct_td > 0 else 0
    fixed_care = fixed_non_care * care_fixed_ratio

    # Volumetric revenue remaining
    vol_revenue = R - fixed_revenue - wildfire_adj - roe_adj

    # Scaling factor
    scale = vol_revenue / R

    # Scale actual TOU tariff rates (both weekday and weekend)
    new_weekday = {k: v * scale for k, v in ACTUAL_TOU_WEEKDAY.items()}
    new_weekend = {k: v * scale for k, v in ACTUAL_TOU_WEEKEND.items()}

    # Scale baseline credit proportionally
    baseline_credit = BASELINE_CREDIT_TOU * scale

    # Consumption-weighted average (using blended weekday/weekend for summer peak)
    blended_rates = {}
    for k in ACTUAL_TOU_WEEKDAY:
        blended_rates[k] = new_weekday[k] * WEEKDAY_FRAC + new_weekend[k] * WEEKEND_FRAC
    vol_avg = sum(blended_rates[p] * TOU_WEIGHTS[p] for p in blended_rates)

    # Total revenue after policy adjustments
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
        # Weekday rates
        'summer_peak_wd': round(new_weekday['summer_peak'], 5),
        'summer_offpeak_wd': round(new_weekday['summer_offpeak'], 5),
        'winter_peak_wd': round(new_weekday['winter_peak'], 5),
        'winter_midpeak_wd': round(new_weekday['winter_midpeak'], 5),
        'winter_offpeak_wd': round(new_weekday['winter_offpeak'], 5),
        # Weekend rates
        'summer_peak_we': round(new_weekend['summer_peak'], 5),
        'summer_offpeak_we': round(new_weekend['summer_offpeak'], 5),
        'winter_peak_we': round(new_weekend['winter_peak'], 5),
        'winter_midpeak_we': round(new_weekend['winter_midpeak'], 5),
        'winter_offpeak_we': round(new_weekend['winter_offpeak'], 5),
    }


def validate_against_actual():
    """Validate by comparing designed scenario against actual TOU-D-4-9-F tariff."""
    target_fixed = ACTUAL_FIXED_HIGH  # $24.15
    total_weighted = CUSTOMERS['care'] * CARE_FIXED_RATIO + CUSTOMERS['non_care']
    implied_fixed_revenue = target_fixed * total_weighted * 12
    implied_pct = implied_fixed_revenue / TD_COSTS * 100

    scenario = design_rate_sce(fixed_pct_td=implied_pct)

    print("=" * 80)
    print("VALIDATION: Scenario at implied fixed % vs actual TOU-D-4-9-F")
    print("=" * 80)
    print(f"Implied fixed charge % of T&D: {implied_pct:.1f}%")
    print()

    print("WEEKDAY rates:")
    print(f"{'Period':<20s} {'Actual':>12s} {'Modeled':>12s} {'Error':>10s}")
    print("-" * 60)
    for period in ACTUAL_TOU_WEEKDAY.keys():
        actual = ACTUAL_TOU_F_WEEKDAY[period]
        modeled = scenario[f'{period}_wd']
        error = (modeled - actual) / actual * 100
        print(f"  {period:<18s} ${actual:>10.5f}   ${modeled:>10.5f}   {error:>+7.2f}%")

    print("\nWEEKEND rates:")
    print(f"{'Period':<20s} {'Actual':>12s} {'Modeled':>12s} {'Error':>10s}")
    print("-" * 60)
    for period in ACTUAL_TOU_WEEKEND.keys():
        actual = ACTUAL_TOU_F_WEEKEND[period]
        modeled = scenario[f'{period}_we']
        error = (modeled - actual) / actual * 100
        print(f"  {period:<18s} ${actual:>10.5f}   ${modeled:>10.5f}   {error:>+7.2f}%")

    print(f"\n  {'Fixed (non-CARE)':<18s} ${ACTUAL_FIXED_HIGH:>10.5f}   "
          f"${scenario['Fixed_NonCARE']:>10.5f}")
    print(f"  {'Fixed (CARE)':<18s} ${ACTUAL_FIXED_LOW:>10.5f}   "
          f"${scenario['Fixed_CARE']:>10.5f}")
    print()

    return implied_pct


def generate_excel_rate_rows(scenarios_df, output_excel='retail_rates_data_SCE.xlsx'):
    """
    Generate rate rows in the format expected by corrected_bill_calc.py.

    For each scenario, creates weekday and weekend rows with the same TOU
    period definitions as TOU-D-4-9, but with scaled volumetric rates and
    appropriate fixed charges.
    """
    # Load existing Excel to get TOU period definitions
    existing = pd.read_excel(output_excel, sheet_name='retail_rates_oct32025')
    tou_weekday = existing[(existing['rate_type'] == 'TOU-D-4-9') & (existing['weekday'] == 'weekday')].iloc[0]
    tou_weekend = existing[(existing['rate_type'] == 'TOU-D-4-9') & (existing['weekday'] == 'weekend')].iloc[0]

    new_rows = []
    for _, sc in scenarios_df.iterrows():
        scenario_name = sc['Scenario']
        has_fixed = sc['Fixed_Pct_TD'] > 0

        for day_type, template, suffix in [('weekday', tou_weekday, '_wd'),
                                            ('weekend', tou_weekend, '_we')]:
            row = template.copy()
            row['rate_type'] = f"TOU-{scenario_name}"
            row['Fixed'] = 'Yes' if has_fixed else 'No'

            # Set TOU rates (tier 1 = tier 2 for non-tiered TOU)
            for tier in ['1', '2']:
                row[f'peak_rate_summer{tier}'] = sc[f'summer_peak{suffix}']
                row[f'midpeak_rate_summer{tier}'] = 0  # No summer midpeak in TOU-D-4-9
                row[f'offpeak_rate_summer{tier}'] = sc[f'summer_offpeak{suffix}']
                row[f'peak_rate_winter{tier}'] = sc[f'winter_peak{suffix}']
                row[f'midpeak_rate_winter{tier}'] = sc[f'winter_midpeak{suffix}']
                row[f'offpeak_rate_winter{tier}'] = sc[f'winter_offpeak{suffix}']

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

            # CARE discount (same as TOU-D-4-9)
            row['care_discount'] = -CARE_VOLUMETRIC_DISCOUNT

            row['weekday'] = day_type
            new_rows.append(row)

    return pd.DataFrame(new_rows)


def main():
    print("=" * 80)
    print("SCE RATE BUILDER — Actual Tariff-Based Scenarios")
    print("=" * 80)

    # Print baseline info
    blended = {}
    for k in ACTUAL_TOU_WEEKDAY:
        blended[k] = ACTUAL_TOU_WEEKDAY[k] * WEEKDAY_FRAC + ACTUAL_TOU_WEEKEND[k] * WEEKEND_FRAC
    weighted_avg = sum(blended[p] * TOU_WEIGHTS[p] for p in blended)

    print(f"\nBaseline TOU-D-4-9 weighted average: ${weighted_avg:.4f}/kWh")
    print(f"Revenue/sales average: ${RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH:.4f}/kWh")
    print(f"Ratio (tariff/rev-avg): {weighted_avg / (RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH):.2f}x")
    print(f"\nResidential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"T&D costs (residential): ${TD_COSTS/1e6:.1f}M ({TD_COSTS/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"Wildfire costs (residential): ${REVENUE_COMPONENTS['wildfire']/1e6:.1f}M "
          f"({REVENUE_COMPONENTS['wildfire']/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"ROE impact per 1pp: ${RATE_BASE * 0.01 * EQUITY_SHARE * res_share/1e6:.1f}M "
          f"({RATE_BASE * 0.01 * EQUITY_SHARE * res_share/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"\nCARE fixed charge ratio: {CARE_FIXED_RATIO:.3f} "
          f"(from actual TOU-D-4-9-F: ${ACTUAL_FIXED_LOW:.2f}/${ACTUAL_FIXED_HIGH:.2f})")

    # Validate against actual TOU-D-4-9-F
    print()
    implied_pct = validate_against_actual()

    # Generate scenarios
    fixed_percentages = [0, 25, 50, 75, 100]
    wildfire_options = [False, True]
    roe_reductions = [0, 0.5, 1.0, 1.5]

    scenarios = []
    for fixed_pct, wf, roe in product(fixed_percentages, wildfire_options, roe_reductions):
        scenarios.append(design_rate_sce(
            fixed_pct_td=fixed_pct,
            remove_wildfire=wf,
            roe_reduction=roe,
        ))

    df = pd.DataFrame(scenarios)

    # Save rate scenarios
    df.to_csv('rate_scenarios_sce.csv', index=False)
    print(f"\nGenerated {len(df)} rate scenarios → rate_scenarios_sce.csv")

    # Print summary table
    print("\n" + "=" * 80)
    print("RATE SCENARIO SUMMARY (weekday rates)")
    print("=" * 80)
    print(f"\n{'Scenario':<22s} {'Vol_Avg':>8s} {'S_Peak':>8s} {'S_Off':>8s} "
          f"{'W_Peak':>8s} {'Fixed$':>8s} {'Scale':>7s}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"  {row['Scenario']:<20s} ${row['Vol_Avg']:>6.4f} ${row['summer_peak_wd']:>6.4f} "
              f"${row['summer_offpeak_wd']:>6.4f} ${row['winter_peak_wd']:>6.4f} "
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
