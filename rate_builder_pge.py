"""
rate_builder_pge.py — Build PGE rate scenarios from actual E-TOU-C tariff rates.

Mirrors the SDGE rate builder (rate_builder_sdge.py) structure:
  Starts from ACTUAL PGE E-TOU-C tariff rates and applies proportional
  reductions for each policy lever. At F0_WF0_ROE0 (status quo), the rates
  match what customers actually see on their bills.

Policy levers:
  1. Fixed charge %  — Fraction of T&D costs moved from volumetric to fixed charges
  2. Wildfire removal — Remove wildfire cost component from rates
  3. ROE reduction    — Reduce allowed return on equity (in percentage points)

Data sources:
  - Revenue/customer data: EIA Form 861 (bundled + unbundled)
  - Rate structure: retail_rates_data_oct32025.xlsx
  - Rate base & capital structure: PGE GRC filings
"""

import pandas as pd
import numpy as np
from itertools import product

# ============================================================================
# PGE CONSTANTS (from GRC filings, EIA Form 861)
# ============================================================================

RESIDENTIAL_REVENUE = 8_238_576_000  # Total residential revenue (EIA 861 bundled+unbundled)
RESIDENTIAL_SALES_KWH = 25_987_213_000  # Total residential sales (EIA 861 bundled+unbundled)
TOTAL_UTILITY_REVENUE = 20_340_000_000  # All customer classes (TODO: update from EIA 861)

CUSTOMERS = {
    'care': 1_500_000,      # CARE-eligible (TODO: update from PGE filing)
    'non_care': 3_547_461,  # Non-CARE (TODO: update from PGE filing)
    'total': 5_047_461,     # EIA 861 bundled+unbundled
}

RATE_BASE = 40_000_000_000  # Total utility rate base (TODO: update from PGE GRC)
EQUITY_SHARE = 0.52         # Equity portion of capital structure (TODO: update from PGE GRC)

# Residential share of system-wide costs
res_share = RESIDENTIAL_REVENUE / TOTAL_UTILITY_REVENUE

# Revenue components (system-wide values × residential share)
# TODO: Update wildfire, transmission, distribution from PGE GRC filings
REVENUE_COMPONENTS = {
    'wildfire': 1_000_000_000 * res_share,       # Wildfire fund recovery (TODO)
    'transmission': 2_000_000_000 * res_share,   # Transmission costs (TODO)
    'distribution': 4_000_000_000 * res_share,   # Distribution costs (TODO)
}
TD_COSTS = REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']

# ============================================================================
# ACTUAL PGE TARIFF RATES (from retail_rates_data_oct32025.xlsx)
# ============================================================================

# E-TOU-C: Peak (4-9pm) / Off-peak (all other hours), same weekday/weekend
# Summer: June-October, Winter: November-May
ACTUAL_TOU = {
    'summer_peak': 0.52268,
    'summer_offpeak': 0.39968,
    'winter_peak': 0.39785,
    'winter_offpeak': 0.36785,
}
BASELINE_CREDIT_TOU = 0.1031  # $/kWh baseline credit

# E-TOU-C-F: With fixed charges (for validation)
ACTUAL_TOU_F = {
    'summer_peak': 0.47568,
    'summer_offpeak': 0.35268,
    'winter_peak': 0.35085,
    'winter_offpeak': 0.32085,
}
ACTUAL_FIXED_LOW = 6.00       # $/month (CARE/low income)
ACTUAL_FIXED_MED = 24.15      # $/month (medium income)
ACTUAL_FIXED_HIGH = 24.15     # $/month (high income)

# CARE discount on volumetric rates
CARE_VOLUMETRIC_DISCOUNT = 0.34965  # ~35% discount

# ============================================================================
# TOU CONSUMPTION WEIGHTS
# ============================================================================

# TODO: Calculate from PGE ResStock building profiles once available
# For now, use approximate weights based on PGE's E-TOU-C period structure:
#   Summer (Jun-Oct): Peak 4-9pm, Off-peak all other hours
#   Winter (Nov-May): Peak 4-9pm, Off-peak all other hours
TOU_WEIGHTS = {
    'summer_peak': 0.090,      # Summer 4-9pm (~5/24 of summer hours, load-weighted)
    'summer_offpeak': 0.327,   # Summer other hours
    'winter_peak': 0.108,      # Winter 4-9pm
    'winter_offpeak': 0.475,   # Winter other hours
}

# ============================================================================
# FIXED CHARGE INCOME GRADUATION
# ============================================================================

CARE_FIXED_RATIO = ACTUAL_FIXED_LOW / ACTUAL_FIXED_HIGH  # ≈ 0.248


def design_rate_pge(fixed_pct_td=0, remove_wildfire=False, roe_reduction=0,
                    care_fixed_ratio=CARE_FIXED_RATIO):
    """
    Design PGE rate scenario starting from actual E-TOU-C tariff rates.

    At F0_WF0_ROE0 (status quo), rates equal the actual E-TOU-C tariff.
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

    # Scale actual TOU tariff rates
    new_tou_rates = {k: v * scale for k, v in ACTUAL_TOU.items()}

    # Scale baseline credit proportionally
    baseline_credit = BASELINE_CREDIT_TOU * scale

    # Consumption-weighted average of new tariff rates
    vol_avg = sum(new_tou_rates[p] * TOU_WEIGHTS[p] for p in new_tou_rates)

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
        'summer_peak': round(new_tou_rates['summer_peak'], 5),
        'summer_offpeak': round(new_tou_rates['summer_offpeak'], 5),
        'winter_peak': round(new_tou_rates['winter_peak'], 5),
        'winter_offpeak': round(new_tou_rates['winter_offpeak'], 5),
    }


def validate_against_actual():
    """Validate by comparing designed scenario against actual E-TOU-C-F tariff."""
    target_fixed = ACTUAL_FIXED_HIGH  # $24.15
    total_weighted = CUSTOMERS['care'] * CARE_FIXED_RATIO + CUSTOMERS['non_care']
    implied_fixed_revenue = target_fixed * total_weighted * 12
    implied_pct = implied_fixed_revenue / TD_COSTS * 100

    scenario = design_rate_pge(fixed_pct_td=implied_pct)

    print("=" * 80)
    print("VALIDATION: Scenario at implied fixed % vs actual E-TOU-C-F")
    print("=" * 80)
    print(f"Implied fixed charge % of T&D: {implied_pct:.1f}%")
    print()
    print(f"{'Period':<20s} {'Actual E-TOU-C-F':>15s} {'Modeled':>15s} {'Error':>10s}")
    print("-" * 60)
    for period in ACTUAL_TOU.keys():
        actual = ACTUAL_TOU_F[period]
        modeled = scenario[period]
        error = (modeled - actual) / actual * 100
        print(f"  {period:<18s} ${actual:>12.5f}   ${modeled:>12.5f}   {error:>+7.2f}%")

    print(f"\n  {'Fixed (non-CARE)':<18s} ${ACTUAL_FIXED_HIGH:>12.5f}   "
          f"${scenario['Fixed_NonCARE']:>12.5f}")
    print(f"  {'Fixed (CARE)':<18s} ${ACTUAL_FIXED_LOW:>12.5f}   "
          f"${scenario['Fixed_CARE']:>12.5f}")
    print()

    return implied_pct


def main():
    print("=" * 80)
    print("PGE RATE BUILDER — Actual Tariff-Based Scenarios")
    print("=" * 80)

    # Print baseline info
    weighted_avg = sum(ACTUAL_TOU[p] * TOU_WEIGHTS[p] for p in ACTUAL_TOU)
    print(f"\nBaseline E-TOU-C weighted average: ${weighted_avg:.4f}/kWh")
    print(f"Revenue/sales average: ${RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH:.4f}/kWh")
    print(f"Ratio (tariff/rev-avg): {weighted_avg / (RESIDENTIAL_REVENUE/RESIDENTIAL_SALES_KWH):.2f}x")
    print(f"\nResidential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"T&D costs (residential): ${TD_COSTS/1e6:.1f}M ({TD_COSTS/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"Wildfire costs (residential): ${REVENUE_COMPONENTS['wildfire']/1e6:.1f}M "
          f"({REVENUE_COMPONENTS['wildfire']/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"ROE impact per 1pp: ${RATE_BASE * 0.01 * EQUITY_SHARE * res_share/1e6:.1f}M "
          f"({RATE_BASE * 0.01 * EQUITY_SHARE * res_share/RESIDENTIAL_REVENUE*100:.1f}% of revenue)")
    print(f"\nCARE fixed charge ratio: {CARE_FIXED_RATIO:.3f} "
          f"(from actual E-TOU-C-F: ${ACTUAL_FIXED_LOW:.2f}/${ACTUAL_FIXED_HIGH:.2f})")

    # Validate against actual E-TOU-C-F
    print()
    implied_pct = validate_against_actual()

    # Generate scenarios
    fixed_percentages = [0, 25, 50, 75, 100]
    wildfire_options = [False, True]
    roe_reductions = [0, 0.5, 1.0, 1.5]

    scenarios = []
    for fixed_pct, wf, roe in product(fixed_percentages, wildfire_options, roe_reductions):
        scenarios.append(design_rate_pge(
            fixed_pct_td=fixed_pct,
            remove_wildfire=wf,
            roe_reduction=roe,
        ))

    df = pd.DataFrame(scenarios)

    # Save rate scenarios
    df.to_csv('rate_scenarios_pge.csv', index=False)
    print(f"\nGenerated {len(df)} rate scenarios → rate_scenarios_pge.csv")

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

    return df


if __name__ == '__main__':
    df = main()
