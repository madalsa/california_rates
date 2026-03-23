"""
rate_designer_pge.py - Revenue-neutral retail rate designer for PGE

Sample-based revenue neutrality (R_sample approach):
  R_sample = weighted sum of actual E-TOU-C bills from ResStock building sample.
  Policy cost adjustments (wildfire, ROE) are expressed as SHARES of filed
  revenue and applied proportionally to R_sample.
  Baseline scenario F0_WF0_ROE0 produces scaling=1.0 (rates = E-TOU-C tariff).

Design parameters:
  - Fixed cost allocation (% of T&D costs to fixed charges)
  - Wildfire cost removal
  - ROE (Return on Equity) reduction — applied to equity share only

Data sources:
  - TOU weights: from ResStock building profiles (tou_weights_pge.csv)
  - Revenue/customer data: EIA Form 861, PGE filings
  - Rate base & capital structure: PGE 2024 GRC
"""

import sys
import pandas as pd
import numpy as np
from itertools import product


# =============================================================================
# PGE UTILITY DATA (from EIA and PGE filings)
# =============================================================================

RESIDENTIAL_REVENUE = 8_238_576_000      # Annual residential revenue ($, EIA 861 bundled+unbundled)
RESIDENTIAL_SALES_KWH = 25_987_213_000   # Annual residential sales (kWh, EIA 861 bundled+unbundled)
TOTAL_REVENUE = 20_341_236_000           # Total utility revenue ($)

CUSTOMERS = {
    'care': 1_371_555,
    'non_care': 3_675_906,
    'total': 5_047_461                   # EIA 861 bundled+unbundled
}

# Capital structure (PGE 2024 GRC)
RATE_BASE = 41_987_991_000   # Total rate base ($)
EQUITY_SHARE = 0.52          # Equity portion of capital structure
AUTHORIZED_ROE = 0.1028      # Current authorized ROE (10.28%)

# Residential share of total revenue
RES_SHARE = RESIDENTIAL_REVENUE / TOTAL_REVENUE

# Revenue components — absolute costs (from PGE revenue requirement filings)
REVENUE_COMPONENTS = {
    'wildfire': 5_404_304_000 * RES_SHARE,       # Wildfire fund recovery
    'transmission': 2_663_244_000 * RES_SHARE,   # Transmission costs
    'distribution': 8_741_084_000 * RES_SHARE,   # Distribution costs
}

# Cost component SHARES of filed residential revenue (used in R_sample approach)
WILDFIRE_SHARE = REVENUE_COMPONENTS['wildfire'] / RESIDENTIAL_REVENUE
TD_SHARE = (REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']) / RESIDENTIAL_REVENUE
ROE_SHARE_PER_PP = (RATE_BASE * EQUITY_SHARE * 0.01 * RES_SHARE) / RESIDENTIAL_REVENUE

# Current PGE E-TOU-C baseline rate structure ($/kWh)
# 4 periods: summer peak/offpeak, winter peak/offpeak (no midpeak)
BASELINE_TOU_RATES = {
    'summer_peak': 0.58943,
    'summer_offpeak': 0.46643,
    'winter_peak': 0.46460,
    'winter_offpeak': 0.43460,
}


def load_tou_weights(csv_path='tou_weights_pge.csv'):
    """
    Load TOU consumption weights computed from ResStock building profiles.

    These weights represent the fraction of total residential consumption
    occurring in each TOU period, based on ~22,700 simulated PGE buildings.
    """
    df = pd.read_csv(csv_path)
    weights = dict(zip(df['period'], df['weight']))

    # Validate weights sum to ~1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.001, f"TOU weights sum to {total}, expected ~1.0"

    return weights


def design_rate(fixed_pct_td=0, remove_wildfire=False, roe_reduction=0,
                care_fixed_ratio=0.248, tou_weights=None,
                r_sample=None, r_gross_vol=None,
                sample_n_care=None, sample_n_noncare=None):
    """
    Design a revenue-neutral rate scenario using R_sample approach.

    Parameters
    ----------
    fixed_pct_td : float
        Percentage of T&D costs allocated to fixed charges (0-100).
    remove_wildfire : bool
        If True, remove wildfire fund recovery costs from revenue requirement.
    roe_reduction : float
        Reduction in ROE in percentage points (e.g., 0.5 = 50 basis points).
    care_fixed_ratio : float
        CARE fixed charge as fraction of non-CARE (default 0.248 from actual E-TOU-C-F).
    tou_weights : dict
        TOU consumption weights from ResStock profiles.
    r_sample : float
        Weighted sample E-TOU-C revenue (R_sample). Required.
    r_gross_vol : float, optional
        Gross volumetric revenue (with CARE, without baseline credits).
    sample_n_care : int
        Number of CARE customers in sample (weighted).
    sample_n_noncare : int
        Number of non-CARE customers in sample (weighted).

    Returns
    -------
    dict
        Rate scenario with all components.
    """
    if r_sample is None:
        raise ValueError(
            "r_sample is required. Compute it as: "
            "sum(e_tou_c_bills) * BUILDING_WEIGHT from baseline bill output."
        )
    if tou_weights is None:
        tou_weights = load_tou_weights()

    # Use sample customer counts if provided, else fall back to utility counts
    n_care = sample_n_care if sample_n_care is not None else CUSTOMERS['care']
    n_noncare = sample_n_noncare if sample_n_noncare is not None else CUSTOMERS['non_care']

    # --- Step 1: Revenue target = R_sample minus policy adjustments (as shares) ---
    r_target = r_sample
    if remove_wildfire:
        r_target -= r_sample * WILDFIRE_SHARE
    if roe_reduction > 0:
        r_target -= r_sample * ROE_SHARE_PER_PP * roe_reduction

    # --- Step 2: Fixed charge revenue = share of T&D costs ---
    r_fixed = r_sample * TD_SHARE * (fixed_pct_td / 100)

    # Per-customer fixed charges (using sample counts)
    total_weighted_customers = n_noncare + care_fixed_ratio * n_care
    fixed_non_care = r_fixed / total_weighted_customers / 12  # monthly
    fixed_care = fixed_non_care * care_fixed_ratio

    # --- Step 3: Volumetric rates (TOU) ---
    r_vol = r_target - r_fixed

    # Scale E-TOU-C rates so that designed scenario bills match R_vol.
    scale_denom = r_gross_vol if r_gross_vol is not None else r_sample
    scaling = r_vol / scale_denom
    new_tou_rates = {k: v * scaling for k, v in BASELINE_TOU_RATES.items()}

    # Weighted average volumetric rate (for verification)
    vol_avg = sum(new_tou_rates[p] * tou_weights[p] for p in new_tou_rates)

    return {
        'Scenario': f'F{fixed_pct_td}_WF{int(remove_wildfire)}_ROE{roe_reduction}',
        'Fixed_Pct_TD': fixed_pct_td,
        'Remove_Wildfire': remove_wildfire,
        'ROE_Reduction': roe_reduction,
        'Fixed_CARE': fixed_care,
        'Fixed_NonCARE': fixed_non_care,
        'Scaling': scaling,
        'Vol_Avg': vol_avg,
        'Total_Revenue': r_target,
        **new_tou_rates
    }


def generate_all_scenarios(fixed_percentages=None, wildfire_options=None,
                           roe_reductions=None, output_csv=None,
                           r_sample=None, r_gross_vol=None,
                           sample_n_care=None, sample_n_noncare=None):
    """
    Generate all rate scenarios from parameter grid.

    Parameters
    ----------
    fixed_percentages : list
        Fixed cost allocation levels (default: [0, 25, 50, 75, 100]).
    wildfire_options : list
        Wildfire removal options (default: [False, True]).
    roe_reductions : list
        ROE reduction levels in pp (default: [0, 0.5, 1.0, 1.5]).
    output_csv : str
        Path to save output CSV.
    r_sample : float
        Weighted sample E-TOU-C revenue. Required.
    sample_n_care : int
        Number of CARE customers in sample (weighted count).
    sample_n_noncare : int
        Number of non-CARE customers in sample (weighted count).

    Returns
    -------
    pd.DataFrame
        All rate scenarios.
    """
    if fixed_percentages is None:
        fixed_percentages = [0, 25, 50, 75, 100]
    if wildfire_options is None:
        wildfire_options = [False, True]
    if roe_reductions is None:
        roe_reductions = [0, 0.5, 1.0, 1.5]
    if output_csv is None:
        output_csv = 'rate_scenarios_pge_fresh.csv'

    # Load ResStock TOU weights
    tou_weights = load_tou_weights()

    print("=" * 80)
    print("PGE RETAIL RATE DESIGNER — Sample-Based Revenue Neutrality")
    print("=" * 80)

    print(f"\nR_sample (weighted E-TOU-C revenue): ${r_sample/1e9:.4f}B")
    if r_gross_vol is not None:
        print(f"R_gross_vol (gross volumetric w/ CARE, no BL credits): ${r_gross_vol/1e9:.4f}B")
        print(f"  Baseline credit + fixed gap: ${(r_gross_vol - r_sample)/1e9:.4f}B")
    if sample_n_care is not None:
        print(f"Sample customers: {sample_n_care:,} CARE, {sample_n_noncare:,} non-CARE")

    print(f"\nCost shares from GRC filing:")
    print(f"  Wildfire:  {WILDFIRE_SHARE*100:.2f}%")
    print(f"  T&D:       {TD_SHARE*100:.2f}%")
    print(f"  ROE/pp:    {ROE_SHARE_PER_PP*100:.2f}%")

    print(f"\nResStock TOU consumption weights:")
    for period, weight in tou_weights.items():
        print(f"  {period:20s}: {weight * 100:5.2f}%")

    # Generate all scenarios
    scenarios = []
    for fixed_pct, wf, roe in product(fixed_percentages, wildfire_options, roe_reductions):
        scenarios.append(design_rate(
            fixed_pct_td=fixed_pct,
            remove_wildfire=wf,
            roe_reduction=roe,
            tou_weights=tou_weights,
            r_sample=r_sample,
            r_gross_vol=r_gross_vol,
            sample_n_care=sample_n_care,
            sample_n_noncare=sample_n_noncare,
        ))

    df = pd.DataFrame(scenarios).round(4)
    df.to_csv(output_csv, index=False)

    print(f"\nGenerated {len(df)} rate scenarios")
    print(f"  Fixed charge levels: {fixed_percentages}")
    print(f"  Wildfire options: {wildfire_options}")
    print(f"  ROE reductions: {roe_reductions} pp")

    # Revenue neutrality check
    print("\n" + "=" * 80)
    print("REVENUE NEUTRALITY CHECK")
    print("=" * 80)

    baseline = df[(df['Remove_Wildfire'] == False) & (df['ROE_Reduction'] == 0)]
    print(f"\nBaseline scenarios (same R_target = R_sample = ${r_sample/1e9:.4f}B):")
    for _, row in baseline.iterrows():
        print(f"  {row['Scenario']:15s}: scaling={row['Scaling']:.4f}  "
              f"FC_nonCARE=${row['Fixed_NonCARE']:.2f}/mo  "
              f"Vol avg=${row['Vol_Avg']:.4f}")

    expected_scaling = r_sample / r_gross_vol if r_gross_vol else 1.0
    print(f"\nF0_WF0_ROE0 scaling = {baseline.iloc[0]['Scaling']:.4f} "
          f"(expected {expected_scaling:.4f}; <1.0 accounts for E-TOU-C baseline credits)")

    # Show policy effects
    print(f"\nPolicy effects on revenue target:")
    for _, row in df[df['Fixed_Pct_TD'] == 0].iterrows():
        pct_reduction = (1 - row['Total_Revenue'] / r_sample) * 100
        print(f"  {row['Scenario']:20s}: ${row['Total_Revenue']/1e9:.4f}B  "
              f"({pct_reduction:+.2f}% vs baseline)")

    print(f"\nSaved to: {output_csv}")

    return df


if __name__ == "__main__":
    # Standalone usage: load R_sample from baseline bills if available
    import os
    bills_file = 'baseline_bills_pge_fresh.csv'
    if not os.path.exists(bills_file):
        bills_file = 'post_adoption_bills_pge.csv'
    if os.path.exists(bills_file):
        bills_df = pd.read_csv(bills_file)
        building_weight = 252.3
        r_sample = bills_df['e_tou_c_bill'].dropna().sum() * building_weight
        n_care = int((bills_df['is_care'] == True).sum() * building_weight)
        n_noncare = int((bills_df['is_care'] == False).sum() * building_weight)
        print(f"Loaded R_sample from {bills_file}: ${r_sample/1e9:.4f}B")
        df = generate_all_scenarios(
            r_sample=r_sample,
            sample_n_care=n_care,
            sample_n_noncare=n_noncare,
        )
    else:
        print(f"ERROR: No bills file found ({bills_file}).")
        print("Run the pipeline first to generate baseline E-TOU-C bills,")
        print("then re-run this script.")
        sys.exit(1)
