"""
rate_designer.py - Revenue-neutral retail rate designer for SDGE

Sample-based revenue neutrality (R_sample approach):
  R_sample = weighted sum of actual TOU-DR bills from ResStock building sample.
  Policy cost adjustments (wildfire, ROE) are expressed as SHARES of filed
  revenue and applied proportionally to R_sample.
  Baseline scenario F0_WF0_ROE0 produces scaling=1.0 (rates = TOU-DR tariff).

Design parameters:
  - Fixed cost allocation (% of T&D costs to fixed charges)
  - Wildfire cost removal
  - ROE (Return on Equity) reduction — applied to equity share only

Capital structure note:
  ROE ≠ ROR (Rate of Return). ROE applies only to the equity portion of
  the rate base. SDGE's capital structure is ~52% equity / ~48% debt.
  Revenue impact of ROE change = Rate_Base × Equity_Share × ΔROE × Res_Share

Data sources:
  - TOU weights: from ResStock building profiles (tou_weights_sdge.csv)
  - Revenue/customer data: EIA, SDGE filings
  - Rate base & capital structure: SDGE 2024 GRC
"""

import sys
import pandas as pd
import numpy as np
from itertools import product


# =============================================================================
# SDGE UTILITY DATA (from EIA and SDGE filings)
# =============================================================================

RESIDENTIAL_REVENUE = 1_561_695_600     # Annual residential revenue ($, EIA 861 bundled+unbundled)
RESIDENTIAL_SALES_KWH = 4_809_988_000   # Annual residential sales (kWh, EIA 861 bundled+unbundled)
TOTAL_REVENUE = 4_233_072_000           # Total utility revenue ($)

CUSTOMERS = {
    'care': 372_135,
    'non_care': 992_226,              # includes ~40k unbundled (CCA) customers
    'total': 1_364_361                # EIA 861 bundled+unbundled
}

# Capital structure (SDGE 2024 GRC)
RATE_BASE = 13_590_538_000   # Total rate base ($)
EQUITY_SHARE = 0.52          # Equity portion of capital structure
AUTHORIZED_ROE = 0.1022      # Current authorized ROE (10.22%)

# Residential share of total revenue
RES_SHARE = RESIDENTIAL_REVENUE / TOTAL_REVENUE

# Revenue components — absolute costs (from SDGE revenue requirement filings)
REVENUE_COMPONENTS = {
    'wildfire': 413_873_000 * RES_SHARE,       # Wildfire fund recovery
    'transmission': 685_245_000 * RES_SHARE,   # Transmission costs
    'distribution': 1_722_187_000 * RES_SHARE, # Distribution costs
}

# Cost component SHARES of filed residential revenue (used in R_sample approach)
WILDFIRE_SHARE = REVENUE_COMPONENTS['wildfire'] / RESIDENTIAL_REVENUE
TD_SHARE = (REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']) / RESIDENTIAL_REVENUE
ROE_SHARE_PER_PP = (RATE_BASE * EQUITY_SHARE * 0.01 * RES_SHARE) / RESIDENTIAL_REVENUE

# Current SDGE TOU-DR baseline rate structure ($/kWh)
BASELINE_TOU_RATES = {
    'summer_peak': 0.60,
    'summer_midpeak': 0.527,
    'summer_offpeak': 0.45,
    'winter_peak': 0.58155,
    'winter_midpeak': 0.51899,
    'winter_offpeak': 0.50084
}


def load_tou_weights(csv_path='tou_weights_sdge.csv'):
    """
    Load TOU consumption weights computed from ResStock building profiles.

    These weights represent the fraction of total residential consumption
    occurring in each TOU period, based on ~4,300 simulated SDGE buildings.
    """
    df = pd.read_csv(csv_path)
    weights = dict(zip(df['period'], df['weight']))

    # Validate weights sum to ~1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.001, f"TOU weights sum to {total}, expected ~1.0"

    return weights


def design_rate(fixed_pct_td=0, remove_wildfire=False, roe_reduction=0,
                care_fixed_ratio=0.4, tou_weights=None,
                r_sample=None, r_gross_vol=None,
                sample_n_care=None, sample_n_noncare=None):
    """
    Design a revenue-neutral rate scenario using R_sample approach.

    R_sample is the weighted sum of actual TOU-DR bills from the building
    sample. Policy cost adjustments (wildfire, ROE) are applied as SHARES
    of filed revenue, proportionally to R_sample.

    R_gross_vol is the gross volumetric revenue from the sample (with CARE
    discounts applied, but WITHOUT baseline credits or fixed charges). This
    is used as the scaling denominator because designed scenario bills are
    computed as pure volumetric × rate (no baseline credits). Using R_sample
    as the denominator would produce rates that are too high since R_sample
    is lower than R_gross_vol (baseline credits reduce TOU-DR bills).

    Parameters
    ----------
    fixed_pct_td : float
        Percentage of T&D costs allocated to fixed charges (0-100).
    remove_wildfire : bool
        If True, remove wildfire fund recovery costs from revenue requirement.
    roe_reduction : float
        Reduction in ROE in percentage points (e.g., 0.5 = 50 basis points).
        Applied only to equity portion of rate base.
    care_fixed_ratio : float
        CARE fixed charge as fraction of non-CARE (default 0.4 = 40%).
    tou_weights : dict
        TOU consumption weights from ResStock profiles.
    r_sample : float
        Weighted sample TOU-DR revenue (R_sample). Required.
    r_gross_vol : float, optional
        Gross volumetric revenue (with CARE, without baseline credits).
        Used as scaling denominator. If None, falls back to r_sample.
    sample_n_care : int
        Number of CARE buildings in sample (weighted by BUILDING_WEIGHT
        to get population estimate for fixed charge allocation).
    sample_n_noncare : int
        Number of non-CARE buildings in sample.

    Returns
    -------
    dict
        Rate scenario with all components.
    """
    if r_sample is None:
        raise ValueError(
            "r_sample is required. Compute it as: "
            "sum(tou_dr_bills) * BUILDING_WEIGHT from baseline bill output."
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

    # Scale TOU-DR rates so that designed scenario bills match R_vol.
    # Designed bills = sum(load × rate × scaling) × care_factor (no baseline credits).
    # The denominator must be the gross volumetric revenue (with CARE, without
    # baseline credits) so that: scaling × R_gross_vol + R_fixed = R_target.
    # If r_gross_vol is not provided, fall back to r_sample (legacy behavior).
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
        Path to save output CSV (default: rate_scenarios_all_corrected.csv).
    r_sample : float
        Weighted sample TOU-DR revenue. Required.
    sample_n_care : int
        Number of CARE buildings in sample (weighted count).
    sample_n_noncare : int
        Number of non-CARE buildings in sample (weighted count).

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
        output_csv = 'rate_scenarios_all_corrected.csv'

    # Load ResStock TOU weights
    tou_weights = load_tou_weights()

    print("=" * 80)
    print("SDGE RETAIL RATE DESIGNER — Sample-Based Revenue Neutrality")
    print("=" * 80)

    print(f"\nR_sample (weighted TOU-DR revenue): ${r_sample/1e9:.4f}B")
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
          f"(expected {expected_scaling:.4f}; <1.0 accounts for TOU-DR baseline credits)")

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
    bills_file = 'baseline_bills_sdge_fresh.csv'
    if not os.path.exists(bills_file):
        bills_file = 'post_adoption_bills_sdge.csv'
    if os.path.exists(bills_file):
        bills_df = pd.read_csv(bills_file)
        building_weight = 252.3
        r_sample = bills_df['tou_dr_bill'].dropna().sum() * building_weight
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
        print("Run the pipeline first to generate baseline TOU-DR bills,")
        print("then re-run this script.")
        sys.exit(1)
