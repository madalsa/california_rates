"""
rate_designer.py - Revenue-neutral retail rate designer for SDGE

Uses ResStock simulated building consumption profiles (TOU weights)
to design revenue-neutral rate scenarios with:
  - Fixed cost allocation (% of T&D costs to fixed charges)
  - Wildfire cost removal
  - ROE (Return on Equity) reduction — correctly applied to equity share only

Capital structure note:
  ROE ≠ ROR (Rate of Return). ROE applies only to the equity portion of
  the rate base. SDGE's capital structure is ~52% equity / ~48% debt.
  Revenue impact of ROE change = Rate_Base × Equity_Share × ΔROE × Res_Share

Data sources:
  - TOU weights: from ResStock building profiles (tou_weights_sdge.csv)
  - Revenue/customer data: EIA, SDGE filings
  - Rate base & capital structure: SDGE 2024 GRC
"""

import pandas as pd
import numpy as np
from itertools import product


# =============================================================================
# SDGE UTILITY DATA (from EIA and SDGE filings)
# =============================================================================

RESIDENTIAL_REVENUE = 1_561_695_600     # Annual residential revenue ($)
RESIDENTIAL_SALES_KWH = 4_810_000_000   # Annual residential sales (kWh)
TOTAL_REVENUE = 4_233_072_000           # Total utility revenue ($)

CUSTOMERS = {
    'care': 372_135,
    'non_care': 951_477,
    'total': 1_323_612
}

# Capital structure (SDGE 2024 GRC)
RATE_BASE = 13_590_538_000   # Total rate base ($)
EQUITY_SHARE = 0.52          # Equity portion of capital structure
AUTHORIZED_ROE = 0.1022      # Current authorized ROE (10.22%)

# Residential share of total revenue
RES_SHARE = RESIDENTIAL_REVENUE / TOTAL_REVENUE

# Revenue components (from SDGE revenue requirement filings)
REVENUE_COMPONENTS = {
    'wildfire': 413_873_000 * RES_SHARE,       # Wildfire fund recovery
    'transmission': 685_245_000 * RES_SHARE,   # Transmission costs
    'distribution': 1_722_187_000 * RES_SHARE, # Distribution costs
}

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
                care_fixed_ratio=0.4, tou_weights=None):
    """
    Design a revenue-neutral rate scenario.

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

    Returns
    -------
    dict
        Rate scenario with all components.
    """
    if tou_weights is None:
        tou_weights = load_tou_weights()

    # --- Step 1: Calculate target revenue requirement ---
    revenue = RESIDENTIAL_REVENUE

    if remove_wildfire:
        revenue -= REVENUE_COMPONENTS['wildfire']

    if roe_reduction > 0:
        # ROE reduction applies only to EQUITY portion of rate base
        # Revenue impact = Rate_Base × Equity_Share × ΔROE × Res_Share
        roe_revenue_impact = RATE_BASE * EQUITY_SHARE * (roe_reduction / 100) * RES_SHARE
        revenue -= roe_revenue_impact

    # --- Step 2: Allocate T&D costs to fixed charges ---
    td_costs = REVENUE_COMPONENTS['transmission'] + REVENUE_COMPONENTS['distribution']
    fixed_revenue = td_costs * (fixed_pct_td / 100)

    # Income-graduated fixed charges (CARE pays care_fixed_ratio × non-CARE)
    total_weighted_customers = (CUSTOMERS['care'] * care_fixed_ratio) + CUSTOMERS['non_care']
    fixed_non_care = fixed_revenue / total_weighted_customers / 12  # monthly
    fixed_care = fixed_non_care * care_fixed_ratio

    # --- Step 3: Design volumetric rates (TOU) ---
    volumetric_revenue = revenue - fixed_revenue
    target_avg_rate = volumetric_revenue / RESIDENTIAL_SALES_KWH

    # Scale baseline TOU rates to hit target average (consumption-weighted)
    current_weighted_avg = sum(
        BASELINE_TOU_RATES[period] * tou_weights[period]
        for period in BASELINE_TOU_RATES
    )
    scaling_factor = target_avg_rate / current_weighted_avg

    new_tou_rates = {k: v * scaling_factor for k, v in BASELINE_TOU_RATES.items()}

    # --- Step 4: Verify revenue neutrality ---
    verification_avg = sum(
        new_tou_rates[period] * tou_weights[period]
        for period in new_tou_rates
    )

    # Revenue check
    reconstructed_revenue = (
        verification_avg * RESIDENTIAL_SALES_KWH +
        (fixed_care * CUSTOMERS['care'] + fixed_non_care * CUSTOMERS['non_care']) * 12
    )

    return {
        'Scenario': f'F{fixed_pct_td}_WF{int(remove_wildfire)}_ROE{roe_reduction}',
        'Fixed_Pct_TD': fixed_pct_td,
        'Remove_Wildfire': remove_wildfire,
        'ROE_Reduction': roe_reduction,
        'Fixed_CARE': fixed_care,
        'Fixed_NonCARE': fixed_non_care,
        'Vol_Avg': target_avg_rate,
        'Vol_Avg_Check': verification_avg,
        'Total_Revenue': revenue,
        'Reconstructed_Revenue': reconstructed_revenue,
        **new_tou_rates
    }


def generate_all_scenarios(fixed_percentages=None, wildfire_options=None,
                           roe_reductions=None, output_csv=None):
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
    print("SDGE RETAIL RATE DESIGNER")
    print("Revenue-neutral scenarios using ResStock consumption profiles")
    print("=" * 80)

    print("\nResStock TOU consumption weights:")
    for period, weight in tou_weights.items():
        print(f"  {period:20s}: {weight * 100:5.2f}%")

    print(f"\nCapital structure: {EQUITY_SHARE*100:.0f}% equity / "
          f"{(1-EQUITY_SHARE)*100:.0f}% debt")
    print(f"Rate base: ${RATE_BASE/1e9:.2f}B")
    print(f"Baseline residential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")

    # Generate all scenarios
    scenarios = []
    for fixed_pct, wf, roe in product(fixed_percentages, wildfire_options, roe_reductions):
        scenarios.append(design_rate(
            fixed_pct_td=fixed_pct,
            remove_wildfire=wf,
            roe_reduction=roe,
            tou_weights=tou_weights
        ))

    df = pd.DataFrame(scenarios).round(4)

    # Drop Reconstructed_Revenue from output (just for internal validation)
    df_out = df.drop(columns=['Reconstructed_Revenue'])
    df_out.to_csv(output_csv, index=False)

    print(f"\nGenerated {len(df)} rate scenarios")
    print(f"  Fixed charge levels: {fixed_percentages}")
    print(f"  Wildfire options: {wildfire_options}")
    print(f"  ROE reductions: {roe_reductions} pp")

    # Revenue neutrality check
    print("\n" + "=" * 80)
    print("REVENUE NEUTRALITY CHECK")
    print("=" * 80)

    baseline = df[(df['Remove_Wildfire'] == False) & (df['ROE_Reduction'] == 0)]
    print("\nBaseline scenarios (should have identical total revenue):")
    for _, row in baseline.iterrows():
        rev_error = abs(row['Reconstructed_Revenue'] - row['Total_Revenue'])
        print(f"  {row['Scenario']:15s}: ${row['Total_Revenue']/1e9:.4f}B  "
              f"Vol avg: ${row['Vol_Avg']:.4f}  "
              f"Fixed non-CARE: ${row['Fixed_NonCARE']:.2f}/mo  "
              f"Rev error: ${rev_error:.2f}")

    cv = baseline['Total_Revenue'].std() / baseline['Total_Revenue'].mean() * 100
    print(f"\nCoefficient of Variation: {cv:.6f}% (should be ~0%)")

    # Show ROE impact
    print("\n" + "=" * 80)
    print("ROE REDUCTION IMPACT (corrected: equity share only)")
    print("=" * 80)

    f0_scenarios = df[(df['Fixed_Pct_TD'] == 0) & (df['Remove_Wildfire'] == False)]
    base_rev = f0_scenarios[f0_scenarios['ROE_Reduction'] == 0]['Total_Revenue'].values[0]

    for _, row in f0_scenarios.iterrows():
        rev_change = row['Total_Revenue'] - base_rev
        pct_change = rev_change / base_rev * 100
        print(f"  ROE -{row['ROE_Reduction']:.1f}pp: "
              f"Revenue ${row['Total_Revenue']/1e9:.4f}B  "
              f"({pct_change:+.2f}%, ${rev_change/1e6:+.1f}M)")

    # Summary table
    print("\n" + "=" * 80)
    print("SCENARIO SUMMARY")
    print("=" * 80)
    print(f"\n{'Scenario':<20} {'Fixed NC':<12} {'Fixed CARE':<12} "
          f"{'Vol Avg':<10} {'Revenue':<15}")
    print("-" * 70)
    for _, row in df_out.iterrows():
        print(f"{row['Scenario']:<20} ${row['Fixed_NonCARE']:<10.2f} "
              f"${row['Fixed_CARE']:<10.2f} ${row['Vol_Avg']:<8.4f} "
              f"${row['Total_Revenue']/1e9:.4f}B")

    print(f"\nSaved to: {output_csv}")

    return df_out


if __name__ == "__main__":
    df = generate_all_scenarios()
