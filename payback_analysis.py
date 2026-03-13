"""
Simple payback period analysis for technology adoption scenarios.

Scenarios:
  S1: EV only (add EV charging to baseline load)
  S2: PV + Storage (add solar + battery to baseline load)
  S3: Full electrification bundle (HP HVAC + HP WH + induction + dryer + envelope)
      + PV (sized on pre-electrification load) + Storage + EV
  S4: Same as S3 but PV sized on post-electrification load

Payback = Net upfront cost / Annual net savings
"""

import pandas as pd
import numpy as np

# =============================================================================
# COST ASSUMPTIONS (2024-2025 dollars, SDGE / California)
# =============================================================================

# Solar PV (installed, before incentives)
SOLAR_COST_PER_W = 3.50          # $/W installed
ITC_RATE = 0.30                   # 30% federal investment tax credit (solar + storage)

# Battery storage (Tesla Powerwall-class, 13.5 kWh)
BATTERY_COST = 12_000             # $ installed (before ITC)

# EV
EV_PREMIUM = 10_000              # $ price premium over comparable ICE vehicle
ANNUAL_MILES = 12_000            # miles/year (average US driver)
GASOLINE_PRICE = 5.50            # $/gallon (California average)
ICE_MPG = 25                     # miles per gallon (average ICE)

# Full electrification bundle (Upgrade 11)
# HP HVAC ($8-12k) + HP Water Heater ($3-5k) + Induction ($2-3k)
# + Electric Dryer ($1-2k) + Envelope improvements ($5-8k)
ELECTRIFICATION_BUNDLE = 25_000  # $ total installed cost

# Natural gas (SDGE territory, from ResStock metadata)
GAS_FIXED_CHARGE_MONTHLY = 11.25       # $/month
GAS_MARGINAL_RATE = 1.2515             # $/therm
AVG_GAS_THERMS_YEAR = 250              # therms/year (SDGE average)
# Default annual gas bill if building-specific data unavailable
DEFAULT_GAS_BILL = GAS_FIXED_CHARGE_MONTHLY * 12 + AVG_GAS_THERMS_YEAR * GAS_MARGINAL_RATE

# =============================================================================
# SCENARIO COST FUNCTIONS
# =============================================================================

def s1_costs():
    """S1: EV only. Returns (upfront_cost, annual_fuel_savings)."""
    upfront = EV_PREMIUM
    # Gasoline savings (fixed 12k miles/year assumption)
    gallons_saved = ANNUAL_MILES / ICE_MPG
    gas_savings = gallons_saved * GASOLINE_PRICE
    return upfront, gas_savings


def s2_costs(pv_kw):
    """S2: PV + Storage. Returns upfront cost (after ITC)."""
    solar_cost = pv_kw * SOLAR_COST_PER_W * 1000  # kW -> W
    total_before_itc = solar_cost + BATTERY_COST
    total_after_itc = total_before_itc * (1 - ITC_RATE)
    return total_after_itc


def s3s4_costs(pv_kw):
    """S3/S4: Full electrification + PV + Storage + EV. Returns (upfront, gas_savings, fuel_savings)."""
    # Solar + storage (ITC eligible)
    solar_storage_before = pv_kw * SOLAR_COST_PER_W * 1000 + BATTERY_COST
    solar_storage_after = solar_storage_before * (1 - ITC_RATE)

    # EV
    ev_cost = EV_PREMIUM

    # Electrification bundle (not ITC eligible, though some rebates may apply)
    electrification = ELECTRIFICATION_BUNDLE

    upfront = solar_storage_after + ev_cost + electrification

    # Annual savings from dropping gas entirely
    gas_bill_savings = DEFAULT_GAS_BILL

    # Gasoline savings (fixed 12k miles/year)
    fuel_savings = (ANNUAL_MILES / ICE_MPG) * GASOLINE_PRICE

    return upfront, gas_bill_savings, fuel_savings


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def compute_payback(post_df, rate_scenario='F0_WF0_ROE0'):
    """
    Compute simple payback periods for each building and scenario.

    Parameters
    ----------
    post_df : DataFrame
        post_adoption_bills_sdge.csv
    rate_scenario : str
        Which rate scenario to use for bill comparison (default: status quo)

    Returns
    -------
    DataFrame with payback results
    """
    baseline_col = f'{rate_scenario}_bill'
    s1_col = f'{rate_scenario}_bill_s1_ev'
    s2_col = f'{rate_scenario}_bill_s2_pv_stor'
    s3_col = f'{rate_scenario}_bill_s3_full_pre'
    s4_col = f'{rate_scenario}_bill_s4_full_post'

    results = []

    for _, row in post_df.iterrows():
        bid = row['building_id']
        income = row['income']
        baseline_bill = row[baseline_col]
        r = {'building_id': bid, 'income': income, 'annual_kwh': row['annual_kwh'],
             'baseline_bill': baseline_bill, 'rate_scenario': rate_scenario}

        # --- S1: EV only ---
        if pd.notna(row.get(s1_col)):
            s1_bill = row[s1_col]
            upfront, fuel_savings = s1_costs()
            bill_delta = s1_bill - baseline_bill  # positive = bill increase
            annual_savings = fuel_savings - bill_delta
            r['s1_upfront'] = upfront
            r['s1_bill'] = s1_bill
            r['s1_bill_delta'] = bill_delta
            r['s1_fuel_savings'] = fuel_savings
            r['s1_net_annual_savings'] = annual_savings
            r['s1_payback'] = upfront / annual_savings if annual_savings > 0 else np.nan

        # --- S2: PV + Storage ---
        if pd.notna(row.get(s2_col)) and pd.notna(row.get('pv_size_kw_s2')):
            s2_bill = row[s2_col]
            upfront = s2_costs(row['pv_size_kw_s2'])
            bill_savings = baseline_bill - s2_bill
            r['s2_upfront'] = upfront
            r['s2_bill'] = s2_bill
            r['s2_bill_savings'] = bill_savings
            r['s2_net_annual_savings'] = bill_savings
            r['s2_payback'] = upfront / bill_savings if bill_savings > 0 else np.nan

        # --- S3: Full electrification (pre-PV sizing) ---
        if pd.notna(row.get(s3_col)) and pd.notna(row.get('pv_size_kw_s3')):
            s3_bill = row[s3_col]
            upfront, gas_savings, fuel_savings = s3s4_costs(row['pv_size_kw_s3'])
            bill_delta = s3_bill - baseline_bill
            annual_savings = gas_savings + fuel_savings - bill_delta
            r['s3_upfront'] = upfront
            r['s3_bill'] = s3_bill
            r['s3_bill_delta'] = bill_delta
            r['s3_gas_savings'] = gas_savings
            r['s3_fuel_savings'] = fuel_savings
            r['s3_net_annual_savings'] = annual_savings
            r['s3_payback'] = upfront / annual_savings if annual_savings > 0 else np.nan

        # --- S4: Full electrification (post-PV sizing) ---
        if pd.notna(row.get(s4_col)) and pd.notna(row.get('pv_size_kw_s4')):
            s4_bill = row[s4_col]
            upfront, gas_savings, fuel_savings = s3s4_costs(row['pv_size_kw_s4'])
            bill_delta = s4_bill - baseline_bill
            annual_savings = gas_savings + fuel_savings - bill_delta
            r['s4_upfront'] = upfront
            r['s4_bill'] = s4_bill
            r['s4_bill_delta'] = bill_delta
            r['s4_gas_savings'] = gas_savings
            r['s4_fuel_savings'] = fuel_savings
            r['s4_net_annual_savings'] = annual_savings
            r['s4_payback'] = upfront / annual_savings if annual_savings > 0 else np.nan

        results.append(r)

    return pd.DataFrame(results)


def summarize_payback(payback_df):
    """Print summary statistics of payback periods by scenario and income."""

    scenarios = [
        ('s1', 'S1: EV Only'),
        ('s2', 'S2: PV + Storage'),
        ('s3', 'S3: Full Electrification (pre-PV sizing)'),
        ('s4', 'S4: Full Electrification (post-PV sizing)'),
    ]

    print("=" * 90)
    print("SIMPLE PAYBACK PERIOD ANALYSIS")
    print("=" * 90)
    print(f"\nRate scenario: {payback_df['rate_scenario'].iloc[0]}")

    for prefix, label in scenarios:
        pb_col = f'{prefix}_payback'
        sav_col = f'{prefix}_net_annual_savings'
        up_col = f'{prefix}_upfront'

        valid = payback_df[payback_df[pb_col].notna()].copy() if pb_col in payback_df.columns else pd.DataFrame()

        if valid.empty:
            continue

        print(f"\n{'─' * 90}")
        print(f"  {label}  (n={len(valid)} buildings)")
        print(f"{'─' * 90}")

        # Overall
        print(f"  {'Metric':<35s} {'Mean':>10s} {'Median':>10s} {'P25':>10s} {'P75':>10s}")
        for col, lbl in [(up_col, 'Upfront cost ($)'),
                         (sav_col, 'Net annual savings ($)'),
                         (pb_col, 'Simple payback (years)')]:
            vals = valid[col].dropna()
            # Cap payback at 50 years for stats
            if col == pb_col:
                vals = vals.clip(upper=50)
            print(f"  {lbl:<35s} {vals.mean():>10,.0f} {vals.median():>10,.0f} "
                  f"{vals.quantile(0.25):>10,.0f} {vals.quantile(0.75):>10,.0f}")

        # By income
        print(f"\n  Payback by income group (median years):")
        for inc in ['low', 'medium', 'high']:
            subset = valid[valid['income'] == inc][pb_col].dropna().clip(upper=50)
            if len(subset) > 0:
                print(f"    {inc:<10s}: {subset.median():>6.1f} yrs  (n={len(subset)})")

        # Fraction with payback < 10, 15, 20 years
        pb_vals = valid[pb_col].dropna()
        print(f"\n  Payback distribution:")
        for threshold in [10, 15, 20, 25]:
            frac = (pb_vals <= threshold).mean() * 100
            print(f"    < {threshold} years: {frac:5.1f}%")
        neg_savings = (valid[sav_col] <= 0).sum()
        if neg_savings > 0:
            print(f"    Never pays back (net savings <= 0): {neg_savings} buildings ({neg_savings/len(valid)*100:.1f}%)")


def compare_rate_scenarios(post_df, scenarios=None):
    """Compare payback across multiple rate scenarios."""
    if scenarios is None:
        scenarios = ['F0_WF0_ROE0', 'F50_WF0_ROE0', 'F100_WF0_ROE0',
                     'F0_WF1_ROE0', 'F50_WF1_ROE1.0']

    # Filter to scenarios that exist in the data
    bill_cols = [c for c in post_df.columns if c.endswith('_bill') and '_s' not in c]
    available = [s for s in scenarios if f'{s}_bill' in post_df.columns]

    print("\n" + "=" * 90)
    print("PAYBACK COMPARISON ACROSS RATE SCENARIOS")
    print("=" * 90)

    summary_rows = []
    for scen in available:
        pb = compute_payback(post_df, scen)
        for prefix, label in [('s1', 'S1:EV'), ('s2', 'S2:PV+Stor'),
                              ('s3', 'S3:Full(pre)'), ('s4', 'S4:Full(post)')]:
            col = f'{prefix}_payback'
            sav = f'{prefix}_net_annual_savings'
            if col not in pb.columns:
                continue
            valid = pb[col].dropna().clip(upper=50)
            neg = (pb[sav].dropna() <= 0).sum() if sav in pb.columns else 0
            if len(valid) == 0:
                continue
            summary_rows.append({
                'rate_scenario': scen,
                'adoption_scenario': label,
                'n': len(valid),
                'median_payback_yrs': valid.median(),
                'mean_payback_yrs': valid.mean(),
                'pct_under_15yrs': (valid <= 15).mean() * 100,
                'pct_never_payback': neg / (len(valid) + neg) * 100 if (len(valid) + neg) > 0 else 0,
            })

    summary = pd.DataFrame(summary_rows)
    print(f"\n{'Rate Scenario':<22s} {'Adoption':<15s} {'n':>5s} {'Median':>8s} {'Mean':>8s} {'<15yr':>7s} {'Never':>7s}")
    print("─" * 75)
    for _, r in summary.iterrows():
        print(f"{r['rate_scenario']:<22s} {r['adoption_scenario']:<15s} {r['n']:>5.0f} "
              f"{r['median_payback_yrs']:>7.1f}y {r['mean_payback_yrs']:>7.1f}y "
              f"{r['pct_under_15yrs']:>6.1f}% {r['pct_never_payback']:>6.1f}%")

    return summary


# =============================================================================
# RUN
# =============================================================================

if __name__ == '__main__':
    print("Loading post-adoption bills...")
    post_df = pd.read_csv('post_adoption_bills_sdge.csv')
    print(f"  {len(post_df)} buildings loaded\n")

    # Default analysis under status quo rates
    payback_df = compute_payback(post_df, rate_scenario='F0_WF0_ROE0')
    summarize_payback(payback_df)

    # Compare across rate scenarios
    summary = compare_rate_scenarios(post_df)

    # Save results
    payback_df.to_csv('payback_results_sdge.csv', index=False)
    summary.to_csv('payback_summary_by_scenario.csv', index=False)
    print(f"\nSaved: payback_results_sdge.csv, payback_summary_by_scenario.csv")

    # Print cost assumptions
    print("\n" + "=" * 90)
    print("COST ASSUMPTIONS")
    print("=" * 90)
    print(f"  Solar PV:              ${SOLAR_COST_PER_W}/W installed (${SOLAR_COST_PER_W*(1-ITC_RATE):.2f}/W after {ITC_RATE:.0%} ITC)")
    print(f"  Battery (13.5 kWh):    ${BATTERY_COST:,} (${BATTERY_COST*(1-ITC_RATE):,.0f} after ITC)")
    print(f"  EV premium over ICE:   ${EV_PREMIUM:,}")
    print(f"  Annual driving:        {ANNUAL_MILES:,} miles/year")
    print(f"  Electrification bundle: ${ELECTRIFICATION_BUNDLE:,} (HP HVAC + HP WH + induction + dryer + envelope)")
    print(f"  Gasoline:              ${GASOLINE_PRICE}/gal, {ICE_MPG} mpg average ICE")
    print(f"  Annual gasoline cost:  ${ANNUAL_MILES/ICE_MPG*GASOLINE_PRICE:,.0f}/yr")
    print(f"  Natural gas bill:      ${DEFAULT_GAS_BILL:,.0f}/yr (${GAS_FIXED_CHARGE_MONTHLY}/mo fixed + {AVG_GAS_THERMS_YEAR} therms @ ${GAS_MARGINAL_RATE}/therm)")
