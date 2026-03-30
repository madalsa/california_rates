"""
pge_summary.py — Stage 7: Distributional analysis for PGE

Reports revenue neutrality, bill changes by CARE status,
tech adoption impact across 4 adoption scenarios (S1-S4),
PV sizing summary, and income-based breakdowns.

Uses PGE rate names (e_tou_c_bill, e_tou_c_f_bill).
"""

import numpy as np
import pandas as pd

from pge_config import (
    RESIDENTIAL_REVENUE, CUSTOMERS, DESIGNED_SCENARIOS,
    ACTUAL_PGE_RATES, SUMMARY_OUT,
)


# PGE adoption scenario suffixes and labels
ADOPT_SUFFIXES = {
    's1_ev': 'S1: EV only',
    's2_pv_stor': 'S2: PV + storage',
    's3_pv_stor_ev': 'S3: PV + storage + EV',
    's4_full_elec': 'S4: Full elec + PV + storage + EV',
}


def stage7_summary(final_df, rate_scenarios_df):
    """Generate distributional analysis tables."""
    print("\n" + "=" * 80)
    print("STAGE 7: DISTRIBUTIONAL ANALYSIS")
    print("=" * 80)

    all_bill_cols = [c for c in final_df.columns if c.endswith('_bill') and 'postadopt' not in c]

    total_customers = CUSTOMERS['total']

    print("\n--- Revenue Neutrality Check (population-level) ---")
    print(f"  EIA/PGE filed residential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"  Total residential customers: {total_customers:,}")

    etoc_col = 'e_tou_c_bill'
    designed_cols = [f'{s}_bill' for s in DESIGNED_SCENARIOS if f'{s}_bill' in final_df.columns]

    if etoc_col in final_df.columns and final_df[etoc_col].notna().sum() > 0:
        valid_mask = final_df[etoc_col].notna()
        n_valid = valid_mask.sum()

        etoc_mean = final_df.loc[valid_mask, etoc_col].mean()
        etoc_pop_rev = etoc_mean * total_customers

        print(f"\n  Sample size: {n_valid} buildings "
              f"(each represents ~{total_customers/n_valid:.0f} households)")
        print(f"\n  {'Rate':<25s} {'Mean Bill':>12s} {'Pop Revenue':>14s} "
              f"{'vs E-TOU-C':>10s} {'vs Filed':>10s}")
        print(f"  {'-'*71}")

        pct_vs_filed = (etoc_pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
        print(f"  {'Actual E-TOU-C':<25s} ${etoc_mean:>10,.0f} "
              f"${etoc_pop_rev/1e9:>12.4f}B {'---':>10s} "
              f"{pct_vs_filed:>+9.2f}%")

        etocf_col = 'e_tou_c_f_bill'
        if etocf_col in final_df.columns:
            drf_valid = valid_mask & final_df[etocf_col].notna()
            if drf_valid.sum() > 0:
                drf_mean = final_df.loc[drf_valid, etocf_col].mean()
                drf_pop_rev = drf_mean * total_customers
                pct_vs_etoc = (drf_pop_rev - etoc_pop_rev) / etoc_pop_rev * 100
                pct_vs_filed = (drf_pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
                print(f"  {'Actual E-TOU-C-F':<25s} ${drf_mean:>10,.0f} "
                      f"${drf_pop_rev/1e9:>12.4f}B {pct_vs_etoc:>+9.2f}% "
                      f"{pct_vs_filed:>+9.2f}%")

        for col in designed_cols:
            mean_bill = final_df.loc[valid_mask, col].mean()
            pop_rev = mean_bill * total_customers
            pct_vs_etoc = (pop_rev - etoc_pop_rev) / etoc_pop_rev * 100
            pct_vs_filed = (pop_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
            label = col.replace('_bill', '')
            print(f"  {label:<25s} ${mean_bill:>10,.0f} "
                  f"${pop_rev/1e9:>12.4f}B {pct_vs_etoc:>+9.2f}% "
                  f"{pct_vs_filed:>+9.2f}%")
    else:
        print("\n  WARNING: No valid E-TOU-C bills — comparing designed scenarios only")
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

    # Bill change from baseline
    base_col = 'e_tou_c_bill' if 'e_tou_c_bill' in final_df.columns else 'F0_WF0_ROE0_bill'
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

    # Tech adoption impact
    if 'assigned_pv' in final_df.columns or 'assigned_ev' in final_df.columns:
        print("\n--- Tech Adoption Bill Impact by Scenario ---")
        for sfx, sfx_label in ADOPT_SUFFIXES.items():
            adopt_cols = [c for c in final_df.columns if c.endswith(f'_bill_{sfx}')]
            if not adopt_cols:
                continue
            mask = final_df[adopt_cols[0]].notna()
            n_bldgs = mask.sum()
            if n_bldgs == 0:
                continue
            print(f"\n  {sfx_label} (n={n_bldgs}):")
            for acol in adopt_cols:
                rate_name = acol.replace(f'_bill_{sfx}', '')
                base_col_tech = f'{rate_name}_bill'
                if base_col_tech not in final_df.columns:
                    continue
                valid = mask & final_df[base_col_tech].notna()
                if valid.sum() == 0:
                    continue
                change = final_df.loc[valid, base_col_tech] - final_df.loc[valid, acol]
                print(f"    {rate_name:<25s}: mean savings ${change.mean():+,.0f}/yr  "
                      f"median ${change.median():+,.0f}/yr")
            for inc in ['low', 'medium', 'high']:
                inc_mask = mask & (final_df['income'] == inc)
                if inc_mask.sum() == 0:
                    continue
                ref_acol = adopt_cols[0]
                ref_rate = ref_acol.replace(f'_bill_{sfx}', '')
                ref_base = f'{ref_rate}_bill'
                if 'e_tou_c_bill' in final_df.columns and f'e_tou_c_bill_{sfx}' in final_df.columns:
                    ref_base = 'e_tou_c_bill'
                    ref_acol = f'e_tou_c_bill_{sfx}'
                valid_inc = inc_mask & final_df[ref_base].notna() & final_df[ref_acol].notna()
                if valid_inc.sum() > 0:
                    ch = final_df.loc[valid_inc, ref_base] - final_df.loc[valid_inc, ref_acol]
                    print(f"      {inc:>8s} (n={valid_inc.sum()}): "
                          f"mean savings ${ch.mean():+,.0f}/yr")

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
