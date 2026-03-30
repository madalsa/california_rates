"""
sdge_summary.py — Stage 7: Distributional analysis for SDGE

Reports revenue neutrality, bill changes by CARE status,
tech adoption impact across 4 adoption scenarios (S1-S4),
PV sizing summary, and income-based breakdowns.

Uses SDGE rate names (tou_dr_bill, tou_dr_f_bill).
"""

import numpy as np
import pandas as pd

from sdge_config import (
    RESIDENTIAL_REVENUE, CUSTOMERS, DESIGNED_SCENARIOS,
    ACTUAL_SDGE_RATES, SUMMARY_OUT,
)

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

    all_bill_cols = [c for c in final_df.columns
                     if c.endswith('_bill') and 'postadopt' not in c]
    total_customers = CUSTOMERS['total']

    # Revenue neutrality
    print("\n--- Revenue Neutrality Check ---")
    print(f"  Filed residential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"  Total customers: {total_customers:,}")

    base_col = 'tou_dr_bill'
    designed_cols = [f'{s}_bill' for s in DESIGNED_SCENARIOS
                     if f'{s}_bill' in final_df.columns]

    if base_col in final_df.columns and final_df[base_col].notna().sum() > 0:
        valid = final_df[base_col].notna()
        n_valid = valid.sum()
        base_mean = final_df.loc[valid, base_col].mean()
        base_rev = base_mean * total_customers

        print(f"\n  Sample: {n_valid} buildings")
        print(f"\n  {'Rate':<25s} {'Mean Bill':>12s} {'Pop Revenue':>14s} "
              f"{'vs TOU-DR':>10s} {'vs Filed':>10s}")
        print(f"  {'-'*71}")

        pct_filed = (base_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
        print(f"  {'TOU-DR':<25s} ${base_mean:>10,.0f} "
              f"${base_rev/1e9:>12.4f}B {'—':>10s} {pct_filed:>+9.2f}%")

        f_col = 'tou_dr_f_bill'
        if f_col in final_df.columns:
            fv = valid & final_df[f_col].notna()
            if fv.sum() > 0:
                f_mean = final_df.loc[fv, f_col].mean()
                f_rev = f_mean * total_customers
                pct_dr = (f_rev - base_rev) / base_rev * 100
                pct_f = (f_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
                print(f"  {'TOU-DR-F':<25s} ${f_mean:>10,.0f} "
                      f"${f_rev/1e9:>12.4f}B {pct_dr:>+9.2f}% {pct_f:>+9.2f}%")

        for col in designed_cols:
            mean = final_df.loc[valid, col].mean()
            rev = mean * total_customers
            pct_dr = (rev - base_rev) / base_rev * 100
            pct_f = (rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
            label = col.replace('_bill', '')
            print(f"  {label:<25s} ${mean:>10,.0f} "
                  f"${rev/1e9:>12.4f}B {pct_dr:>+9.2f}% {pct_f:>+9.2f}%")

    # Bill by CARE status
    print("\n--- Mean Annual Bill by CARE Status ---")
    for col in all_bill_cols:
        print(f"\n  {col}:")
        for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
            sub = final_df[final_df['is_care'] == care_val]
            if len(sub) > 0:
                v = sub[col].dropna()
                if len(v) > 0:
                    print(f"    {care_label:>8s}: mean=${v.mean():,.0f}  "
                          f"median=${v.median():,.0f}  (n={len(v)})")

    # Bill change from baseline
    if base_col in final_df.columns:
        print(f"\n--- Bill Change from {base_col} by CARE Status ---")
        compare_cols = [c for c in all_bill_cols if c != base_col]
        for col in compare_cols:
            print(f"\n  {col} vs {base_col}:")
            for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
                sub = final_df[final_df['is_care'] == care_val]
                if len(sub) > 0:
                    vm = sub[col].notna() & sub[base_col].notna()
                    if vm.sum() > 0:
                        change = sub.loc[vm, col] - sub.loc[vm, base_col]
                        print(f"    {care_label:>8s}: mean=${change.mean():+,.0f}  "
                              f"median=${change.median():+,.0f}  "
                              f"winners={(change < 0).sum()}/{vm.sum()}")

    # Tech adoption impact
    if 'assigned_pv' in final_df.columns or 'assigned_ev' in final_df.columns:
        print("\n--- Tech Adoption Bill Impact ---")
        for sfx, sfx_label in ADOPT_SUFFIXES.items():
            acols = [c for c in final_df.columns if c.endswith(f'_bill_{sfx}')]
            if not acols:
                continue
            mask = final_df[acols[0]].notna()
            n = mask.sum()
            if n == 0:
                continue
            print(f"\n  {sfx_label} (n={n}):")
            for acol in acols:
                rate_name = acol.replace(f'_bill_{sfx}', '')
                bc = f'{rate_name}_bill'
                if bc not in final_df.columns:
                    continue
                v = mask & final_df[bc].notna()
                if v.sum() == 0:
                    continue
                ch = final_df.loc[v, bc] - final_df.loc[v, acol]
                print(f"    {rate_name:<25s}: savings ${ch.mean():+,.0f}/yr  "
                      f"median ${ch.median():+,.0f}/yr")

            for inc in ['low', 'medium', 'high']:
                im = mask & (final_df['income'] == inc)
                if im.sum() == 0:
                    continue
                ref_acol = acols[0]
                ref_rate = ref_acol.replace(f'_bill_{sfx}', '')
                ref_base = f'{ref_rate}_bill'
                if f'tou_dr_bill_{sfx}' in final_df.columns:
                    ref_base = 'tou_dr_bill'
                    ref_acol = f'tou_dr_bill_{sfx}'
                vi = im & final_df[ref_base].notna() & final_df[ref_acol].notna()
                if vi.sum() > 0:
                    ch = final_df.loc[vi, ref_base] - final_df.loc[vi, ref_acol]
                    print(f"      {inc:>8s} (n={vi.sum()}): savings ${ch.mean():+,.0f}/yr")

        pv_cols = [c for c in final_df.columns if c.startswith('pv_size_kw_s')]
        if pv_cols:
            print("\n  PV Sizing:")
            for pc in pv_cols:
                v = final_df[pc].dropna()
                if len(v) > 0:
                    sfx_key = pc.replace('pv_size_kw_', '')
                    label = ADOPT_SUFFIXES.get(sfx_key, sfx_key)
                    print(f"    {label}: mean={v.mean():.1f} kW  "
                          f"median={v.median():.1f} kW  [{v.min():.1f}, {v.max():.1f}]")

    # Save summary
    summary_rows = []
    for col in all_bill_cols:
        sname = col.replace('_bill', '')
        for care_label, care_val in [('CARE', True), ('non-CARE', False)]:
            sub = final_df[final_df['is_care'] == care_val]
            if len(sub) > 0:
                summary_rows.append({
                    'scenario': sname,
                    'care_status': care_label,
                    'n_buildings': len(sub),
                    'mean_bill': sub[col].mean(),
                    'median_bill': sub[col].median(),
                    'p10_bill': sub[col].quantile(0.1),
                    'p90_bill': sub[col].quantile(0.9),
                    'mean_kwh': sub['annual_kwh'].mean(),
                })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        print(f"\n  Summary saved to: {SUMMARY_OUT}")
