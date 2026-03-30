"""
sce_summary.py — Stage 7: Enhanced distributional analysis for SCE

Reports for 5 customer types:
  1. Non-adopters
  2. EV owners (only)
  3. PV + storage owners
  4. PV + EV + storage owners
  5. Fully electrified (HP + PV + EV + storage, from ResStock baseline)

Metrics tracked:
  - Bill changes ($) and (%) from baseline
  - Breakdowns by CEC climate zone
  - Changes in overall grid demand
  - Exported electricity and monetized export value
  - Self-sufficiency ratio (self-generation / native demand)
"""

import numpy as np
import pandas as pd

from sce_config import (
    RESIDENTIAL_REVENUE, CUSTOMERS, DESIGNED_SCENARIOS,
    ACTUAL_SCE_RATES, SUMMARY_OUT, BUILDING_WEIGHT,
)


# 5 customer type definitions
CUSTOMER_TYPES = {
    'non_adopter': 'Non-adopter',
    's1_ev': 'EV only',
    's2_pv_stor': 'PV + storage',
    's3_pv_ev_stor': 'PV + EV + storage',
    'fully_elec': 'Fully electrified (HP+PV+EV+stor)',
}


def _classify_customers(df):
    """Add customer_type column based on tech assignments."""
    conditions = []
    # Fully electrified: HP + PV + EV + battery
    is_full = ((df.get('assigned_hp', 0) == 1) &
               (df.get('assigned_pv', 0) == 1) &
               (df.get('assigned_ev', 0) == 1) &
               (df.get('assigned_battery', 0) == 1))
    # PV + EV + storage (not fully elec)
    is_pv_ev = ((df.get('assigned_pv', 0) == 1) &
                (df.get('assigned_ev', 0) == 1) &
                ~is_full)
    # PV + storage only
    is_pv = ((df.get('assigned_pv', 0) == 1) &
             (df.get('assigned_ev', 0) == 0))
    # EV only
    is_ev = ((df.get('assigned_ev', 0) == 1) &
             (df.get('assigned_pv', 0) == 0))
    # Non-adopter
    is_none = ~(df.get('assigned_pv', 0).astype(bool) |
                df.get('assigned_ev', 0).astype(bool))

    df['customer_type'] = 'non_adopter'
    df.loc[is_ev, 'customer_type'] = 's1_ev'
    df.loc[is_pv, 'customer_type'] = 's2_pv_stor'
    df.loc[is_pv_ev, 'customer_type'] = 's3_pv_ev_stor'
    df.loc[is_full, 'customer_type'] = 'fully_elec'
    return df


def stage7_summary(final_df, rate_scenarios_df):
    """Generate enhanced distributional analysis."""
    print("\n" + "=" * 80)
    print("STAGE 7: DISTRIBUTIONAL ANALYSIS")
    print("=" * 80)

    total_customers = CUSTOMERS['total']
    final_df = _classify_customers(final_df)

    # ===================================================================
    # 1. Revenue neutrality check
    # ===================================================================
    print("\n--- Revenue Neutrality Check ---")
    print(f"  Filed residential revenue: ${RESIDENTIAL_REVENUE/1e9:.4f}B")
    print(f"  Total customers: {total_customers:,}")

    base_col = 'tou_d_4_9_bill'
    designed_cols = [f'{s}_bill' for s in DESIGNED_SCENARIOS
                     if f'{s}_bill' in final_df.columns]

    if base_col in final_df.columns and final_df[base_col].notna().sum() > 0:
        valid = final_df[base_col].notna()
        n_valid = valid.sum()
        base_mean = final_df.loc[valid, base_col].mean()
        base_rev = base_mean * total_customers

        print(f"\n  Sample: {n_valid} buildings")
        print(f"\n  {'Rate':<25s} {'Mean Bill':>12s} {'Pop Revenue':>14s} "
              f"{'vs TOU-D-4-9':>12s} {'vs Filed':>10s}")
        print(f"  {'-'*73}")

        pct_filed = (base_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
        print(f"  {'TOU-D-4-9':<25s} ${base_mean:>10,.0f} "
              f"${base_rev/1e9:>12.4f}B {'—':>12s} {pct_filed:>+9.2f}%")

        # TOU-D-4-9-F
        f_col = 'tou_d_4_9_f_bill'
        if f_col in final_df.columns:
            f_valid = valid & final_df[f_col].notna()
            if f_valid.sum() > 0:
                f_mean = final_df.loc[f_valid, f_col].mean()
                f_rev = f_mean * total_customers
                pct_base = (f_rev - base_rev) / base_rev * 100
                pct_filed = (f_rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
                print(f"  {'TOU-D-4-9-F':<25s} ${f_mean:>10,.0f} "
                      f"${f_rev/1e9:>12.4f}B {pct_base:>+11.2f}% {pct_filed:>+9.2f}%")

        for col in designed_cols:
            mean = final_df.loc[valid, col].mean()
            rev = mean * total_customers
            pct_base = (rev - base_rev) / base_rev * 100
            pct_filed = (rev - RESIDENTIAL_REVENUE) / RESIDENTIAL_REVENUE * 100
            label = col.replace('_bill', '')
            print(f"  {label:<25s} ${mean:>10,.0f} "
                  f"${rev/1e9:>12.4f}B {pct_base:>+11.2f}% {pct_filed:>+9.2f}%")

    # ===================================================================
    # 2. Bill changes by 5 customer types ($ and %)
    # ===================================================================
    print("\n--- Bill Changes by Customer Type ---")

    # Map adoption suffixes to bill column patterns
    suffix_map = {
        's1_ev': '_bill_s1_ev',
        's2_pv_stor': '_bill_s2_pv_stor',
        's3_pv_ev_stor': '_bill_s3_pv_ev_stor',
        'fully_elec': '_bill_s3_pv_ev_stor',  # same bills, filtered to HP buildings
    }

    for ctype, clabel in CUSTOMER_TYPES.items():
        mask = final_df['customer_type'] == ctype
        n = mask.sum()
        if n == 0:
            continue

        print(f"\n  {clabel} (n={n}):")
        subset = final_df[mask]

        if ctype == 'non_adopter':
            # Non-adopters: compare designed scenario bills to baseline
            if base_col in final_df.columns:
                for col in designed_cols:
                    v = subset[col].notna() & subset[base_col].notna()
                    if v.sum() == 0:
                        continue
                    change_dollar = subset.loc[v, col] - subset.loc[v, base_col]
                    change_pct = change_dollar / subset.loc[v, base_col] * 100
                    label = col.replace('_bill', '')
                    print(f"    {label:<25s}: Δ${change_dollar.mean():+,.0f}/yr "
                          f"({change_pct.mean():+.1f}%)  "
                          f"winners={(change_dollar < 0).sum()}/{v.sum()}")
        else:
            # Adopters: compare post-adoption bill to baseline bill
            sfx = suffix_map.get(ctype, '')
            if not sfx:
                continue

            # Use baseline TOU-D-4-9 as reference
            if base_col not in final_df.columns:
                continue

            adopt_col = f'tou_d_4_9{sfx}'
            if adopt_col not in subset.columns:
                # Try designed scenario
                for ds in DESIGNED_SCENARIOS[:1]:
                    adopt_col = f'{ds}{sfx}'
                    if adopt_col in subset.columns:
                        break

            if adopt_col not in subset.columns:
                # Show whatever adoption columns exist
                acols = [c for c in subset.columns if sfx in c and c.endswith(sfx)]
                if acols:
                    adopt_col = acols[0]
                else:
                    print(f"    (no post-adoption bill columns found)")
                    continue

            # Show per-scenario bill changes
            for scen_name in list(ACTUAL_SCE_RATES.values()) + DESIGNED_SCENARIOS:
                base_c = f'{scen_name}_bill'
                adopt_c = f'{scen_name}{sfx}'
                if base_c not in subset.columns or adopt_c not in subset.columns:
                    continue
                v = subset[base_c].notna() & subset[adopt_c].notna()
                if v.sum() == 0:
                    continue
                savings = subset.loc[v, base_c] - subset.loc[v, adopt_c]
                pct = savings / subset.loc[v, base_c].replace(0, np.nan) * 100
                print(f"    {scen_name:<25s}: savings ${savings.mean():+,.0f}/yr "
                      f"({pct.mean():+.1f}%)")

    # ===================================================================
    # 3. Bill changes by CARE status
    # ===================================================================
    print("\n--- Mean Annual Bill by CARE Status ---")
    all_bill_cols = [c for c in final_df.columns
                     if c.endswith('_bill') and 'postadopt' not in c]
    for col in all_bill_cols[:8]:  # limit output
        for care_val, care_label in [(True, 'CARE'), (False, 'non-CARE')]:
            sub = final_df[final_df['is_care'] == care_val]
            if len(sub) > 0:
                v = sub[col].dropna()
                if len(v) > 0:
                    print(f"    {col:<35s} {care_label:>8s}: "
                          f"mean=${v.mean():,.0f}  median=${v.median():,.0f}")

    # ===================================================================
    # 4. Analysis by CEC climate zone
    # ===================================================================
    if 'cec_cz' in final_df.columns:
        print("\n--- Bill Changes by CEC Climate Zone ---")
        czs = sorted(final_df['cec_cz'].dropna().unique())

        if base_col in final_df.columns:
            ref_designed = designed_cols[0] if designed_cols else None
            for cz in czs:
                cz_mask = final_df['cec_cz'] == cz
                n = cz_mask.sum()
                if n == 0:
                    continue
                cz_sub = final_df[cz_mask]
                base_mean = cz_sub[base_col].mean()
                line = f"  CZ {int(cz):>2d} (n={n:>4d}): TOU-D-4-9=${base_mean:,.0f}"
                if ref_designed and ref_designed in cz_sub.columns:
                    d_mean = cz_sub[ref_designed].mean()
                    delta = d_mean - base_mean
                    line += f"  {ref_designed.replace('_bill','')}=${d_mean:,.0f} (Δ${delta:+,.0f})"
                print(line)

    # ===================================================================
    # 5. Grid demand changes
    # ===================================================================
    print("\n--- Grid Demand & Export Metrics ---")
    for sfx, label in [('s1_ev', 'EV only'), ('s2_pv_stor', 'PV+stor'),
                        ('s3_pv_ev_stor', 'PV+EV+stor')]:
        gi_col = f'grid_import_kwh_{sfx}'
        ge_col = f'grid_export_kwh_{sfx}'
        ev_col = f'export_value_{sfx}'

        if gi_col not in final_df.columns:
            continue

        mask = final_df[gi_col].notna()
        n = mask.sum()
        if n == 0:
            continue

        sub = final_df[mask]
        native = sub['annual_kwh'] if 'annual_kwh' in sub.columns else sub.get('native_annual_kwh', pd.Series())
        gi_total = sub[gi_col].sum()
        ge_total = sub[ge_col].sum() if ge_col in sub.columns else 0
        ev_total = sub[ev_col].sum() if ev_col in sub.columns else 0
        native_total = native.sum() if len(native) > 0 else 0

        grid_change = gi_total - native_total
        grid_change_pct = grid_change / native_total * 100 if native_total > 0 else 0

        print(f"\n  {label} (n={n}):")
        print(f"    Native demand:  {native_total:>12,.0f} kWh")
        print(f"    Grid import:    {gi_total:>12,.0f} kWh (Δ{grid_change:+,.0f}, {grid_change_pct:+.1f}%)")
        print(f"    Grid export:    {ge_total:>12,.0f} kWh")
        print(f"    Export value:   ${ev_total:>11,.0f} (EEC)")

    # ===================================================================
    # 6. Self-sufficiency / energy resilience metric
    # ===================================================================
    print("\n--- Self-Sufficiency (Energy Resilience) ---")
    print("  Fraction of native demand met by self-generation + self-optimization")

    for sfx, label in [('s2', 'PV+stor'), ('s3', 'PV+EV+stor')]:
        ss_col = f'self_sufficiency_{sfx}'
        if ss_col not in final_df.columns:
            continue
        mask = final_df[ss_col].notna()
        if mask.sum() == 0:
            continue
        vals = final_df.loc[mask, ss_col]
        print(f"\n  {label} (n={mask.sum()}):")
        print(f"    Mean:   {vals.mean():.1%}")
        print(f"    Median: {vals.median():.1%}")
        print(f"    P10:    {vals.quantile(0.1):.1%}")
        print(f"    P90:    {vals.quantile(0.9):.1%}")

        # By CZ
        if 'cec_cz' in final_df.columns:
            for cz in sorted(final_df.loc[mask, 'cec_cz'].dropna().unique()):
                cz_vals = final_df.loc[mask & (final_df['cec_cz'] == cz), ss_col]
                if len(cz_vals) > 0:
                    print(f"      CZ {int(cz):>2d}: {cz_vals.mean():.1%} "
                          f"(n={len(cz_vals)})")

    # ===================================================================
    # 7. PV sizing summary
    # ===================================================================
    pv_cols = [c for c in final_df.columns if c.startswith('pv_size_kw_')]
    if pv_cols:
        print("\n--- PV System Sizing (90% native offset) ---")
        for pc in pv_cols:
            v = final_df[pc].dropna()
            if len(v) > 0:
                print(f"  {pc}: mean={v.mean():.1f} kW, "
                      f"median={v.median():.1f} kW, "
                      f"range=[{v.min():.1f}, {v.max():.1f}]")

    # ===================================================================
    # Save summary CSV
    # ===================================================================
    summary_rows = []
    all_base_cols = [c for c in final_df.columns if c.endswith('_bill')
                     and '_s1_' not in c and '_s2_' not in c and '_s3_' not in c]

    for col in all_base_cols:
        sname = col.replace('_bill', '')
        for ctype, clabel in CUSTOMER_TYPES.items():
            mask = final_df['customer_type'] == ctype
            sub = final_df[mask]
            if len(sub) == 0 or col not in sub.columns:
                continue
            v = sub[col].dropna()
            if len(v) == 0:
                continue
            summary_rows.append({
                'scenario': sname,
                'customer_type': clabel,
                'n_buildings': len(v),
                'mean_bill': v.mean(),
                'median_bill': v.median(),
                'p10_bill': v.quantile(0.1),
                'p90_bill': v.quantile(0.9),
                'mean_kwh': sub['annual_kwh'].mean() if 'annual_kwh' in sub.columns else np.nan,
            })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(SUMMARY_OUT, index=False)
        print(f"\n  Summary saved to: {SUMMARY_OUT}")
