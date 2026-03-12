"""
test_one_building.py — Single-building test: baseline bills + all 4 adoption scenarios.

Picks first building with both Baseline and Upgrade 11 data, runs everything, prints results.

Usage:
  python test_one_building.py
  python test_one_building.py --building 100190
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from run_sdge_pipeline import (
    BASELINE_DIR, UPGRADE11_DIR, METADATA_FILE, EXCEL_FILE, EEC_FILE,
    RATE_SCENARIOS_OUT, ACTUAL_SDGE_RATES, DESIGNED_SCENARIOS,
    PV_OFFSET_TARGET, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW,
    build_tou_rate_array, build_tou_rate_array_from_dict,
    calculate_actual_sdge_bill_vectorized,
    size_pv_system, make_ev_profile, _synthetic_solar_profile,
    stage1_generate_rate_scenarios,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--building', type=int, default=None)
    args = parser.parse_args()

    baseline_dir = Path(BASELINE_DIR)
    upgrade11_dir = Path(UPGRADE11_DIR)

    # ── Check data ──
    if not baseline_dir.exists() or not any(baseline_dir.glob('*.parquet')):
        sys.exit(f"ERROR: No parquets in {BASELINE_DIR}")

    has_u11 = upgrade11_dir.exists() and any(upgrade11_dir.glob('*.parquet'))
    if not has_u11:
        print(f"WARNING: No Upgrade 11 data — S3/S4 will be skipped\n")

    # ── Pick building ──
    if args.building:
        bid = args.building
        if not (baseline_dir / f"{bid}-0.parquet").exists():
            sys.exit(f"ERROR: {bid}-0.parquet not found in {BASELINE_DIR}")
    else:
        for bf in sorted(baseline_dir.glob('*-0.parquet')):
            b = int(bf.stem.split('-')[0])
            if has_u11 and (upgrade11_dir / f"{b}-11.parquet").exists():
                bid = b
                break
        else:
            bid = int(sorted(baseline_dir.glob('*-0.parquet'))[0].stem.split('-')[0])

    print(f"{'='*70}")
    print(f"  TESTING BUILDING {bid}")
    print(f"{'='*70}")

    # ── Load baseline profile ──
    df_base = pd.read_parquet(baseline_dir / f"{bid}-0.parquet")
    hourly_load = df_base['out.electricity.total.energy_consumption'].values.reshape(-1, 4).sum(axis=1)

    # ── Load upgrade 11 profile ──
    u11_load = None
    u11_file = upgrade11_dir / f"{bid}-11.parquet"
    if u11_file.exists():
        df_u11 = pd.read_parquet(u11_file)
        u11_load = df_u11['out.electricity.total.energy_consumption'].values.reshape(-1, 4).sum(axis=1)

    # ── Metadata ──
    meta = pd.read_parquet(METADATA_FILE)
    bldg_meta = meta[meta['building_id'] == bid]
    if bldg_meta.empty:
        print(f"  WARNING: building {bid} not in metadata — using defaults")
        income, is_care, puma_str, sf = 'medium', False, '', 1.0
    else:
        row = bldg_meta.iloc[0]
        income = row.get('income', 'medium')
        is_care = (income == 'low')
        puma_str = str(row.get('puma', ''))
        sf = row.get('scaling_factor', 1.0)

    hourly_load *= sf
    if u11_load is not None:
        u11_load *= sf

    annual_kwh = hourly_load.sum()
    print(f"\n  PUMA: {puma_str} | Income: {income} | CARE: {is_care} | SF: {sf:.2f}")
    print(f"  Baseline: {annual_kwh:,.0f} kWh/yr | Peak: {hourly_load.max():.2f} kW")
    if u11_load is not None:
        print(f"  Upgrade 11: {u11_load.sum():,.0f} kWh/yr | Delta: {u11_load.sum()-annual_kwh:+,.0f} kWh")

    # ── Rate scenarios ──
    if Path(RATE_SCENARIOS_OUT).exists():
        rate_scenarios = pd.read_csv(RATE_SCENARIOS_OUT)
        print(f"  Loaded {len(rate_scenarios)} rate scenarios from {RATE_SCENARIOS_OUT}")
    else:
        print(f"  Generating rate scenarios...")
        rate_scenarios = stage1_generate_rate_scenarios()

    # ── Load actual rate structures ──
    from corrected_bill_calc import load_excel_data
    rates_df, baseline_df_xl = load_excel_data(EXCEL_FILE)

    bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
    bl_row = bl_entry.iloc[0].to_dict() if not bl_entry.empty else None

    def _s(v):
        return 0.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    def _load_rate(rate_code):
        wd = rates_df[(rates_df['rate_type'] == rate_code) &
                       (rates_df['weekday'] == 'weekday')].iloc[0].to_dict()
        rd = {k: _s(wd.get(v, 0)) for k, v in {
            'summer_peak': 'peak_rate_summer1', 'summer_midpeak': 'midpeak_rate_summer1',
            'summer_offpeak': 'offpeak_rate_summer1', 'winter_peak': 'peak_rate_winter1',
            'winter_midpeak': 'midpeak_rate_winter1', 'winter_offpeak': 'offpeak_rate_winter1',
        }.items()}
        return {
            'rate_arr': build_tou_rate_array_from_dict(rd),
            'baseline_credit': _s(wd.get('baseline_credit', 0)),
            'care_discount': _s(wd.get('care_discount', 0)),
            'base_svc_daily': _s(wd.get('base_service_charge', 0)),
            'has_fixed': _s(wd.get('fixed_charge_low', 0)) > 0,
            'fc_monthly': {
                'low': _s(wd.get('fixed_charge_low', 0)),
                'medium': _s(wd.get('fixed_charge_medium', 0)),
                'high': _s(wd.get('fixed_charge_high', 0)),
            },
        }

    actual_rates = {rc: _load_rate(rc) for rc in ACTUAL_SDGE_RATES}
    tou_dr_rate_arr = actual_rates['TOU-DR']['rate_arr']
    tou_dr_bl_credit = actual_rates['TOU-DR']['baseline_credit']
    tou_dr_care_disc = actual_rates['TOU-DR']['care_discount']

    # ── EEC rates ──
    try:
        eec_rates = pd.read_csv(EEC_FILE, parse_dates=['datetime'])['sdge_total'].values[:8760]
        print(f"  Loaded EEC rates from {EEC_FILE}")
    except Exception as e:
        print(f"  EEC not available ({e}) — using zeros")
        eec_rates = np.zeros(8760)

    # ── Solar profile ──
    print(f"  Generating synthetic solar profile...")
    synth = _synthetic_solar_profile()
    solar_per_kw = synth / 5.0
    annual_kwh_per_kw = solar_per_kw.sum()
    print(f"  Annual solar yield: {annual_kwh_per_kw:,.0f} kWh/kW")

    # ── EV profile ──
    ev_profile = make_ev_profile(daily_miles=30.0)
    print(f"  EV: 30 mi/day → {ev_profile.sum():,.0f} kWh/yr")

    # ── Helpers ──
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    month_bounds = np.concatenate(([0], np.cumsum(days_per_month * 24)))

    def calc_import_cost(hourly_import, rate_arr, bl_credit, care_disc):
        cost = np.dot(hourly_import, rate_arr)
        if bl_row is not None:
            bl_c = 0.0
            for m in range(12):
                mi = hourly_import[month_bounds[m]:month_bounds[m+1]].sum()
                mbl = (bl_row['summer_baseline_allowance'] if 6 <= (m+1) <= 10
                       else bl_row['winter_baseline_allowance']) * days_per_month[m]
                bl_c += bl_credit * min(mi, mbl)
            cost -= bl_c
        if is_care and care_disc > 0:
            cost *= (1 - care_disc)
        return cost

    def print_bills(label, suffix, load, solar_gen=None):
        """Print all bills (designed + TOU-DR + TOU-DR-F) for a scenario."""
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        print(f"  Load: {load.sum():,.0f} kWh/yr")

        if solar_gen is not None:
            net = load - solar_gen
            hourly_import = np.maximum(net, 0)
            hourly_export = np.maximum(-net, 0)
            export_credit = np.dot(hourly_export, eec_rates)
            print(f"  Solar: {solar_gen.sum():,.0f} kWh | Import: {hourly_import.sum():,.0f} | "
                  f"Export: {hourly_export.sum():,.0f} | EEC credit: ${export_credit:.0f}")

            # Designed scenarios (anchored billing)
            base_import_cost = calc_import_cost(hourly_import, tou_dr_rate_arr,
                                                 tou_dr_bl_credit, tou_dr_care_disc)
            for sname in DESIGNED_SCENARIOS:
                sc = rate_scenarios[rate_scenarios['Scenario'] == sname]
                if sc.empty:
                    continue
                sc = sc.iloc[0]
                alpha = sc.get('alpha', 1.0) if 'alpha' in sc.index else 1.0
                fc = sc.get('Fixed_CARE', 0.0) if is_care else sc.get('Fixed_NonCARE', 0.0)
                bill = max(alpha * base_import_cost - export_credit, 0) + fc
                print(f"  {sname+'_bill_'+suffix:<45s}: ${bill:>8,.0f}")

            # Actual tariffs (TOU-DR, TOU-DR-F)
            for rc, rc_info in actual_rates.items():
                col_prefix = ACTUAL_SDGE_RATES[rc]
                rc_cost = calc_import_cost(hourly_import, rc_info['rate_arr'],
                                           rc_info['baseline_credit'], rc_info['care_discount'])
                rc_fixed = rc_info['base_svc_daily'] * 365
                if rc_info['has_fixed']:
                    rc_fixed += rc_info['fc_monthly'].get(income, 0.0) * 12
                rc_bill = max(rc_cost - export_credit, 0) + rc_fixed
                print(f"  {col_prefix+'_bill_'+suffix:<45s}: ${rc_bill:>8,.0f}")
        else:
            # Volumetric only (no PV) — S1 EV case
            for sname in DESIGNED_SCENARIOS:
                sc = rate_scenarios[rate_scenarios['Scenario'] == sname]
                if sc.empty:
                    continue
                sc = sc.iloc[0]
                rate_arr = build_tou_rate_array(sc)
                vol = np.dot(load, rate_arr)
                fc = sc.get('Fixed_CARE', 0.0) if is_care else sc.get('Fixed_NonCARE', 0.0)
                bill = vol + fc
                print(f"  {sname+'_bill_'+suffix:<45s}: ${bill:>8,.0f}")

            for rc in ACTUAL_SDGE_RATES:
                col_prefix = ACTUAL_SDGE_RATES[rc]
                bill = calculate_actual_sdge_bill_vectorized(
                    load, rc, puma_str, income, is_care)
                print(f"  {col_prefix+'_bill_'+suffix:<45s}: ${bill:>8,.0f}")

    # ══════════════════════════════════════════════════════════════════
    # BASELINE (no tech)
    # ══════════════════════════════════════════════════════════════════
    print_bills("BASELINE (no tech adoption)", "baseline", hourly_load)

    # ══════════════════════════════════════════════════════════════════
    # S1: EV only
    # ══════════════════════════════════════════════════════════════════
    s1_load = hourly_load + ev_profile
    print_bills("S1: EV ONLY (baseline + EV)", "s1_ev", s1_load)

    # ══════════════════════════════════════════════════════════════════
    # S2: PV + storage (no EV)
    # ══════════════════════════════════════════════════════════════════
    pv_s2 = size_pv_system(annual_kwh, annual_kwh_per_kw)
    solar_s2 = solar_per_kw * pv_s2
    print(f"\n  S2 PV size: {pv_s2:.1f} kW (sized on baseline {annual_kwh:,.0f} kWh)")
    print_bills("S2: PV + STORAGE (baseline + PV, no battery dispatch in test)",
                "s2_pv_stor", hourly_load, solar_gen=solar_s2)

    # ══════════════════════════════════════════════════════════════════
    # S3: Full electrification + PV sized PRE-electrification
    # ══════════════════════════════════════════════════════════════════
    if u11_load is not None:
        s3_load = u11_load + ev_profile
        pre_kwh = (hourly_load + ev_profile).sum()
        pv_s3 = size_pv_system(pre_kwh, annual_kwh_per_kw)
        solar_s3 = solar_per_kw * pv_s3
        print(f"\n  S3 PV size: {pv_s3:.1f} kW (sized on baseline+EV {pre_kwh:,.0f} kWh)")
        print_bills("S3: FULL ELEC + EV + PV (sized PRE-electrification)",
                    "s3_full_pre", s3_load, solar_gen=solar_s3)

        # ══════════════════════════════════════════════════════════════
        # S4: Full electrification + PV sized POST-electrification
        # ══════════════════════════════════════════════════════════════
        s4_load = u11_load + ev_profile
        post_kwh = s4_load.sum()
        pv_s4 = size_pv_system(post_kwh, annual_kwh_per_kw)
        solar_s4 = solar_per_kw * pv_s4
        print(f"\n  S4 PV size: {pv_s4:.1f} kW (sized on U11+EV {post_kwh:,.0f} kWh)")
        print_bills("S4: FULL ELEC + EV + PV (sized POST-electrification)",
                    "s4_full_post", s4_load, solar_gen=solar_s4)
    else:
        print(f"\n  Skipping S3/S4 — no Upgrade 11 data for building {bid}")

    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE — building {bid}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
