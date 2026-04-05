"""
sdge_post_adoption.py — Stage 6: Post-adoption bill calculation for SDGE

Computes bills for tech-adopted buildings under 4 adoption scenarios:
  S1 (ev_only):       baseline + EV charging (no PV/battery)
  S2 (pv_storage):    baseline + PV + battery (no EV)
  S3 (pv_stor_ev):    baseline + EV + PV + battery
  S4 (full_elec):     Upgrade11 load + EV + PV + battery

SDGE-specific:
  - Uses native demand (RASS sf stored but not applied)
  - Single solar centroid (not per-CZ)
  - 6 TOU periods (with midpeak)
  - LP-only battery dispatch
  - Baseline credits included in designed scenario bills
  - Tracks grid import/export, export value, self-sufficiency
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from sdge_config import (
    BASELINE_DIR, UPGRADE11_DIR, EXCEL_FILE, EEC_FILE, POSTADOPT_BILLS_OUT,
    ACTUAL_SDGE_RATES, DESIGNED_SCENARIOS, TOU_PERIODS,
    SDGE_ANNUAL_KWH_PER_KW, PV_OFFSET_TARGET,
    BEV_DVMT_CDF, EV_MILES_PER_KWH, EV_CHARGE_START_HOUR, EV_CHARGER_KW,
    build_tou_rate_array, safe_float,
)
from sdge_baseline_bills import load_sdge_metadata, calculate_actual_sdge_bill_vectorized
from sdge_solar import size_pv_system
from sdge_battery_lp import battery_lp_dispatch


def sample_ev_dvmt(n, seed=44):
    """Sample daily VMT for n EV buildings from BEV empirical CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    return np.interp(u, BEV_DVMT_CDF[:, 1], BEV_DVMT_CDF[:, 0])


def make_ev_profile(daily_miles=None):
    """Generate Level 2 EV charging profile (8760 hours).

    Charges at fixed Level 2 rate (7.2 kW) starting at 10 PM.
    Duration varies by daily miles — ends when daily kWh is replenished.
    """
    if daily_miles is None:
        daily_miles = np.trapz(BEV_DVMT_CDF[:, 0], BEV_DVMT_CDF[:, 1])
    daily_kwh = daily_miles / EV_MILES_PER_KWH
    charge_hours = daily_kwh / EV_CHARGER_KW
    full_hours = int(charge_hours)
    partial = charge_hours - full_hours

    profile = np.zeros(8760)
    for day in range(365):
        for h in range(full_hours):
            hour = day * 24 + (EV_CHARGE_START_HOUR + h) % 24
            if hour < 8760:
                profile[hour] = EV_CHARGER_KW
        if partial > 0:
            hour = day * 24 + (EV_CHARGE_START_HOUR + full_hours) % 24
            if hour < 8760:
                profile[hour] = EV_CHARGER_KW * partial
    return profile


def stage6_post_adoption_bills(bills_df, tech_df, solar_profile, rate_scenarios_df,
                               use_lp=True, annual_kwh_per_kw=None,
                               skip_s3=False):
    """
    Compute post-adoption bills for SDGE buildings.

    Parameters
    ----------
    solar_profile : np.ndarray (8760,)
        Per-kW hourly generation — SDGE uses single centroid.
    annual_kwh_per_kw : float or None
        Annual kWh per kW DC for PV sizing.
    """
    print("\n" + "=" * 80)
    print("STAGE 6: POST-ADOPTION BILLS")
    print("=" * 80)

    if skip_s3:
        print("  Skipping S3 (PV+storage+EV)")

    _dispatch = battery_lp_dispatch
    print(f"  Battery dispatch: LP (scipy/HiGHS)")

    # Merge tech assignments
    tech_cols = ['building_id', 'assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']
    tech_cols = [c for c in tech_cols if c in tech_df.columns]
    merged = bills_df.merge(tech_df[tech_cols], on='building_id', how='left')
    for col in ['assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)

    has_tech = merged[(merged['assigned_pv'] == 1) |
                      (merged['assigned_ev'] == 1)].copy()
    print(f"  Buildings with tech adoption: {len(has_tech)}")

    if len(has_tech) == 0:
        merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
        return merged

    # Load EEC rates
    try:
        eec_df = pd.read_csv(EEC_FILE, parse_dates=['datetime'])
        eec_rates = eec_df['sdge_total'].values[:8760]
        print(f"  Loaded EEC rates from {EEC_FILE} (sdge_total)")
    except Exception as e:
        print(f"  Could not load EEC: {e} — using 0 export credit")
        eec_rates = np.zeros(8760)

    # Sample EV VMT
    ev_buildings = has_tech[has_tech['assigned_ev'] == 1]
    ev_dvmt_map = {}
    if len(ev_buildings) > 0:
        dvmt_samples = sample_ev_dvmt(len(ev_buildings))
        for i, (_, ev_row) in enumerate(ev_buildings.iterrows()):
            ev_dvmt_map[int(ev_row['building_id'])] = dvmt_samples[i]
        print(f"  EV VMT: mean={dvmt_samples.mean():.1f} mi/day")

    if annual_kwh_per_kw is None:
        annual_kwh_per_kw = SDGE_ANNUAL_KWH_PER_KW

    # Build designed scenario rate arrays (6-period SDGE)
    selected = rate_scenarios_df[rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)]
    designed_rate_arrays = {}
    scenario_fixed_charges = {}
    for _, scen in selected.iterrows():
        sname = scen['Scenario']
        rate_dict = {p: scen[p] for p in TOU_PERIODS}
        designed_rate_arrays[sname] = build_tou_rate_array(rate_dict)
        scenario_fixed_charges[sname] = {
            'care': scen['Fixed_CARE'] * 12,
            'noncare': scen['Fixed_NonCARE'] * 12,
        }
    print(f"  Built rate arrays for {len(designed_rate_arrays)} designed scenarios")

    # Load actual tariff rate info
    from corrected_bill_calc import load_excel_data
    rates_df_xl, baseline_df_xl = load_excel_data(EXCEL_FILE)

    def _load_actual_rate(rate_code):
        entries = rates_df_xl[rates_df_xl['rate_type'] == rate_code]
        wd = entries[entries['weekday'] == 'weekday'].iloc[0].to_dict()
        rd = {p: safe_float(wd.get(f'{p.split("_")[1]}_rate_{p.split("_")[0]}1', 0))
              for p in TOU_PERIODS}
        # More explicit extraction for safety
        rd = {
            'summer_peak': safe_float(wd.get('peak_rate_summer1', 0)),
            'summer_midpeak': safe_float(wd.get('midpeak_rate_summer1', 0)),
            'summer_offpeak': safe_float(wd.get('offpeak_rate_summer1', 0)),
            'winter_peak': safe_float(wd.get('peak_rate_winter1', 0)),
            'winter_midpeak': safe_float(wd.get('midpeak_rate_winter1', 0)),
            'winter_offpeak': safe_float(wd.get('offpeak_rate_winter1', 0)),
        }
        bl_credit = safe_float(wd.get('baseline_credit', 0))
        care_disc = abs(safe_float(wd.get('care_discount', 0)))
        base_svc = safe_float(wd.get('base_service_charge_per_day', 0))
        has_fixed = wd.get('Fixed', '') == 'Yes'
        fc = {
            'low': safe_float(wd.get('fixedcharge_low', 0)) if has_fixed else 0.0,
            'medium': safe_float(wd.get('fixedcharge_med', 0)) if has_fixed else 0.0,
            'high': safe_float(wd.get('fixedcharge_high', 0)) if has_fixed else 0.0,
        }
        return {
            'rate_arr': build_tou_rate_array(rd),
            'baseline_credit': bl_credit,
            'care_discount': care_disc,
            'base_svc_daily': base_svc,
            'has_fixed': has_fixed,
            'fc_monthly': fc,
        }

    actual_rates = {}
    for rc in ACTUAL_SDGE_RATES:
        actual_rates[rc] = _load_actual_rate(rc)
        print(f"  Loaded {rc}: bl_credit={actual_rates[rc]['baseline_credit']:.4f}")

    designed_care_discount = actual_rates['TOU-DR']['care_discount']

    # Check Upgrade 11
    upgrade11_dir = Path(UPGRADE11_DIR)
    has_upgrade11 = upgrade11_dir.exists() and any(upgrade11_dir.glob('*.parquet'))
    if has_upgrade11:
        print(f"  Upgrade 11 data found in {UPGRADE11_DIR}")
    else:
        print(f"  No Upgrade 11 data — S4 skipped")

    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))

    baseline_dir = Path(BASELINE_DIR)
    results_update = {}
    start_time = time.time()
    processed = 0
    lp_failures = 0
    pv_sizes = []

    def _compute_all_bills(load_profile, solar_gen, is_care, income, puma_str,
                           use_battery, prefix):
        """Compute bills for all rate scenarios (designed + actual).

        Solves LP ONCE using actual tariff (TOU-DR) rate array, then reuses
        the dispatch profile for all 8 rate scenarios. Valid because all scenarios
        share the same TOU period structure — only rate levels differ.
        """
        result = {}
        lp_fail = 0

        # Solve LP once with actual tariff rates
        actual_rate_arr = actual_rates['TOU-DR']['rate_arr']
        batt = None
        if use_battery:
            batt = _dispatch(load_profile, solar_gen, actual_rate_arr, eec_rates)
            if batt is None:
                lp_fail = 1

        if batt is not None:
            gi, ge = batt['grid_import'], batt['grid_export']
        else:
            net = load_profile - solar_gen
            gi, ge = np.maximum(net, 0), np.maximum(-net, 0)

        # Track grid/export/self-sufficiency metrics
        result[f'grid_import_kwh_{prefix}'] = gi.sum()
        result[f'grid_export_kwh_{prefix}'] = ge.sum()
        result[f'export_value_{prefix}'] = np.dot(ge, eec_rates)
        solar_total = solar_gen.sum()
        self_consumed = solar_total - ge.sum()
        native_kwh = load_profile.sum()
        result[f'self_consumption_kwh_{prefix}'] = max(self_consumed, 0)
        result[f'self_sufficiency_{prefix}'] = max(self_consumed, 0) / native_kwh if native_kwh > 0 else 0

        # Baseline credit on grid import (same dispatch for all rates)
        bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
        designed_bl_credit = 0.0
        if not bl_entry.empty:
            d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
            d_win = bl_entry.iloc[0]['winter_baseline_allowance']
            bl_rate = actual_rates['TOU-DR']['baseline_credit']
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                mi = gi[s:e].sum()
                bl = (d_sum if 6 <= (m+1) <= 10 else d_win) * days_per_month[m]
                designed_bl_credit += bl_rate * min(mi, bl)

        # Designed scenario bills (reuse dispatch)
        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            vol_after_credit = np.dot(gi, rate_arr) - designed_bl_credit
            exp = np.dot(ge, eec_rates)
            if is_care and designed_care_discount > 0:
                vol_after_credit *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = max(vol_after_credit - exp, 0) + fixed

        # Actual tariff bills (reuse same dispatch)
        for rc, rc_info in actual_rates.items():
            col_pfx = ACTUAL_SDGE_RATES[rc]
            imp = np.dot(gi, rc_info['rate_arr'])
            exp = np.dot(ge, eec_rates)

            # Baseline credit on import
            if not bl_entry.empty:
                tot_bl = 0.0
                for m in range(12):
                    s, e = month_boundaries[m], month_boundaries[m + 1]
                    mi = gi[s:e].sum()
                    bl = (d_sum if 6 <= (m+1) <= 10 else d_win) * days_per_month[m]
                    tot_bl += rc_info['baseline_credit'] * min(mi, bl)
                imp -= tot_bl

            if is_care and rc_info['care_discount'] > 0:
                imp *= (1 - rc_info['care_discount'])

            rc_fixed = rc_info['base_svc_daily'] * 365
            if rc_info['has_fixed']:
                rc_fixed += rc_info['fc_monthly'].get(income, 0.0) * 12
            result[f'{col_pfx}_bill_{prefix}'] = max(imp - exp, 0) + rc_fixed

        return result, lp_fail

    def _bill_volumetric(load_profile, is_care, income, puma_str, prefix):
        """Compute bills for EV-only (no PV/battery)."""
        result = {}
        result[f'grid_import_kwh_{prefix}'] = load_profile.sum()
        result[f'grid_export_kwh_{prefix}'] = 0.0
        result[f'export_value_{prefix}'] = 0.0
        result[f'self_consumption_kwh_{prefix}'] = 0.0
        result[f'self_sufficiency_{prefix}'] = 0.0

        # Baseline credit for designed scenarios (load_profile = grid import)
        bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
        designed_bl_credit = 0.0
        if not bl_entry.empty:
            d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
            d_win = bl_entry.iloc[0]['winter_baseline_allowance']
            bl_rate = actual_rates['TOU-DR']['baseline_credit']
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                mi = load_profile[s:e].sum()
                bl = (d_sum if 6 <= (m+1) <= 10 else d_win) * days_per_month[m]
                designed_bl_credit += bl_rate * min(mi, bl)

        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            vol = np.dot(load_profile, rate_arr)
            vol -= designed_bl_credit  # subtract baseline credit
            if is_care and designed_care_discount > 0:
                vol *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = vol + fixed

        for rc in ACTUAL_SDGE_RATES:
            col_pfx = ACTUAL_SDGE_RATES[rc]
            bill = calculate_actual_sdge_bill_vectorized(
                load_profile, rc, puma_str, income, is_care)
            result[f'{col_pfx}_bill_{prefix}'] = bill
        return result

    for idx, row in has_tech.iterrows():
        bid = int(row['building_id'])
        pq_file = baseline_dir / f"{bid}-0.parquet"
        if not pq_file.exists():
            continue

        try:
            df = pd.read_parquet(pq_file)
            load_15min = df['out.electricity.total.energy_consumption'].values
            hourly_load = load_15min.reshape(-1, 4).sum(axis=1)
            sf = row.get('scaling_factor', 1.0)  # stored but NOT applied

            gas_col = 'out.natural_gas.total.energy_consumption'
            baseline_gas = 0.0
            if gas_col in df.columns:
                baseline_gas = df[gas_col].values.sum() / 100.0

            income = row.get('income', 'medium')
            is_care = (income == 'low')
            puma_str = row.get('puma', '')

            bldg_dvmt = ev_dvmt_map.get(bid, 30.0) if row['assigned_ev'] == 1 else 0.0
            ev_profile = make_ev_profile(daily_miles=bldg_dvmt) if bldg_dvmt > 0 else np.zeros(8760)

            update_row = {
                'building_id': bid,
                'ev_daily_miles': bldg_dvmt,
                'baseline_gas_therms': baseline_gas,
                'gas_savings_therms': baseline_gas,
            }

            # S1: EV only
            if row['assigned_ev'] == 1:
                s1_load = hourly_load + ev_profile
                s1_bills = _bill_volumetric(s1_load, is_care, income, puma_str, 's1_ev')
                update_row.update(s1_bills)

            # S2: PV + storage
            if row['assigned_pv'] == 1:
                bldg_kwh = row.get('annual_kwh', hourly_load.sum())
                pv_size = size_pv_system(bldg_kwh, annual_kwh_per_kw)
                bldg_solar = solar_profile * pv_size
                update_row['pv_size_kw_s2'] = pv_size
                pv_sizes.append(pv_size)

                s2_bills, s2_lp = _compute_all_bills(
                    hourly_load, bldg_solar, is_care, income, puma_str,
                    use_battery=(row['assigned_battery'] == 1), prefix='s2_pv_stor')
                update_row.update(s2_bills)
                lp_failures += s2_lp

            # S3: PV + Storage + EV
            if row['assigned_pv'] == 1 and row['assigned_ev'] == 1 and not skip_s3:
                s3_load = hourly_load + ev_profile
                pv_size_s3 = size_pv_system(row.get('annual_kwh', hourly_load.sum()), annual_kwh_per_kw)
                bldg_solar_s3 = solar_profile * pv_size_s3
                update_row['pv_size_kw_s3'] = pv_size_s3
                update_row['annual_kwh_s3'] = s3_load.sum()

                s3_bills, s3_lp = _compute_all_bills(
                    s3_load, bldg_solar_s3, is_care, income, puma_str,
                    use_battery=(row['assigned_battery'] == 1), prefix='s3_pv_stor_ev')
                update_row.update(s3_bills)
                lp_failures += s3_lp

            # S4: Full electrification (Upgrade 11) — DISABLED
            # Not running Upgrade 11 scenarios
            # if row['assigned_pv'] == 1 and has_upgrade11:
            #     u11_file = upgrade11_dir / f"{bid}-11.parquet"
            #     if u11_file.exists():
            #         u11_df = pd.read_parquet(u11_file)
            #         u11_15min = u11_df['out.electricity.total.energy_consumption'].values
            #         u11_load = u11_15min.reshape(-1, 4).sum(axis=1)
            #         s4_load = u11_load + ev_profile
            #         pv_size_s4 = size_pv_system(s4_load.sum(), annual_kwh_per_kw)
            #         bldg_solar_s4 = solar_profile * pv_size_s4
            #         update_row['pv_size_kw_s4'] = pv_size_s4
            #         update_row['annual_kwh_s4'] = s4_load.sum()
            #
            #         s4_bills, s4_lp = _compute_all_bills(
            #             s4_load, bldg_solar_s4, is_care, income, puma_str,
            #             use_battery=(row['assigned_battery'] == 1), prefix='s4_full_elec')
            #         update_row.update(s4_bills)
            #         lp_failures += s4_lp

            results_update[bid] = update_row
            processed += 1

            if processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(has_tech)} | {elapsed:.0f}s")

        except Exception as e:
            if processed < 5:
                print(f"  Error on building {bid}: {e}")

    if results_update:
        post_df = pd.DataFrame(results_update.values())
        merged = merged.merge(post_df, on='building_id', how='left')

    elapsed = time.time() - start_time
    print(f"\n  Processed {processed} buildings in {elapsed:.1f}s")
    if pv_sizes:
        pv_arr = np.array(pv_sizes)
        print(f"  PV sizing ({PV_OFFSET_TARGET*100:.0f}% offset):")
        print(f"    Mean: {pv_arr.mean():.1f} kW | Median: {np.median(pv_arr):.1f} kW")
    if lp_failures > 0:
        print(f"  LP failures: {lp_failures}")

    merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
    print(f"  Saved to: {POSTADOPT_BILLS_OUT}")
    return merged
