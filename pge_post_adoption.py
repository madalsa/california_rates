"""
pge_post_adoption.py — Stage 6: Post-adoption bill calculation

Computes bills for tech-adopted buildings under 4 adoption scenarios:
  S1 (ev_only):       baseline + EV charging (no PV/battery)
  S2 (pv_storage):    baseline + PV + battery (no EV)
  S3 (pv_stor_ev):    baseline + EV + PV + battery
  S4 (full_elec):     Upgrade11 load + EV + PV + battery

PGE-specific:
  - Uses native demand (no RASS scaling; sf stored for population extrapolation)
  - PV sized to 80% of native demand
  - Has Upgrade 11 (S4 scenario) for full electrification
  - LP-only battery dispatch
  - Tracks grid import/export, export value, self-sufficiency
  - 4 TOU periods (no midpeak)
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from pge_config import (
    BASELINE_DIR, UPGRADE11_DIR, EXCEL_FILE, EEC_FILE, POSTADOPT_BILLS_OUT,
    ACTUAL_PGE_RATES, DESIGNED_SCENARIOS,
    PGE_ANNUAL_KWH_PER_KW, PV_OFFSET_TARGET,
    BEV_DVMT_CDF, EV_MILES_PER_KWH, EV_CHARGE_START_HOUR, EV_CHARGE_HOURS,
    BATTERY_EFFICIENCY,
    safe_float,
)
from pge_baseline_bills import load_pge_metadata, calculate_actual_pge_bill_vectorized
from pge_solar import size_pv_system
from pge_battery_lp import battery_lp_dispatch


def _build_tou_rate_array_from_dict(rate_dict):
    """Build 8760-length rate array from a dict with keys like 'summer_peak', etc."""
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1

    hour_of_day = hours % 24
    is_summer = (months >= 6) & (months <= 10)
    is_peak = (hour_of_day >= 16) & (hour_of_day < 21)

    rates = np.where(
        is_summer,
        np.where(is_peak, rate_dict['summer_peak'], rate_dict['summer_offpeak']),
        np.where(is_peak, rate_dict['winter_peak'], rate_dict['winter_offpeak'])
    )
    return rates


def sample_ev_dvmt(n, seed=44):
    """Sample daily VMT for n EV buildings from the empirical BEV CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    dvmt = np.interp(u, BEV_DVMT_CDF[:, 1], BEV_DVMT_CDF[:, 0])
    return dvmt


def make_ev_profile(daily_miles=None):
    """Generate a Level 2 EV charging profile (8760 hours)."""
    if daily_miles is None:
        daily_miles = np.trapz(BEV_DVMT_CDF[:, 0], BEV_DVMT_CDF[:, 1])
    daily_kwh = daily_miles / EV_MILES_PER_KWH
    hourly_charge_rate = daily_kwh / EV_CHARGE_HOURS

    profile = np.zeros(8760)
    for day in range(365):
        for h in range(EV_CHARGE_HOURS):
            hour = day * 24 + (EV_CHARGE_START_HOUR + h) % 24
            if hour < 8760:
                profile[hour] = hourly_charge_rate

    return profile


def stage6_post_adoption_bills(bills_df, tech_df, solar_profiles, rate_scenarios_df,
                               use_lp=False, annual_kwh_per_kw_by_cz=None,
                               skip_s3=False):
    """
    Compute post-adoption bills under 4 technology adoption scenarios.

    Uses heuristic battery dispatch (no LP).
    Tracks grid import/export, export value, and self-sufficiency.
    """
    print("\n" + "=" * 80)
    print("STAGE 6: POST-ADOPTION BILLS")
    print("=" * 80)

    if skip_s3:
        print("  Skipping S3 (PV+storage+EV)")

    # Merge tech assignments with bills
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
        print("  No tech-adopted buildings — skipping")
        merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
        return merged

    # Load EEC rates for net billing (PGE column)
    try:
        eec_df = pd.read_csv(EEC_FILE, parse_dates=['datetime'])
        eec_rates = eec_df['pge_total'].values[:8760]
        print(f"  Loaded EEC rates from {EEC_FILE} (pge_total)")
    except Exception as e:
        print(f"  Could not load EEC data: {e}")
        print("  Using 0 export credit (conservative)")
        eec_rates = np.zeros(8760)

    # Sample per-building daily VMT
    ev_buildings = has_tech[has_tech['assigned_ev'] == 1]
    ev_dvmt_map = {}
    if len(ev_buildings) > 0:
        dvmt_samples = sample_ev_dvmt(len(ev_buildings))
        for i, (_, ev_row) in enumerate(ev_buildings.iterrows()):
            ev_dvmt_map[int(ev_row['building_id'])] = dvmt_samples[i]
        print(f"  EV daily VMT: mean={dvmt_samples.mean():.1f} mi, "
              f"median={np.median(dvmt_samples):.1f} mi")

    # Build CZ lookup for buildings
    pge_meta = load_pge_metadata()
    cz_lookup = dict(zip(pge_meta['building_id'].astype(int),
                         pge_meta['in.cec_climate_zone'].astype(int)))

    # Default annual kWh/kW for sizing
    if annual_kwh_per_kw_by_cz is None:
        annual_kwh_per_kw_by_cz = {}

    # Filter designed scenarios
    selected_designed = rate_scenarios_df[
        rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)
    ]

    # Build rate arrays and fixed charges for designed scenarios (4-period PGE)
    tou_periods = ['summer_peak', 'summer_offpeak', 'winter_peak', 'winter_offpeak']
    designed_rate_arrays = {}
    scenario_fixed_charges = {}
    for _, scenario in selected_designed.iterrows():
        sname = scenario['Scenario']
        rate_dict = {p: scenario[p] for p in tou_periods}
        designed_rate_arrays[sname] = _build_tou_rate_array_from_dict(rate_dict)
        fc_care = scenario['Fixed_CARE'] * 12
        fc_noncare = scenario['Fixed_NonCARE'] * 12
        scenario_fixed_charges[sname] = {'care': fc_care, 'noncare': fc_noncare}
    print(f"  Built rate arrays for {len(designed_rate_arrays)} designed scenarios")

    # Load actual E-TOU-C and E-TOU-C-F rate info
    from corrected_bill_calc import load_excel_data
    rates_df_xl, baseline_df_xl = load_excel_data(EXCEL_FILE)

    def _safe(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        return float(val)

    def _load_actual_rate(rate_code):
        rate_entries = rates_df_xl[rates_df_xl['rate_type'] == rate_code]
        wd = rate_entries[rate_entries['weekday'] == 'weekday'].iloc[0].to_dict()
        rd = {
            'summer_peak': _safe(wd.get('peak_rate_summer1', 0)),
            'summer_offpeak': _safe(wd.get('offpeak_rate_summer1', 0)),
            'winter_peak': _safe(wd.get('peak_rate_winter1', 0)),
            'winter_offpeak': _safe(wd.get('offpeak_rate_winter1', 0)),
        }
        bl_credit = _safe(wd.get('baseline_credit', 0))
        care_disc = abs(_safe(wd.get('care_discount', 0)))
        base_svc = _safe(wd.get('base_service_charge_per_day', 0))
        min_bill_daily = _safe(wd.get('minimum_bill_per_day', 0))
        has_fixed = wd.get('Fixed', '') == 'Yes'
        fc_low = _safe(wd.get('fixedcharge_low', 0)) if has_fixed else 0.0
        fc_med = _safe(wd.get('fixedcharge_med', 0)) if has_fixed else 0.0
        fc_high = _safe(wd.get('fixedcharge_high', 0)) if has_fixed else 0.0
        return {
            'rate_dict': rd,
            'rate_arr': _build_tou_rate_array_from_dict(rd),
            'baseline_credit': bl_credit,
            'care_discount': care_disc,
            'base_svc_daily': base_svc,
            'min_bill_daily': min_bill_daily,
            'has_fixed': has_fixed,
            'fc_monthly': {'low': fc_low, 'medium': fc_med, 'high': fc_high},
        }

    actual_rates = {}
    for rc in ACTUAL_PGE_RATES:
        actual_rates[rc] = _load_actual_rate(rc)
        print(f"  Loaded {rc}: baseline_credit={actual_rates[rc]['baseline_credit']:.4f}, "
              f"has_fixed={actual_rates[rc]['has_fixed']}")

    designed_care_discount = actual_rates['E-TOU-C']['care_discount']
    print(f"  CARE discount for designed scenarios: {designed_care_discount:.2%}")

    # Check for Upgrade 11 directory
    upgrade11_dir = Path(UPGRADE11_DIR)
    has_upgrade11 = upgrade11_dir.exists() and any(upgrade11_dir.glob('*.parquet'))
    if has_upgrade11:
        print(f"  Upgrade 11 data found in {UPGRADE11_DIR}")
    else:
        print(f"  WARNING: No Upgrade 11 data — S4 scenario will be skipped")

    baseline_dir = Path(BASELINE_DIR)
    results_update = {}
    start_time = time.time()
    processed = 0
    lp_failures = 0
    pv_sizes = []

    # LP-only battery dispatch
    _battery_dispatch = battery_lp_dispatch
    print(f"  Battery dispatch: LP (scipy/HiGHS)")

    def _compute_bill_for_rate(load_profile, solar_gen, rate_arr, eec_rate_arr,
                               is_care, care_disc, bl_entry_row, bl_credit_rate,
                               min_bill_daily, use_battery=False):
        """Compute net bill for a single rate scenario."""
        net = load_profile - solar_gen
        hourly_import = np.maximum(net, 0)

        days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month = days_per_month * 24
        month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))

        if use_battery:
            batt_result = _battery_dispatch(load_profile, solar_gen, rate_arr, eec_rate_arr)
            if batt_result is not None:
                import_cost = batt_result['bill_energy']
                export_credit = batt_result['export_credit']
                batt_import = batt_result['grid_import']

                if bl_entry_row is not None:
                    d_sum_bl = bl_entry_row['summer_baseline_allowance']
                    d_win_bl = bl_entry_row['winter_baseline_allowance']
                    total_bl_credit = 0.0
                    for m in range(12):
                        s, e = month_boundaries[m], month_boundaries[m + 1]
                        monthly_import = batt_import[s:e].sum()
                        if 6 <= (m + 1) <= 10:
                            monthly_bl = d_sum_bl * days_per_month[m]
                        else:
                            monthly_bl = d_win_bl * days_per_month[m]
                        total_bl_credit += bl_credit_rate * min(monthly_import, monthly_bl)
                    import_cost -= total_bl_credit

                if is_care and care_disc > 0:
                    import_cost *= (1 - care_disc)
                return import_cost, export_credit, batt_import

        # No battery or LP failed
        import_cost = np.dot(hourly_import, rate_arr)
        hourly_export = np.maximum(-net, 0)
        export_credit = np.dot(hourly_export, eec_rate_arr)

        if bl_entry_row is not None:
            d_sum_bl = bl_entry_row['summer_baseline_allowance']
            d_win_bl = bl_entry_row['winter_baseline_allowance']
            total_bl_credit = 0.0
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                monthly_import = hourly_import[s:e].sum()
                if 6 <= (m + 1) <= 10:
                    monthly_bl = d_sum_bl * days_per_month[m]
                else:
                    monthly_bl = d_win_bl * days_per_month[m]
                total_bl_credit += bl_credit_rate * min(monthly_import, monthly_bl)
            import_cost -= total_bl_credit

        if is_care and care_disc > 0:
            import_cost *= (1 - care_disc)

        return import_cost, export_credit, None

    def _compute_all_scenario_bills(load_profile, solar_gen, is_care, income,
                                     bl_entry_row, use_battery, prefix):
        """Compute post-adoption bills for all rate scenarios."""
        result = {}
        lp_fail = 0

        # Designed scenarios: solve LP once, reuse dispatch
        ref_rate_name = list(designed_rate_arrays.keys())[0]
        ref_rate_arr = designed_rate_arrays[ref_rate_name]

        batt_dispatch = None
        if use_battery:
            batt_dispatch = _battery_dispatch(
                load_profile, solar_gen, ref_rate_arr, eec_rates)
            if batt_dispatch is None:
                lp_fail = 1

        if batt_dispatch is not None:
            grid_import_arr = batt_dispatch['grid_import']
            grid_export_arr = batt_dispatch['grid_export']
        else:
            net = load_profile - solar_gen
            grid_import_arr = np.maximum(net, 0)
            grid_export_arr = np.maximum(-net, 0)

        # Track grid/export/self-sufficiency metrics
        result[f'grid_import_kwh_{prefix}'] = grid_import_arr.sum()
        result[f'grid_export_kwh_{prefix}'] = grid_export_arr.sum()
        result[f'export_value_{prefix}'] = np.dot(grid_export_arr, eec_rates)
        solar_total = solar_gen.sum()
        self_consumed = solar_total - grid_export_arr.sum()
        native_kwh = load_profile.sum()
        result[f'self_consumption_kwh_{prefix}'] = max(self_consumed, 0)
        result[f'self_sufficiency_{prefix}'] = max(self_consumed, 0) / native_kwh if native_kwh > 0 else 0

        # Compute monthly grid import for baseline credit on designed scenarios
        days_per_month_arr = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month_arr = days_per_month_arr * 24
        mb = np.concatenate(([0], np.cumsum(hours_per_month_arr)))
        designed_bl_credit = 0.0
        if bl_entry_row is not None:
            d_sum_bl = bl_entry_row['summer_baseline_allowance']
            d_win_bl = bl_entry_row['winter_baseline_allowance']
            bl_cr = actual_rates['E-TOU-C']['baseline_credit']
            for m_idx in range(12):
                s_h, e_h = mb[m_idx], mb[m_idx + 1]
                monthly_import = grid_import_arr[s_h:e_h].sum()
                if 6 <= (m_idx + 1) <= 10:
                    monthly_bl = d_sum_bl * days_per_month_arr[m_idx]
                else:
                    monthly_bl = d_win_bl * days_per_month_arr[m_idx]
                designed_bl_credit += bl_cr * min(monthly_import, monthly_bl)

        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            import_cost = np.dot(grid_import_arr, rate_arr)
            export_credit = np.dot(grid_export_arr, eec_rates)
            # Subtract baseline credit (applied to grid import)
            import_cost -= designed_bl_credit
            if is_care and designed_care_discount > 0:
                import_cost *= (1 - designed_care_discount)
            bill = max(import_cost - export_credit, 0) + fixed
            result[f'{sname}_bill_{prefix}'] = bill

        # Actual tariff bills: separate LP dispatch per tariff
        for rc, rc_info in actual_rates.items():
            col_prefix = ACTUAL_PGE_RATES[rc]
            import_cost, export_credit, batt_import = _compute_bill_for_rate(
                load_profile, solar_gen, rc_info['rate_arr'], eec_rates,
                is_care, rc_info['care_discount'], bl_entry_row,
                rc_info['baseline_credit'], rc_info['min_bill_daily'],
                use_battery=use_battery)
            rc_fixed = rc_info['base_svc_daily'] * 365
            if rc_info['has_fixed']:
                rc_fixed += rc_info['fc_monthly'].get(income, 0.0) * 12
            bill = max(import_cost - export_credit, 0) + rc_fixed
            result[f'{col_prefix}_bill_{prefix}'] = bill
            if use_battery and batt_import is None:
                lp_fail += 1

        return result, lp_fail

    def _bill_volumetric_only(load_profile, is_care, income, puma_str,
                              bl_entry_row, prefix):
        """Compute bills for a load profile with no PV (EV-only scenario)."""
        result = {}
        result[f'grid_import_kwh_{prefix}'] = load_profile.sum()
        result[f'grid_export_kwh_{prefix}'] = 0.0
        result[f'export_value_{prefix}'] = 0.0
        result[f'self_consumption_kwh_{prefix}'] = 0.0
        result[f'self_sufficiency_{prefix}'] = 0.0

        # Compute monthly baseline credit for designed scenarios (no PV, load = grid import)
        days_per_month_arr = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        hours_per_month_arr = days_per_month_arr * 24
        mb = np.concatenate(([0], np.cumsum(hours_per_month_arr)))
        designed_bl_credit = 0.0
        if bl_entry_row is not None:
            d_sum_bl = bl_entry_row['summer_baseline_allowance']
            d_win_bl = bl_entry_row['winter_baseline_allowance']
            bl_cr = actual_rates['E-TOU-C']['baseline_credit']
            for m_idx in range(12):
                s_h, e_h = mb[m_idx], mb[m_idx + 1]
                monthly_kwh = load_profile[s_h:e_h].sum()
                if 6 <= (m_idx + 1) <= 10:
                    monthly_bl = d_sum_bl * days_per_month_arr[m_idx]
                else:
                    monthly_bl = d_win_bl * days_per_month_arr[m_idx]
                designed_bl_credit += bl_cr * min(monthly_kwh, monthly_bl)

        # Designed scenarios
        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            vol_cost = np.dot(load_profile, rate_arr)
            # Subtract baseline credit
            vol_cost -= designed_bl_credit
            if is_care and designed_care_discount > 0:
                vol_cost *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = vol_cost + fixed

        # Actual tariff bills
        for rc in ACTUAL_PGE_RATES:
            col_prefix = ACTUAL_PGE_RATES[rc]
            rc_bill = calculate_actual_pge_bill_vectorized(
                load_profile, rc, puma_str, income, is_care)
            result[f'{col_prefix}_bill_{prefix}'] = rc_bill
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
            # NATIVE demand — no RASS scaling
            # sf stored for population extrapolation only

            # Natural gas
            gas_col = 'out.natural_gas.total.energy_consumption'
            baseline_gas_therms = 0.0
            if gas_col in df.columns:
                gas_15min_kbtu = df[gas_col].values
                baseline_gas_therms = gas_15min_kbtu.sum() / 100.0

            income = row.get('income', 'medium')
            is_care = (income == 'low')
            puma_str = row.get('puma', '')

            # Get building's CEC climate zone for solar profile lookup
            bldg_cz = cz_lookup.get(bid, 12)  # default to CZ12 (Sacramento)
            bldg_solar_profile = solar_profiles.get(bldg_cz, solar_profiles.get(12, np.zeros(8760)))
            bldg_annual_kwh_per_kw = annual_kwh_per_kw_by_cz.get(bldg_cz, PGE_ANNUAL_KWH_PER_KW)

            # EV charging profile
            bldg_dvmt = ev_dvmt_map.get(bid, 30.0) if row['assigned_ev'] == 1 else 0.0
            ev_profile = make_ev_profile(daily_miles=bldg_dvmt) if bldg_dvmt > 0 else np.zeros(8760)

            # Baseline allowance lookup
            bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
            bl_row = bl_entry.iloc[0].to_dict() if not bl_entry.empty else None

            # Load Upgrade 11 profile
            u11_file = upgrade11_dir / f"{bid}-11.parquet"
            u11_load = None
            if has_upgrade11 and u11_file.exists():
                u11_df = pd.read_parquet(u11_file)
                u11_15min = u11_df['out.electricity.total.energy_consumption'].values
                u11_load = u11_15min.reshape(-1, 4).sum(axis=1)  # native demand

            update_row = {'building_id': bid,
                          'ev_daily_miles': bldg_dvmt,
                          'baseline_gas_therms': baseline_gas_therms,
                          'gas_savings_therms': baseline_gas_therms}

            # S1: EV only
            if row['assigned_ev'] == 1:
                s1_load = hourly_load + ev_profile
                s1_bills = _bill_volumetric_only(
                    s1_load, is_care, income, puma_str, bl_row, 's1_ev')
                update_row.update(s1_bills)

            # S2: PV + storage
            if row['assigned_pv'] == 1:
                bldg_annual_kwh = row.get('annual_kwh', hourly_load.sum())
                pv_size_s2 = size_pv_system(bldg_annual_kwh, bldg_annual_kwh_per_kw)
                bldg_solar_s2 = bldg_solar_profile * pv_size_s2
                update_row['pv_size_kw_s2'] = pv_size_s2
                pv_sizes.append(pv_size_s2)

                s2_bills, s2_lp_fail = _compute_all_scenario_bills(
                    hourly_load, bldg_solar_s2, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s2_pv_stor')
                update_row.update(s2_bills)
                lp_failures += s2_lp_fail

            # S3: PV + Storage + EV
            if row['assigned_pv'] == 1 and row['assigned_ev'] == 1 and not skip_s3:
                s3_load = hourly_load + ev_profile
                pv_size_s3 = size_pv_system(row.get('annual_kwh', hourly_load.sum()), bldg_annual_kwh_per_kw)
                bldg_solar_s3 = bldg_solar_profile * pv_size_s3
                update_row['pv_size_kw_s3'] = pv_size_s3

                s3_bills, s3_lp_fail = _compute_all_scenario_bills(
                    s3_load, bldg_solar_s3, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s3_pv_stor_ev')
                update_row.update(s3_bills)
                update_row['annual_kwh_s3'] = s3_load.sum()
                lp_failures += s3_lp_fail

            # S4: Full electrification + PV + Storage + EV
            if row['assigned_pv'] == 1 and u11_load is not None:
                s4_load = u11_load + ev_profile
                post_elec_kwh = s4_load.sum()
                pv_size_s4 = size_pv_system(post_elec_kwh, bldg_annual_kwh_per_kw)
                bldg_solar_s4 = bldg_solar_profile * pv_size_s4
                update_row['pv_size_kw_s4'] = pv_size_s4

                s4_bills, s4_lp_fail = _compute_all_scenario_bills(
                    s4_load, bldg_solar_s4, is_care, income,
                    bl_row, use_battery=(row['assigned_battery'] == 1),
                    prefix='s4_full_elec')
                update_row.update(s4_bills)
                update_row['annual_kwh_s4'] = s4_load.sum()
                lp_failures += s4_lp_fail

            results_update[bid] = update_row
            processed += 1

            if processed % 500 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(has_tech)} | {elapsed:.0f}s")

        except Exception as e:
            if processed < 5:
                print(f"  Error on building {bid}: {e}")

    # Merge post-adoption bills back
    if results_update:
        post_df = pd.DataFrame(results_update.values())
        merged = merged.merge(post_df, on='building_id', how='left')

    elapsed = time.time() - start_time
    print(f"\n  Processed {processed} buildings in {elapsed:.1f}s")
    if pv_sizes:
        pv_arr = np.array(pv_sizes)
        print(f"  PV sizing ({PV_OFFSET_TARGET*100:.0f}% offset target):")
        print(f"    Mean: {pv_arr.mean():.1f} kW | Median: {np.median(pv_arr):.1f} kW")
        print(f"    Range: {pv_arr.min():.1f}--{pv_arr.max():.1f} kW")
    if lp_failures > 0:
        print(f"  LP failures: {lp_failures}")

    merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
    print(f"  Saved to: {POSTADOPT_BILLS_OUT}")

    return merged
