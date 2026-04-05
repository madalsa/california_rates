"""
sce_post_adoption.py — Stage 6: Post-adoption bill calculation

Computes bills for tech-adopted buildings under 4 adoption scenarios:
  S1 (ev_only):       baseline + EV charging (no PV/battery)
  S2 (pv_storage):    baseline + PV + battery (no EV)
  S3 (pv_ev_storage): baseline + EV + PV + battery
  S4 (fully_elec):    baseline (already has HP) + EV + PV + battery

S4 uses the same baseline load (which already includes heat pump from
ResStock) — NO Upgrade 11. It's S3 filtered to HP-equipped buildings.

PV sized to 90% of NATIVE demand. LP battery dispatch on NATIVE demand.
Tracks grid imports, exports, export value, and self-sufficiency metrics.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

from sce_config import (
    BASELINE_DIR, METADATA_FILE, PUMA_UTILITY_FILE, EXCEL_FILE,
    EEC_FILE, POSTADOPT_BILLS_OUT,
    ACTUAL_SCE_RATES, DESIGNED_SCENARIOS, TOU_PERIODS,
    WEEKDAY_FRAC, WEEKEND_FRAC, SCE_ANNUAL_KWH_PER_KW,
    BEV_DVMT_CDF, EV_MILES_PER_KWH, EV_CHARGE_START_HOUR, EV_CHARGE_HOURS,
    build_time_arrays, build_tou_rate_array, build_actual_tariff_rate_array,
    safe_float,
)
from sce_baseline_bills import load_sce_metadata
from sce_solar import size_pv_system
from sce_battery_lp import battery_lp_dispatch


def sample_ev_dvmt(n, seed=44):
    """Sample daily VMT for n EV buildings from BEV empirical CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, size=n)
    return np.interp(u, BEV_DVMT_CDF[:, 1], BEV_DVMT_CDF[:, 0])


def make_ev_profile(daily_miles=None):
    """Generate Level 2 EV charging profile (8760 hours)."""
    if daily_miles is None:
        daily_miles = np.trapz(BEV_DVMT_CDF[:, 0], BEV_DVMT_CDF[:, 1])
    daily_kwh = daily_miles / EV_MILES_PER_KWH
    hourly_rate = daily_kwh / EV_CHARGE_HOURS

    profile = np.zeros(8760)
    for day in range(365):
        for h in range(EV_CHARGE_HOURS):
            hour = day * 24 + (EV_CHARGE_START_HOUR + h) % 24
            if hour < 8760:
                profile[hour] = hourly_rate
    return profile


def stage6_post_adoption_bills(bills_df, tech_df, solar_profiles,
                               rate_scenarios_df, annual_kwh_per_kw_by_cz=None):
    """
    Compute post-adoption bills for all tech-adopted SCE buildings.

    All load profiles use NATIVE (unscaled) demand.
    LP-only battery dispatch.
    """
    print("\n" + "=" * 80)
    print("STAGE 6: POST-ADOPTION BILLS")
    print("=" * 80)

    ta = build_time_arrays()

    # Merge tech assignments
    tech_cols = ['building_id', 'assigned_pv', 'assigned_battery',
                 'assigned_ev', 'assigned_hp']
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

    # Load EEC rates
    try:
        eec_df = pd.read_csv(EEC_FILE, parse_dates=['datetime'])
        # Use SCE column if available, else sce_total, else pge_total as proxy
        for col_name in ['sce_total', 'pge_total']:
            if col_name in eec_df.columns:
                eec_rates = eec_df[col_name].values[:8760]
                print(f"  Loaded EEC rates from {EEC_FILE} ({col_name})")
                break
        else:
            eec_rates = np.zeros(8760)
            print("  No SCE EEC column found — using 0 export credit")
    except Exception as e:
        print(f"  Could not load EEC: {e} — using 0 export credit")
        eec_rates = np.zeros(8760)

    # Sample EV daily VMT
    ev_buildings = has_tech[has_tech['assigned_ev'] == 1]
    ev_dvmt_map = {}
    if len(ev_buildings) > 0:
        dvmt_samples = sample_ev_dvmt(len(ev_buildings))
        for i, (_, ev_row) in enumerate(ev_buildings.iterrows()):
            ev_dvmt_map[int(ev_row['building_id'])] = dvmt_samples[i]
        print(f"  EV VMT: mean={dvmt_samples.mean():.1f} mi/day, "
              f"median={np.median(dvmt_samples):.1f}")

    # CZ lookup
    sce_meta = load_sce_metadata()
    cz_lookup = dict(zip(sce_meta['building_id'].astype(int),
                         sce_meta['in.cec_climate_zone'].astype(int)))

    if annual_kwh_per_kw_by_cz is None:
        annual_kwh_per_kw_by_cz = {}

    # Load rate data for actual tariff bills
    from corrected_bill_calc import load_excel_data
    rates_df_xl, baseline_df_xl = load_excel_data(EXCEL_FILE)

    def _load_actual_rate(rate_code):
        entries = rates_df_xl[rates_df_xl['rate_type'] == rate_code]
        wd = entries[entries['weekday'] == 'weekday'].iloc[0].to_dict()
        we = entries[entries['weekday'] == 'weekend'].iloc[0].to_dict()
        wd_rates = {
            'summer_peak': safe_float(wd.get('peak_rate_summer1', 0)),
            'summer_offpeak': safe_float(wd.get('offpeak_rate_summer1', 0)),
            'winter_peak': safe_float(wd.get('peak_rate_winter1', 0)),
            'winter_midpeak': safe_float(wd.get('midpeak_rate_winter1', 0)),
            'winter_offpeak': safe_float(wd.get('offpeak_rate_winter1', 0)),
        }
        we_rates = {
            'summer_peak': safe_float(we.get('peak_rate_summer1', 0)),
            'summer_offpeak': safe_float(we.get('offpeak_rate_summer1', 0)),
            'winter_peak': safe_float(we.get('peak_rate_winter1', 0)),
            'winter_midpeak': safe_float(we.get('midpeak_rate_winter1', 0)),
            'winter_offpeak': safe_float(we.get('offpeak_rate_winter1', 0)),
        }
        bl_credit = safe_float(wd.get('baseline_credit', 0))
        care_disc = abs(safe_float(wd.get('care_discount', 0)))
        base_svc = safe_float(wd.get('base_service_charge_per_day', 0))
        min_bill = safe_float(wd.get('minimum_bill_per_day', 0))
        has_fixed = wd.get('Fixed', '') == 'Yes'
        fc = {
            'low': safe_float(wd.get('fixedcharge_low', 0)) if has_fixed else 0.0,
            'medium': safe_float(wd.get('fixedcharge_med', 0)) if has_fixed else 0.0,
            'high': safe_float(wd.get('fixedcharge_high', 0)) if has_fixed else 0.0,
        }
        return {
            'rate_arr': build_actual_tariff_rate_array(wd_rates, we_rates, ta),
            'baseline_credit': bl_credit,
            'care_discount': care_disc,
            'base_svc_daily': base_svc,
            'min_bill_daily': min_bill,
            'has_fixed': has_fixed,
            'fc_monthly': fc,
        }

    actual_rates = {}
    for rc in ACTUAL_SCE_RATES:
        actual_rates[rc] = _load_actual_rate(rc)
        print(f"  Loaded {rc}: bl_credit={actual_rates[rc]['baseline_credit']:.3f}, "
              f"has_fixed={actual_rates[rc]['has_fixed']}")

    designed_care_discount = actual_rates['TOU-D-4-9']['care_discount']
    print(f"  CARE discount: {designed_care_discount:.2%}")

    # Build designed scenario rate arrays (blended weekday/weekend)
    selected = rate_scenarios_df[
        rate_scenarios_df['Scenario'].isin(DESIGNED_SCENARIOS)]
    designed_rate_arrays = {}
    scenario_fixed_charges = {}
    for _, scen in selected.iterrows():
        sname = scen['Scenario']
        blended = {}
        for p in TOU_PERIODS:
            wd_val = scen.get(f'{p}_wd', scen.get(p, 0))
            we_val = scen.get(f'{p}_we', wd_val)
            blended[p] = wd_val * WEEKDAY_FRAC + we_val * WEEKEND_FRAC
        designed_rate_arrays[sname] = build_tou_rate_array(blended, ta)
        scenario_fixed_charges[sname] = {
            'care': scen['Fixed_CARE'] * 12,
            'noncare': scen['Fixed_NonCARE'] * 12,
        }
    print(f"  Built rate arrays for {len(designed_rate_arrays)} designed scenarios")

    # Precompute arrays for baseline credit calculation
    days_per_month = ta['days_per_month']
    month_boundaries = ta['month_boundaries']

    baseline_dir = Path(BASELINE_DIR)
    results_update = {}
    start_time = time.time()
    processed = 0
    lp_failures = 0
    pv_sizes = []

    def _compute_designed_bills(load_profile, solar_gen, is_care, use_battery, prefix):
        """Compute post-adoption bills for all rate scenarios (designed + actual).

        Solves LP ONCE using actual tariff (TOU-D-4-9) rate array, then reuses
        the dispatch profile for all 8 rate scenarios. Valid because all scenarios
        share the same TOU period structure (peak 4-9pm) — only rate levels differ.
        """
        result = {}
        lp_fail = 0

        # Solve LP once with actual tariff rates
        actual_rate_arr = actual_rates['TOU-D-4-9']['rate_arr']
        batt_dispatch = None
        if use_battery:
            batt_dispatch = battery_lp_dispatch(load_profile, solar_gen, actual_rate_arr, eec_rates)
            if batt_dispatch is None:
                lp_fail = 1

        # Precompute baseline credit for designed scenarios (based on grid import)
        def _designed_bl_credit(grid_import_profile):
            bl_entry = baseline_df_xl[baseline_df_xl['puma'] == _current_puma]
            if bl_entry.empty:
                return 0.0
            d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
            d_win = bl_entry.iloc[0]['winter_baseline_allowance']
            bl_credit_rate = actual_rates['TOU-D-4-9']['baseline_credit']
            tot = 0.0
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                mi = grid_import_profile[s:e].sum()
                bl = (d_sum if 6 <= (m + 1) <= 10 else d_win) * days_per_month[m]
                tot += bl_credit_rate * min(mi, bl)
            return tot

        if batt_dispatch is not None:
            gi = batt_dispatch['grid_import']
            ge = batt_dispatch['grid_export']
        else:
            net = load_profile - solar_gen
            gi = np.maximum(net, 0)
            ge = np.maximum(-net, 0)

        # Store grid metrics (same dispatch for all rates)
        result[f'grid_import_kwh_{prefix}'] = gi.sum()
        result[f'grid_export_kwh_{prefix}'] = ge.sum()
        result[f'export_value_{prefix}'] = np.dot(ge, eec_rates)
        result[f'self_consumption_kwh_{prefix}'] = solar_gen.sum() - ge.sum()

        bl_credit = _designed_bl_credit(gi)

        # Designed scenario bills (reuse dispatch)
        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            imp_cost = np.dot(gi, rate_arr) - bl_credit
            exp_credit = np.dot(ge, eec_rates)
            if is_care and designed_care_discount > 0:
                imp_cost *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = max(imp_cost - exp_credit, 0) + fixed

        # Actual tariff bills (reuse same dispatch)
        for rc, rc_info in actual_rates.items():
            col_pfx = ACTUAL_SCE_RATES[rc]
            imp_cost = np.dot(gi, rc_info['rate_arr'])
            exp_credit = np.dot(ge, eec_rates)

            # Baseline credit on import
            bl_entry = baseline_df_xl[baseline_df_xl['puma'] == _current_puma]
            if not bl_entry.empty:
                d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
                d_win = bl_entry.iloc[0]['winter_baseline_allowance']
                tot_bl = 0.0
                for m in range(12):
                    s, e = month_boundaries[m], month_boundaries[m + 1]
                    mi = gi[s:e].sum()
                    bl = (d_sum if 6 <= (m+1) <= 10 else d_win) * days_per_month[m]
                    tot_bl += rc_info['baseline_credit'] * min(mi, bl)
                imp_cost -= tot_bl

            if is_care and rc_info['care_discount'] > 0:
                imp_cost *= (1 - rc_info['care_discount'])

            rc_fixed = rc_info['base_svc_daily'] * 365
            if rc_info['has_fixed']:
                rc_fixed += rc_info['fc_monthly'].get(_current_income, 0.0) * 12
            bill = max(imp_cost - exp_credit, 0) + rc_fixed
            # minimum bill removed — compare designed vs actual on same basis
            result[f'{col_pfx}_bill_{prefix}'] = bill

        return result, lp_fail

    def _bill_volumetric_only(load_profile, is_care, income, puma_str, prefix):
        """Compute bills for EV-only scenario (no PV/battery)."""
        result = {}
        result[f'grid_import_kwh_{prefix}'] = load_profile.sum()
        result[f'grid_export_kwh_{prefix}'] = 0.0
        result[f'export_value_{prefix}'] = 0.0
        result[f'self_consumption_kwh_{prefix}'] = 0.0

        # Baseline credit for designed scenarios (all consumption is grid import)
        bl_credit_designed = 0.0
        bl_entry = baseline_df_xl[baseline_df_xl['puma'] == puma_str]
        if not bl_entry.empty:
            d_sum = bl_entry.iloc[0]['summer_baseline_allowance']
            d_win = bl_entry.iloc[0]['winter_baseline_allowance']
            bl_credit_rate = actual_rates['TOU-D-4-9']['baseline_credit']
            for m in range(12):
                s, e = month_boundaries[m], month_boundaries[m + 1]
                mkwh = load_profile[s:e].sum()
                bl = (d_sum if 6 <= (m + 1) <= 10 else d_win) * days_per_month[m]
                bl_credit_designed += bl_credit_rate * min(mkwh, bl)

        for sname, rate_arr in designed_rate_arrays.items():
            fc = scenario_fixed_charges[sname]
            fixed = fc['care'] if is_care else fc['noncare']
            vol_cost = np.dot(load_profile, rate_arr) - bl_credit_designed
            if is_care and designed_care_discount > 0:
                vol_cost *= (1 - designed_care_discount)
            result[f'{sname}_bill_{prefix}'] = vol_cost + fixed

        from sce_baseline_bills import calculate_actual_sce_bill
        for rc in ACTUAL_SCE_RATES:
            col_pfx = ACTUAL_SCE_RATES[rc]
            bill = calculate_actual_sce_bill(
                load_profile, rc, puma_str, income, is_care,
                rates_df_xl, baseline_df_xl, ta)
            result[f'{col_pfx}_bill_{prefix}'] = bill
        return result

    # --- Main loop ---
    _current_puma = ''
    _current_income = 'medium'

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

            income = row.get('income', 'medium')
            is_care = (income == 'low')
            puma_str = row.get('puma', '')
            _current_puma = puma_str
            _current_income = income

            bldg_cz = cz_lookup.get(bid, 9)
            bldg_solar = solar_profiles.get(bldg_cz, solar_profiles.get(9, np.zeros(8760)))
            bldg_kwh_per_kw = annual_kwh_per_kw_by_cz.get(bldg_cz, SCE_ANNUAL_KWH_PER_KW)

            # EV profile
            bldg_dvmt = ev_dvmt_map.get(bid, 30.0) if row['assigned_ev'] == 1 else 0.0
            ev_profile = make_ev_profile(daily_miles=bldg_dvmt) if bldg_dvmt > 0 else np.zeros(8760)

            native_annual_kwh = hourly_load.sum()

            update_row = {
                'building_id': bid,
                'ev_daily_miles': bldg_dvmt,
                'native_annual_kwh': native_annual_kwh,
            }

            # S1: EV only
            if row['assigned_ev'] == 1:
                s1_load = hourly_load + ev_profile
                s1_bills = _bill_volumetric_only(s1_load, is_care, income, puma_str, 's1_ev')
                update_row.update(s1_bills)
                update_row['annual_kwh_s1'] = s1_load.sum()

            # S2: PV + storage (PV sized to 90% of native demand)
            if row['assigned_pv'] == 1:
                pv_size = size_pv_system(native_annual_kwh, bldg_kwh_per_kw)
                bldg_solar_gen = bldg_solar * pv_size
                update_row['pv_size_kw_s2'] = pv_size
                pv_sizes.append(pv_size)

                # Self-sufficiency: fraction of native demand met by self-generation
                solar_total = bldg_solar_gen.sum()
                self_consumed = min(solar_total, native_annual_kwh)
                update_row['solar_gen_kwh_s2'] = solar_total
                update_row['self_sufficiency_s2'] = self_consumed / native_annual_kwh if native_annual_kwh > 0 else 0

                s2_bills, s2_lp = _compute_designed_bills(
                    hourly_load, bldg_solar_gen, is_care,
                    use_battery=(row['assigned_battery'] == 1), prefix='s2_pv_stor')
                update_row.update(s2_bills)
                lp_failures += s2_lp

            # S3: PV + EV + storage
            if row['assigned_pv'] == 1 and row['assigned_ev'] == 1:
                s3_load = hourly_load + ev_profile
                # PV still sized on native baseline demand (before EV)
                pv_size_s3 = size_pv_system(native_annual_kwh, bldg_kwh_per_kw)
                bldg_solar_s3 = bldg_solar * pv_size_s3
                update_row['pv_size_kw_s3'] = pv_size_s3

                solar_total_s3 = bldg_solar_s3.sum()
                self_consumed_s3 = min(solar_total_s3, s3_load.sum())
                update_row['solar_gen_kwh_s3'] = solar_total_s3
                update_row['self_sufficiency_s3'] = self_consumed_s3 / s3_load.sum() if s3_load.sum() > 0 else 0
                update_row['annual_kwh_s3'] = s3_load.sum()

                s3_bills, s3_lp = _compute_designed_bills(
                    s3_load, bldg_solar_s3, is_care,
                    use_battery=(row['assigned_battery'] == 1), prefix='s3_pv_ev_stor')
                update_row.update(s3_bills)
                lp_failures += s3_lp

            # Tag for fully electrified (HP already in baseline + PV + EV + battery)
            if (row.get('assigned_hp', 0) == 1 and row['assigned_pv'] == 1 and
                    row['assigned_ev'] == 1):
                update_row['is_fully_electrified'] = True
            else:
                update_row['is_fully_electrified'] = False

            results_update[bid] = update_row
            processed += 1

            if processed % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  {processed}/{len(has_tech)} | {elapsed:.0f}s")

        except Exception as e:
            if processed < 5:
                print(f"  Error on building {bid}: {e}")

    # Merge back
    if results_update:
        post_df = pd.DataFrame(results_update.values())
        merged = merged.merge(post_df, on='building_id', how='left')

    elapsed = time.time() - start_time
    print(f"\n  Processed {processed} buildings in {elapsed:.1f}s")
    if pv_sizes:
        pv_arr = np.array(pv_sizes)
        print(f"  PV sizing (90% native offset):")
        print(f"    Mean: {pv_arr.mean():.1f} kW | Median: {np.median(pv_arr):.1f} kW")
        print(f"    Range: {pv_arr.min():.1f}–{pv_arr.max():.1f} kW")
    if lp_failures > 0:
        print(f"  LP failures: {lp_failures}")

    merged.to_csv(POSTADOPT_BILLS_OUT, index=False)
    print(f"  Saved to: {POSTADOPT_BILLS_OUT}")

    return merged
