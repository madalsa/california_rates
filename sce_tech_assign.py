"""
sce_tech_assign.py — Stage 3: Technology adoption assignments for SCE buildings

Assigns PV, battery, EV, heat pump adoption.
Battery = all PV homes. Heat pump = ResStock baseline flag.
Uses survey-based probabilities when available, else simplified assignment.
"""

import os
import numpy as np
import pandas as pd

from sce_config import (
    METADATA_FILE, PUMA_UTILITY_FILE, TECH_ASSIGNMENTS_OUT,
)


def stage3_tech_assignments(bills_df):
    """Assign PV, battery, EV, heat pump adoption to SCE buildings."""
    print("\n" + "=" * 80)
    print("STAGE 3: TECHNOLOGY ADOPTION ASSIGNMENTS")
    print("=" * 80)

    sce_building_ids = set(bills_df['building_id'].astype(str))

    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    sce_pumas = puma_util[puma_util['utility_acronym'] == 'SCE']['PUMA'].tolist()
    sce_meta = meta[meta['puma20'].isin(sce_pumas)].copy()
    sce_meta = sce_meta[sce_meta['building_id'].astype(str).isin(sce_building_ids)].copy()
    print(f"  Buildings with bills: {len(sce_meta)}")

    survey_path = 'survey_responses.csv'
    try:
        from assign_technologies import (
            prepare_survey, compute_survey_adoption_rates,
            score_buildings, assign_technology,
            map_income_bracket, income_to_bin, resstock_home_type, resstock_tenure,
        )

        sce_meta['income_numeric'] = sce_meta['in.income'].apply(map_income_bracket)
        sce_meta['inc_bin'] = sce_meta['income_numeric'].apply(income_to_bin)
        sce_meta['home_type_bin'] = sce_meta['in.geometry_building_type_acs'].apply(resstock_home_type)
        sce_meta['own_bin'] = sce_meta['in.tenure'].apply(resstock_tenure)
        sce_meta['cz'] = sce_meta['in.cec_climate_zone']

        if 'weight' in sce_meta.columns:
            weights = np.array(sce_meta['weight'], dtype=float)
        elif 'in.units_represented' in sce_meta.columns:
            weights = np.array(sce_meta['in.units_represented'], dtype=float)
        else:
            weights = np.ones(len(sce_meta))

        if os.path.exists(survey_path):
            print("  Using survey-based adoption probabilities")
            sv = prepare_survey(survey_path)

            # Survey-derived adoption rates by income × home_type × CZ.
            # No tenure conditioning, no hand-coded scalars.
            pv_groupby = ['inc_bin', 'home_type_bin', 'cz']
            pv_rates = compute_survey_adoption_rates(sv, 'PV', pv_groupby)
            pv_rates_coarse = compute_survey_adoption_rates(sv, 'PV', ['inc_bin', 'home_type_bin'])
            pv_scores_fine = score_buildings(sce_meta, pv_rates, pv_groupby)
            pv_scores_coarse = score_buildings(sce_meta, pv_rates_coarse, ['inc_bin', 'home_type_bin'])
            pv_scores = np.where(np.isnan(pv_scores_fine) | (pv_scores_fine == 0),
                                 pv_scores_coarse, pv_scores_fine)
            pv_scores = np.sqrt(np.maximum(pv_scores, 0))
            sce_meta['assigned_pv'] = assign_technology(pv_scores, weights, 0.17, seed=42)

            ev_groupby = ['inc_bin', 'home_type_bin', 'cz']
            ev_rates = compute_survey_adoption_rates(sv, 'EV', ev_groupby)
            ev_rates_coarse = compute_survey_adoption_rates(sv, 'EV', ['inc_bin', 'home_type_bin'])
            ev_scores_fine = score_buildings(sce_meta, ev_rates, ev_groupby)
            ev_scores_coarse = score_buildings(sce_meta, ev_rates_coarse, ['inc_bin', 'home_type_bin'])
            ev_scores = np.where(np.isnan(ev_scores_fine) | (ev_scores_fine == 0),
                                 ev_scores_coarse, ev_scores_fine)
            ev_scores = np.sqrt(np.maximum(ev_scores, 0))
            sce_meta['assigned_ev'] = assign_technology(ev_scores, weights, 0.12, seed=43)
        else:
            print("  Survey data not found — using simplified adoption")
            _assign_simplified(sce_meta, weights)

        # Battery = all PV homes
        sce_meta['assigned_battery'] = sce_meta['assigned_pv'].copy()

        # Heat pump = ResStock baseline flag
        sce_meta['assigned_hp'] = sce_meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)

    except Exception as e:
        print(f"  Survey-based assignment failed: {e}")
        print("  Falling back to simplified assignment")
        weights = np.ones(len(sce_meta))
        _assign_simplified(sce_meta, weights)

    # Summary
    if 'weight' in sce_meta.columns:
        w = np.array(sce_meta['weight'], dtype=float)
    else:
        w = np.ones(len(sce_meta))

    print(f"\n  Adoption rates (weighted):")
    for tech in ['assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']:
        if tech in sce_meta.columns:
            rate = np.average(sce_meta[tech], weights=w)
            count = sce_meta[tech].sum()
            print(f"    {tech}: {rate*100:.1f}% ({count} buildings)")

    out_cols = ['building_id', 'puma20', 'income_category',
                'assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']
    out_cols = [c for c in out_cols if c in sce_meta.columns]
    sce_meta[out_cols].to_csv(TECH_ASSIGNMENTS_OUT, index=False)
    print(f"  Saved to: {TECH_ASSIGNMENTS_OUT}")

    return sce_meta


def _assign_simplified(meta, weights):
    """Simplified assignment without survey data."""
    rng = np.random.RandomState(42)
    n = len(meta)

    # PV: ~17%, biased toward SF owners
    pv_scores = np.ones(n) * 0.17
    if 'in.geometry_building_type_acs' in meta.columns:
        sf_mask = meta['in.geometry_building_type_acs'].isin(
            ['Single-Family Detached', 'Single-Family Attached', 'Mobile Home'])
        pv_scores[~sf_mask.values] *= 0.1
    if 'in.tenure' in meta.columns:
        renter = meta['in.tenure'] == 'Renter'
        pv_scores[renter.values] *= 0.15

    pv_probs = pv_scores / np.average(pv_scores, weights=weights) * 0.17
    pv_probs = np.clip(pv_probs, 0, 1)
    meta['assigned_pv'] = (rng.random(n) < pv_probs).astype(int)

    meta['assigned_battery'] = meta['assigned_pv'].copy()

    # EV: ~12%
    ev_scores = np.ones(n) * 0.12
    if 'income_category' in meta.columns:
        high_inc = meta['income_category'] == 'High'
        ev_scores[high_inc.values] *= 2.0
    ev_probs = ev_scores / np.average(ev_scores, weights=weights) * 0.12
    ev_probs = np.clip(ev_probs, 0, 1)
    meta['assigned_ev'] = (rng.random(n) < ev_probs).astype(int)

    # HP = ResStock baseline
    if 'in.hvac_heating_type' in meta.columns:
        meta['assigned_hp'] = meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)
    else:
        meta['assigned_hp'] = 0
