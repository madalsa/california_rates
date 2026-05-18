"""
pge_tech_assign.py — Stage 3: Technology adoption assignments for PGE buildings

Assigns PV, battery, EV, heat pump adoption.
Battery = all PV homes. Heat pump = ResStock baseline flag.
Uses survey-based probabilities when available, else simplified assignment.
Filtered to PGE PUMAs.
"""

import os
import numpy as np
import pandas as pd

from pge_config import (
    METADATA_FILE, PUMA_UTILITY_FILE, TECH_ASSIGNMENTS_OUT,
)


def stage3_tech_assignments(bills_df):
    """Assign PV, battery, EV, heat pump adoption to PGE buildings."""
    print("\n" + "=" * 80)
    print("STAGE 3: TECHNOLOGY ADOPTION ASSIGNMENTS")
    print("=" * 80)

    pge_building_ids = set(bills_df['building_id'].astype(str))

    meta = pd.read_parquet(METADATA_FILE).reset_index(drop=True)
    puma_util = pd.read_csv(PUMA_UTILITY_FILE)
    pge_pumas = puma_util[puma_util['utility_acronym'] == 'PGE']['PUMA'].tolist()
    pge_meta = meta[meta['puma20'].isin(pge_pumas)].copy()

    pge_meta = pge_meta[pge_meta['building_id'].astype(str).isin(pge_building_ids)].copy()
    print(f"  Buildings with bills: {len(pge_meta)}")

    survey_path = 'survey_responses.csv'
    try:
        from assign_technologies import (
            prepare_survey, compute_survey_adoption_rates,
            score_buildings, assign_technology,
            map_income_bracket, income_to_bin, resstock_home_type, resstock_tenure,
        )

        pge_meta['income_numeric'] = pge_meta['in.income'].apply(map_income_bracket)
        pge_meta['inc_bin'] = pge_meta['income_numeric'].apply(income_to_bin)
        pge_meta['home_type_bin'] = pge_meta['in.geometry_building_type_acs'].apply(resstock_home_type)
        pge_meta['own_bin'] = pge_meta['in.tenure'].apply(resstock_tenure)
        pge_meta['cz'] = pge_meta['in.cec_climate_zone']

        if 'weight' in pge_meta.columns:
            weights = np.array(pge_meta['weight'], dtype=float)
        elif 'in.units_represented' in pge_meta.columns:
            weights = np.array(pge_meta['in.units_represented'], dtype=float)
        else:
            weights = np.ones(len(pge_meta))

        if os.path.exists(survey_path):
            print("  Using survey-based adoption probabilities")
            sv = prepare_survey(survey_path)

            # Survey-derived adoption rates by income × home_type × CZ.
            # No tenure conditioning, no hand-coded scalars.
            pv_groupby = ['inc_bin', 'home_type_bin', 'cz']
            pv_rates = compute_survey_adoption_rates(sv, 'PV', pv_groupby)
            pv_rates_coarse = compute_survey_adoption_rates(sv, 'PV', ['inc_bin', 'home_type_bin'])
            pv_scores_fine = score_buildings(pge_meta, pv_rates, pv_groupby)
            pv_scores_coarse = score_buildings(pge_meta, pv_rates_coarse, ['inc_bin', 'home_type_bin'])
            pv_scores = np.where(np.isnan(pv_scores_fine) | (pv_scores_fine == 0),
                                 pv_scores_coarse, pv_scores_fine)
            pv_scores = np.sqrt(np.maximum(pv_scores, 0))
            pge_meta['assigned_pv'] = assign_technology(pv_scores, weights, 0.17, seed=42)

            ev_groupby = ['inc_bin', 'home_type_bin', 'cz']
            ev_rates = compute_survey_adoption_rates(sv, 'EV', ev_groupby)
            ev_rates_coarse = compute_survey_adoption_rates(sv, 'EV', ['inc_bin', 'home_type_bin'])
            ev_scores_fine = score_buildings(pge_meta, ev_rates, ev_groupby)
            ev_scores_coarse = score_buildings(pge_meta, ev_rates_coarse, ['inc_bin', 'home_type_bin'])
            ev_scores = np.where(np.isnan(ev_scores_fine) | (ev_scores_fine == 0),
                                 ev_scores_coarse, ev_scores_fine)
            ev_scores = np.sqrt(np.maximum(ev_scores, 0))
            pge_meta['assigned_ev'] = assign_technology(ev_scores, weights, 0.12, seed=43)
        else:
            print("  Survey data not found — using simplified adoption")
            _assign_simplified(pge_meta, weights)

        # Battery = all PV homes
        pge_meta['assigned_battery'] = pge_meta['assigned_pv'].copy()

        # Heat pump = keep ResStock
        pge_meta['assigned_hp'] = pge_meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)

    except Exception as e:
        print(f"  Survey-based assignment failed: {e}")
        print("  Falling back to simplified assignment")
        weights = np.ones(len(pge_meta))
        _assign_simplified(pge_meta, weights)

    # Summary
    if 'weight' in pge_meta.columns:
        w = np.array(pge_meta['weight'], dtype=float)
    else:
        w = np.ones(len(pge_meta))

    print(f"\n  Adoption rates (weighted):")
    for tech in ['assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']:
        if tech in pge_meta.columns:
            rate = np.average(pge_meta[tech], weights=w)
            count = pge_meta[tech].sum()
            print(f"    {tech}: {rate*100:.1f}% ({count} buildings)")

    out_cols = ['building_id', 'puma20', 'income_category',
                'assigned_pv', 'assigned_battery', 'assigned_ev', 'assigned_hp']
    out_cols = [c for c in out_cols if c in pge_meta.columns]
    pge_meta[out_cols].to_csv(TECH_ASSIGNMENTS_OUT, index=False)
    print(f"  Saved to: {TECH_ASSIGNMENTS_OUT}")

    return pge_meta


def _assign_simplified(meta, weights):
    """Simplified assignment without survey data."""
    rng = np.random.RandomState(42)
    n = len(meta)

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

    ev_scores = np.ones(n) * 0.12
    if 'income_category' in meta.columns:
        high_inc = meta['income_category'] == 'High'
        ev_scores[high_inc.values] *= 2.0
    ev_probs = ev_scores / np.average(ev_scores, weights=weights) * 0.12
    ev_probs = np.clip(ev_probs, 0, 1)
    meta['assigned_ev'] = (rng.random(n) < ev_probs).astype(int)

    if 'in.hvac_heating_type' in meta.columns:
        meta['assigned_hp'] = meta['in.hvac_heating_type'].str.contains(
            'Heat Pump', na=False).astype(int)
    else:
        meta['assigned_hp'] = 0
