"""
Assign technology adoption (PV, battery, EV) to ResStock building stock
using survey-derived adoption probabilities calibrated to administrative estimates.

Approach:
- Heat pumps: keep ResStock assignments as-is
- Solar PV: survey covariates → relative probabilities, calibrated to ~17% admin rate
- Battery storage: paired with ALL solar PV homes
- EV: survey covariates → relative probabilities, calibrated to ~12% admin rate
"""

import numpy as np
import pandas as pd
from pathlib import Path


def map_income_bracket(income_str):
    """Map ResStock income string to a numeric midpoint for binning."""
    income_map = {
        '<10000': 5000, '10000-14999': 12500, '15000-19999': 17500,
        '20000-24999': 22500, '25000-29999': 27500, '30000-34999': 32500,
        '35000-39999': 37500, '40000-44999': 42500, '45000-49999': 47500,
        '50000-59999': 55000, '60000-69999': 65000, '70000-79999': 75000,
        '80000-99999': 90000, '100000-119999': 110000, '120000-139999': 130000,
        '140000-159999': 150000, '160000-179999': 170000, '180000-199999': 190000,
        '200000+': 225000,
    }
    return income_map.get(income_str, np.nan)


def income_to_bin(income_val):
    """Bin income into categories matching survey granularity."""
    if pd.isna(income_val):
        return 'unknown'
    if income_val < 50000:
        return '<50k'
    elif income_val < 100000:
        return '50-100k'
    elif income_val < 150000:
        return '100-150k'
    else:
        return '150k+'


def resstock_home_type(building_type):
    """Map ResStock building type to SF/MF."""
    sf_types = {'Single-Family Detached', 'Single-Family Attached', 'Mobile Home'}
    if building_type in sf_types:
        return 'SF'
    else:
        return 'MF'


def resstock_tenure(tenure_str):
    """Map ResStock tenure to 0/1 ownership."""
    if tenure_str == 'Owner':
        return 1
    elif tenure_str == 'Renter':
        return 0
    else:
        return np.nan


def compute_survey_adoption_rates(survey_df, tech_col, groupby_cols, weight_col='wt_ca'):
    """
    Compute weighted adoption rates from survey by covariate groups.

    Returns a DataFrame with group-level adoption rates.
    """
    # Drop rows with missing weights or tech values
    df = survey_df.dropna(subset=[weight_col, tech_col] + groupby_cols).copy()

    # Weighted adoption by group
    df['weighted_adopt'] = df[tech_col] * df[weight_col]

    grouped = df.groupby(groupby_cols).agg(
        weighted_adopt=('weighted_adopt', 'sum'),
        total_weight=(weight_col, 'sum'),
        n=('weighted_adopt', 'count')
    ).reset_index()

    grouped['adoption_rate'] = grouped['weighted_adopt'] / grouped['total_weight']

    return grouped


def prepare_survey(survey_path='survey_responses.csv'):
    """Load and prepare survey data with harmonized covariates."""
    sv = pd.read_csv(survey_path)

    # Harmonize covariates
    sv['inc_bin'] = sv['income'].apply(income_to_bin)
    # home_type already SF/MF/Others - map Others to MF
    sv['home_type_bin'] = sv['home_type'].map({'SF': 'SF', 'MF': 'MF', 'Others': 'MF'})
    sv['own_bin'] = sv['home_own']  # already 0/1
    sv['cz'] = sv['climatezone']

    return sv


def prepare_resstock(meta_path='CA_baseline_metadata_rescaled.parquet'):
    """Load and prepare ResStock metadata with harmonized covariates."""
    meta = pd.read_parquet(meta_path)
    meta = meta.reset_index()  # ensure we have a clean index

    # Harmonize covariates
    meta['income_numeric'] = meta['in.income'].apply(map_income_bracket)
    meta['inc_bin'] = meta['income_numeric'].apply(income_to_bin)
    meta['home_type_bin'] = meta['in.geometry_building_type_acs'].apply(resstock_home_type)
    meta['own_bin'] = meta['in.tenure'].apply(resstock_tenure)
    meta['cz'] = meta['in.cec_climate_zone']

    # Existing tech
    meta['has_hp'] = meta['in.hvac_heating_type'].str.contains('Heat Pump', na=False).astype(int)
    meta['has_pv_resstock'] = (meta['in.has_pv'] == 'Yes').astype(int)

    return meta


def score_buildings(meta, survey_rates, groupby_cols, fallback_rate=None):
    """
    Score each ResStock building using survey adoption rates.

    Merges survey group-level rates onto ResStock buildings.
    Buildings that don't match any survey group get the fallback rate.
    """
    scored = meta.merge(
        survey_rates[groupby_cols + ['adoption_rate']],
        on=groupby_cols,
        how='left'
    )

    if fallback_rate is None:
        fallback_rate = survey_rates['adoption_rate'].mean()

    scored['adoption_rate'] = scored['adoption_rate'].fillna(fallback_rate)

    return scored['adoption_rate'].values


def assign_technology(scores, weights, target_rate, seed=42):
    """
    Assign technology adoption using calibrated probabilistic sampling.

    Each building's adoption probability is proportional to its score,
    scaled so the overall weighted adoption rate matches target_rate.

    Parameters
    ----------
    scores : array-like
        Relative adoption probability for each building
    weights : array-like
        Building weight (units represented)
    target_rate : float
        Target adoption rate (e.g., 0.17 for 17%)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Boolean array of assignments
    """
    n = len(scores)
    rng = np.random.RandomState(seed)
    scores = np.array(scores, dtype=float)
    weights = np.array(weights, dtype=float)

    # Calibrate: scale scores so weighted mean equals target_rate
    # P(adopt_i) = scores_i * calibration_factor
    weighted_mean_score = np.average(scores, weights=weights)
    if weighted_mean_score > 0:
        calibration_factor = target_rate / weighted_mean_score
    else:
        calibration_factor = 1.0

    probs = scores * calibration_factor
    # Clip to [0, 1]
    probs = np.clip(probs, 0, 1)

    # Sample assignments
    assigned = rng.random(n) < probs

    # Check achieved rate and report
    achieved_rate = np.average(assigned, weights=weights)

    return assigned


def run_assignment(
    survey_path='survey_responses.csv',
    meta_path='CA_baseline_metadata_rescaled.parquet',
    pv_target_rate=0.17,
    ev_target_rate=0.12,
    battery_all_pv=True,
    output_path=None,
    seed=42
):
    """
    Run the full technology assignment pipeline.

    Parameters
    ----------
    survey_path : str
        Path to survey_responses.csv
    meta_path : str
        Path to ResStock metadata parquet
    pv_target_rate : float
        Target PV adoption rate (calibrated to admin data)
    ev_target_rate : float
        Target EV adoption rate (calibrated to admin data)
    battery_all_pv : bool
        If True, all PV homes get battery storage
    output_path : str or None
        If provided, save results to this parquet file
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        ResStock metadata with new columns: assigned_pv, assigned_battery, assigned_ev
    """
    print("Loading data...")
    sv = prepare_survey(survey_path)
    meta = prepare_resstock(meta_path)

    # Get building weights (convert to numpy float to avoid Arrow issues)
    if 'weight' in meta.columns:
        weights = np.array(meta['weight'], dtype=float)
    elif 'in.units_represented' in meta.columns:
        weights = np.array(meta['in.units_represented'], dtype=float)
    else:
        weights = np.ones(len(meta))

    # === PV Assignment ===
    print(f"\nAssigning PV (target: {pv_target_rate*100:.0f}%)...")

    # Compute survey adoption rates by covariate groups
    # Use inc_bin × home_type × cz for fine-grained scoring
    pv_groupby = ['inc_bin', 'home_type_bin', 'cz']
    pv_rates = compute_survey_adoption_rates(sv, 'PV', pv_groupby)

    # Also compute coarser rates as fallback
    pv_rates_coarse = compute_survey_adoption_rates(sv, 'PV', ['inc_bin', 'home_type_bin'])

    # Score buildings - try fine-grained first, fill with coarse
    pv_scores_fine = score_buildings(meta, pv_rates, pv_groupby, fallback_rate=None)
    pv_scores_coarse = score_buildings(meta, pv_rates_coarse, ['inc_bin', 'home_type_bin'], fallback_rate=None)

    # Use fine scores where available, coarse otherwise
    pv_scores = np.where(np.isnan(pv_scores_fine) | (pv_scores_fine == 0),
                         pv_scores_coarse, pv_scores_fine)

    # Soften PV scores to avoid extreme concentration
    pv_scores = np.sqrt(np.maximum(pv_scores, 0))

    # Additional adjustments: renters much less likely to have PV
    renter_mask = meta['own_bin'] == 0
    pv_scores[renter_mask] *= 0.15  # renters very unlikely to have rooftop PV

    # MF already penalized in survey rates, but add extra for large MF
    large_mf_mask = meta['in.geometry_building_type_acs'].isin(['50 or more Unit', '20 to 49 Unit'])
    pv_scores[large_mf_mask] *= 0.05  # very few large MF have individual PV

    # Assign PV
    meta['assigned_pv'] = assign_technology(pv_scores, weights, pv_target_rate, seed=seed)

    actual_pv_rate = np.average(meta['assigned_pv'], weights=weights)
    print(f"  Assigned PV rate: {actual_pv_rate*100:.1f}%")

    # === Battery Assignment ===
    print(f"\nAssigning battery (all PV homes)...")
    if battery_all_pv:
        meta['assigned_battery'] = meta['assigned_pv'].copy()
    else:
        # Could add partial battery assignment logic here
        meta['assigned_battery'] = meta['assigned_pv'].copy()

    actual_battery_rate = np.average(meta['assigned_battery'], weights=weights)
    print(f"  Assigned battery rate: {actual_battery_rate*100:.1f}%")

    # === EV Assignment ===
    print(f"\nAssigning EV (target: {ev_target_rate*100:.0f}%)...")

    ev_groupby = ['inc_bin', 'home_type_bin', 'cz']
    ev_rates = compute_survey_adoption_rates(sv, 'EV', ev_groupby)
    ev_rates_coarse = compute_survey_adoption_rates(sv, 'EV', ['inc_bin', 'home_type_bin'])

    ev_scores_fine = score_buildings(meta, ev_rates, ev_groupby, fallback_rate=None)
    ev_scores_coarse = score_buildings(meta, ev_rates_coarse, ['inc_bin', 'home_type_bin'], fallback_rate=None)

    ev_scores = np.where(np.isnan(ev_scores_fine) | (ev_scores_fine == 0),
                         ev_scores_coarse, ev_scores_fine)

    # Soften the EV scores to avoid extreme concentration
    # Use sqrt to compress the range while preserving ordering
    ev_scores = np.sqrt(np.maximum(ev_scores, 0))

    # Renters can have EVs but somewhat less likely
    ev_scores[renter_mask] *= 0.5

    # Assign EV
    meta['assigned_ev'] = assign_technology(ev_scores, weights, ev_target_rate, seed=seed+1)

    actual_ev_rate = np.average(meta['assigned_ev'], weights=weights)
    print(f"  Assigned EV rate: {actual_ev_rate*100:.1f}%")

    # === Heat Pump: keep ResStock as-is ===
    meta['assigned_hp'] = meta['has_hp']
    actual_hp_rate = np.average(meta['assigned_hp'], weights=weights)
    print(f"\nHeat pump rate (ResStock, unchanged): {actual_hp_rate*100:.1f}%")

    # === Summary ===
    print("\n" + "="*60)
    print("ASSIGNMENT SUMMARY (weighted)")
    print("="*60)
    print(f"  Solar PV:  {actual_pv_rate*100:.1f}% (target: {pv_target_rate*100:.0f}%)")
    print(f"  Battery:   {actual_battery_rate*100:.1f}% (all PV homes)")
    print(f"  EV:        {actual_ev_rate*100:.1f}% (target: {ev_target_rate*100:.0f}%)")
    print(f"  Heat Pump: {actual_hp_rate*100:.1f}% (ResStock)")

    # Cross-tabulations
    print(f"\n--- By home type (weighted) ---")
    for ht in ['SF', 'MF']:
        mask = meta['home_type_bin'] == ht
        w = weights[mask]
        print(f"  {ht}:")
        for tech in ['assigned_pv', 'assigned_ev', 'assigned_hp']:
            rate = np.average(meta.loc[mask, tech], weights=w)
            print(f"    {tech}: {rate*100:.1f}%")

    print(f"\n--- By income (weighted) ---")
    for inc in ['<50k', '50-100k', '100-150k', '150k+']:
        mask = meta['inc_bin'] == inc
        if mask.sum() == 0:
            continue
        w = weights[mask]
        print(f"  {inc}:")
        for tech in ['assigned_pv', 'assigned_ev']:
            rate = np.average(meta.loc[mask, tech], weights=w)
            print(f"    {tech}: {rate*100:.1f}%")

    # Save if requested
    if output_path:
        # Select key columns to save
        output_cols = [c for c in meta.columns if c.startswith(('in.', 'assigned_', 'has_'))]
        output_cols += ['weight', 'inc_bin', 'home_type_bin', 'own_bin', 'cz',
                        'income_numeric', 'scaling_factor']
        output_cols = [c for c in output_cols if c in meta.columns]
        meta[output_cols].to_parquet(output_path, index=False)
        print(f"\nSaved to {output_path}")

    return meta


if __name__ == '__main__':
    result = run_assignment(
        pv_target_rate=0.17,
        ev_target_rate=0.12,
        output_path='resstock_with_tech_assignments.parquet'
    )
