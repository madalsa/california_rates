"""Stratified sampling + clustering to pick representative buildings.

POPULATION SCOPE (paper):
  Include: IOU customers (PGE, SCE, SDGE) where NPV is a useful question.
  Exclude:
    - POU territories (LADWP, SMUD, IID, etc.) - filtered via puma_utility
    - Renters - capex / payback decisions belong to property owners
    - Households with "Not Available" income tier - cannot classify
    - EBD-eligible: <=80% AMI in CEC priority CZs (turnkey free retrofit)

Approach:
  1. Load CA_baseline_tmy_metadata_and_annual_results.parquet (has metadata
     + annual electricity / NG / peak kW). Join utility/CZ via puma_utility.
  2. Apply scope filter, write population_excluded_summary.csv.
  3. Stratify by (utility, cec_cz, heating_fuel, building_type, ami_bin,
     vintage_decade).
  4. Within each stratum (size > 1) k-means with k = min(N_per_stratum/30, 5)
     on standardized features:
        annual_kwh, annual_therms, summer_peak_kw, winter_peak_kw,
        cooling_share, hvac_share, sqft
  5. Pick medoid building (closest to centroid in feature space).
  6. Write representatives parquet: medoid building_id + features + weight.

CLI:
    python -m electrification_economics.src.representative_buildings \
        [--target N] [--out PATH]

Defaults: ~500 medoids, write to data/representative_buildings.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


# AMI string bin -> midpoint (fraction of 100% AMI)
AMI_BIN_TO_FRAC = {
    "0-30%":   0.15,
    "30-60%":  0.45,
    "60-80%":  0.70,
    "80-100%": 0.90,
    "100-120%": 1.10,
    "120-150%": 1.35,
    "150%+":   2.00,
    "Not Available": np.nan,
}

VINTAGE_DECADE = {
    "<1940": "pre1940", "1940s": "pre1960", "1950s": "pre1960",
    "1960s": "1960_70s", "1970s": "1960_70s",
    "1980s": "1980_90s", "1990s": "1980_90s",
    "2000s": "post2000", "2010s": "post2000",
}


def load_metadata() -> pd.DataFrame:
    """Load metadata + annual results, attach utility, return slim frame."""
    df = pd.read_parquet(
        config.CR_ROOT / "CA_baseline_tmy_metadata_and_annual_results.parquet")
    # Promote ResStock bldg_id (the metadata index) into a column so it
    # survives the puma merge below.
    df = df.reset_index()  # creates column named after index ("bldg_id")

    # PUMA: take the second token of "in.county_and_puma" (e.g. "G06003729")
    puma_full = df["in.county_and_puma"].str.split(", ").str[1]
    df["puma_full"] = puma_full

    # Join utility info
    pum = pd.read_csv(config.PUMA_UTILITY)
    pum = pum[["PUMA", "utility_acronym", "utility_type", "climate_zone"]]
    pum = pum.rename(columns={"PUMA": "puma_full",
                              "utility_acronym": "utility",
                              "climate_zone": "puma_cec_cz"})
    df = df.merge(pum, on="puma_full", how="left")

    # Slim columns we need
    keep = [
        "bldg_id",
        "weight",
        "puma_full", "utility", "utility_type",
        "in.cec_climate_zone",
        "in.area_median_income", "in.federal_poverty_level",
        "in.heating_fuel", "in.geometry_building_type_recs",
        "in.vintage", "in.tenure", "in.sqft", "in.county_name",
        "out.electricity.total.energy_consumption.kwh",
        "out.natural_gas.total.energy_consumption.kwh",
        "out.electricity.cooling.energy_consumption.kwh",
        "out.electricity.heating.energy_consumption.kwh",
        "out.electricity.hot_water.energy_consumption.kwh",
        "out.electricity.summer.peak.kw",
        "out.electricity.winter.peak.kw",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def apply_scope_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply paper scope. Returns (kept, excluded_summary)."""
    rows = []
    n0 = len(df)
    rows.append(("total_metadata_rows", n0, df["weight"].sum()))

    # IOU only
    df["utility_low"] = df["utility"].str.lower()
    iou = df[df["utility_low"].isin(config.INCLUDED_UTILITIES)].copy()
    rows.append(("after_iou_only", len(iou), iou["weight"].sum()))

    # Drop POU explicitly (utility_type != 'IOU')
    if "utility_type" in iou.columns:
        iou = iou[iou["utility_type"] == "IOU"]
        rows.append(("after_drop_pou", len(iou), iou["weight"].sum()))

    # Owner only (renters can't make capex decisions)
    own = iou[iou["in.tenure"] == "Owner"].copy()
    rows.append(("after_owner_only", len(own), own["weight"].sum()))

    # Drop unknown income
    own = own[own["in.area_median_income"] != "Not Available"]
    rows.append(("after_drop_unknown_income", len(own), own["weight"].sum()))

    # EBD-eligible: <=80% AMI AND in priority CZ
    own["ami_frac"] = own["in.area_median_income"].map(AMI_BIN_TO_FRAC)
    own["cec_cz"] = own["in.cec_climate_zone"].astype(int)
    ebd = (own["ami_frac"] <= config.EBD_AMI_THRESHOLD) & (
        own["cec_cz"].isin(config.EBD_PRIORITY_CEC_CZS))
    excluded_ebd = own[ebd]
    rows.append(("excluded_ebd_eligible",
                 len(excluded_ebd), excluded_ebd["weight"].sum()))
    own = own[~ebd]
    rows.append(("kept_for_analysis", len(own), own["weight"].sum()))

    summary = pd.DataFrame(rows, columns=["step", "rows", "weighted_pop"])
    return own, summary


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature columns for clustering."""
    df = df.copy()
    e_total = df["out.electricity.total.energy_consumption.kwh"].astype(float)
    df["annual_kwh"] = e_total
    df["annual_therms"] = (
        df["out.natural_gas.total.energy_consumption.kwh"].astype(float)
        / 29.3001)  # kWh -> therms (1 therm = 29.3001 kWh)
    e_cool = df["out.electricity.cooling.energy_consumption.kwh"].astype(float)
    e_heat = df["out.electricity.heating.energy_consumption.kwh"].astype(float)
    df["cooling_share"] = np.where(e_total > 0, e_cool / e_total, 0)
    df["hvac_share"] = np.where(e_total > 0, (e_cool + e_heat) / e_total, 0)
    df["summer_peak_kw"] = df["out.electricity.summer.peak.kw"].astype(float)
    df["winter_peak_kw"] = df["out.electricity.winter.peak.kw"].astype(float)
    df["sqft"] = df["in.sqft"].astype(float)
    df["vintage_decade"] = df["in.vintage"].map(VINTAGE_DECADE).fillna("post2000")
    df["ami_bin"] = df["in.area_median_income"]
    df["building_type"] = df["in.geometry_building_type_recs"]
    df["heating_fuel"] = df["in.heating_fuel"]
    return df


FEATURE_COLS = [
    "annual_kwh", "annual_therms", "summer_peak_kw", "winter_peak_kw",
    "cooling_share", "hvac_share", "sqft",
]


def cluster_and_pick_medoids(
    df: pd.DataFrame,
    target_total: int = 500,
) -> pd.DataFrame:
    """Stratify, cluster within strata, pick medoids.

    target_total guides per-stratum k selection so we land near target_total
    medoids globally.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    strata_cols = [
        "utility", "cec_cz", "heating_fuel", "building_type",
        "ami_bin", "vintage_decade",
    ]
    grouped = df.groupby(strata_cols, dropna=False)
    n_strata = len(grouped)
    # Per-stratum k - small strata get k=1, large get up to 5
    avg_k = max(1, target_total // max(n_strata, 1))
    rng = np.random.default_rng(42)

    medoids = []
    for keys, sub in grouped:
        if len(sub) <= 1:
            sub = sub.copy()
            sub["cluster_id"] = 0
            sub["n_in_cluster"] = len(sub)
            sub["cluster_weight"] = sub["weight"].sum()
            medoids.append(sub.iloc[[0]])
            continue
        k = min(max(avg_k, 1), 5, len(sub))
        X = sub[FEATURE_COLS].fillna(0).values
        Xs = StandardScaler().fit_transform(X)
        # KMeans++ with fixed seed for reproducibility
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
        sub = sub.copy()
        sub["cluster_id"] = km.labels_
        # Medoid = point closest to its cluster centroid in standardized space
        centroids = km.cluster_centers_
        dists = np.linalg.norm(Xs - centroids[km.labels_], axis=1)
        sub["_dist"] = dists
        for cid, csub in sub.groupby("cluster_id"):
            med = csub.loc[csub["_dist"].idxmin()].copy()
            med["n_in_cluster"] = len(csub)
            med["cluster_weight"] = csub["weight"].sum()
            medoids.append(med.to_frame().T)

    out = pd.concat(medoids, ignore_index=True)
    out = out.drop(columns=[c for c in ("_dist",) if c in out.columns])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500,
                    help="Approx target number of representative buildings.")
    ap.add_argument("--out", default=str(
        config.DATA_DIR / "representative_buildings.parquet"))
    ap.add_argument("--summary-out", default=str(
        config.DATA_DIR / "population_excluded_summary.csv"))
    args = ap.parse_args()

    config.assert_safe_out_dir(Path(args.out).parent)
    config.assert_safe_out_dir(Path(args.summary_out).parent)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata + annual results ...")
    df = load_metadata()
    print(f"  {len(df):,} rows")

    print("Applying scope filter ...")
    df_kept, summary = apply_scope_filter(df)
    print(summary.to_string(index=False))

    print("Building features ...")
    df_feat = build_features(df_kept)

    print("Clustering by stratum + picking medoids ...")
    medoids = cluster_and_pick_medoids(df_feat, target_total=args.target)
    print(f"  {len(medoids)} medoids; weighted pop "
          f"{medoids['cluster_weight'].sum():,.0f}")

    medoids.to_parquet(args.out, index=False)
    summary.to_csv(args.summary_out, index=False)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
