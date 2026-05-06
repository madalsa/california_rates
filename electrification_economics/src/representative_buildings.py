"""Stratified sampling + clustering to pick representative buildings.

POPULATION SCOPE (paper):
  Include: IOU customers (PGE, SCE, SDGE) where NPV is a useful question.
  Exclude:
    - POU territories (handled upstream; not in pipeline output anyway)
    - EBD-eligible households: <=80% AMI in CEC priority climate zones
      (turnkey free retrofit -> payback question doesn't apply).
      Reported as a population share, not optimized.

Approach:
  1. Load ResStock metadata; filter to INCLUDED_UTILITIES; drop EBD-eligible.
  2. Stratify by (climate_zone, heating_fuel, dwelling_type, income_tier,
     vintage_bin). ~100-200 strata.
  3. Within each stratum, K-means (k=3-5) on load-shape features:
       - annual kWh
       - summer/winter kWh ratio
       - peak-period share (utility TOU windows)
       - weekday/weekend ratio
       - load factor
  4. For each cluster pick the medoid (closest to centroid) as the
     representative building. Carry a population weight = sum of RASS
     expansion weights of cluster members.

Output: parquet with one row per representative building, columns:
  building_id, utility, climate_zone, heating_fuel, dwelling_type,
  income_tier, vintage_bin, cluster_id, n_in_cluster, weight, features...

Also writes population_excluded_summary.csv with EBD-eligible counts and
weights (for reporting in the paper's methods section).
"""

# TODO(impl): load metadata + load profiles, filter, stratify, cluster,
# write medoids + excluded-population summary.
