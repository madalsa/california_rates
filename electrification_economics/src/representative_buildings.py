"""Stratified sampling + clustering to pick representative buildings.

Approach:
  1. Stratify ResStock metadata by (climate_zone, heating_fuel, dwelling_type,
     income_tier, vintage_bin). ~100-200 strata.
  2. Within each stratum, K-means (k=3-5) on load-shape features:
       - annual kWh
       - summer/winter kWh ratio
       - peak-period share (utility TOU windows)
       - weekday/weekend ratio
       - load factor
  3. For each cluster pick the medoid (closest to centroid) as the
     representative building. Carry a population weight = sum of RASS
     expansion weights of cluster members.

Output: parquet with one row per representative building, columns:
  building_id, utility, climate_zone, heating_fuel, dwelling_type,
  income_tier, vintage_bin, cluster_id, n_in_cluster, weight, features...
"""

# TODO(impl): load metadata + load profiles, stratify, cluster, write medoids.
