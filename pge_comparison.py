"""
pge_comparison.py - Compare simulated ResStock sample against actual PGE utility data.

Generates comparison tables and statistics for the paper's data validation section.
Mirrors sdge_comparison.py structure for consistency.
"""

import pandas as pd
import numpy as np

# ============================================================================
# ACTUAL PGE DATA (from GRC filings, EIA, CPUC)
# ============================================================================
ACTUAL_PGE = {
    'total_customers': 5_047_461,             # EIA 861 bundled+unbundled
    'care_customers': 1_371_555,
    'non_care_customers': 3_675_906,
    'care_pct': 1_371_555 / 5_047_461 * 100,  # 27.2%
    'total_sales_gwh': 25_987.213,             # EIA 861 bundled+unbundled
    'avg_consumption_kwh': 25_987_213_000 / 5_047_461,  # ~5,149 kWh
    'residential_revenue_usd': 8_238_576_000,
    'avg_rate_usd_kwh': 8_238_576_000 / 25_987_213_000,
    'rate_base': 41_987_991_000,
}

# PGE climate zones in service territory
PGE_CLIMATE_ZONES = [1, 2, 3, 4, 5, 11, 12, 13, 16]

# CZ labels for PGE territory
PGE_CZ_LABELS = {
    1: 'North Coast (Arcata)',
    2: 'North Coast (Santa Rosa)',
    3: 'SF Bay Coast',
    4: 'SF Bay Inland',
    5: 'Central Coast (Santa Maria)',
    11: 'Sacramento Valley',
    12: 'Central Valley (Stockton)',
    13: 'Central Valley (Fresno)',
    16: 'Mountain (Sierra)',
}


def load_pge_sample():
    """Load and filter metadata to PGE territory using PUMA-utility mapping."""
    puma_util = pd.read_csv('puma_utility_data.csv')
    pge_pumas = set(puma_util[puma_util['utility_acronym'] == 'PGE']['PUMA'].values)

    df = pd.read_parquet('CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet')
    pge = df[df['puma20'].isin(pge_pumas)].copy()
    return pge


def compute_simulated_stats(pge_df):
    """Compute key statistics from the simulated sample."""
    n = len(pge_df)
    weight = pge_df['weight']
    scaled_kwh = pge_df['scaled_out.electricity.total.energy_consumption.kwh']
    raw_kwh = pge_df['out.electricity.total.energy_consumption.kwh']

    # Housing type summary
    housing = pge_df['in.geometry_building_type_acs'].value_counts()
    sf_det = housing.get('Single-Family Detached', 0)
    sf_att = housing.get('Single-Family Attached', 0)
    mobile = housing.get('Mobile Home', 0)
    mf = n - sf_det - sf_att - mobile

    # Tenure (excluding Not Available)
    tenure_known = pge_df[pge_df['in.tenure'] != 'Not Available']
    owner_pct = (tenure_known['in.tenure'] == 'Owner').sum() / len(tenure_known) * 100

    # CARE proxy: income_category == 'Low'
    low_income_pct = (pge_df['income_category'] == 'Low').sum() / n * 100

    # FPL-based CARE proxy (<200% FPL)
    fpl_known = pge_df[pge_df['in.federal_poverty_level'] != 'Not Available']
    care_fpl = fpl_known['in.federal_poverty_level'].isin(
        ['0-100%', '100-150%', '150-200%']
    ).sum() / len(fpl_known) * 100

    stats = {
        'n_buildings': n,
        'total_represented': weight.sum(),
        'raw_total_gwh': (raw_kwh * weight).sum() / 1e6,
        'scaled_total_gwh': (scaled_kwh * weight).sum() / 1e6,
        'raw_avg_kwh': raw_kwh.mean(),
        'scaled_avg_kwh': scaled_kwh.mean(),
        'scaled_median_kwh': scaled_kwh.median(),
        'scaled_std_kwh': scaled_kwh.std(),
        'scaled_p10': scaled_kwh.quantile(0.10),
        'scaled_p25': scaled_kwh.quantile(0.25),
        'scaled_p75': scaled_kwh.quantile(0.75),
        'scaled_p90': scaled_kwh.quantile(0.90),
        'sf_detached_pct': sf_det / n * 100,
        'sf_attached_pct': sf_att / n * 100,
        'mf_pct': mf / n * 100,
        'mobile_pct': mobile / n * 100,
        'owner_pct': owner_pct,
        'renter_pct': 100 - owner_pct,
        'low_income_pct': low_income_pct,
        'care_fpl_pct': care_fpl,
        'pv_pct': (pge_df['in.has_pv'] == 'Yes').sum() / n * 100,
        'heating_natgas_pct': (pge_df['in.heating_fuel'] == 'Natural Gas').sum() / n * 100,
        'heating_elec_pct': (pge_df['in.heating_fuel'] == 'Electricity').sum() / n * 100,
        'cooling_central_pct': (pge_df['in.hvac_cooling_type'] == 'Central AC').sum() / n * 100,
        'cooling_none_pct': (pge_df['in.hvac_cooling_type'] == 'None').sum() / n * 100,
    }

    # Climate zone distribution and consumption
    for cz in PGE_CLIMATE_ZONES:
        sub = pge_df[pge_df['in.cec_climate_zone'] == cz]
        stats[f'cz{cz}_pct'] = len(sub) / n * 100
        stats[f'cz{cz}_n'] = len(sub)
        if len(sub) > 0:
            stats[f'cz{cz}_avg_kwh'] = sub['scaled_out.electricity.total.energy_consumption.kwh'].mean()
            stats[f'cz{cz}_raw_avg_kwh'] = sub['out.electricity.total.energy_consumption.kwh'].mean()
            stats[f'cz{cz}_scale'] = sub['scaling_factor'].mean()
        else:
            stats[f'cz{cz}_avg_kwh'] = 0
            stats[f'cz{cz}_raw_avg_kwh'] = 0
            stats[f'cz{cz}_scale'] = 0

    return stats


def print_comparison_table(stats):
    """Print formatted comparison table."""
    A = ACTUAL_PGE
    S = stats

    print("\n" + "=" * 85)
    print("SIMULATED SAMPLE vs ACTUAL PGE — COMPARISON FOR PAPER")
    print("=" * 85)

    rows = [
        ("Panel A: Service Territory", "", "", ""),
        ("Total residential customers", f"{A['total_customers']:,}", f"{S['n_buildings']:,} (sample)", ""),
        ("Weighted households represented", f"{A['total_customers']:,}", f"{S['total_represented']:,.0f}", f"{S['total_represented']/A['total_customers']*100:.1f}%"),
        ("", "", "", ""),
        ("Panel B: Consumption", "", "", ""),
        ("Total sales, raw ResStock (GWh/yr)", f"{A['total_sales_gwh']:,.0f}", f"{S['raw_total_gwh']:,.0f}", f"{S['raw_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("Total sales, RASS-scaled (GWh/yr)", f"{A['total_sales_gwh']:,.0f}", f"{S['scaled_total_gwh']:,.0f}", f"{S['scaled_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("Mean consumption, raw (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['raw_avg_kwh']:,.0f}", f"{S['raw_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("Mean consumption, RASS-scaled (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['scaled_avg_kwh']:,.0f}", f"{S['scaled_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("Median consumption, RASS-scaled (kWh/yr)", "N/A", f"{S['scaled_median_kwh']:,.0f}", ""),
        ("10th percentile (kWh/yr)", "N/A", f"{S['scaled_p10']:,.0f}", ""),
        ("90th percentile (kWh/yr)", "N/A", f"{S['scaled_p90']:,.0f}", ""),
        ("", "", "", ""),
        ("Panel C: Customer Composition", "", "", ""),
        ("CARE-eligible (%)", f"{A['care_pct']:.1f}%", f"{S['low_income_pct']:.1f}% (low-income)", ""),
        ("  (<200% FPL proxy)", "", f"{S['care_fpl_pct']:.1f}%", ""),
        ("Homeownership rate (%)", "~55%*", f"{S['owner_pct']:.1f}%", ""),
        ("Solar PV adoption (%)", "~12%*", f"{S['pv_pct']:.1f}%", ""),
        ("", "", "", ""),
        ("Panel D: Housing Stock", "", "", ""),
        ("Single-family detached (%)", "~57%*", f"{S['sf_detached_pct']:.1f}%", ""),
        ("Single-family attached (%)", "~7%*", f"{S['sf_attached_pct']:.1f}%", ""),
        ("Multi-family (%)", "~32%*", f"{S['mf_pct']:.1f}%", ""),
        ("Mobile home (%)", "~4%*", f"{S['mobile_pct']:.1f}%", ""),
        ("", "", "", ""),
        ("Panel E: Climate Zone Distribution", "", "", ""),
    ]

    for cz in PGE_CLIMATE_ZONES:
        label = PGE_CZ_LABELS.get(cz, '')
        rows.append((f"CZ {cz} - {label} (%)", "---", f"{S[f'cz{cz}_pct']:.1f}%", ""))

    rows.extend([
        ("", "", "", ""),
        ("Panel F: Climate Zone Consumption (kWh/yr)", "", "", ""),
    ])
    for cz in PGE_CLIMATE_ZONES:
        if S[f'cz{cz}_n'] > 10:
            label = PGE_CZ_LABELS.get(cz, '')
            rows.append((
                f"CZ {cz} mean (raw / scaled)",
                "---",
                f"{S[f'cz{cz}_raw_avg_kwh']:,.0f} / {S[f'cz{cz}_avg_kwh']:,.0f}",
                ""
            ))

    rows.extend([
        ("", "", "", ""),
        ("Panel G: Building Systems", "", "", ""),
        ("Natural gas heating (%)", "---", f"{S['heating_natgas_pct']:.1f}%", ""),
        ("Electric heating (%)", "---", f"{S['heating_elec_pct']:.1f}%", ""),
        ("Central AC (%)", "---", f"{S['cooling_central_pct']:.1f}%", ""),
        ("No cooling (%)", "---", f"{S['cooling_none_pct']:.1f}%", ""),
    ])

    print(f"{'Characteristic':<50s} {'Actual PGE':>15s} {'Simulated':>15s} {'Ratio':>8s}")
    print("-" * 90)
    for row in rows:
        if row[1] == "" and row[2] == "" and row[3] == "":
            if row[0]:
                print(f"\n{row[0]}")
                print("-" * 90)
        else:
            print(f"  {row[0]:<48s} {row[1]:>15s} {row[2]:>15s} {row[3]:>8s}")

    print("\n* ACS 2020-2024 estimates for PGE service territory counties")
    print("  PGE service territory spans ~48 counties in northern and central California")


def generate_latex_table(stats):
    """Generate LaTeX table for the paper."""
    A = ACTUAL_PGE
    S = stats

    # Group CZs into regions for cleaner presentation
    coastal_czs = [1, 2, 3, 4, 5]
    valley_czs = [11, 12, 13]
    mountain_czs = [16]

    coastal_pct = sum(S[f'cz{cz}_pct'] for cz in coastal_czs)
    valley_pct = sum(S[f'cz{cz}_pct'] for cz in valley_czs)
    mountain_pct = sum(S[f'cz{cz}_pct'] for cz in mountain_czs)

    # Weighted average consumption by region
    def region_avg(czs, key):
        total_n = sum(S[f'cz{cz}_n'] for cz in czs)
        if total_n == 0:
            return 0
        return sum(S[f'cz{cz}_{key}'] * S[f'cz{cz}_n'] for cz in czs) / total_n

    coastal_scaled = region_avg(coastal_czs, 'avg_kwh')
    valley_scaled = region_avg(valley_czs, 'avg_kwh')
    mountain_scaled = region_avg(mountain_czs, 'avg_kwh')

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of Simulated Building Sample with Actual PG\&E Service Territory Characteristics}
\label{tab:sample_comparison_pge}
\begin{tabular}{lrr}
\toprule
\textbf{Characteristic} & \textbf{Actual PG\&E} & \textbf{Simulated Sample} \\
\midrule
\multicolumn{3}{l}{\textit{Panel A: Service Territory}} \\
Total residential customers & """ + f"{A['total_customers']:,}" + r""" & """ + f"{S['n_buildings']:,}" + r""" (sample) \\
Weighted households represented & """ + f"{A['total_customers']:,}" + r""" & """ + f"{S['total_represented']:,.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel B: Electricity Consumption}} \\
Total residential sales (GWh/yr) & """ + f"{A['total_sales_gwh']:,.0f}" + r""" & """ + f"{S['scaled_total_gwh']:,.0f}" + r"""$^a$ \\
Mean consumption (kWh/cust/yr) & """ + f"{A['avg_consumption_kwh']:,.0f}" + r""" & """ + f"{S['scaled_avg_kwh']:,.0f}" + r""" \\
Median consumption (kWh/cust/yr) & --- & """ + f"{S['scaled_median_kwh']:,.0f}" + r""" \\
10th percentile (kWh/yr) & --- & """ + f"{S['scaled_p10']:,.0f}" + r""" \\
90th percentile (kWh/yr) & --- & """ + f"{S['scaled_p90']:,.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel C: Customer Composition}} \\
CARE-eligible (\%) & """ + f"{A['care_pct']:.1f}" + r""" & """ + f"{S['low_income_pct']:.1f}" + r"""$^b$ \\
Homeownership rate (\%) & $\sim$55$^c$ & """ + f"{S['owner_pct']:.1f}" + r""" \\
Solar PV adoption (\%) & $\sim$12$^c$ & """ + f"{S['pv_pct']:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel D: Housing Stock}} \\
Single-family detached (\%) & $\sim$57$^c$ & """ + f"{S['sf_detached_pct']:.1f}" + r""" \\
Single-family attached (\%) & $\sim$7$^c$ & """ + f"{S['sf_attached_pct']:.1f}" + r""" \\
Multi-family (\%) & $\sim$32$^c$ & """ + f"{S['mf_pct']:.1f}" + r""" \\
Mobile home (\%) & $\sim$4$^c$ & """ + f"{S['mobile_pct']:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel E: Climate Zone Distribution}} \\
CZ 1--5 --- Coastal (\%) & --- & """ + f"{coastal_pct:.1f}" + r""" \\
CZ 11--13 --- Central Valley (\%) & --- & """ + f"{valley_pct:.1f}" + r""" \\
CZ 16 --- Mountain/Sierra (\%) & --- & """ + f"{mountain_pct:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel F: Mean Consumption by Region (kWh/yr)}} \\
CZ 1--5 (Coastal) & --- & """ + f"{coastal_scaled:,.0f}" + r""" \\
CZ 11--13 (Central Valley) & --- & """ + f"{valley_scaled:,.0f}" + r""" \\
CZ 16 (Mountain/Sierra) & --- & """ + f"{mountain_scaled:,.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel G: Building Systems}} \\
Natural gas heating (\%) & --- & """ + f"{S['heating_natgas_pct']:.1f}" + r""" \\
Electric heating (\%) & --- & """ + f"{S['heating_elec_pct']:.1f}" + r""" \\
Central AC (\%) & --- & """ + f"{S['cooling_central_pct']:.1f}" + r""" \\
No cooling (\%) & --- & """ + f"{S['cooling_none_pct']:.1f}" + r""" \\
\bottomrule
\end{tabular}
\begin{minipage}{\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Actual PG\&E data from utility General Rate Case filings and EIA Form 861. Simulated sample from NREL ResStock with RASS-calibrated scaling factors applied to consumption. \\
$^a$ Weighted total using ResStock sampling weights; exceeds actual PG\&E sales due to higher per-building consumption in the simulation (ResStock models occupied, full-year households only). \\
$^b$ Based on ResStock income classification (low income category); actual CARE enrollment is """ + f"{A['care_pct']:.1f}" + r"""\%. \\
$^c$ ACS 2020--2024 estimates for PG\&E service territory counties; not precisely service-territory-specific.
\end{minipage}
\end{table}
"""
    return latex


if __name__ == '__main__':
    print("Loading PGE sample from metadata...")
    pge_df = load_pge_sample()
    print(f"Loaded {len(pge_df)} buildings")

    stats = compute_simulated_stats(pge_df)
    print_comparison_table(stats)

    latex = generate_latex_table(stats)
    print("\n\n" + "=" * 85)
    print("LATEX TABLE (copy into paper)")
    print("=" * 85)
    print(latex)

    # Save LaTeX table to file
    with open('table_sample_comparison_pge.tex', 'w') as f:
        f.write(latex)
    print("\nSaved to: table_sample_comparison_pge.tex")
