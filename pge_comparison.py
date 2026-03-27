"""
pge_comparison.py - Compare simulated ResStock sample against actual PGE utility data.

Generates comparison tables and statistics for the paper's data validation section.
Mirrors sdge_comparison.py structure for consistency.

Note on BTM solar adjustment:
  EIA Form 861 "sales" report net electricity delivered to customers. Customers with
  behind-the-meter (BTM) solar generate electricity that is either self-consumed on-site
  or exported to the grid. Neither component appears in utility sales figures. Since
  ResStock's out.electricity.total.energy_consumption.kwh reports gross building load
  (before any PV offset), comparing it to net utility sales overstates the gap. We
  estimate total BTM solar generation and add it back to utility sales to compute
  "gross consumption" for a fairer apples-to-apples comparison.
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

# ============================================================================
# BTM SOLAR ESTIMATES (for adjusting utility sales to gross consumption)
# Sources: CA DG Stats (californiadgstats.ca.gov), CEC, CPUC NEM reports
# ============================================================================
BTM_SOLAR_PGE = {
    'pv_adoption_pct': 12.0,                  # ~12% of residential customers (CA DG Stats)
    'avg_system_kw': 6.5,                     # avg residential system size (CEC/CSI data)
    'annual_gen_kwh_per_kw': 1_550,           # PG&E territory avg (coastal+valley+mountain)
    # Derived: ~606K solar customers × 6.5 kW × 1,550 kWh/kW = ~6,100 GWh BTM generation
}

def compute_btm_adjustment(actual_data, btm_data):
    """Compute BTM solar generation and adjusted gross consumption figures."""
    n_solar = actual_data['total_customers'] * btm_data['pv_adoption_pct'] / 100
    total_capacity_mw = n_solar * btm_data['avg_system_kw'] / 1_000
    btm_gen_gwh = n_solar * btm_data['avg_system_kw'] * btm_data['annual_gen_kwh_per_kw'] / 1e9
    gross_sales_gwh = actual_data['total_sales_gwh'] + btm_gen_gwh
    gross_avg_kwh = gross_sales_gwh * 1e6 / actual_data['total_customers']
    return {
        'n_solar_customers': n_solar,
        'total_capacity_mw': total_capacity_mw,
        'btm_gen_gwh': btm_gen_gwh,
        'gross_sales_gwh': gross_sales_gwh,
        'gross_avg_kwh': gross_avg_kwh,
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


def print_comparison_table(stats, btm_adj):
    """Print formatted comparison table."""
    A = ACTUAL_PGE
    S = stats
    B = btm_adj

    print("\n" + "=" * 95)
    print("SIMULATED SAMPLE vs ACTUAL PGE — COMPARISON FOR PAPER")
    print("=" * 95)

    print(f"\n  BTM Solar Adjustment:")
    print(f"    Est. solar customers: {B['n_solar_customers']:,.0f} ({BTM_SOLAR_PGE['pv_adoption_pct']:.0f}% adoption)")
    print(f"    Est. BTM generation:  {B['btm_gen_gwh']:,.0f} GWh/yr")
    print(f"    Utility net sales:    {A['total_sales_gwh']:,.0f} GWh/yr")
    print(f"    Gross consumption:    {B['gross_sales_gwh']:,.0f} GWh/yr (net sales + BTM solar)")
    print(f"    Gross avg per cust:   {B['gross_avg_kwh']:,.0f} kWh/yr")

    rows = [
        ("Panel A: Service Territory", "", "", ""),
        ("Total residential customers", f"{A['total_customers']:,}", f"{S['n_buildings']:,} (sample)", ""),
        ("Weighted households represented", f"{A['total_customers']:,}", f"{S['total_represented']:,.0f}", f"{S['total_represented']/A['total_customers']*100:.1f}%"),
        ("", "", "", ""),
        ("Panel B: Consumption", "", "", ""),
        ("Total sales, raw ResStock (GWh/yr)", f"{A['total_sales_gwh']:,.0f}", f"{S['raw_total_gwh']:,.0f}", f"{S['raw_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("Total sales, RASS-scaled (GWh/yr)", f"{A['total_sales_gwh']:,.0f}", f"{S['scaled_total_gwh']:,.0f}", f"{S['scaled_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("  BTM-adj gross (GWh/yr)", f"{B['gross_sales_gwh']:,.0f}", f"{S['scaled_total_gwh']:,.0f}", f"{S['scaled_total_gwh']/B['gross_sales_gwh']*100:.1f}%"),
        ("Mean consumption, raw (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['raw_avg_kwh']:,.0f}", f"{S['raw_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("Mean consumption, RASS-scaled (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['scaled_avg_kwh']:,.0f}", f"{S['scaled_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("  BTM-adj gross mean (kWh/yr)", f"{B['gross_avg_kwh']:,.0f}", f"{S['scaled_avg_kwh']:,.0f}", f"{S['scaled_avg_kwh']/B['gross_avg_kwh']*100:.1f}%"),
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


def generate_latex_table(stats, btm_adj):
    """Generate LaTeX table for the paper."""
    A = ACTUAL_PGE
    S = stats
    B = btm_adj

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

    coastal_raw = region_avg(coastal_czs, 'raw_avg_kwh')
    valley_raw = region_avg(valley_czs, 'raw_avg_kwh')
    mountain_raw = region_avg(mountain_czs, 'raw_avg_kwh')

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of Simulated Building Sample with Actual PG\&E Service Territory Characteristics}
\label{tab:sample_comparison_pge}
\begin{tabular}{lrrr}
\toprule
\textbf{Characteristic} & \textbf{Actual PG\&E} & \textbf{ResStock (Raw)} & \textbf{ResStock (RASS-Adjusted)} \\
\midrule
\multicolumn{4}{l}{\textit{Panel A: Service Territory}} \\
Total residential customers & """ + f"{A['total_customers']:,}" + r""" & \multicolumn{2}{c}{""" + f"{S['n_buildings']:,}" + r""" (sample)} \\
Weighted households represented & """ + f"{A['total_customers']:,}" + r""" & \multicolumn{2}{c}{""" + f"{S['total_represented']:,.0f}" + r"""} \\
\midrule
\multicolumn{4}{l}{\textit{Panel B: Electricity Consumption}} \\
Total residential sales (GWh/yr) & """ + f"{A['total_sales_gwh']:,.0f}" + r""" & """ + f"{S['raw_total_gwh']:,.0f}" + r""" & """ + f"{S['scaled_total_gwh']:,.0f}" + r"""$^a$ \\
\quad + Est.\ BTM solar gen.\ (GWh/yr) & """ + f"{B['btm_gen_gwh']:,.0f}" + r"""$^d$ & --- & --- \\
\quad = Gross consumption (GWh/yr) & """ + f"{B['gross_sales_gwh']:,.0f}" + r""" & """ + f"{S['raw_total_gwh']:,.0f}" + r""" & """ + f"{S['scaled_total_gwh']:,.0f}" + r""" \\
Mean consumption (kWh/cust/yr) & """ + f"{A['avg_consumption_kwh']:,.0f}" + r""" & """ + f"{S['raw_avg_kwh']:,.0f}" + r""" & """ + f"{S['scaled_avg_kwh']:,.0f}" + r""" \\
\quad BTM-adjusted gross mean & """ + f"{B['gross_avg_kwh']:,.0f}" + r""" & --- & """ + f"{S['scaled_avg_kwh']:,.0f}" + r""" \\
Median consumption (kWh/cust/yr) & --- & --- & """ + f"{S['scaled_median_kwh']:,.0f}" + r""" \\
10th percentile (kWh/yr) & --- & --- & """ + f"{S['scaled_p10']:,.0f}" + r""" \\
90th percentile (kWh/yr) & --- & --- & """ + f"{S['scaled_p90']:,.0f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textit{Panel C: Customer Composition}} \\
CARE-eligible (\%) & """ + f"{A['care_pct']:.1f}" + r""" & \multicolumn{2}{c}{""" + f"{S['low_income_pct']:.1f}" + r"""$^b$} \\
Homeownership rate (\%) & $\sim$55$^c$ & \multicolumn{2}{c}{""" + f"{S['owner_pct']:.1f}" + r"""} \\
Solar PV adoption (\%) & $\sim$12$^c$ & \multicolumn{2}{c}{""" + f"{S['pv_pct']:.1f}" + r"""} \\
\midrule
\multicolumn{4}{l}{\textit{Panel D: Housing Stock}} \\
Single-family detached (\%) & $\sim$57$^c$ & \multicolumn{2}{c}{""" + f"{S['sf_detached_pct']:.1f}" + r"""} \\
Single-family attached (\%) & $\sim$7$^c$ & \multicolumn{2}{c}{""" + f"{S['sf_attached_pct']:.1f}" + r"""} \\
Multi-family (\%) & $\sim$32$^c$ & \multicolumn{2}{c}{""" + f"{S['mf_pct']:.1f}" + r"""} \\
Mobile home (\%) & $\sim$4$^c$ & \multicolumn{2}{c}{""" + f"{S['mobile_pct']:.1f}" + r"""} \\
\midrule
\multicolumn{4}{l}{\textit{Panel E: Climate Zone Distribution}} \\
CZ 1--5 --- Coastal (\%) & --- & \multicolumn{2}{c}{""" + f"{coastal_pct:.1f}" + r"""} \\
CZ 11--13 --- Central Valley (\%) & --- & \multicolumn{2}{c}{""" + f"{valley_pct:.1f}" + r"""} \\
CZ 16 --- Mountain/Sierra (\%) & --- & \multicolumn{2}{c}{""" + f"{mountain_pct:.1f}" + r"""} \\
\midrule
\multicolumn{4}{l}{\textit{Panel F: Mean Consumption by Region (kWh/yr)}} \\
CZ 1--5 (Coastal) & --- & """ + f"{coastal_raw:,.0f}" + r""" & """ + f"{coastal_scaled:,.0f}" + r""" \\
CZ 11--13 (Central Valley) & --- & """ + f"{valley_raw:,.0f}" + r""" & """ + f"{valley_scaled:,.0f}" + r""" \\
CZ 16 (Mountain/Sierra) & --- & """ + f"{mountain_raw:,.0f}" + r""" & """ + f"{mountain_scaled:,.0f}" + r""" \\
\midrule
\multicolumn{4}{l}{\textit{Panel G: Building Systems}} \\
Natural gas heating (\%) & --- & \multicolumn{2}{c}{""" + f"{S['heating_natgas_pct']:.1f}" + r"""} \\
Electric heating (\%) & --- & \multicolumn{2}{c}{""" + f"{S['heating_elec_pct']:.1f}" + r"""} \\
Central AC (\%) & --- & \multicolumn{2}{c}{""" + f"{S['cooling_central_pct']:.1f}" + r"""} \\
No cooling (\%) & --- & \multicolumn{2}{c}{""" + f"{S['cooling_none_pct']:.1f}" + r"""} \\
\bottomrule
\end{tabular}
\begin{minipage}{\textwidth}
\vspace{0.3em}
\footnotesize
\textit{Notes:} Actual PG\&E data from utility General Rate Case filings and EIA Form 861. ``Raw'' values are direct ResStock simulation output; ``RASS-Adjusted'' applies climate-zone-specific scaling factors calibrated to the 2019 Residential Appliance Saturation Survey (RASS). \\
$^a$ Weighted total using ResStock sampling weights. ResStock reports gross building load (before any PV offset), while utility sales are net of BTM solar generation. \\
$^b$ Based on ResStock income classification (low income category); actual CARE enrollment is """ + f"{A['care_pct']:.1f}" + r"""\%. \\
$^c$ ACS 2020--2024 estimates for PG\&E service territory counties; not precisely service-territory-specific. \\
$^d$ Estimated BTM solar generation: """ + f"{B['n_solar_customers']:,.0f}" + r""" solar customers (""" + f"{BTM_SOLAR_PGE['pv_adoption_pct']:.0f}" + r"""\% adoption $\times$ """ + f"{A['total_customers']:,}" + r""" customers) $\times$ """ + f"{BTM_SOLAR_PGE['avg_system_kw']}" + r"""\,kW avg system $\times$ """ + f"{BTM_SOLAR_PGE['annual_gen_kwh_per_kw']:,}" + r"""\,kWh/kW/yr. Sources: CA DG Stats, CEC interconnection data. Adding BTM generation back to utility sales yields gross consumption, which is the appropriate comparator for ResStock's gross building load.
\end{minipage}
\end{table}
"""
    return latex


if __name__ == '__main__':
    print("Loading PGE sample from metadata...")
    pge_df = load_pge_sample()
    print(f"Loaded {len(pge_df)} buildings")

    stats = compute_simulated_stats(pge_df)
    btm_adj = compute_btm_adjustment(ACTUAL_PGE, BTM_SOLAR_PGE)
    print_comparison_table(stats, btm_adj)

    latex = generate_latex_table(stats, btm_adj)
    print("\n\n" + "=" * 85)
    print("LATEX TABLE (copy into paper)")
    print("=" * 85)
    print(latex)

    # Save LaTeX table to file
    with open('table_sample_comparison_pge.tex', 'w') as f:
        f.write(latex)
    print("\nSaved to: table_sample_comparison_pge.tex")
