"""
sdge_comparison.py - Compare simulated ResStock sample against actual SDGE utility data.

Generates comparison tables and statistics for the paper's data validation section.
"""

import pandas as pd
import numpy as np

# ============================================================================
# ACTUAL SDGE DATA (from GRC filings, EIA, CPUC)
# ============================================================================
ACTUAL_SDGE = {
    'total_customers': 1_364_361,           # EIA 861 bundled+unbundled
    'care_customers': 372_135,
    'non_care_customers': 992_226,          # includes ~40k unbundled/CCA
    'care_pct': 372_135 / 1_364_361 * 100,
    'total_sales_gwh': 4_809.988,
    'avg_consumption_kwh': 4_809_988_000 / 1_364_361,
    'residential_revenue_usd': 1_561_695_600,
    'avg_rate_usd_kwh': 1_561_695_600 / 4_809_988_000,
    'rate_base': 13_590_538_000,
}

# SDGE climate zone baseline allowances (from puma_utility_data.csv)
SDGE_CZ_BASELINES = {
    7: {'summer': 11.70, 'winter': 11.97},
    10: {'summer': 13.53, 'winter': 12.47},
    14: {'summer': 17.67, 'winter': 16.77},
}


def load_sdge_sample():
    """Load and filter metadata to SDGE territory."""
    df = pd.read_parquet('CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet')
    sdge = df[df['in.county_name'].str.contains('San Diego', na=False)].copy()
    return sdge


def compute_simulated_stats(sdge_df):
    """Compute key statistics from the simulated sample."""
    n = len(sdge_df)
    weight = sdge_df['weight']
    scaled_kwh = sdge_df['scaled_out.electricity.total.energy_consumption.kwh']
    raw_kwh = sdge_df['out.electricity.total.energy_consumption.kwh']

    # Housing type summary
    housing = sdge_df['in.geometry_building_type_acs'].value_counts()
    sf_det = housing.get('Single-Family Detached', 0)
    sf_att = housing.get('Single-Family Attached', 0)
    mobile = housing.get('Mobile Home', 0)
    mf = n - sf_det - sf_att - mobile

    # Tenure (excluding Not Available)
    tenure_known = sdge_df[sdge_df['in.tenure'] != 'Not Available']
    owner_pct = (tenure_known['in.tenure'] == 'Owner').sum() / len(tenure_known) * 100

    # CARE proxy: income_category == 'Low'
    low_income_pct = (sdge_df['income_category'] == 'Low').sum() / n * 100

    # FPL-based CARE proxy (<200% FPL)
    fpl_known = sdge_df[sdge_df['in.federal_poverty_level'] != 'Not Available']
    care_fpl = fpl_known['in.federal_poverty_level'].isin(
        ['0-100%', '100-150%', '150-200%']
    ).sum() / len(fpl_known) * 100

    # Climate zone distribution (main SDGE zones only)
    main_cz = sdge_df[sdge_df['in.cec_climate_zone'].isin([7, 10, 14])]

    stats = {
        'n_buildings': n,
        'total_represented': weight.sum(),
        'weight_per_building': weight.iloc[0],
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
        'pv_pct': (sdge_df['in.has_pv'] == 'Yes').sum() / n * 100,
        'cz7_pct': (sdge_df['in.cec_climate_zone'] == 7).sum() / n * 100,
        'cz10_pct': (sdge_df['in.cec_climate_zone'] == 10).sum() / n * 100,
        'cz14_pct': (sdge_df['in.cec_climate_zone'] == 14).sum() / n * 100,
        'main_cz_pct': len(main_cz) / n * 100,
        'heating_natgas_pct': (sdge_df['in.heating_fuel'] == 'Natural Gas').sum() / n * 100,
        'heating_elec_pct': (sdge_df['in.heating_fuel'] == 'Electricity').sum() / n * 100,
        'cooling_central_pct': (sdge_df['in.hvac_cooling_type'] == 'Central AC').sum() / n * 100,
        'cooling_none_pct': (sdge_df['in.hvac_cooling_type'] == 'None').sum() / n * 100,
    }

    # Climate zone consumption
    for cz in [7, 10, 14]:
        sub = sdge_df[sdge_df['in.cec_climate_zone'] == cz]
        stats[f'cz{cz}_avg_kwh'] = sub['scaled_out.electricity.total.energy_consumption.kwh'].mean()
        stats[f'cz{cz}_raw_avg_kwh'] = sub['out.electricity.total.energy_consumption.kwh'].mean()
        stats[f'cz{cz}_scale'] = sub['scaling_factor'].mean()
        stats[f'cz{cz}_n'] = len(sub)

    return stats


def print_comparison_table(stats):
    """Print formatted comparison table."""
    A = ACTUAL_SDGE
    S = stats

    print("\n" + "=" * 85)
    print("SIMULATED SAMPLE vs ACTUAL SDGE — COMPARISON FOR PAPER")
    print("=" * 85)

    rows = [
        ("Panel A: Service Territory", "", "", ""),
        ("Total residential customers", f"{A['total_customers']:,}", f"{S['n_buildings']:,} (sample)", ""),
        ("Weighted households represented", f"{A['total_customers']:,}", f"{S['total_represented']:,.0f}", f"{S['total_represented']/A['total_customers']*100:.1f}%"),
        ("", "", "", ""),
        ("Panel B: Consumption", "", "", ""),
        ("Total sales, raw ResStock (GWh/yr)", f"{A['total_sales_gwh']:,}", f"{S['raw_total_gwh']:,.0f}", f"{S['raw_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("Total sales, RASS-scaled (GWh/yr)", f"{A['total_sales_gwh']:,}", f"{S['scaled_total_gwh']:,.0f}", f"{S['scaled_total_gwh']/A['total_sales_gwh']*100:.1f}%"),
        ("Mean consumption, raw (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['raw_avg_kwh']:,.0f}", f"{S['raw_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("Mean consumption, RASS-scaled (kWh/yr)", f"{A['avg_consumption_kwh']:,.0f}", f"{S['scaled_avg_kwh']:,.0f}", f"{S['scaled_avg_kwh']/A['avg_consumption_kwh']*100:.1f}%"),
        ("Median consumption, RASS-scaled (kWh/yr)", "N/A", f"{S['scaled_median_kwh']:,.0f}", ""),
        ("", "", "", ""),
        ("Panel C: Customer Composition", "", "", ""),
        ("CARE-eligible (%)", f"{A['care_pct']:.1f}%", f"{S['low_income_pct']:.1f}% (low-income)", ""),
        ("  (<200% FPL proxy)", "", f"{S['care_fpl_pct']:.1f}%", ""),
        ("Homeownership rate (%)", "~54%*", f"{S['owner_pct']:.1f}%", ""),
        ("Solar PV adoption (%)", "~9%*", f"{S['pv_pct']:.1f}%", ""),
        ("", "", "", ""),
        ("Panel D: Housing Stock", "", "", ""),
        ("Single-family detached (%)", "~51%*", f"{S['sf_detached_pct']:.1f}%", ""),
        ("Single-family attached (%)", "~10%*", f"{S['sf_attached_pct']:.1f}%", ""),
        ("Multi-family (%)", "~35%*", f"{S['mf_pct']:.1f}%", ""),
        ("Mobile home (%)", "~4%*", f"{S['mobile_pct']:.1f}%", ""),
        ("", "", "", ""),
        ("Panel E: Climate Zone Distribution", "", "", ""),
        ("CZ 7 - Coastal (%)", "~70%*", f"{S['cz7_pct']:.1f}%", ""),
        ("CZ 10 - Inland (%)", "~28%*", f"{S['cz10_pct']:.1f}%", ""),
        ("CZ 14 - Mountain (%)", "~2%*", f"{S['cz14_pct']:.1f}%", ""),
        ("", "", "", ""),
        ("Panel F: Climate Zone Consumption (kWh/yr)", "", "", ""),
        ("CZ 7 mean (raw / scaled)", "~4,800*", f"{S['cz7_raw_avg_kwh']:,.0f} / {S['cz7_avg_kwh']:,.0f}", ""),
        ("CZ 10 mean (raw / scaled)", "~6,900*", f"{S['cz10_raw_avg_kwh']:,.0f} / {S['cz10_avg_kwh']:,.0f}", ""),
        ("CZ 14 mean (raw / scaled)", "~7,800*", f"{S['cz14_raw_avg_kwh']:,.0f} / {S['cz14_avg_kwh']:,.0f}", ""),
    ]

    print(f"{'Characteristic':<45s} {'Actual SDGE':>15s} {'Simulated':>15s} {'Ratio':>8s}")
    print("-" * 85)
    for row in rows:
        if row[1] == "" and row[2] == "" and row[3] == "":
            if row[0]:
                print(f"\n{row[0]}")
                print("-" * 85)
        else:
            print(f"  {row[0]:<43s} {row[1]:>15s} {row[2]:>15s} {row[3]:>8s}")

    print("\n* ACS/Census estimates for San Diego County; not SDGE-specific")
    print("  SDGE service territory covers most but not all of San Diego County")


def generate_latex_table(stats):
    """Generate LaTeX table for the paper."""
    A = ACTUAL_SDGE
    S = stats

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Simulated Building Sample with Actual SDGE Service Territory Characteristics}
\label{tab:sample_comparison}
\begin{tabular}{lrr}
\toprule
\textbf{Characteristic} & \textbf{Actual SDGE} & \textbf{Simulated Sample} \\
\midrule
\multicolumn{3}{l}{\textit{Panel A: Service Territory}} \\
Total residential customers & """ + f"{A['total_customers']:,}" + r""" & """ + f"{S['n_buildings']:,}" + r""" (sample) \\
Weighted households represented & """ + f"{A['total_customers']:,}" + r""" & """ + f"{S['total_represented']:,.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel B: Electricity Consumption}} \\
Total sales, raw ResStock (GWh/yr) & """ + f"{A['total_sales_gwh']:,}" + r""" & """ + f"{S['raw_total_gwh']:,.0f}" + r""" \\
Total sales, RASS-scaled (GWh/yr) & """ + f"{A['total_sales_gwh']:,}" + r""" & """ + f"{S['scaled_total_gwh']:,.0f}" + r"""$^a$ \\
Mean consumption, raw (kWh/cust/yr) & """ + f"{A['avg_consumption_kwh']:,.0f}" + r""" & """ + f"{S['raw_avg_kwh']:,.0f}" + r""" \\
Mean consumption, RASS-scaled (kWh/cust/yr) & """ + f"{A['avg_consumption_kwh']:,.0f}" + r""" & """ + f"{S['scaled_avg_kwh']:,.0f}" + r""" \\
Median consumption, RASS-scaled (kWh/cust/yr) & --- & """ + f"{S['scaled_median_kwh']:,.0f}" + r""" \\
10th percentile (kWh/yr) & --- & """ + f"{S['scaled_p10']:,.0f}" + r""" \\
90th percentile (kWh/yr) & --- & """ + f"{S['scaled_p90']:,.0f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel C: Customer Composition}} \\
CARE-eligible (\%) & """ + f"{A['care_pct']:.1f}" + r""" & """ + f"{S['low_income_pct']:.1f}" + r"""$^b$ \\
Homeownership rate (\%) & $\sim$54$^c$ & """ + f"{S['owner_pct']:.1f}" + r""" \\
Solar PV adoption (\%) & $\sim$9$^c$ & """ + f"{S['pv_pct']:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel D: Housing Stock}} \\
Single-family detached (\%) & $\sim$51$^c$ & """ + f"{S['sf_detached_pct']:.1f}" + r""" \\
Single-family attached (\%) & $\sim$10$^c$ & """ + f"{S['sf_attached_pct']:.1f}" + r""" \\
Multi-family (\%) & $\sim$35$^c$ & """ + f"{S['mf_pct']:.1f}" + r""" \\
Mobile home (\%) & $\sim$4$^c$ & """ + f"{S['mobile_pct']:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel E: Climate Zone Distribution}} \\
CZ 7 --- Coastal (\%) & --- & """ + f"{S['cz7_pct']:.1f}" + r""" \\
CZ 10 --- Inland (\%) & --- & """ + f"{S['cz10_pct']:.1f}" + r""" \\
CZ 14 --- Mountain (\%) & --- & """ + f"{S['cz14_pct']:.1f}" + r""" \\
\midrule
\multicolumn{3}{l}{\textit{Panel F: Mean Consumption by Climate Zone (kWh/yr)}} \\
CZ 7 --- raw / RASS-scaled & --- & """ + f"{S['cz7_raw_avg_kwh']:,.0f}" + r""" / """ + f"{S['cz7_avg_kwh']:,.0f}" + r""" \\
CZ 10 --- raw / RASS-scaled & --- & """ + f"{S['cz10_raw_avg_kwh']:,.0f}" + r""" / """ + f"{S['cz10_avg_kwh']:,.0f}" + r""" \\
CZ 14 --- raw / RASS-scaled & --- & """ + f"{S['cz14_raw_avg_kwh']:,.0f}" + r""" / """ + f"{S['cz14_avg_kwh']:,.0f}" + r""" \\
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
\textit{Notes:} Actual SDGE data from utility General Rate Case filings and EIA Form 861. ``Raw'' values are direct ResStock simulation output; ``RASS-scaled'' values apply RASS-calibrated per-building scaling factors to better match observed consumption patterns by building type. \\
$^a$ Weighted total using ResStock sampling weights; RASS-scaled total exceeds actual SDGE sales due to overcounting in some building types. \\
$^b$ Based on ResStock income classification (low income category); actual CARE eligibility is 28.1\%. \\
$^c$ ACS 2020--2024 estimates for San Diego County; not SDGE service-territory-specific.
\end{minipage}
\end{table}
"""
    return latex


if __name__ == '__main__':
    print("Loading SDGE sample from metadata...")
    sdge_df = load_sdge_sample()
    print(f"Loaded {len(sdge_df)} buildings")

    stats = compute_simulated_stats(sdge_df)
    print_comparison_table(stats)

    latex = generate_latex_table(stats)
    print("\n\n" + "=" * 85)
    print("LATEX TABLE (copy into paper)")
    print("=" * 85)
    print(latex)

    # Save LaTeX table to file
    with open('table_sample_comparison.tex', 'w') as f:
        f.write(latex)
    print("\nSaved to: table_sample_comparison.tex")
