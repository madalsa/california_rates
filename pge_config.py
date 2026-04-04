"""
pge_config.py — Configuration and constants for PGE pipeline

Utility data from utility_data_inputs.tex (2026-03-23).
PGE E-TOU-C tariff structure with 4 TOU periods (no midpeak).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------

BASELINE_DIR = './Baseline_PGE'
UPGRADE11_DIR = './Upgrade11_PGE'
METADATA_FILE = 'CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet'
PUMA_UTILITY_FILE = 'puma_utility_data.csv'
TOU_WEIGHTS_FILE = 'tou_weights_pge.csv'
EEC_FILE = 'eec_hourly_2025_wide.csv'
EXCEL_FILE = 'retail_rates_data_PGE.xlsx'

# Output files
RATE_SCENARIOS_OUT = 'rate_scenarios_pge_fresh.csv'
BASELINE_BILLS_OUT = 'baseline_bills_pge_fresh.csv'
TECH_ASSIGNMENTS_OUT = 'tech_assignments_pge.csv'
POSTADOPT_BILLS_OUT = 'post_adoption_bills_pge.csv'
SUMMARY_OUT = 'pipeline_summary_pge.csv'

# ---------------------------------------------------------------------------
# PGE Utility Data (from utility_data_inputs.tex)
# ---------------------------------------------------------------------------

RESIDENTIAL_REVENUE = 8_240_000_000       # $8.24B
RESIDENTIAL_SALES_KWH = 25_987_000_000    # 25,987 GWh
TOTAL_RESIDENTIAL_CUSTOMERS = 5_047_461
CARE_CUSTOMERS = 1_371_555
BUILDING_WEIGHT = 252.3

CUSTOMERS = {
    'care': CARE_CUSTOMERS,
    'non_care': TOTAL_RESIDENTIAL_CUSTOMERS - CARE_CUSTOMERS,
    'total': TOTAL_RESIDENTIAL_CUSTOMERS,
}

# Capital structure
RATE_BASE = 41_990_000_000       # $41.99B
EQUITY_SHARE = 0.52
AUTHORIZED_ROE = 0.1028          # 10.28%

# Cost components ($)
TRANSMISSION_COST = 2_660_000_000    # $2.66B
DISTRIBUTION_COST = 8_740_000_000    # $8.74B
WILDFIRE_COST = 5_400_000_000        # $5.40B

# ---------------------------------------------------------------------------
# PGE E-TOU-C tariff structure
# ---------------------------------------------------------------------------

ACTUAL_PGE_RATES = {
    'E-TOU-C': 'e_tou_c',
    'E-TOU-C-F': 'e_tou_c_f',
}

DESIGNED_SCENARIOS = [
    'F0_WF0_ROE0',
    'F50_WF0_ROE0',
    'F100_WF0_ROE0',
    'F0_WF1_ROE0',
    'F0_WF0_ROE1.0',
    'F50_WF1_ROE1.0',
]

# PGE TOU: 4 periods (NO midpeak)
# Summer (Jun-Oct): Peak 4-9pm, Offpeak all other
# Winter (Nov-May): Peak 4-9pm, Offpeak all other
SUMMER_MONTHS = set(range(6, 11))
TOU_PERIODS = ['summer_peak', 'summer_offpeak', 'winter_peak', 'winter_offpeak']

# ---------------------------------------------------------------------------
# PGE CEC climate zone coordinates for pvlib
# ---------------------------------------------------------------------------

PGE_CZ_COORDINATES = {
    1:  (41.40, -124.10, 10,  'Arcata'),
    2:  (40.48, -122.30, 150, 'Redding'),
    3:  (37.75, -122.40, 20,  'San Francisco'),
    4:  (37.40, -122.10, 30,  'Sunnyvale'),
    5:  (37.77, -122.00, 100, 'Livermore'),
    6:  (36.30, -119.30, 90,  'Visalia'),
    11: (35.40, -119.05, 120, 'Bakersfield'),
    12: (38.55, -121.50, 10,  'Sacramento'),
    13: (37.72, -121.05, 30,  'Modesto'),
    14: (36.77, -119.72, 100, 'Fresno'),
    16: (39.80, -121.60, 600, 'Paradise'),
}

# ---------------------------------------------------------------------------
# Solar sizing
# ---------------------------------------------------------------------------

DEFAULT_PV_SIZE_KW = 5.0
PV_OFFSET_TARGET = 0.90           # 90% of native annual demand
PV_MIN_SIZE_KW = 2.0
PV_MAX_SIZE_KW = 12.0
PGE_ANNUAL_KWH_PER_KW = 1600.0

# ---------------------------------------------------------------------------
# Battery parameters
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 13.5
BATTERY_POWER_KW = 5.0
BATTERY_EFFICIENCY = 0.90

# ---------------------------------------------------------------------------
# EV parameters
# ---------------------------------------------------------------------------

EV_MILES_PER_KWH = 3.0
EV_CHARGE_START_HOUR = 22
EV_CHARGE_HOURS = 5

BEV_DVMT_CDF = np.array([
    [0,  0.00], [5,  0.03], [10, 0.12], [15, 0.25], [20, 0.42],
    [25, 0.58], [30, 0.70], [35, 0.80], [40, 0.87], [50, 0.95],
    [60, 0.98], [70, 1.00],
])

# ---------------------------------------------------------------------------
# Precomputed time arrays
# ---------------------------------------------------------------------------

def build_time_arrays():
    """Build hour→month, hour_of_day, is_weekend, is_summer arrays."""
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1

    hour_of_day = hours % 24
    day_of_year = hours // 24
    day_of_week = (day_of_year + 2) % 7  # Jan 1 = Wednesday
    is_weekend = day_of_week >= 5
    is_summer = (months >= 6) & (months <= 10)

    return {
        'hours': hours,
        'months': months,
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'is_summer': is_summer,
        'days_per_month': days_per_month,
        'month_boundaries': month_boundaries,
    }


def build_pge_period_masks(ta=None):
    """Build boolean masks for 4 PGE E-TOU-C periods (no midpeak)."""
    if ta is None:
        ta = build_time_arrays()
    hod = ta['hour_of_day']
    is_summer = ta['is_summer']
    is_peak = (hod >= 16) & (hod < 21)

    return {
        'summer_peak':    is_summer & is_peak,
        'summer_offpeak': is_summer & ~is_peak,
        'winter_peak':    ~is_summer & is_peak,
        'winter_offpeak': ~is_summer & ~is_peak,
    }


def build_tou_rate_array(rate_dict, ta=None):
    """Build 8760-length rate array from dict with 4 period keys."""
    masks = build_pge_period_masks(ta)
    rates = np.zeros(8760)
    for period, mask in masks.items():
        rates[mask] = rate_dict[period]
    return rates


def safe_float(val):
    """Convert NaN/None to 0.0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)
