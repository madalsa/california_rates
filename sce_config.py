"""
sce_config.py — Configuration and constants for SCE pipeline

Utility data from utility_data_inputs.tex (2026-03-23).
SCE TOU-D-4-9 tariff structure with 5 TOU periods.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------

BASELINE_DIR = './Baseline_SCE'
METADATA_FILE = 'CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet'
PUMA_UTILITY_FILE = 'puma_utility_data.csv'
TOU_WEIGHTS_FILE = 'tou_weights_sce.csv'
EEC_FILE = 'eec_hourly_2025_wide.csv'
EXCEL_FILE = 'retail_rates_data_SCE.xlsx'
BUILDING_IDS_FILE = 'sce_building_ids.txt'

# Output files
RATE_SCENARIOS_OUT = 'rate_scenarios_sce_fresh.csv'
BASELINE_BILLS_OUT = 'baseline_bills_sce_fresh.csv'
TECH_ASSIGNMENTS_OUT = 'tech_assignments_sce.csv'
POSTADOPT_BILLS_OUT = 'post_adoption_bills_sce.csv'
SUMMARY_OUT = 'pipeline_summary_sce.csv'

# ---------------------------------------------------------------------------
# SCE Utility Data (from utility_data_inputs.tex)
# ---------------------------------------------------------------------------

RESIDENTIAL_REVENUE = 7_750_000_000       # $7.75B (EIA 861)
RESIDENTIAL_SALES_KWH = 27_414_000_000    # 27,414 GWh
TOTAL_RESIDENTIAL_CUSTOMERS = 4_594_415
CARE_CUSTOMERS = 1_353_981
BUILDING_WEIGHT = 252.3                   # uniform ResStock weight

CUSTOMERS = {
    'care': CARE_CUSTOMERS,
    'non_care': TOTAL_RESIDENTIAL_CUSTOMERS - CARE_CUSTOMERS,
    'total': TOTAL_RESIDENTIAL_CUSTOMERS,
}

# Capital structure (from utility_data_inputs.tex)
RATE_BASE = 41_430_000_000       # $41.43B
EQUITY_SHARE = 0.52
AUTHORIZED_ROE = 0.1033          # 10.33%

# Cost components ($B → $)
TRANSMISSION_COST = 1_120_000_000    # $1.12B
DISTRIBUTION_COST = 8_940_000_000    # $8.94B
WILDFIRE_COST = 2_930_000_000        # $2.93B

# ---------------------------------------------------------------------------
# SCE TOU-D-4-9 tariff structure
# ---------------------------------------------------------------------------

# Actual tariff rates (from retail_rates_data_SCE.xlsx)
ACTUAL_SCE_RATES = {
    'TOU-D-4-9': 'tou_d_4_9',
    'TOU-D-4-9-F': 'tou_d_4_9_f',
}

# Designed rate scenarios to evaluate
DESIGNED_SCENARIOS = [
    'F0_WF0_ROE0',
    'F50_WF0_ROE0',
    'F100_WF0_ROE0',
    'F0_WF1_ROE0',
    'F0_WF0_ROE1.0',
    'F50_WF1_ROE1.0',
]

# TOU periods: 5 periods (summer has no midpeak)
# Summer (Jun-Oct): Peak 4-9pm, Off-peak all other hours
# Winter (Nov-May): Peak 4-9pm, Mid-peak 9pm-8am, Off-peak 8am-4pm
SUMMER_MONTHS = set(range(6, 11))  # June-October (1-indexed)
TOU_PERIODS = ['summer_peak', 'summer_offpeak',
               'winter_peak', 'winter_midpeak', 'winter_offpeak']

# Weekday/weekend fractions (for blending summer peak)
WEEKDAY_FRAC = 5 / 7
WEEKEND_FRAC = 2 / 7

# ---------------------------------------------------------------------------
# SCE CEC Climate Zone coordinates for pvlib
# ---------------------------------------------------------------------------

SCE_CZ_COORDINATES = {
    5:  (34.00, -118.50, 30,  'Santa Monica'),
    6:  (33.77, -118.19, 10,  'Long Beach'),
    8:  (34.05, -118.25, 90,  'Downtown LA'),
    9:  (34.05, -118.25, 90,  'LA Basin'),       # same as CZ8 fallback
    10: (33.95, -117.40, 310, 'Riverside'),
    13: (37.72, -121.05, 30,  'Modesto'),
    14: (34.27, -118.40, 380, 'Palmdale area'),
    15: (33.75, -116.53, 150, 'Palm Springs'),
    16: (34.75, -118.35, 800, 'Mountain/high desert'),
}

# ---------------------------------------------------------------------------
# Solar sizing parameters
# ---------------------------------------------------------------------------

DEFAULT_PV_SIZE_KW = 7.0          # Fallback
PV_OFFSET_TARGET = 0.90           # 90% of native annual demand
PV_MIN_SIZE_KW = 4.0              # Floor
PV_MAX_SIZE_KW = 15.0             # Cap (larger to reflect CA median sizes)
SCE_ANNUAL_KWH_PER_KW = 1700.0   # Default; updated at runtime from pvlib

# ---------------------------------------------------------------------------
# Battery parameters (Tesla Powerwall equiv)
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 13.5
BATTERY_POWER_KW = 5.0
BATTERY_EFFICIENCY = 0.90  # round-trip

# ---------------------------------------------------------------------------
# EV parameters
# ---------------------------------------------------------------------------

EV_MILES_PER_KWH = 3.0
EV_CHARGE_START_HOUR = 22  # 10 PM
EV_CHARGER_KW = 7.2        # Level 2 charger (240V × 30A)

# BEV daily VMT empirical CDF (vehicletrends.us, BEV All)
BEV_DVMT_CDF = np.array([
    [0,  0.00],
    [5,  0.03],
    [10, 0.12],
    [15, 0.25],
    [20, 0.42],
    [25, 0.58],
    [30, 0.70],
    [35, 0.80],
    [40, 0.87],
    [50, 0.95],
    [60, 0.98],
    [70, 1.00],
])

# ---------------------------------------------------------------------------
# Precomputed time arrays (shared across modules)
# ---------------------------------------------------------------------------

def build_time_arrays():
    """Build hour→month, hour_of_day, is_weekend, is_summer arrays."""
    hours = np.arange(8760)
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_boundaries = np.concatenate(([0], np.cumsum(hours_per_month)))
    months = np.searchsorted(month_boundaries[1:], hours) + 1  # 1-indexed

    hour_of_day = hours % 24
    # Jan 1 = Wednesday (day_of_week=2, 0=Monday)
    day_of_year = hours // 24
    day_of_week = (day_of_year + 2) % 7
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


def build_sce_period_masks(ta=None):
    """
    Build boolean masks for 5 SCE TOU-D-4-9 periods.

    Summer (Jun-Oct): Peak 4-9pm, Off-peak all other
    Winter (Nov-May): Peak 4-9pm, Mid-peak 9pm-8am, Off-peak 8am-4pm
    """
    if ta is None:
        ta = build_time_arrays()

    hod = ta['hour_of_day']
    is_summer = ta['is_summer']

    is_peak = (hod >= 16) & (hod < 21)
    # Winter midpeak: 9pm-8am → hour >= 21 or hour < 8
    is_winter_midpeak = (hod >= 21) | (hod < 8)

    masks = {
        'summer_peak':    is_summer & is_peak,
        'summer_offpeak': is_summer & ~is_peak,
        'winter_peak':    ~is_summer & is_peak,
        'winter_midpeak': ~is_summer & ~is_peak & is_winter_midpeak,
        'winter_offpeak': ~is_summer & ~is_peak & ~is_winter_midpeak,
    }
    return masks


def build_tou_rate_array(rate_dict, ta=None):
    """
    Build 8760-length rate array from dict with keys:
    summer_peak, summer_offpeak, winter_peak, winter_midpeak, winter_offpeak.

    Uses blended (constant across weekday/weekend) rates for designed scenarios.
    """
    masks = build_sce_period_masks(ta)
    rates = np.zeros(8760)
    for period, mask in masks.items():
        rates[mask] = rate_dict[period]
    return rates


def build_actual_tariff_rate_array(weekday_rates, weekend_rates, ta=None):
    """
    Build 8760-length rate array with weekday/weekend distinction.

    Used for actual SCE TOU-D-4-9 tariff where summer peak differs
    by weekday ($0.627) vs weekend ($0.507).
    """
    if ta is None:
        ta = build_time_arrays()
    masks = build_sce_period_masks(ta)
    is_weekend = ta['is_weekend']

    rates = np.zeros(8760)
    for period, mask in masks.items():
        wd_rate = weekday_rates.get(period, 0.0)
        we_rate = weekend_rates.get(period, wd_rate)
        rates[mask & ~is_weekend] = wd_rate
        rates[mask & is_weekend] = we_rate
    return rates


def safe_float(val):
    """Convert NaN/None to 0.0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)
