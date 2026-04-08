"""
sdge_config.py — Configuration and constants for SDGE pipeline

Utility data from utility_data_inputs.tex (2026-03-23).
SDGE TOU-DR tariff structure with 6 TOU periods (includes midpeak).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------

BASELINE_DIR = './Baseline_SDGE'
UPGRADE11_DIR = './Upgrade11_SDGE'
METADATA_FILE = 'CA_Baseline_metadata_rescaled_twoincomes_puma20.parquet'
PUMA_UTILITY_FILE = 'puma_utility_data.csv'
TOU_WEIGHTS_FILE = 'tou_weights_sdge.csv'
EEC_FILE = 'eec_hourly_2025_wide.csv'
EXCEL_FILE = 'retail_rates_data_SDGE.xlsx'

# Output files
RATE_SCENARIOS_OUT = 'rate_scenarios_sdge_fresh.csv'
BASELINE_BILLS_OUT = 'baseline_bills_sdge_fresh.csv'
TECH_ASSIGNMENTS_OUT = 'tech_assignments_sdge.csv'
POSTADOPT_BILLS_OUT = 'post_adoption_bills_sdge.csv'
SUMMARY_OUT = 'pipeline_summary_sdge.csv'

# ---------------------------------------------------------------------------
# SDGE Utility Data (from utility_data_inputs.tex)
# ---------------------------------------------------------------------------

RESIDENTIAL_REVENUE = 1_560_000_000       # $1.56B
RESIDENTIAL_SALES_KWH = 4_810_000_000     # 4,810 GWh
TOTAL_RESIDENTIAL_CUSTOMERS = 1_364_361
CARE_CUSTOMERS = 305_902
BUILDING_WEIGHT = 252.3

CUSTOMERS = {
    'care': CARE_CUSTOMERS,
    'non_care': TOTAL_RESIDENTIAL_CUSTOMERS - CARE_CUSTOMERS,
    'total': TOTAL_RESIDENTIAL_CUSTOMERS,
}

# Capital structure
RATE_BASE = 13_590_000_000       # $13.59B
EQUITY_SHARE = 0.52
AUTHORIZED_ROE = 0.1022          # 10.22%

# Cost components ($)
TRANSMISSION_COST = 685_000_000      # $0.685B
DISTRIBUTION_COST = 1_720_000_000    # $1.72B
WILDFIRE_COST = 414_000_000          # $0.414B

# ---------------------------------------------------------------------------
# SDGE TOU-DR tariff structure
# ---------------------------------------------------------------------------

ACTUAL_SDGE_RATES = {
    'TOU-DR': 'tou_dr',
    'TOU-DR-F': 'tou_dr_f',
}

DESIGNED_SCENARIOS = [
    'F0_WF0_ROE0',
    'F50_WF0_ROE0',
    'F100_WF0_ROE0',
    'F0_WF1_ROE0',
    'F0_WF0_ROE1.0',
    'F50_WF1_ROE1.0',
]

# SDGE TOU: 6 periods (with midpeak)
# Summer (Jun-Oct): Peak 4-9pm, Midpeak 6am-4pm + 9-10pm, Offpeak 10pm-6am
# Winter (Nov-May): same structure
SUMMER_MONTHS = set(range(6, 11))
TOU_PERIODS = ['summer_peak', 'summer_midpeak', 'summer_offpeak',
               'winter_peak', 'winter_midpeak', 'winter_offpeak']

# SDGE single centroid for pvlib
SDGE_LATITUDE = 32.9
SDGE_LONGITUDE = -117.1
SDGE_ALTITUDE = 130

# ---------------------------------------------------------------------------
# Solar sizing
# ---------------------------------------------------------------------------

DEFAULT_PV_SIZE_KW = 5.0
PV_OFFSET_TARGET = 0.90           # 90% of native annual demand
PV_MIN_SIZE_KW = 4.0
PV_MAX_SIZE_KW = 15.0
SDGE_ANNUAL_KWH_PER_KW = 1700.0

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
EV_CHARGE_START_HOUR = 22  # 10 PM
EV_CHARGER_KW = 7.2        # Level 2 charger (240V × 30A)

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


def build_sdge_period_masks(ta=None):
    """Build boolean masks for 6 SDGE TOU-DR periods (with midpeak)."""
    if ta is None:
        ta = build_time_arrays()
    hod = ta['hour_of_day']
    is_summer = ta['is_summer']
    is_peak = (hod >= 16) & (hod < 21)
    is_midpeak = ((hod >= 6) & (hod < 16)) | ((hod >= 21) & (hod < 22))

    return {
        'summer_peak':    is_summer & is_peak,
        'summer_midpeak': is_summer & is_midpeak,
        'summer_offpeak': is_summer & ~is_peak & ~is_midpeak,
        'winter_peak':    ~is_summer & is_peak,
        'winter_midpeak': ~is_summer & is_midpeak,
        'winter_offpeak': ~is_summer & ~is_peak & ~is_midpeak,
    }


def build_tou_rate_array(rate_dict, ta=None):
    """Build 8760-length rate array from dict with 6 period keys."""
    masks = build_sdge_period_masks(ta)
    rates = np.zeros(8760)
    for period, mask in masks.items():
        rates[mask] = rate_dict[period]
    return rates


def safe_float(val):
    """Convert NaN/None to 0.0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return float(val)
