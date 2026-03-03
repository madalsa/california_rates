import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import pulp as plp
from datetime import datetime, timedelta
from calendar import isleap
import os

# Global cache for Excel data
_EXCEL_CACHE = {}

def load_excel_data(excel_file="retail_rates_data_oct32025.xlsx"):
    """
    Load and cache Excel data to avoid repeated file I/O.
    
    Parameters:
    -----------
    excel_file : str
        Path to the Excel file with rate data
    
    Returns:
    --------
    tuple
        (rates_df, baseline_df)
    """
    global _EXCEL_CACHE
    
    # Check if data is already cached
    if excel_file in _EXCEL_CACHE:
        return _EXCEL_CACHE[excel_file]
    
    # Load the data
    rates_df = pd.read_excel(excel_file, sheet_name="retail_rates_oct32025")
    baseline_df = pd.read_excel(excel_file, sheet_name="baseline_puma")
    
    # Cache the data
    _EXCEL_CACHE[excel_file] = (rates_df, baseline_df)
    
    return rates_df, baseline_df


def calculate_hourly_rates_with_consumption(puma, rate_code, hourly_consumption, excel_file=None, care_eligible=False):
    """
    Calculate the precise $/kWh rate for each hour of the year (8760 hours) 
    based on the customer's consumption pattern.
    Edge cases that cross tiers are placed in tier 2.
    
    CORRECTED VERSION: Properly handles all rate structures and CARE discounts
    
    Key fixes:
    1. Tiered rates correctly use baseline allowances for tier determination
    2. TOU + Tiered rates apply both baseline tiers AND time-of-use periods
    3. Fixed charges are calculated based on income level
    4. CARE discount is ALWAYS applied to volumetric rates ($/kWh), not final bill
    """
    # Set default Excel file if not provided
    if excel_file is None:
        excel_file = "retail_rates_data_may202025.xlsx"
    
    # Read the rates and baseline data (using cache)
    rates_df, baseline_df = load_excel_data(excel_file)
    
    # 1. Find the baseline allowance for the customer's PUMA
    baseline_entry = baseline_df[baseline_df['puma'] == puma]
    if baseline_entry.empty:
        raise ValueError(f"No baseline data found for PUMA: {puma}")
    
    # Extract baseline values (these are daily values)
    daily_summer_baseline = baseline_entry['summer_baseline_allowance'].values[0]
    daily_winter_baseline = baseline_entry['winter_baseline_allowance'].values[0]
    
    # 2. Find rate structure entries for the given rate code
    rate_entries = rates_df[rates_df['rate_type'] == rate_code]
    if rate_entries.empty:
        raise ValueError(f"Rate code not found: {rate_code}")
    
    # Find weekday and weekend rate entries
    weekday_rate = rate_entries[rate_entries['weekday'] == 'weekday']
    weekend_rate = rate_entries[rate_entries['weekday'] == 'weekend']
    
    # If weekend rate not found, use weekday rate
    if weekend_rate.empty:
        weekend_rate = weekday_rate
    
    # Check if we have valid rate entries
    if weekday_rate.empty:
        raise ValueError(f"No weekday rate found for rate code: {rate_code}")
    
    # Convert to dictionaries for easier access
    weekday_rate = weekday_rate.iloc[0].to_dict()
    weekend_rate = weekend_rate.iloc[0].to_dict()
    
    # Extract CARE discount information
    care_discount = weekday_rate.get('care_discount', 0)
    if pd.isna(care_discount) or care_discount is None:
        care_discount = 0
    
    # Determine the rate type
    rate_type = weekday_rate.get("type of rate", "")
    
    # Pre-calculate month arrays
    days_per_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_per_month = days_per_month * 24
    month_start_hours = np.concatenate(([0], np.cumsum(hours_per_month)))
    
    # Pre-calculate which month each hour belongs to
    hour_to_month = np.searchsorted(month_start_hours[1:], np.arange(8760))
    
    # Pre-calculate weekend/weekday array
    # January 1, 2025 is a Wednesday (day_of_week = 2)
    day_of_year = np.arange(8760) // 24
    day_of_week = (day_of_year + 2) % 7  # 0 = Monday
    is_weekend_array = day_of_week >= 5  # 5 = Saturday, 6 = Sunday
    
    # Pre-calculate seasons for all hours
    seasons = pre_calculate_seasons(weekday_rate, weekend_rate, is_weekend_array)
    
    # Pre-calculate time periods for all hours
    time_periods = pre_calculate_time_periods(seasons, is_weekend_array, weekday_rate, weekend_rate, rate_type)
    
    # Pre-calculate all cumulative consumption for all hours
    all_cumulative_consumption = calculate_all_cumulative_consumption(hourly_consumption, hour_to_month, month_start_hours)
    
    # Pre-calculate monthly baseline allowances
    monthly_baselines = np.where(
        seasons == 0,  # 0 = winter, 1 = summer
        daily_winter_baseline * days_per_month[hour_to_month],
        daily_summer_baseline * days_per_month[hour_to_month]
    )
    
    # Initialize arrays to store results
    hourly_rates = np.zeros(8760)  # Hourly rate in $/kWh (BEFORE CARE discount)
    hourly_rates_with_care = np.zeros(8760)  # Hourly rate in $/kWh (AFTER CARE discount)
    period_info = [None] * 8760     # Store period, tier, etc. for each hour
    
    # Track monthly consumption
    monthly_consumption = np.array([
        np.sum(hourly_consumption[month_start_hours[i]:month_start_hours[i+1]]) 
        for i in range(12)
    ])
    
    # Calculate rates for each hour
    for hour in range(8760):
        month = hour_to_month[hour]
        is_weekend = is_weekend_array[hour]
        applicable_rate = weekend_rate if is_weekend else weekday_rate
        season = seasons[hour]
        season_str = 'summer' if season == 1 else 'winter'
        time_period = time_periods[hour]
        
        # Get cumulative consumption up to this hour in the month
        cumulative_consumption = all_cumulative_consumption[hour]
        current_consumption = hourly_consumption[hour]
        monthly_baseline = monthly_baselines[hour]
        
        # Initialize hour info dictionary
        hour_info = {
            'season': season_str,
            'month': month,
            'period': time_period,
            'tier': 1,  # Default to tier 1
            'consumption': current_consumption
        }
        
        # ===== CORRECTED: Calculate the applicable rate based on rate structure =====
        if rate_type == "Tiered":
            # NUANCE 1: Tiered rates use baseline allowances for tier determination
            # Tier 1 = within baseline, Tier 2 = above baseline
            if cumulative_consumption < monthly_baseline:
                # Check if this hour's consumption crosses into tier 2
                remaining_tier1 = monthly_baseline - cumulative_consumption
                
                if current_consumption <= remaining_tier1:
                    # All consumption in tier 1 (within baseline)
                    hourly_rates[hour] = applicable_rate.get(f'{season_str}_charge_1', 0) or 0
                    hour_info['tier'] = 1
                else:
                    # Edge case: consumption crosses tier - place all in tier 2
                    hourly_rates[hour] = applicable_rate.get(f'{season_str}_charge_2', 0) or 0
                    hour_info['tier'] = 2
            else:
                # All consumption in tier 2 (above baseline)
                hourly_rates[hour] = applicable_rate.get(f'{season_str}_charge_2', 0) or 0
                hour_info['tier'] = 2
        
        elif rate_type == "TOU":
            # For TOU rates, use the appropriate rate based on time period
            rate_key = f'{time_period.replace("-", "")}_rate_{season_str}1'
            # Try with and without the '1' suffix
            hourly_rates[hour] = (applicable_rate.get(rate_key, 0) or 
                                 applicable_rate.get(rate_key[:-1], 0) or 0)
        
        elif rate_type in ["TOU + Tiered", "Tiered+TOU"]:
            # NUANCE 2: TOU + Tiered combines baseline tiers AND time-of-use periods
            # First determine tier based on baseline allowance
            tier = 1
            if cumulative_consumption >= monthly_baseline:
                tier = 2
            elif cumulative_consumption + current_consumption > monthly_baseline:
                tier = 2  # Edge case: place all in tier 2
            
            hour_info['tier'] = tier
            
            # Then apply the appropriate TOU rate for that tier
            rate_key = f'{time_period.replace("-", "")}_rate_{season_str}{tier}'
            # Try with fallback to tier 1 if tier 2 doesn't exist
            hourly_rates[hour] = (applicable_rate.get(rate_key, 0) or 
                                 applicable_rate.get(f'{time_period.replace("-", "")}_rate_{season_str}1', 0) or 
                                 applicable_rate.get(f'{time_period.replace("-", "")}_rate_{season_str}', 0) or 0)
        
        # ===== CORRECTED: CARE discount ALWAYS applied to volumetric rates ($/kWh) =====
        # NUANCE 4: CARE discount is applied to the $/kWh rate, not the final bill
        if care_eligible and care_discount > 0:
            # Apply the discount to the volumetric rate
            # If care_discount = 0.35, customer pays 65% of the rate
            hourly_rates_with_care[hour] = hourly_rates[hour] * (1 - care_discount)
        else:
            # No CARE discount
            hourly_rates_with_care[hour] = hourly_rates[hour]
        
        # Store the hour's information
        period_info[hour] = hour_info
    
    # Initialize fixed charges with explicit 0 values
    fixed_charges = {
        'base_service_charge_per_day': 0,
        'fixedcharge_low': 0,
        'fixedcharge_med': 0,
        'fixedcharge_high': 0,
        'fixedcharge_fera': 0
    }
    
    # Extract available fixed charge values
    for field in fixed_charges:
        if field in weekday_rate and weekday_rate[field] is not None:
            # Convert to float and handle NaN
            value = weekday_rate[field]
            if isinstance(value, (int, float)) and not pd.isna(value):
                fixed_charges[field] = value
    
    # Compile rate structure information with default 0 values for missing fields
    rate_info = {
        'rate_code': rate_code,
        'rate_type': rate_type,
        'utility': weekday_rate.get('utility', ''),
        'has_fixed_component': weekday_rate.get('Fixed') == 'Yes',
        'fixed_charges': fixed_charges,
        'minimum_bill_per_day': 0 if pd.isna(weekday_rate.get('minimum_bill_per_day')) else weekday_rate.get('minimum_bill_per_day', 0),
        'baseline_allowances': {
            'summer_daily': daily_summer_baseline,
            'winter_daily': daily_winter_baseline
        },
        'care_discount': care_discount,
        'care_eligible': care_eligible
    }
    
    return {
        'hourly_rates': hourly_rates_with_care,  # Return rates WITH CARE discount applied
        'hourly_rates_before_care': hourly_rates,  # Also return rates before CARE for tracking
        'rate_info': rate_info,
        'period_info': period_info,
        'monthly_consumption': monthly_consumption
    }


def pre_calculate_seasons(weekday_rate, weekend_rate, is_weekend_array):
    """Pre-calculate seasons for all hours to avoid repeated calculations."""
    seasons = np.zeros(8760, dtype=int)  # 0 = winter, 1 = summer
    
    for hour in range(8760):
        applicable_rate = weekend_rate if is_weekend_array[hour] else weekday_rate
        summer_start = applicable_rate.get('summer_start', 0) or 0
        summer_end = applicable_rate.get('summer_end', 0) or 0
        
        # Check if hour is within summer range
        if summer_start <= summer_end:
            seasons[hour] = 1 if (hour >= summer_start and hour < summer_end) else 0
        else:
            # Handle case where summer spans across year boundary
            seasons[hour] = 1 if (hour >= summer_start or hour < summer_end) else 0
    
    return seasons


def pre_calculate_time_periods(seasons, is_weekend_array, weekday_rate, weekend_rate, rate_type):
    """Pre-calculate time periods for all hours."""
    time_periods = ['off-peak'] * 8760
    
    if rate_type not in ["TOU", "TOU + Tiered", "Tiered+TOU"]:
        return time_periods
    
    for hour in range(8760):
        hour_of_day = hour % 24
        season_str = 'summer' if seasons[hour] == 1 else 'winter'
        applicable_rate = weekend_rate if is_weekend_array[hour] else weekday_rate
        
        # Check if it's in peak period
        peak_start = applicable_rate.get(f'{season_str}_peak_start')
        peak_end = applicable_rate.get(f'{season_str}_peak_end')
        
        if peak_start is not None and peak_end is not None:
            if is_in_time_range(hour_of_day, peak_start, peak_end):
                time_periods[hour] = 'peak'
                continue
        
        # Check if it's in mid-peak periods
        for i in range(1, 4):
            mid_peak_start = applicable_rate.get(f'{season_str}_midpeak{i}_start')
            mid_peak_end = applicable_rate.get(f'{season_str}_midpeak{i}_end')
            
            if (mid_peak_start is not None and mid_peak_end is not None and 
                mid_peak_start != 0 and mid_peak_end != 0):
                if is_in_time_range(hour_of_day, mid_peak_start, mid_peak_end):
                    time_periods[hour] = 'mid-peak'
                    break
    
    return time_periods


def calculate_all_cumulative_consumption(hourly_consumption, hour_to_month, month_start_hours):
    """Calculate cumulative consumption up to each hour within its month for all 8760 hours."""
    all_cumulative = np.zeros(8760)
    
    for month in range(12):
        start_hour = month_start_hours[month]
        end_hour = month_start_hours[month + 1]
        
        # Get consumption for this month
        month_consumption = hourly_consumption[start_hour:end_hour]
        
        # Calculate cumulative consumption for each hour in the month
        # We want cumulative consumption up to (but not including) each hour
        monthly_cumulative = np.cumsum(np.concatenate(([0], month_consumption)))[:-1]
        
        # Store in the all_cumulative array
        all_cumulative[start_hour:end_hour] = monthly_cumulative
    
    return all_cumulative


def calculate_total_bill(hourly_rates_result, hourly_consumption, income_level):
    """
    Calculate the total electricity bill including energy charges, fixed charges, and CARE discounts.
    
    CORRECTED VERSION: Properly tracks CARE discount amount
    
    Parameters:
    -----------
    hourly_rates_result : dict
        Output from calculate_hourly_rates_with_consumption
    hourly_consumption : array-like
        Hourly consumption in kWh for 8760 hours
    income_level : str
        Income level ('low', 'medium', 'high') for fixed charge calculation (NUANCE 3)
    
    Returns:
    --------
    dict
        Detailed billing information
    """
    # Extract data from hourly rates result
    hourly_rates = hourly_rates_result['hourly_rates']  # These already have CARE applied
    hourly_rates_before_care = hourly_rates_result.get('hourly_rates_before_care', hourly_rates)
    rate_info = hourly_rates_result['rate_info']
    period_info = hourly_rates_result['period_info']
    monthly_consumption = hourly_rates_result.get('monthly_consumption')
    
    # Convert to numpy arrays for faster operations
    hourly_consumption = np.array(hourly_consumption)
    hourly_rates = np.array(hourly_rates)
    hourly_rates_before_care = np.array(hourly_rates_before_care)
    
    # CARE discount information
    care_eligible = rate_info.get('care_eligible', False)
    care_discount = rate_info.get('care_discount', 0)
    
    # Initialize the billing result
    billing_result = {
        'total_bill': 0,
        'energy_charges': {
            'total': 0,
            'by_month': {},
            'by_tier': {
                'tier1': 0,
                'tier2': 0
            },
            'by_tou': {
                'peak': 0,
                'mid-peak': 0,
                'off-peak': 0
            },
            'by_season': {
                'summer': 0,
                'winter': 0
            }
        },
        'fixed_charges': {
            'total': 0,
            'monthly': [],
            'daily': 0
        },
        'care_discount': {
            'eligible': care_eligible,
            'amount': 0,
            'applied_to': 'volumetric_rates'  # NUANCE 4: Always applied to $/kWh
        },
        'consumption': {
            'total_kwh': np.sum(hourly_consumption),
            'monthly_kwh': {}
        },
        'rate_details': {
            'rate_code': rate_info['rate_code'],
            'rate_type': rate_info['rate_type'],
            'utility': rate_info['utility']
        }
    }
    
    # Calculate hourly charges using rates that already have CARE discount applied
    hourly_charges = hourly_consumption * hourly_rates
    hourly_charges_before_care = hourly_consumption * hourly_rates_before_care
    
    # Setup month boundaries for 2025 (non-leap year)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_per_month = [days * 24 for days in days_per_month]
    month_start_hours = [0]
    for hours in hours_per_month:
        month_start_hours.append(month_start_hours[-1] + hours)
    
    # Calculate monthly energy charges using vectorized operations
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month in range(12):
        start_hour = month_start_hours[month]
        end_hour = month_start_hours[month + 1]
        
        # Vectorized calculation for monthly charges
        month_charges = np.sum(hourly_charges[start_hour:end_hour])
        month_consumption = np.sum(hourly_consumption[start_hour:end_hour])
        
        billing_result['energy_charges']['by_month'][month_names[month]] = month_charges
        billing_result['consumption']['monthly_kwh'][month_names[month]] = month_consumption
        
        # Process period_info for this month
        for hour in range(start_hour, end_hour):
            if hour < len(period_info) and period_info[hour]:
                info = period_info[hour]
                hour_charge = hourly_charges[hour]
                
                # By season
                if info['season'] == 'summer':
                    billing_result['energy_charges']['by_season']['summer'] += hour_charge
                else:
                    billing_result['energy_charges']['by_season']['winter'] += hour_charge
                
                # By TOU period
                if 'period' in info:
                    period = info['period']
                    if period == 'peak':
                        billing_result['energy_charges']['by_tou']['peak'] += hour_charge
                    elif period == 'mid-peak':
                        billing_result['energy_charges']['by_tou']['mid-peak'] += hour_charge
                    else:  # off-peak
                        billing_result['energy_charges']['by_tou']['off-peak'] += hour_charge
                
                # By tier
                if 'tier' in info:
                    tier = info['tier']
                    if tier == 1:
                        billing_result['energy_charges']['by_tier']['tier1'] += hour_charge
                    elif tier == 2:
                        billing_result['energy_charges']['by_tier']['tier2'] += hour_charge
    
    # Sum up all energy charges (already includes CARE discount)
    billing_result['energy_charges']['total'] = sum(
        billing_result['energy_charges']['by_month'].values()
    )
    
    # Calculate CARE discount amount by comparing before and after
    if care_eligible and care_discount > 0:
        total_before_care = np.sum(hourly_charges_before_care)
        total_after_care = billing_result['energy_charges']['total']
        billing_result['care_discount']['amount'] = total_before_care - total_after_care
    
    # Apply minimum bill if necessary (with safer handling)
    minimum_bill_per_day = rate_info.get('minimum_bill_per_day', 0)
    if pd.isna(minimum_bill_per_day) or minimum_bill_per_day is None:
        minimum_bill_per_day = 0
    total_minimum_bill = minimum_bill_per_day * 365  # 365 days in 2025
    
    if billing_result['energy_charges']['total'] < total_minimum_bill:
        # If energy charges are below minimum, adjust to minimum
        billing_result['minimum_bill_applied'] = True
        billing_result['energy_charges']['total'] = total_minimum_bill
    else:
        billing_result['minimum_bill_applied'] = False
    
    # ===== CORRECTED: Calculate fixed charges based on income level =====
    # NUANCE 3: Fixed charges calculated based on income level
    fixed_charges = rate_info['fixed_charges']
    
    # Daily fixed charges (base service charge)
    daily_fixed_charge = fixed_charges['base_service_charge_per_day']
    annual_daily_charges = daily_fixed_charge * 365  # 365 days in 2025
    
    # Monthly fixed charges based on income level
    monthly_fixed_charge = 0
    if rate_info['has_fixed_component']:
        if income_level.lower() == 'low':
            monthly_fixed_charge = fixed_charges['fixedcharge_low']
        elif income_level.lower() == 'medium':
            monthly_fixed_charge = fixed_charges['fixedcharge_med']
        elif income_level.lower() == 'high':
            monthly_fixed_charge = fixed_charges['fixedcharge_high']
    
    # Calculate total fixed charges
    total_monthly_fixed_charges = monthly_fixed_charge * 12  # 12 months
    total_fixed_charges = annual_daily_charges + total_monthly_fixed_charges
    
    # Store fixed charge details
    billing_result['fixed_charges']['total'] = total_fixed_charges
    billing_result['fixed_charges']['daily'] = annual_daily_charges
    billing_result['fixed_charges']['monthly_rate'] = monthly_fixed_charge
    for month in month_names:
        billing_result['fixed_charges']['monthly'].append({
            'month': month,
            'fixed_charge': monthly_fixed_charge
        })
    
    # Calculate final total bill
    # Energy charges already have CARE discount applied (it was applied to the rates)
    final_total = billing_result['energy_charges']['total'] + billing_result['fixed_charges']['total']
    
    # Set the final total bill
    billing_result['total_bill'] = final_total
        
    return billing_result


def determine_season(hour, rate_structure):
    """
    Determine whether a given hour is in summer or winter season.
    
    Parameters:
    -----------
    hour : int
        Hour of year (0-8759)
    rate_structure : dict
        Rate structure dictionary
    
    Returns:
    --------
    str
        'summer' or 'winter'
    """
    summer_start = rate_structure.get('summer_start', 0) or 0
    summer_end = rate_structure.get('summer_end', 0) or 0
    
    # Check if hour is within summer range
    if summer_start <= summer_end:
        return 'summer' if (hour >= summer_start and hour < summer_end) else 'winter'
    else:
        # Handle case where summer spans across year boundary
        return 'summer' if (hour >= summer_start or hour < summer_end) else 'winter'


def determine_time_period(hour_of_day, season, rate_structure):
    """
    Determine the time period (peak, mid-peak, off-peak) for a given hour.
    
    Parameters:
    -----------
    hour_of_day : int
        Hour of day (0-23)
    season : str
        'summer' or 'winter'
    rate_structure : dict
        Rate structure dictionary
    
    Returns:
    --------
    str
        'peak', 'mid-peak', or 'off-peak'
    """
    # Check if it's in peak period
    peak_start = rate_structure.get(f'{season}_peak_start')
    peak_end = rate_structure.get(f'{season}_peak_end')
    
    if peak_start is not None and peak_end is not None:
        if is_in_time_range(hour_of_day, peak_start, peak_end):
            return 'peak'
    
    # Check if it's in mid-peak periods (can have multiple mid-peak periods)
    for i in range(1, 4):
        mid_peak_start = rate_structure.get(f'{season}_midpeak{i}_start')
        mid_peak_end = rate_structure.get(f'{season}_midpeak{i}_end')
        
        if (mid_peak_start is not None and mid_peak_end is not None and 
            mid_peak_start != 0 and mid_peak_end != 0):
            if is_in_time_range(hour_of_day, mid_peak_start, mid_peak_end):
                return 'mid-peak'
    
    # If not in peak or mid-peak, it's off-peak
    return 'off-peak'


def is_in_time_range(hour, start_hour, end_hour):
    """
    Check if an hour is within a time range.
    
    Parameters:
    -----------
    hour : int
        Hour to check (0-23)
    start_hour : int
        Start hour of range
    end_hour : int
        End hour of range
    
    Returns:
    --------
    bool
        True if hour is in range, False otherwise
    """
    if start_hour is None or end_hour is None:
        return False
    
    if start_hour < end_hour:
        # Normal range (e.g., 9-17)
        return hour >= start_hour and hour < end_hour
    else:
        # Overnight range (e.g., 22-6)
        return hour >= start_hour or hour < end_hour


# Utility function to clear cache if needed
def clear_excel_cache():
    """Clear the Excel data cache."""
    global _EXCEL_CACHE
    _EXCEL_CACHE.clear()