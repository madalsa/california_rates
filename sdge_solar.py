"""
sdge_solar.py — Stage 4: Generate solar profiles for SDGE territory

Uses pvlib to generate per-kW 8760 hourly solar profile.
SDGE uses a SINGLE centroid (San Diego area), not per-CZ like PGE/SCE.
Falls back to synthetic profile if pvlib/PVGIS unavailable.

Returns (solar_per_kw, annual_kwh_per_kw) — single arrays, not dicts.
"""

import numpy as np

from sdge_config import (
    SDGE_LATITUDE, SDGE_LONGITUDE, SDGE_ALTITUDE,
    SDGE_ANNUAL_KWH_PER_KW,
    DEFAULT_PV_SIZE_KW, PV_OFFSET_TARGET, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW,
)


def size_pv_system(annual_kwh, annual_kwh_per_kw):
    """Size a PV system to offset PV_OFFSET_TARGET of annual consumption.

    Parameters
    ----------
    annual_kwh : float
        Building annual electricity consumption (kWh).
    annual_kwh_per_kw : float
        Annual generation per kW of installed PV (kWh/kW), from the
        pvlib profile or SDGE_ANNUAL_KWH_PER_KW default.

    Returns
    -------
    float
        PV system size in kW DC, clamped to [PV_MIN_SIZE_KW, PV_MAX_SIZE_KW].
    """
    if annual_kwh_per_kw <= 0:
        return DEFAULT_PV_SIZE_KW
    ideal_kw = (annual_kwh * PV_OFFSET_TARGET) / annual_kwh_per_kw
    return float(np.clip(ideal_kw, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW))


def stage4_solar_profiles(tech_df, bills_df):
    """
    Generate a *per-kW* 8760 hourly solar generation profile using pvlib.

    SDGE uses a single centroid for all buildings (San Diego area).

    Returns
    -------
    solar_per_kw : np.ndarray, shape (8760,)
        Hourly generation in kWh per 1 kW DC installed.
    annual_kwh_per_kw : float
        Sum of solar_per_kw — used by stage 6 to size each building's PV.
    """
    print("\n" + "=" * 80)
    print("STAGE 4: GENERATE SOLAR PROFILES (pvlib)")
    print("=" * 80)

    pv_buildings = tech_df[tech_df['assigned_pv'] == 1]['building_id'].values
    print(f"  PV-adopted buildings: {len(pv_buildings)}")

    if len(pv_buildings) == 0:
        print("  No PV buildings — skipping")
        return np.zeros(8760), SDGE_ANNUAL_KWH_PER_KW

    try:
        import pvlib
        from pvlib.pvsystem import PVSystem
        from pvlib.location import Location
        from pvlib.modelchain import ModelChain
        from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    except ImportError:
        print("  pvlib not installed — using synthetic solar profile")
        profile = _synthetic_solar_profile()
        per_kw = profile / DEFAULT_PV_SIZE_KW
        annual = per_kw.sum()
        return per_kw, annual

    print(f"  Location: lat={SDGE_LATITUDE}, lon={SDGE_LONGITUDE}")
    print(f"  Solar sizing: {PV_OFFSET_TARGET*100:.0f}% offset target, "
          f"{PV_MIN_SIZE_KW}-{PV_MAX_SIZE_KW} kW range")

    # Create location and get solar data
    location = Location(SDGE_LATITUDE, SDGE_LONGITUDE, 'US/Pacific',
                        SDGE_ALTITUDE, 'San Diego')

    try:
        tmy_result = pvlib.iotools.get_pvgis_tmy(
            SDGE_LATITUDE, SDGE_LONGITUDE, map_variables=True)
        tmy_data, tmy_meta = tmy_result[0], tmy_result[1]
        print("  Retrieved TMY data from PVGIS")
    except Exception as e:
        print(f"  Could not fetch TMY data: {e}")
        print("  Using synthetic solar profile instead")
        profile = _synthetic_solar_profile()
        per_kw = profile / DEFAULT_PV_SIZE_KW
        annual = per_kw.sum()
        return per_kw, annual

    # Set up PV system
    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = cec_modules.iloc[:, 0]
    inverter = cec_inverters.iloc[:, 0]

    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    system = PVSystem(
        surface_tilt=SDGE_LATITUDE,
        surface_azimuth=180,
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temp_params,
    )

    mc = ModelChain(system, location, dc_model='cec', aoi_model='physical',
                    spectral_model='no_loss')

    # Prepare weather data (ensure 8760 hours)
    weather = tmy_data.copy()
    if len(weather) != 8760:
        weather = weather.resample('h').mean()
    weather = weather.iloc[:8760]

    mc.run_model(weather)

    ac_power = mc.results.ac.values
    ac_power = np.nan_to_num(ac_power, nan=0.0)
    ac_power = np.maximum(ac_power, 0)

    # Normalize: module STC rating -> per kW
    # CEC modules use I_mp_ref/V_mp_ref; Sandia uses Impo/Vmpo
    try:
        stc_rating = module['I_mp_ref'] * module['V_mp_ref']
    except KeyError:
        stc_rating = module.get('Impo', 0) * module.get('Vmpo', 0)
    if stc_rating > 0:
        solar_per_kw = ac_power / stc_rating
    else:
        solar_per_kw = ac_power / 1000.0

    annual_kwh_per_kw = solar_per_kw.sum()

    capacity_factor = annual_kwh_per_kw / 8760 * 100
    print(f"  Per-kW annual generation: {annual_kwh_per_kw:,.0f} kWh/kW "
          f"({capacity_factor:.1f}% CF)")
    print(f"  Example sizes at {PV_OFFSET_TARGET*100:.0f}% offset: "
          f"5000 kWh/yr -> {size_pv_system(5000, annual_kwh_per_kw):.1f} kW, "
          f"10000 kWh/yr -> {size_pv_system(10000, annual_kwh_per_kw):.1f} kW, "
          f"20000 kWh/yr -> {size_pv_system(20000, annual_kwh_per_kw):.1f} kW")

    return solar_per_kw, annual_kwh_per_kw


def _synthetic_solar_profile():
    """Generate a synthetic solar profile for San Diego (per 1 kW DC).

    Returns total-system kWh array for backward compatibility when called
    directly; stage4 divides by DEFAULT_PV_SIZE_KW to get per-kW.
    """
    print("  Generating synthetic solar profile for San Diego")
    hours = np.arange(8760)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    declination = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
    hour_angle = (hour_of_day - 12) * 15

    lat_rad = np.radians(SDGE_LATITUDE)
    decl_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    sin_alt = (np.sin(lat_rad) * np.sin(decl_rad) +
               np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    sin_alt = np.maximum(sin_alt, 0)

    ghi = 1000 * sin_alt ** 1.2

    # Per-kW: 1 kW nameplate -> panel_area = 1/0.18 m^2
    panel_area_per_kw = 1.0 / 0.18
    hourly_gen = ghi * panel_area_per_kw * 0.18 * 0.85 / 1000  # kWh per kW

    # Scale to DEFAULT_PV_SIZE_KW for backward compatibility
    hourly_gen_total = hourly_gen * DEFAULT_PV_SIZE_KW

    annual = hourly_gen_total.sum()
    cf = annual / (DEFAULT_PV_SIZE_KW * 8760) * 100
    print(f"  Synthetic annual generation: {annual:,.0f} kWh ({cf:.1f}% CF)")

    return hourly_gen_total
