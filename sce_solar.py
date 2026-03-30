"""
sce_solar.py — Stage 4: Generate solar profiles for SCE CEC climate zones

Uses pvlib to generate per-kW 8760 hourly solar profiles.
Falls back to synthetic profiles if pvlib/PVGIS unavailable.
"""

import numpy as np

from sce_config import (
    SCE_CZ_COORDINATES, SCE_ANNUAL_KWH_PER_KW,
    DEFAULT_PV_SIZE_KW, PV_OFFSET_TARGET, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW,
)


def size_pv_system(annual_kwh, annual_kwh_per_kw):
    """Size PV to offset PV_OFFSET_TARGET (90%) of native annual demand."""
    if annual_kwh_per_kw <= 0:
        return DEFAULT_PV_SIZE_KW
    ideal_kw = (annual_kwh * PV_OFFSET_TARGET) / annual_kwh_per_kw
    return float(np.clip(ideal_kw, PV_MIN_SIZE_KW, PV_MAX_SIZE_KW))


def _generate_solar_profile_for_location(lat, lon, alt, name):
    """Generate a per-kW 8760 solar profile using pvlib for one location."""
    try:
        import pvlib
        from pvlib.pvsystem import PVSystem
        from pvlib.location import Location
        from pvlib.modelchain import ModelChain
        from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    except ImportError:
        return None

    location = Location(lat, lon, 'US/Pacific', alt, name)

    try:
        tmy_result = pvlib.iotools.get_pvgis_tmy(lat, lon, map_variables=True)
        tmy_data = tmy_result[0]
    except Exception as e:
        print(f"    Could not fetch TMY for {name} ({lat},{lon}): {e}")
        return None

    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = cec_modules.iloc[:, 0]
    inverter = cec_inverters.iloc[:, 0]
    temp_params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    system = PVSystem(
        surface_tilt=lat,
        surface_azimuth=180,
        module_parameters=module,
        inverter_parameters=inverter,
        temperature_model_parameters=temp_params,
    )

    mc = ModelChain(system, location, dc_model='cec', aoi_model='physical',
                    spectral_model='no_loss')

    weather = tmy_data.copy()
    if len(weather) != 8760:
        weather = weather.resample('h').mean()
    weather = weather.iloc[:8760]

    mc.run_model(weather)

    ac_power = mc.results.ac.values
    ac_power = np.nan_to_num(ac_power, nan=0.0)
    ac_power = np.maximum(ac_power, 0)

    try:
        stc_rating = module['I_mp_ref'] * module['V_mp_ref']
    except KeyError:
        stc_rating = module.get('Impo', 0) * module.get('Vmpo', 0)
    if stc_rating > 0:
        solar_per_kw = ac_power / stc_rating
    else:
        solar_per_kw = ac_power / 1000.0

    return solar_per_kw


def _synthetic_solar_profile(latitude=34.0):
    """Generate a synthetic solar profile for a given latitude (per 1 kW DC)."""
    hours = np.arange(8760)
    day_of_year = hours // 24
    hour_of_day = hours % 24

    declination = 23.45 * np.sin(np.radians((284 + day_of_year) * 360 / 365))
    hour_angle = (hour_of_day - 12) * 15

    lat_rad = np.radians(latitude)
    decl_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)

    sin_alt = (np.sin(lat_rad) * np.sin(decl_rad) +
               np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    sin_alt = np.maximum(sin_alt, 0)

    ghi = 1000 * sin_alt ** 1.2
    panel_area_per_kw = 1.0 / 0.18
    hourly_gen = ghi * panel_area_per_kw * 0.18 * 0.85 / 1000  # kWh per kW

    return hourly_gen


def stage4_solar_profiles(tech_df, bills_df):
    """
    Generate per-kW 8760 solar profiles for each CEC climate zone in SCE territory.

    Returns dict {cz: solar_per_kw_array} and {cz: annual_kWh_per_kW}.
    """
    print("\n" + "=" * 80)
    print("STAGE 4: GENERATE SOLAR PROFILES (pvlib, per CEC climate zone)")
    print("=" * 80)

    pv_buildings = tech_df[tech_df['assigned_pv'] == 1]
    print(f"  PV-adopted buildings: {len(pv_buildings)}")

    if len(pv_buildings) == 0:
        print("  No PV buildings — skipping")
        return {}, {}

    # Determine which CEC zones have PV buildings
    if 'in.cec_climate_zone' in tech_df.columns:
        pv_czs = sorted(pv_buildings['in.cec_climate_zone'].dropna().unique())
    else:
        pv_czs = sorted(SCE_CZ_COORDINATES.keys())
    print(f"  CEC climate zones with PV: {pv_czs}")

    solar_profiles = {}
    annual_kwh_per_kw_by_cz = {}

    for cz in pv_czs:
        cz_int = int(cz)
        if cz_int in SCE_CZ_COORDINATES:
            lat, lon, alt, name = SCE_CZ_COORDINATES[cz_int]
        else:
            lat, lon, alt, name = 34.05, -118.25, 90, f'CZ{cz_int}_fallback'

        print(f"  CZ {cz_int} ({name}): lat={lat}, lon={lon}")
        profile = _generate_solar_profile_for_location(lat, lon, alt, name)

        if profile is not None:
            solar_profiles[cz_int] = profile
            annual = profile.sum()
            annual_kwh_per_kw_by_cz[cz_int] = annual
            cf = annual / 8760 * 100
            print(f"    → {annual:,.0f} kWh/kW/yr ({cf:.1f}% CF)")
        else:
            print(f"    → pvlib failed, using synthetic profile")
            synth = _synthetic_solar_profile(lat)
            solar_profiles[cz_int] = synth
            annual_kwh_per_kw_by_cz[cz_int] = synth.sum()

    if annual_kwh_per_kw_by_cz:
        avg = np.mean(list(annual_kwh_per_kw_by_cz.values()))
        print(f"\n  Average: {avg:,.0f} kWh/kW across {len(solar_profiles)} zones")

    return solar_profiles, annual_kwh_per_kw_by_cz
