"""Shared paths, constants, and economic assumptions.

All numeric assumptions are sourced; see comments. Last verified: 2026-05-06.
Major 2026 policy changes (vs prior years) are flagged "POLICY 2026:".

Reads pipeline outputs from the parent `california_rates` repo so we don't
duplicate data. When extracted to a standalone repo, point CR_ROOT at the
local clone.
"""

from pathlib import Path

CR_ROOT = Path(__file__).resolve().parents[2]
EE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = EE_ROOT / "data"

PIPELINE_OUTPUTS = {
    "pge": {
        "baseline_bills": CR_ROOT / "baseline_bills_pge_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_pge.csv",
        "summary":        CR_ROOT / "pipeline_summary_pge.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_pge.csv",
        "tou_weights":    CR_ROOT / "tou_weights_pge.csv",
        "baseline_parquets": CR_ROOT / "Baseline_PGE",
        "upgrade11_parquets": CR_ROOT / "Upgrade11_PGE",
    },
    "sce": {
        "baseline_bills": CR_ROOT / "baseline_bills_sce_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_sce.csv",
        "summary":        CR_ROOT / "pipeline_summary_sce.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_sce.csv",
        "tou_weights":    CR_ROOT / "tou_weights_sce.csv",
        "baseline_parquets": CR_ROOT / "Baseline_SCE",
        "upgrade11_parquets": None,
    },
    "sdge": {
        "baseline_bills": CR_ROOT / "baseline_bills_sdge_fresh.csv",
        "post_adoption":  CR_ROOT / "post_adoption_bills_sdge.csv",
        "summary":        CR_ROOT / "pipeline_summary_sdge.csv",
        "rate_scenarios": CR_ROOT / "rate_scenarios_sdge.csv",
        "tou_weights":    CR_ROOT / "tou_weights_sdge.csv",
        "baseline_parquets": CR_ROOT / "Baseline_SDGE",
        "upgrade11_parquets": CR_ROOT / "Upgrade11_SDGE",
    },
}

METADATA_PARQUET = CR_ROOT / "CA_baseline_metadata_rescaled.parquet"
RASS_SURVEY = CR_ROOT / "Final19_SW_CleanedSurvey.csv"
PUMA_UTILITY = CR_ROOT / "puma_utility_data.csv"

# -----------------------------------------------------------------------------
# Population scoping for the paper
# -----------------------------------------------------------------------------
# Include only IOU customers where NPV is a meaningful question.
# Exclude:
#   - POU territories (LADWP, SMUD, IID, etc.) - handled upstream by
#     PUMA-utility mapping; only PGE / SCE / SDGE carried through pipeline.
#   - EBD-eligible households (<=80% AMI in CEC priority climate zones).
#     They receive turnkey free retrofit; payback / NPV not applicable.
#     They are reported as a population share in summary stats but
#     excluded from the optimization population.
#
# CEC EBD priority climate zones - TODO: verify against current CEC list.
# Initial list based on DAC overlap and high-heat / poor-AQ CZs.
EBD_PRIORITY_CEC_CZS = {8, 9, 10, 13, 14, 15}

# Income-tier threshold: <=80% AMI defined by HUD-published county income
# limits; in ResStock metadata typically encoded as `income_tier_pct_ami`
# or two-income variant.
EBD_AMI_THRESHOLD = 0.80

INCLUDED_UTILITIES = ("pge", "sce", "sdge")


def is_ebd_eligible(cec_cz: int, ami_pct: float) -> bool:
    """True if household is excluded from the analysis as EBD-eligible."""
    return (cec_cz in EBD_PRIORITY_CEC_CZS) and (ami_pct <= EBD_AMI_THRESHOLD)

# -----------------------------------------------------------------------------
# Financial framework
# -----------------------------------------------------------------------------
# CPUC E3 ACC and DER cost-effectiveness work uses 3-5% real for customer
# perspective. Use 5% real as base; sweep 3-8% in sensitivity.
# Source: CPUC 2024 ACC Documentation v1b.
DISCOUNT_RATE_REAL = 0.05
INFLATION = 0.025
DISCOUNT_RATE_NOMINAL = (1 + DISCOUNT_RATE_REAL) * (1 + INFLATION) - 1

# 20-yr horizon matches NEM 2.0 grandfathering and PV manufacturer warranty;
# sensitivity 15 / 25 yr.
ANALYSIS_YEARS = 20

# Real bill escalator: CPUC GRC filings typically project 2-2.5%/yr real;
# 2020-2024 actuals exceeded inflation. Base 2% real; sweep 0-4%.
# Source: Cal Advocates Q1 2025 Rates Report; SCE 2025 GRC.
BILL_ESCALATOR_REAL = 0.02

# -----------------------------------------------------------------------------
# CAPEX (CA installed, 2026 averages, pre-incentive)
# -----------------------------------------------------------------------------
# Sources: EnergySage CA 2026 marketplace; SolarReviews 2026; Reliable HVAC
# LA/Ventura 2026; Today's Homeowner 2026; Custom Home Bay Area 2026;
# Expert Electric Group CA 2026; Cox Auto / KBB Mar 2026 ATP.
CAPEX = {
    "pv_per_kw":         2500,   # $/kW DC turnkey; range $2,400-$2,600 (CA)
    "battery_per_kwh":   1050,   # installed; Powerwall 3 ~$1,000-$1,100/kWh
    "ev_premium":        5800,   # KBB Mar 2026 ATP: EV $54.5K vs $49.3K avg
    "ev_charger":        1500,   # L2 EVSE installed; range $800-$2,700
    "heat_pump_space":  15000,   # 3-ton ducted CA install; range $12-18K
    "heat_pump_water":   5500,   # 50-80 gal HPWH installed; range $4-8K
    "induction_range":   3500,   # range + 240V install; SF Bay $2.5-6K
    "panel_upgrade_200a": 3500,  # 200A service upgrade; range $2-4.5K
}

# -----------------------------------------------------------------------------
# EV acquisition scenarios
# -----------------------------------------------------------------------------
# Households face different effective premiums depending on what they
# replace and what they qualify for. Run all three in the paper.
#
# 1. NEW_NEW: buy new EV instead of new ICE at next purchase.
#    net_premium = CAPEX.ev_premium                              ($5,800)
#
# 2. KEEP_OLD_VS_NEW: keep an old gas car, no replacement decision yet.
#    Not a payback question — excluded from base case.
#
# 3. SCRAP_AND_REPLACE_CC4A: scrap an old high-emitter under Clean Cars
#    4 All (income-qualified, varies by air district). The CC4A rebate
#    plus avoided ICE-purchase cost can flip the premium to negative
#    for qualifying households.
#    Approx: net_premium = EV_price - ICE_alt_price - cc4a_rebate
#                        - assumed_salvage_of_scrapped_car
#    Many CC4A programs also disqualify the trade-in from resale, so
#    salvage is effectively $0 (vehicle is scrapped).
#
# 4. NEW_EV_WITH_DCAP: low-income (<=300% FPL) household qualifies for
#    DCAP $7,500 (+$4,500 if DAC) when buying any new ZEV.
#    net_premium = CAPEX.ev_premium - dcap_total
#
# CC4A rebates vary by air district; see assumptions_sources.md.
EV_SCENARIOS = {
    "new_new":            {"premium": CAPEX["ev_premium"], "rebate": 0},
    "scrap_replace_cc4a": {"premium": CAPEX["ev_premium"],
                           "rebate_by_district": "CC4A_BY_DISTRICT",
                           "salvage": 0,
                           "income_eligible": True},
    "new_ev_dcap":        {"premium": CAPEX["ev_premium"], "rebate": 7500,
                           "income_eligible": True},
    "new_ev_dcap_dac":    {"premium": CAPEX["ev_premium"], "rebate": 12000,
                           "income_eligible": True,
                           "dac_required": True},
}

# Inverter replacement cost at year ~13 for string inverters; Enphase
# microinverters typically last full PV life (set to 0 if microinverter).
INVERTER_REPLACEMENT_COST = 2500
INVERTER_REPLACEMENT_YEAR = 13

# -----------------------------------------------------------------------------
# INCENTIVES (status as of 2026-05-06)
# -----------------------------------------------------------------------------
# POLICY 2026: One Big Beautiful Bill (OBBB / P.L. 119-21, July 4, 2025)
# repealed Sections 25C, 25D, 30D for installations / vehicles after
# Dec 31, 2025 (30D after Sept 30, 2025). Section 30C (EVSE) sunsets
# Jun 30, 2026. Federal residential clean-energy credits are effectively
# zero for the 2026 base case.
# Source: IRS OBBB FAQ; Electrification Coalition summary.
INCENTIVES_2026 = {
    # Federal — repealed for 2026 installations
    "itc_pv":              0.0,    # was 30% under Section 25D
    "itc_battery":         0.0,    # was 30% under Section 25D
    "fed_25c_hp_max":      0,      # was $2,000/yr
    "fed_25c_hpwh_max":    0,      # was $2,000/yr (combined w/ HP)
    "fed_25c_panel_max":   0,      # was $600
    "fed_30d_ev":          0,      # was $7,500 (vehicles after 9/30/25)
    "fed_30c_evse":        0.0,    # 30% / $1,000 cap; sunsets 6/30/2026

    # CA HEAR (HEEHRA): single-family fully reserved as of 2026-02-24.
    # Multifamily still open. Caps shown for record.
    # <80% AMI: 100% cost up to caps. 80-150% AMI: 50% cost up to caps.
    "hear_available":            False,   # set True if user is multifamily
                                          # or new appropriation lands
    "hear_lt80_ami_total_cap":   14000,
    "hear_80_150_ami_total_cap": 4000,
    "hear_hp_cap":               8000,
    "hear_hpwh_cap":             1750,
    "hear_panel_cap":            4000,
    "hear_induction_cap":        840,

    # TECH Clean California (active, but funding-limited; verify per project).
    # Source: TECH Single Family Incentives tracker.
    # HP space: $1,000/system, $2,000/home cap. HPWH: range below.
    "tech_hpwh_market":      2700,   # midpoint of $1,100-$4,300
    "tech_hpwh_equity":      4600,   # midpoint of $3,500-$5,700
    "tech_hp_space_market":  1000,
    "tech_hp_space_equity":  3500,
    "tech_hp_space_home_cap": 2000,

    # SGIP — battery storage. Step varies; check selfgenca.com.
    # General Market mid-step ~$200/kWh. Equity ~$850/kWh.
    # Equity Resiliency ~$1,050/kWh. Cap 30 kWh GM, 80 kWh ER.
    "sgip_battery_general_per_kwh":   200,
    "sgip_battery_equity_per_kwh":    850,
    "sgip_battery_eq_resilience_per_kwh": 1050,
    "sgip_battery_general_cap_kwh":   30,
    "sgip_battery_eq_resilience_cap_kwh": 80,

    # SGIP-HPWH (CPUC, separate from battery program). Stacks w/ TECH.
    # Source: CPUC SGIP, $84.7M authorized.
    "sgip_hpwh_standard":    3800,
    "sgip_hpwh_low_income":  4885,
    "sgip_hpwh_low_gwp_adder": 1500,

    # HOMES (federal IRA, CEC-administered). Performance-based whole-home
    # retrofit; OBBB did NOT repeal HOMES. Stacks w/ TECH.
    # Up to $8,000 for >=35% modeled savings, low-income.
    # Source: CEC IRA rebates page.
    "homes_max_low_income":  8000,
    "homes_max_market":      4000,

    # Equitable Building Decarbonization (EBD) Direct Install — CEC.
    # Launched April 2026; turnkey free (HP + HPWH + panel + induction)
    # for <80% AMI in priority CZs. Mutually exclusive with TECH/HEAR.
    # Budget: $432M (cut from $922M).
    "ebd_direct_install":    True,   # eligibility flag; replaces capex

    # Golden State Rebates — instant retail rebate, stacks. Through 12/31/26.
    "golden_state_hpwh":     300,
    "golden_state_smart_thermostat": 85,

    # Regional Energy Networks (RENs) — stack w/ TECH.
    # BayREN Home+: $250 elec-replace / $400 fuel-substitution (HPWH).
    # 3C-REN (SLO/SB/Ventura): $5,000 contractor incentive for SF HPWH.
    # SoCalREN: multifamily only.
    "bayren_hpwh_fuel_sub":  400,
    "bayren_hpwh_elec_replace": 250,
    "ren_3c_hpwh_sf":        5000,

    # Publicly Owned Utility (POU) rebates — for customers NOT served by
    # PGE / SCE / SDGE / SoCalGas (so generally not in our pipeline).
    # LADWP: up to $2,500/ton HP space (eff. 11/1/2025).
    # SMUD: $3,000 gas->elec HP / $4,000 gas->elec HPWH (Feb 2026 boost).
    "ladwp_hp_per_ton":      2500,
    "smud_hp_gas_to_elec":   3000,
    "smud_hpwh_gas_to_elec": 4000,

    # Income-qualified turnkey programs — replace capex, mutually
    # exclusive with TECH/HEAR.
    # ESA (CPUC IOU, <=200% FPL): free HP/HPWH where existing unsafe.
    # LIWP (CSD): DAC + low-income, free.
    "esa_eligible":          False,  # set True per building income tier
    "liwp_eligible":         False,

    # CA EV: CVRP closed Nov 2023. DCAP and Clean Cars 4 All only.
    # Both are income-restricted (DCAP <=300% FPL; CC4A varies by air
    # district). No broad-market CA EV rebate.
    "dcap_new_ev_max":      7500,    # +$4,500 if DAC -> $12,000 total
    "dcap_dac_bonus":       4500,
    "dcap_l2_charger":      2000,
}

# Clean Cars 4 All — parameterized by air district.
# Each district has its own income-tier schedule. Use the
# upper-bound rebate by tier for each district. Verify per project at
# https://ww2.arb.ca.gov/our-work/programs/clean-cars-4-all
CC4A_BY_DISTRICT = {
    # Bay Area Air Quality Management District
    "BAAQMD":  {"new_ev_max": 9500,  "used_ev_max": 7500,  "evse": 2000,
                "income_cap_pct_fpl": 400},
    # South Coast (LA / Orange / Riverside / SB)
    "SCAQMD":  {"new_ev_max": 12000, "used_ev_max": 9500,  "evse": 2000,
                "income_cap_pct_fpl": 400},
    # San Joaquin Valley
    "SJVAPCD": {"new_ev_max": 9500,  "used_ev_max": 7500,  "evse": 2000,
                "income_cap_pct_fpl": 400},
    # Sacramento Metro
    "SMAQMD":  {"new_ev_max": 9500,  "used_ev_max": 7500,  "evse": 2000,
                "income_cap_pct_fpl": 400},
    # San Diego — runs DCAP only, no CC4A
    "SDAPCD":  None,
}

# Counterfactual: incentives as they stood in 2024 (pre-OBBB). Use this
# scenario to quantify what federal policy reversal cost households.
INCENTIVES_2024_COUNTERFACTUAL = {
    "itc_pv":            0.30,
    "itc_battery":       0.30,
    "fed_25c_hp_max":    2000,
    "fed_25c_hpwh_max":  2000,
    "fed_25c_panel_max":  600,
    "fed_30d_ev":        7500,
    "fed_30c_evse":      0.30,
}

# -----------------------------------------------------------------------------
# FUEL PRICES (CA, 2026)
# -----------------------------------------------------------------------------
# Source: CEC gasoline breakdown, EIA SCA weekly retail; PG&E and SoCalGas
# Jan 2026 rate advisories.
GAS_PRICE_DEFAULT = 4.90        # $/gal CA regular YTD 2026 baseline
GAS_PRICE_RANGE = (3.50, 6.50)  # sweep range for sensitivity

# Residential natural gas $/therm bundled non-CARE, Jan 2026.
# Use utility-specific values when available.
NG_THERM_PRICE = {
    "pge":       2.92,
    "socalgas":  2.08,   # used for SCE territory residential gas
    "sdge":      2.10,   # SDGE gas via SoCalGas-tariff structure
    "default":   2.40,
}

# -----------------------------------------------------------------------------
# EV / VEHICLE
# -----------------------------------------------------------------------------
# Source: Recurrent 2026 efficiency rankings; EPA 2024 Auto Trends Report;
# Caltrans CA VMT (2023 latest).
EV_EFFICIENCY = {
    "sedan":     4.0,    # mi/kWh real-world; range 4.0-4.3 (Model 3, Lucid)
    "crossover": 3.3,    # range 3.0-3.6 (Model Y, Leaf 2026)
    "default":   3.5,
}
ICE_MPG = {
    "sedan":     32,
    "crossover": 27,
    "default":   28,     # combined real-world; CAFE values run higher
}
VMT_DEFAULT = 12000              # mi/yr per household; CA per-vehicle ~10.2K
VMT_GRID = [5000, 8000, 12000, 15000, 20000, 25000]

# -----------------------------------------------------------------------------
# PV / Battery lifetime
# -----------------------------------------------------------------------------
# Sources: NREL ATB 2024b; Tesla Powerwall warranty; Enphase IQ warranty.
PV_LIFE_YEARS = 30
BATTERY_LIFE_YEARS = 15        # NREL ATB 2024b residential battery
BATTERY_CYCLE_WARRANTY = 3000   # Powerwall 3 / 70% retention at 10 yr

# -----------------------------------------------------------------------------
# Export rates / EEC (Energy Export Compensation)
# -----------------------------------------------------------------------------
# POLICY: All new interconnections after Apr 15, 2023 on Net Billing
# Tariff (NBT, "NEM 3.0"). Compensation = utility-published hourly EEC
# values from MIDAS, NOT a flat rate. We use the actual hourly file
# already in the parent repo:
#   ../eec_hourly_2025.csv          (datetime, utility, eec_total $/kWh)
#   ../LY2025 NBT Pricing Upload MIDAS.csv  (CPUC source data)
# Source: CPUC NEM/NBT; PG&E Solar Billing Plan EEC Price Sheet 2025.
EEC_HOURLY_CSV = CR_ROOT / "eec_hourly_2025.csv"
EEC_MIDAS_CSV = CR_ROOT / "LY2025 NBT Pricing Upload MIDAS.csv"

# Annual averages from eec_hourly_2025.csv (computed 2026-05-06):
EEC_ANNUAL_AVG = {
    "pge":   0.0968,   # $/kWh
    "sce":   0.0853,
    "sdge":  0.0782,
}

# Counterfactual export-rate scenarios for policy sensitivity.
# Use these only when overriding the hourly EEC file (e.g., NEM 2.0
# grandfathered customers, or "what-if" full retail).
NEM2_EXPORT_AVG = 0.32    # blended retail offset for NEM 2.0 customers
EXPORT_RATE_FLAT_SWEEP = [0.05, 0.07, 0.10, 0.15, 0.32]
EXPORT_REGIME = ("nbt_hourly", "nem2_retail", "flat_5c", "flat_15c")

# -----------------------------------------------------------------------------
# Sizing search grid (sizing_optimizer.py)
# -----------------------------------------------------------------------------
PV_KW_GRID = [0, 2, 4, 6, 8, 10, 12, 15]
BATT_KWH_GRID = [0, 5, 10, 15, 20, 30]


def utility_paths(utility: str) -> dict:
    return PIPELINE_OUTPUTS[utility.lower()]


def gas_price(utility: str) -> float:
    """Residential NG $/therm by utility."""
    key = {"pge": "pge", "sce": "socalgas", "sdge": "sdge"}.get(
        utility.lower(), "default")
    return NG_THERM_PRICE[key]


def assert_safe_out_dir(path) -> Path:
    """Refuse to write outside the EE data folder.

    Protects the parent california_rates pipeline outputs from accidental
    overwrite when this module is run from `california_rates/` (which it
    must be, for parent imports to work in sizing_optimizer_hourly.py).
    Any --out-dir argument that escapes EE_ROOT/data raises ValueError.
    """
    resolved = Path(path).resolve()
    expected = (EE_ROOT / "data").resolve()
    try:
        resolved.relative_to(expected)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside electrification_economics/data/. "
            f"Got: {resolved}\n"
            f"This guard exists to prevent accidental overwrite of the "
            f"parent california_rates pipeline outputs.") from exc
    return resolved
