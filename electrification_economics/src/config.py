"""Shared paths, constants, and economic assumptions.

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

# Economic assumptions (placeholders; calibrate from NREL ATB / IRA / SGIP)
DISCOUNT_RATE = 0.06
ANALYSIS_YEARS = 20
INFLATION = 0.025

# Capex placeholders ($/unit, post-incentive figures applied separately)
CAPEX = {
    "pv_per_kw":      2800,   # turnkey residential PV
    "battery_per_kwh": 900,   # lithium, installed
    "ev_premium":      8000,  # delta vs comparable ICE
    "ev_charger":      1500,
    "heat_pump_space": 14000, # ducted central HP install
    "heat_pump_water":  4500,
    "induction_range":  2200,
}

# Incentive placeholders (apply to capex; revisit per income tier and year)
INCENTIVES = {
    "itc_pv":               0.30,
    "itc_battery":          0.30,
    "ira_hp_space_max":     2000,
    "ira_hp_water_max":     1750,
    "ira_panel_max":        4000,
    "tech_hp_low_income":   3000,
    "sgip_battery_per_kwh": 200,    # baseline; equity adds more
    "ev_federal_credit":    7500,
    "ev_state_cvrp":        2000,
}

# Fuel + driving assumptions (sweep these in vmt_sensitivity.py)
GAS_PRICE_DEFAULT = 4.75  # $/gal CA average
EV_EFFICIENCY = 3.5       # mi/kWh
ICE_MPG = 28
VMT_DEFAULT = 12000

# Gas (natural gas) commodity for HP economics
NG_THERM_PRICE = 2.20  # $/therm

# Sizing search grid (sizing_optimizer.py)
PV_KW_GRID = [0, 2, 4, 6, 8, 10, 12, 15]
BATT_KWH_GRID = [0, 5, 10, 15, 20, 30]


def utility_paths(utility: str) -> dict:
    return PIPELINE_OUTPUTS[utility.lower()]
