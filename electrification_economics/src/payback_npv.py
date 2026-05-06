"""Capex, incentives, financing, NPV, and payback calculations.

Pure-python financial helpers. Every other module builds cashflow streams
and hands them here so the financial math has one source of truth.

Conventions:
    - Year 0 = upfront capex (negative cashflow).
    - Years 1..N = annual savings (positive cashflow).
    - All flows in real (inflation-adjusted) dollars by default.
    - Discount rate: real if flows are real. Use config.DISCOUNT_RATE_REAL.

Functions:
    apply_capex_stack(...)        -> capex after rebates / credits
    annual_cashflow_series(...)   -> [cf_1, ..., cf_N] with escalator
    npv(cashflows, rate, capex)   -> scalar
    simple_payback(capex, ann)    -> years (or inf)
    discounted_payback(...)       -> years (or inf)
    levelized_cost(...)           -> $/yr equivalent annual cost

EV-specific:
    ev_net_premium(scenario, district)  -> net premium $ for an EV scenario
    ev_annual_fuel_savings(...)         -> $/yr from displacing ICE
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite
from typing import Iterable, Optional, Sequence

from . import config


# -----------------------------------------------------------------------------
# Capex / incentives
# -----------------------------------------------------------------------------

@dataclass
class CapexBreakdown:
    """Per-tech capex line items (gross, pre-incentive)."""
    pv_kw: float = 0.0
    battery_kwh: float = 0.0
    ev: bool = False
    heat_pump_space: bool = False
    heat_pump_water: bool = False
    induction_range: bool = False
    panel_upgrade: bool = False
    ev_charger: bool = False

    def gross_capex(self) -> float:
        c = config.CAPEX
        return (
            self.pv_kw * c["pv_per_kw"]
            + self.battery_kwh * c["battery_per_kwh"]
            + (c["ev_premium"] if self.ev else 0)
            + (c["ev_charger"] if self.ev_charger else 0)
            + (c["heat_pump_space"] if self.heat_pump_space else 0)
            + (c["heat_pump_water"] if self.heat_pump_water else 0)
            + (c["induction_range"] if self.induction_range else 0)
            + (c["panel_upgrade_200a"] if self.panel_upgrade else 0)
        )


@dataclass
class IncentiveContext:
    """Eligibility flags for a household."""
    income_pct_ami: float = 1.0          # 1.0 = 100% AMI; <0.8 unlocks equity
    is_dac: bool = False
    air_district: Optional[str] = None    # for CC4A
    sgip_tier: str = "general"            # general / equity / equity_resilience
    use_2024_counterfactual: bool = False
    ren: Optional[str] = None             # bayren / 3c_ren / none
    homes_eligible: bool = True           # CA HOMES rollout limited; toggleable


def apply_capex_stack(
    capex: CapexBreakdown,
    ctx: IncentiveContext,
) -> tuple[float, dict]:
    """Apply 2026 incentive stack to a capex breakdown.

    Returns (net_capex_after_incentives, itemized_rebates).

    Stacking rule (per assumptions_sources.md):
      IOU customers can stack TECH + SGIP-HPWH + Golden State + REN.
      EBD / ESA replace stack (handled upstream by population filter).
      Federal 25C/25D/30D = 0 in 2026 base case (use counterfactual flag
      to restore).
    """
    inc_set = (config.INCENTIVES_2024_COUNTERFACTUAL
               if ctx.use_2024_counterfactual
               else config.INCENTIVES_2026)
    inc_2026 = config.INCENTIVES_2026  # always need CA programs
    items = {}

    # PV
    if capex.pv_kw > 0:
        pv_gross = capex.pv_kw * config.CAPEX["pv_per_kw"]
        itc = inc_set.get("itc_pv", 0) * pv_gross
        if itc:
            items["fed_25d_pv"] = itc

    # Battery
    if capex.battery_kwh > 0:
        batt_gross = capex.battery_kwh * config.CAPEX["battery_per_kwh"]
        itc_b = inc_set.get("itc_battery", 0) * batt_gross
        if itc_b:
            items["fed_25d_battery"] = itc_b
        # SGIP — capped at tier-specific kWh max
        if ctx.sgip_tier == "general":
            kwh_eligible = min(capex.battery_kwh,
                               inc_2026["sgip_battery_general_cap_kwh"])
            items["sgip_battery"] = (
                kwh_eligible * inc_2026["sgip_battery_general_per_kwh"])
        elif ctx.sgip_tier == "equity":
            items["sgip_battery"] = (
                capex.battery_kwh * inc_2026["sgip_battery_equity_per_kwh"])
        elif ctx.sgip_tier == "equity_resilience":
            kwh_eligible = min(capex.battery_kwh,
                               inc_2026["sgip_battery_eq_resilience_cap_kwh"])
            items["sgip_battery"] = (
                kwh_eligible
                * inc_2026["sgip_battery_eq_resilience_per_kwh"])

    # HP space
    if capex.heat_pump_space:
        if ctx.income_pct_ami < 0.80:
            items["tech_hp_space"] = inc_2026["tech_hp_space_equity"]
        else:
            items["tech_hp_space"] = inc_2026["tech_hp_space_market"]
        if inc_set.get("fed_25c_hp_max", 0):
            items["fed_25c_hp"] = inc_set["fed_25c_hp_max"]

    # HPWH — biggest stack in CA
    if capex.heat_pump_water:
        if ctx.income_pct_ami < 0.80:
            items["tech_hpwh"] = inc_2026["tech_hpwh_equity"]
            items["sgip_hpwh"] = inc_2026["sgip_hpwh_low_income"]
        else:
            items["tech_hpwh"] = inc_2026["tech_hpwh_market"]
            items["sgip_hpwh"] = inc_2026["sgip_hpwh_standard"]
        items["golden_state_hpwh"] = inc_2026["golden_state_hpwh"]
        if ctx.ren == "bayren":
            items["bayren_hpwh"] = inc_2026["bayren_hpwh_fuel_sub"]
        elif ctx.ren == "3c_ren":
            items["3c_ren_hpwh"] = inc_2026["ren_3c_hpwh_sf"]
        if inc_set.get("fed_25c_hpwh_max", 0):
            items["fed_25c_hpwh"] = inc_set["fed_25c_hpwh_max"]

    # Panel upgrade
    if capex.panel_upgrade and inc_set.get("fed_25c_panel_max", 0):
        items["fed_25c_panel"] = inc_set["fed_25c_panel_max"]

    # HOMES (federal IRA, CEC-administered) — performance-based;
    # apply only if a whole-home retrofit (HP + HPWH at minimum) and
    # eligible flag set.
    if (ctx.homes_eligible
            and capex.heat_pump_space and capex.heat_pump_water):
        items["homes"] = (inc_2026["homes_max_low_income"]
                          if ctx.income_pct_ami < 0.80
                          else inc_2026["homes_max_market"])

    # EV
    if capex.ev:
        if inc_set.get("fed_30d_ev", 0):
            items["fed_30d_ev"] = inc_set["fed_30d_ev"]
        # CA: CC4A by district takes precedence over DCAP if eligible
        if ctx.air_district and ctx.income_pct_ami <= 4.0:  # ~400% FPL proxy
            district = config.CC4A_BY_DISTRICT.get(ctx.air_district)
            if district is not None:
                items["cc4a_new_ev"] = district["new_ev_max"]
        elif ctx.income_pct_ami <= 3.0:  # DCAP <=300% FPL proxy
            base = inc_2026["dcap_new_ev_max"]
            if ctx.is_dac:
                base += inc_2026["dcap_dac_bonus"]
            items["dcap_ev"] = base

    # EVSE (Section 30C still active until 6/30/2026; treat as 0 for 2026
    # base case beyond mid-year unless counterfactual set).
    if capex.ev_charger and inc_set.get("fed_30c_evse", 0):
        evse_gross = config.CAPEX["ev_charger"]
        items["fed_30c_evse"] = min(
            inc_set["fed_30c_evse"] * evse_gross, 1000)

    total_rebates = sum(items.values())
    net_capex = max(0.0, capex.gross_capex() - total_rebates)
    return net_capex, items


# -----------------------------------------------------------------------------
# Cashflow / NPV / payback
# -----------------------------------------------------------------------------

def annual_cashflow_series(
    annual_savings_year1: float,
    years: int = config.ANALYSIS_YEARS,
    escalator_real: float = config.BILL_ESCALATOR_REAL,
    midlife_replacement_year: Optional[int] = None,
    midlife_replacement_cost: float = 0.0,
) -> list[float]:
    """Real-dollar annual cashflows. Bill savings rise at escalator_real."""
    flows = []
    for t in range(1, years + 1):
        cf = annual_savings_year1 * ((1 + escalator_real) ** (t - 1))
        if midlife_replacement_year and t == midlife_replacement_year:
            cf -= midlife_replacement_cost
        flows.append(cf)
    return flows


def npv(
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
    capex: float = 0.0,
) -> float:
    """Net present value. capex is paid at year 0 (subtracted)."""
    pv = -capex
    for t, cf in enumerate(cashflows, start=1):
        pv += cf / ((1 + discount_rate) ** t)
    return pv


def simple_payback(capex: float, annual_savings: float) -> float:
    """Years to recover capex assuming flat year-1 savings. inf if never."""
    if annual_savings <= 0 or capex <= 0:
        return inf
    return capex / annual_savings


def discounted_payback(
    capex: float,
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
) -> float:
    """Years to recover capex on a discounted basis. inf if never."""
    if capex <= 0:
        return 0.0
    cumulative = 0.0
    for t, cf in enumerate(cashflows, start=1):
        cumulative += cf / ((1 + discount_rate) ** t)
        if cumulative >= capex:
            # linear interp within year t
            prev = cumulative - cf / ((1 + discount_rate) ** t)
            frac = (capex - prev) / (cumulative - prev)
            return (t - 1) + frac
    return inf


def levelized_annual_cost(
    capex: float,
    cashflows: Sequence[float],
    discount_rate: float = config.DISCOUNT_RATE_REAL,
) -> float:
    """Equivalent annual cost (annuity equivalent of NPV)."""
    n = len(cashflows)
    pv_savings = sum(cf / ((1 + discount_rate) ** t)
                     for t, cf in enumerate(cashflows, start=1))
    net_pv = pv_savings - capex
    if discount_rate == 0:
        return net_pv / n
    crf = (discount_rate * (1 + discount_rate) ** n) / (
        (1 + discount_rate) ** n - 1)
    return net_pv * crf


# -----------------------------------------------------------------------------
# EV economics helpers
# -----------------------------------------------------------------------------

def ev_net_premium(
    scenario: str,
    air_district: Optional[str] = None,
    is_dac: bool = False,
) -> float:
    """Net out-of-pocket premium for an EV scenario, after rebates.

    scenario: one of EV_SCENARIOS keys (config.EV_SCENARIOS).
    air_district: required if scenario == "scrap_replace_cc4a".
    """
    if scenario not in config.EV_SCENARIOS:
        raise ValueError(f"unknown EV scenario: {scenario}")
    s = config.EV_SCENARIOS[scenario]
    premium = s["premium"]

    if scenario == "scrap_replace_cc4a":
        if air_district is None:
            raise ValueError("air_district required for CC4A scenario")
        district = config.CC4A_BY_DISTRICT.get(air_district)
        if district is None:
            return premium  # district doesn't run CC4A
        rebate = district["new_ev_max"]
        # CC4A scraps the old vehicle; salvage = 0 by program rules.
        return premium - rebate
    elif scenario == "new_ev_dcap_dac":
        if not is_dac:
            raise ValueError("new_ev_dcap_dac requires is_dac=True")
        return premium - s["rebate"]
    else:
        return premium - s.get("rebate", 0)


def ev_annual_fuel_savings(
    vmt: float = config.VMT_DEFAULT,
    gas_price: float = config.GAS_PRICE_DEFAULT,
    ice_mpg: float = 28,
    ev_eff_mi_per_kwh: float = 3.5,
    rate_effective_per_kwh: float = 0.30,
) -> float:
    """Annual $ savings from displacing ICE miles with EV miles.

    Positive = savings. Captures the gasoline-spend reduction net of
    additional electricity cost at the rate-effective $/kWh paid for
    EV charging (depends on tariff and charging schedule; computed
    upstream by vmt_sensitivity.py).
    """
    gas_cost = (vmt / ice_mpg) * gas_price
    ev_cost = (vmt / ev_eff_mi_per_kwh) * rate_effective_per_kwh
    return gas_cost - ev_cost
