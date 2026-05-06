"""Unit tests for payback_npv financial helpers."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, payback_npv as p


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_npv_zero_rate():
    flows = [100.0, 100.0, 100.0]
    assert approx(p.npv(flows, discount_rate=0.0, capex=0.0), 300.0)


def test_npv_known_value():
    # PV of $100 for 3 years at 5% real = 100/(1.05) + 100/(1.05^2) + 100/(1.05^3)
    expected = 100/1.05 + 100/1.05**2 + 100/1.05**3
    assert approx(p.npv([100, 100, 100], 0.05, 0.0), expected)


def test_npv_subtracts_capex():
    capex = 1000
    flows = [200] * 6  # 6 yrs of 200 = 1200 undiscounted
    npv_no_cap = p.npv(flows, 0.05, 0)
    npv_with_cap = p.npv(flows, 0.05, capex)
    assert approx(npv_with_cap, npv_no_cap - capex)


def test_simple_payback_basic():
    assert approx(p.simple_payback(1000, 250), 4.0)


def test_simple_payback_zero_savings():
    assert p.simple_payback(1000, 0) == math.inf
    assert p.simple_payback(1000, -50) == math.inf


def test_discounted_payback_longer_than_simple():
    capex = 1000
    flows = [250] * 10
    sp = p.simple_payback(capex, 250)
    dp = p.discounted_payback(capex, flows, 0.05)
    assert dp > sp


def test_discounted_payback_never():
    flows = [10] * 5  # 50 total << 1000
    assert p.discounted_payback(1000, flows, 0.05) == math.inf


def test_annual_cashflow_escalates():
    flows = p.annual_cashflow_series(100, years=3, escalator_real=0.02)
    assert approx(flows[0], 100)
    assert approx(flows[1], 100 * 1.02)
    assert approx(flows[2], 100 * 1.02 ** 2)


def test_annual_cashflow_replacement():
    flows = p.annual_cashflow_series(
        100, years=15, escalator_real=0.0,
        midlife_replacement_year=13,
        midlife_replacement_cost=2500)
    assert approx(flows[12], 100 - 2500)  # year 13 (index 12)
    assert approx(flows[11], 100)


# ----- capex stack -----

def test_capex_stack_2026_zero_federal_for_pv():
    capex = p.CapexBreakdown(pv_kw=6, battery_kwh=10)
    ctx = p.IncentiveContext(income_pct_ami=1.5)
    net, items = p.apply_capex_stack(capex, ctx)
    # No federal ITC in 2026
    assert "fed_25d_pv" not in items
    assert "fed_25d_battery" not in items
    # SGIP applies to battery
    assert "sgip_battery" in items
    expected_sgip = 10 * config.INCENTIVES_2026["sgip_battery_general_per_kwh"]
    assert approx(items["sgip_battery"], expected_sgip)


def test_capex_stack_2024_counterfactual_restores_itc():
    capex = p.CapexBreakdown(pv_kw=6, battery_kwh=10)
    ctx = p.IncentiveContext(income_pct_ami=1.5, use_2024_counterfactual=True)
    net, items = p.apply_capex_stack(capex, ctx)
    pv_gross = 6 * config.CAPEX["pv_per_kw"]
    assert approx(items["fed_25d_pv"], 0.30 * pv_gross)
    batt_gross = 10 * config.CAPEX["battery_per_kwh"]
    assert approx(items["fed_25d_battery"], 0.30 * batt_gross)


def test_hpwh_stack_market_rate():
    capex = p.CapexBreakdown(heat_pump_water=True)
    ctx = p.IncentiveContext(income_pct_ami=1.5, ren="bayren")
    net, items = p.apply_capex_stack(capex, ctx)
    # market: TECH 2700 + SGIP 3800 + GSR 300 + BayREN 400 = 7200
    assert approx(items["tech_hpwh"], 2700)
    assert approx(items["sgip_hpwh"], 3800)
    assert approx(items["golden_state_hpwh"], 300)
    assert approx(items["bayren_hpwh"], 400)
    total = sum(items.values())
    assert approx(net, max(0, config.CAPEX["heat_pump_water"] - total))


def test_hpwh_stack_low_income():
    capex = p.CapexBreakdown(heat_pump_water=True)
    ctx = p.IncentiveContext(income_pct_ami=0.6)
    net, items = p.apply_capex_stack(capex, ctx)
    assert items["tech_hpwh"] == 4600  # equity midpoint
    assert items["sgip_hpwh"] == 4885  # SGIP LI


def test_capex_clamped_to_zero():
    """If rebates exceed capex, net is 0 not negative."""
    # HPWH ($5500) + stacked rebates (likely > $5500)
    capex = p.CapexBreakdown(heat_pump_water=True)
    ctx = p.IncentiveContext(income_pct_ami=0.6, ren="3c_ren")
    net, items = p.apply_capex_stack(capex, ctx)
    assert net >= 0
    assert net == 0  # for low-income with 3C-REN, stack >> $5,500


def test_ev_premium_new_new():
    # Base case: full premium, no rebate
    assert p.ev_net_premium("new_new") == config.CAPEX["ev_premium"]


def test_ev_premium_cc4a_scaqmd():
    # SCAQMD: $12,000 - so net premium = 5800 - 12000 = -6200
    net = p.ev_net_premium("scrap_replace_cc4a", air_district="SCAQMD")
    assert approx(net, config.CAPEX["ev_premium"] - 12000)


def test_ev_premium_cc4a_baaqmd():
    # BAAQMD: $9,500
    net = p.ev_net_premium("scrap_replace_cc4a", air_district="BAAQMD")
    assert approx(net, config.CAPEX["ev_premium"] - 9500)


def test_ev_premium_cc4a_san_diego_no_program():
    # SDAPCD doesn't run CC4A; falls back to gross premium
    net = p.ev_net_premium("scrap_replace_cc4a", air_district="SDAPCD")
    assert approx(net, config.CAPEX["ev_premium"])


def test_ev_fuel_savings_positive():
    # 12K mi, $4.90/gal, 28 mpg, 3.5 mi/kWh, $0.30/kWh
    gas_cost = 12000 / 28 * 4.90
    ev_cost = 12000 / 3.5 * 0.30
    expected = gas_cost - ev_cost
    actual = p.ev_annual_fuel_savings(
        vmt=12000, gas_price=4.90, ice_mpg=28,
        ev_eff_mi_per_kwh=3.5, rate_effective_per_kwh=0.30)
    assert approx(actual, expected)


if __name__ == "__main__":
    # Minimal test runner so we don't require pytest
    failures = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  PASS  {name}")
            except AssertionError as e:
                print(f"  FAIL  {name}  {e}")
                failures += 1
            except Exception as e:
                print(f"  ERR   {name}  {type(e).__name__}: {e}")
                failures += 1
    print(f"\n{failures} failure(s)" if failures else "\nall passed")
    sys.exit(failures)
