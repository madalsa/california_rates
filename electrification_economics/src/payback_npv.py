"""Capex, incentives, financing, NPV, and payback calculations.

Single source of truth for the financial math. Other modules build cashflow
streams and hand them here.

Functions:
    apply_incentives(capex_dict, income_tier, year) -> capex_after
    annual_cashflow(years, savings_year1, escalator) -> [cf_t]
    npv(cashflows, discount_rate) -> float
    simple_payback(capex, annual_savings) -> float
    discounted_payback(capex, cashflows, discount_rate) -> float
    levelized_cost(capex, cashflows, discount_rate, years) -> $/yr

Financing options:
    - cash purchase (default)
    - loan: principal, rate, term -> monthly payment
    - PPA / lease (PV only): $/kWh contract, escalator
    - on-bill financing (utility-specific)

Bill escalator: real bill rises at 1-3%/yr above inflation in CA.
Use INFLATION + REAL_BILL_ESCALATOR (config) for nominal projections.
"""

# TODO(impl): pure-python financial helpers; unit tests in tests/.
