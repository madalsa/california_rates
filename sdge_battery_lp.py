"""
sdge_battery_lp.py — Stage 5: Battery dispatch optimization (LP + heuristic)

Optimizes battery charge/discharge to minimize net electricity cost.
Includes BOTH LP (optimal via scipy linprog/HiGHS) and heuristic (fast) methods.

Uses RASS-scaled demand.
"""

import numpy as np

from sdge_config import BATTERY_CAPACITY_KWH, BATTERY_POWER_KW, BATTERY_EFFICIENCY


def battery_lp_dispatch(hourly_load, solar_gen, rate_array, eec_rates=None):
    """
    Optimize battery dispatch to minimize net electricity cost via LP.

    Uses scipy.optimize.linprog with sparse matrices for fast setup and solve.

    Objective: minimize (import cost - export credit).
    The battery charges when rates are low (or from excess solar)
    and discharges when rates are high.

    Parameters
    ----------
    hourly_load : np.array (8760,)
        Hourly electricity consumption (kWh)
    solar_gen : np.array (8760,)
        Hourly solar generation (kWh)
    rate_array : np.array (8760,)
        Hourly electricity import rate ($/kWh)
    eec_rates : np.array (8760,) or None
        Hourly export compensation rate ($/kWh). If None, export credit = 0.

    Returns
    -------
    dict with optimized load profile, battery actions, and bill.
    None if LP fails.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    if eec_rates is None:
        eec_rates = np.zeros(8760)

    T = 8760
    eta = np.sqrt(BATTERY_EFFICIENCY)  # one-way efficiency
    cap = BATTERY_CAPACITY_KWH
    pmax = BATTERY_POWER_KW

    # Net load after solar
    net_load = hourly_load - solar_gen

    # Decision variables layout (5T total):
    #   g[0:T]       grid_import  >= 0
    #   e[T:2T]      grid_export  >= 0
    #   c[2T:3T]     charge       [0, pmax]
    #   d[3T:4T]     discharge    [0, pmax]
    #   s[4T:5T]     SOC          [0, cap]
    n = 5 * T

    # Objective: min sum(rate*g - eec*e)
    c_obj = np.zeros(n)
    c_obj[0:T] = rate_array        # import cost
    c_obj[T:2*T] = -eec_rates      # export credit (negative = benefit)

    # Bounds
    bounds = np.zeros((n, 2))
    bounds[0:T, 1] = np.inf        # g >= 0, no upper bound
    # Export bounded: can't pull from grid to re-export
    bounds[T:2*T, 1] = np.maximum(solar_gen, 0) + pmax
    bounds[2*T:3*T, 1] = pmax      # 0 <= c <= pmax
    bounds[3*T:4*T, 1] = pmax      # 0 <= d <= pmax
    bounds[4*T:5*T, 1] = cap       # 0 <= s <= cap

    # Equality constraints (2T equations):
    # Row 0..T-1: energy balance
    #   g[t] - e[t] - c[t] + d[t]*eta = net_load[t]
    # Row T..2T-1: SOC dynamics
    #   s[t] - s[t-1] - c[t]*eta + d[t] = 0  (for t>0)
    #   s[0] - c[0]*eta + d[0] = cap*0.5      (for t=0)

    # Build sparse A_eq in COO format
    rows = []
    cols = []
    vals = []

    tt = np.arange(T)

    # Energy balance: g[t] coeff = 1
    rows.append(tt); cols.append(tt); vals.append(np.ones(T))
    # Energy balance: e[t] coeff = -1
    rows.append(tt); cols.append(T + tt); vals.append(-np.ones(T))
    # Energy balance: c[t] coeff = -1
    rows.append(tt); cols.append(2*T + tt); vals.append(-np.ones(T))
    # Energy balance: d[t] coeff = eta
    rows.append(tt); cols.append(3*T + tt); vals.append(np.full(T, eta))

    # SOC dynamics row indices: T + tt
    soc_rows = T + tt
    # s[t] coeff = 1
    rows.append(soc_rows); cols.append(4*T + tt); vals.append(np.ones(T))
    # s[t-1] coeff = -1 (for t > 0)
    rows.append(soc_rows[1:]); cols.append(4*T + tt[:-1]); vals.append(-np.ones(T-1))
    # c[t] coeff = -eta
    rows.append(soc_rows); cols.append(2*T + tt); vals.append(np.full(T, -eta))
    # d[t] coeff = 1
    rows.append(soc_rows); cols.append(3*T + tt); vals.append(np.ones(T))

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    val_arr = np.concatenate(vals)

    A_eq = csc_matrix((val_arr, (row_idx, col_idx)), shape=(2*T, n))

    b_eq = np.zeros(2*T)
    b_eq[0:T] = net_load               # energy balance RHS
    b_eq[T] = cap * 0.5                # SOC(0) = cap*0.5 + c[0]*eta - d[0]
    # b_eq[T+1:2T] = 0                 # already zeros

    result = linprog(c_obj, A_eq=A_eq, b_eq=b_eq,
                     bounds=list(zip(bounds[:, 0], bounds[:, 1])),
                     method='highs', options={'time_limit': 10.0,
                                              'presolve': True,
                                              'dual_feasibility_tolerance': 1e-6,
                                              'primal_feasibility_tolerance': 1e-6})

    # Log any failures for diagnostics
    if result.status not in (0, 1) or result.x is None:
        import sys
        print(f"    LP status={result.status} msg='{result.message}' "
              f"net_load range=[{net_load.min():.1f}, {net_load.max():.1f}]",
              file=sys.stderr)

    # Accept optimal (0) and iteration-limit-with-feasible (1)
    if result.status in (0, 1) and result.x is not None:
        x = result.x
    elif result.status in (2, 3, 4):
        # Numerical difficulties — retry with scaled problem
        scale = max(np.abs(net_load).max(), 1.0)
        b_eq_scaled = b_eq / scale
        result2 = linprog(c_obj, A_eq=A_eq, b_eq=b_eq_scaled,
                          bounds=[(lo/scale, hi/scale if hi != np.inf else hi)
                                  for lo, hi in zip(bounds[:, 0], bounds[:, 1])],
                          method='highs', options={'time_limit': 10.0})
        if result2.status in (0, 1) and result2.x is not None:
            x = result2.x * scale
        else:
            return None
    else:
        return None
    grid_import_arr = x[0:T]
    grid_export_arr = x[T:2*T]
    charge_arr = x[2*T:3*T]
    discharge_arr = x[3*T:4*T]
    soc_arr = x[4*T:5*T]

    import_cost = np.dot(grid_import_arr, rate_array)
    export_credit = np.dot(grid_export_arr, eec_rates)

    return {
        'grid_import': grid_import_arr,
        'grid_export': grid_export_arr,
        'charge': charge_arr,
        'discharge': discharge_arr,
        'soc': soc_arr,
        'bill_energy': import_cost,
        'export_credit': export_credit,
        'net_cost': import_cost - export_credit,
    }


def battery_heuristic_dispatch(hourly_load, solar_gen, rate_array, eec_rates=None):
    """
    Fast heuristic battery dispatch (no LP solver needed).

    Strategy: charge during cheapest net-rate hours, discharge during most expensive.
    Also captures excess solar.
    """
    if eec_rates is None:
        eec_rates = np.zeros(8760)

    T = 8760
    eta = np.sqrt(BATTERY_EFFICIENCY)
    cap = BATTERY_CAPACITY_KWH
    pmax = BATTERY_POWER_KW

    net_load = hourly_load - solar_gen

    # Use net rate (import rate - export rate) for dispatch decisions
    net_rate = rate_array - eec_rates

    grid_import = np.maximum(net_load, 0).copy()
    grid_export = np.maximum(-net_load, 0).copy()
    soc = np.zeros(T)
    charge_arr = np.zeros(T)
    discharge_arr = np.zeros(T)

    # Simple forward pass
    current_soc = cap * 0.5
    sorted_net_rate = np.sort(net_rate)

    for t in range(T):
        # If excess solar, charge battery (reduce export)
        if net_load[t] < 0:
            excess = -net_load[t]
            can_charge = min(excess, pmax, (cap - current_soc) / eta)
            charge_arr[t] = can_charge
            current_soc += can_charge * eta
            grid_import[t] = 0
            grid_export[t] = max(excess - can_charge, 0)
        else:
            # If rate is high and battery has charge, discharge
            rate_pctile = np.searchsorted(sorted_net_rate, net_rate[t]) / T
            if rate_pctile > 0.6 and current_soc > 0:
                can_discharge = min(pmax, current_soc, net_load[t])
                discharge_arr[t] = can_discharge
                current_soc -= can_discharge
                grid_import[t] = max(net_load[t] - can_discharge * eta, 0)
            elif rate_pctile < 0.3 and current_soc < cap:
                # Charge during cheap hours
                can_charge = min(pmax, (cap - current_soc) / eta)
                charge_arr[t] = can_charge
                current_soc += can_charge * eta
                grid_import[t] = net_load[t] + can_charge

        soc[t] = current_soc

    import_cost = np.dot(grid_import, rate_array)
    export_credit = np.dot(grid_export, eec_rates)

    return {
        'grid_import': grid_import,
        'grid_export': grid_export,
        'charge': charge_arr,
        'discharge': discharge_arr,
        'soc': soc,
        'bill_energy': import_cost,
        'export_credit': export_credit,
        'net_cost': import_cost - export_credit,
    }
