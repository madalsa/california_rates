"""
sce_battery_lp.py — Stage 5: LP-only battery dispatch optimization

Optimizes battery charge/discharge to minimize net electricity cost.
Uses scipy.optimize.linprog with sparse matrices (HiGHS solver).

NO heuristic fallback — LP only per user specification.
Runs on NATIVE (unscaled) demand.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from sce_config import BATTERY_CAPACITY_KWH, BATTERY_POWER_KW, BATTERY_EFFICIENCY


def battery_lp_dispatch(hourly_load, solar_gen, rate_array, eec_rates=None):
    """
    Optimize battery dispatch to minimize net electricity cost via LP.

    Objective: minimize (import_cost - export_credit)
    Battery charges when rates are low / excess solar, discharges when high.

    Parameters
    ----------
    hourly_load : np.array (8760,)
        Native hourly electricity consumption (kWh) — NOT RASS-scaled.
    solar_gen : np.array (8760,)
        Hourly solar generation (kWh).
    rate_array : np.array (8760,)
        Hourly electricity import rate ($/kWh).
    eec_rates : np.array (8760,) or None
        Hourly export compensation rate ($/kWh).

    Returns
    -------
    dict with grid_import, grid_export, charge, discharge, soc,
         bill_energy, export_credit, net_cost.
    None if LP fails.
    """
    if eec_rates is None:
        eec_rates = np.zeros(8760)

    T = 8760
    eta = np.sqrt(BATTERY_EFFICIENCY)  # one-way efficiency
    cap = BATTERY_CAPACITY_KWH
    pmax = BATTERY_POWER_KW

    net_load = hourly_load - solar_gen

    # Decision variables (5T): g[0:T], e[T:2T], c[2T:3T], d[3T:4T], s[4T:5T]
    n = 5 * T

    # Objective: min sum(rate*g - eec*e)
    c_obj = np.zeros(n)
    c_obj[0:T] = rate_array
    c_obj[T:2*T] = -eec_rates

    # Bounds
    bounds = np.zeros((n, 2))
    bounds[0:T, 1] = np.inf        # grid import
    # Export bounded: can't pull from grid to re-export
    bounds[T:2*T, 1] = np.maximum(solar_gen, 0) + pmax
    bounds[2*T:3*T, 1] = pmax      # charge
    bounds[3*T:4*T, 1] = pmax      # discharge
    bounds[4*T:5*T, 1] = cap       # SOC

    # Equality constraints (2T):
    # Energy balance: g[t] - e[t] - c[t] + d[t]*eta = net_load[t]
    # SOC dynamics:   s[t] - s[t-1] - c[t]*eta + d[t] = 0
    #                 s[0] - c[0]*eta + d[0] = cap*0.5
    rows, cols, vals = [], [], []
    tt = np.arange(T)

    # Energy balance
    rows.append(tt); cols.append(tt); vals.append(np.ones(T))           # g
    rows.append(tt); cols.append(T + tt); vals.append(-np.ones(T))      # -e
    rows.append(tt); cols.append(2*T + tt); vals.append(-np.ones(T))    # -c
    rows.append(tt); cols.append(3*T + tt); vals.append(np.full(T, eta))  # d*eta

    # SOC dynamics
    soc_rows = T + tt
    rows.append(soc_rows); cols.append(4*T + tt); vals.append(np.ones(T))        # s[t]
    rows.append(soc_rows[1:]); cols.append(4*T + tt[:-1]); vals.append(-np.ones(T-1))  # -s[t-1]
    rows.append(soc_rows); cols.append(2*T + tt); vals.append(np.full(T, -eta))  # -c*eta
    rows.append(soc_rows); cols.append(3*T + tt); vals.append(np.ones(T))        # d

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    val_arr = np.concatenate(vals)

    A_eq = csc_matrix((val_arr, (row_idx, col_idx)), shape=(2*T, n))

    b_eq = np.zeros(2*T)
    b_eq[0:T] = net_load
    b_eq[T] = cap * 0.5  # initial SOC

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
    elif result.status == 4:
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
