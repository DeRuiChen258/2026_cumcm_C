"""四个问题共用的稀疏 LP 建模部分。

储能本体（电量递推、充放功率上下限、日循环 e0=e144=6000）是同一套，只有购电和应急项不同，
所以集中放这儿，不必每个问题抄一遍。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass
class Storage:
    capacity: float = 12000.0
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    soc_init: float = 6000.0
    power: float = 5000.0
    eta_c: float = 0.9
    eta_d: float = 0.9
    dt: float = 1.0 / 6.0

    @classmethod
    def from_config(cls, config: dict) -> "Storage":
        s = config.get("storage", {})
        t = config.get("time", {})
        return cls(
            capacity=s.get("capacity_kwh", 12000.0),
            soc_min=s.get("soc_min_kwh", 1200.0),
            soc_max=s.get("soc_max_kwh", 10800.0),
            soc_init=s.get("soc_init_kwh", 6000.0),
            power=s.get("power_kw", 5000.0),
            eta_c=s.get("eta_charge", 0.9),
            eta_d=s.get("eta_discharge", 0.9),
            dt=t.get("dt_hours", 1.0 / 6.0),
        )

    @property
    def max_charge_energy(self) -> float:
        return self.power * self.dt

    @property
    def max_discharge_energy(self) -> float:
        return self.power * self.dt


def _coo(rows, cols, values, shape):
    return sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()


def solve_stage_lp(net_load: np.ndarray, price: np.ndarray, st: Storage,
                   soc_start: float | None = None, soc_end="cycle",
                   terminal_credit: float = 0.0, throughput_penalty: float = 0.0) -> dict:
    """Single-stage (deterministic) dispatch LP for one horizon.

    Variables (all in slot *energy*, kWh): q(n) >= 0 purchase, c(n) charge,
    d(n) discharge, e(n_e) SoC levels
    (interior levels, plus the terminal level when `soc_end` is None).
    Constraint q >= net_load + c - d: the slack is free curtailment, which is A11-safe
    because the optimum never purchases energy only to discard it (price > 0).

    `throughput_penalty` (CNY per kWh cycled) optionally prices battery wear; the default
    0.0 keeps the historical results bit-identical.
    """
    n = int(net_load.size)
    soc_start = st.soc_init if soc_start is None else float(soc_start)
    # soc_end="cycle"  -> daily cycle e_T = e_0 (main口径, A10)
    # soc_end=None     -> free terminal level (only for the V_T contrast experiment)
    # soc_end=<float>  -> partially fixed horizon
    if soc_end == "cycle":
        soc_end = soc_start
    fixed_end = soc_end is not None
    n_e = n - 1 if fixed_end else n
    nv = 3 * n + n_e

    def c_idx(k):
        return n + k

    def d_idx(k):
        return 2 * n + k

    def e_idx(k):
        return 3 * n + k

    rows, cols, vals, rhs = [], [], [], []
    for k in range(n):
        r = len(rhs)
        if k < n_e:
            rows.append(r), cols.append(e_idx(k)), vals.append(1.0)
        if 1 <= k <= n_e:
            rows.append(r), cols.append(e_idx(k - 1)), vals.append(-1.0)
        rows.append(r), cols.append(c_idx(k)), vals.append(-st.eta_c)
        rows.append(r), cols.append(d_idx(k)), vals.append(1.0 / st.eta_d)
        if k == 0:
            rhs.append(soc_start)
        elif k == n - 1 and fixed_end:
            rhs.append(-float(soc_end))
        else:
            rhs.append(0.0)
    a_eq = _coo(rows, cols, vals, (n, nv))
    b_eq = np.array(rhs)

    # q_k [kWh] >= N_k [kW] * dt + c_k - d_k  (c/d are slot energies, N is a power)
    rows, cols, vals = [], [], []
    for k in range(n):
        rows.extend([k, k, k])
        cols.extend([k, c_idx(k), d_idx(k)])
        vals.extend([-1.0, 1.0, -1.0])
    a_ub = _coo(rows, cols, vals, (n, nv))
    b_ub = -np.asarray(net_load, dtype=float) * st.dt

    c_obj = np.zeros(nv)
    # q holds slot *energy* (kWh), so money = price * q without an extra dt factor.
    c_obj[:n] = np.asarray(price, dtype=float)
    if throughput_penalty:
        c_obj[n:3 * n] += float(throughput_penalty)      # charge + discharge energy
    if not fixed_end and terminal_credit:
        c_obj[e_idx(n - 1)] = -float(terminal_credit)

    bounds = [(0.0, None)] * n
    bounds += [(0.0, st.max_charge_energy)] * n
    bounds += [(0.0, st.max_discharge_energy)] * n
    bounds += [(st.soc_min, st.soc_max)] * n_e

    res = linprog(c_obj, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"stage LP failed: {res.message}")

    x = res.x
    q = x[:n]
    charge = x[n:2 * n]
    disch = x[2 * n:3 * n]
    soc = np.empty(n + 1)
    soc[0] = soc_start
    for k in range(n_e):
        soc[k + 1] = x[e_idx(k)]
    if fixed_end:
        soc[n] = float(soc_end)
    curtail = np.maximum(0.0, disch - charge - np.asarray(net_load, dtype=float) * st.dt)
    return {
        "q": q, "c": charge, "d": disch, "soc": soc,
        "cost": float(res.fun), "curtail": curtail,
        "mu": _balance_duals(res, n), "lambda": _storage_duals(res, n),
        "status": int(res.status),
    }


def _balance_duals(res, n: int) -> np.ndarray:
    """Marginal cost of one extra kWh of net load (dual of q >= N + c - d)."""
    info = getattr(res, "ineqlin", None)
    if info is None or info.marginals is None:
        return np.full(n, np.nan)
    # HiGHS reports the marginals of A_ub x <= b_ub with the minimisation sign flipped.
    return -np.asarray(info.marginals[:n])


def _storage_duals(res, n: int) -> np.ndarray:
    """Positive storage-energy marginal values in CNY/kWh.

    The recurrence rows use ``e_t - e_{t-1} - eta_c*c_t + d_t/eta_d``.
    Negating the solver equality multipliers gives the economic value used by
    the KKT diagnostics while preserving the LP's existing sign convention.
    """
    info = getattr(res, "eqlin", None)
    if info is None or info.marginals is None:
        return np.full(n, np.nan)
    return -np.asarray(info.marginals[:n], dtype=float)


def solve_two_stage_lp(net_scenarios: np.ndarray, price: np.ndarray, st: Storage,
                       weights: np.ndarray | None = None, soc_start: float | None = None,
                       soc_end="cycle", emergency_multiplier: float = 5.0,
                       cvar_beta: float = 0.0, cvar_alpha: float = 0.9,
                       slack_penalty: float = 0.0, terminal_credit: float = 0.0,
                       throughput_penalty: float = 0.0) -> dict:
    """Two-stage LP: first-stage plan (q, c, d) + scenario recourse (emergency purchase g).

    Stage 1 (committed at 0:00): q_k, c_k, d_k and the SoC path.
    Stage 2 (per scenario s): g_{s,k} >= N_{s,k}*dt + c_k - d_k - q_k, g >= 0, billed at
    `emergency_multiplier` x the slot price.  With `cvar_beta > 0` the objective adds the
    Rockafellar-Uryasev CVaR term over scenario costs; with `slack_penalty > 0` a soft
    slack xi_k >= 0 is allowed in the balance (always-feasible MPC comparison).
    """
    net_scenarios = np.atleast_2d(np.asarray(net_scenarios, dtype=float))
    s_count, n = net_scenarios.shape
    price = np.asarray(price, dtype=float)
    if weights is None:
        weights = np.full(s_count, 1.0 / s_count)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    soc_start = st.soc_init if soc_start is None else float(soc_start)
    if soc_end == "cycle":
        soc_end = soc_start
    fixed_end = soc_end is not None
    n_e = n - 1 if fixed_end else n

    idx_q = 0
    idx_c = n
    idx_d = 2 * n
    idx_e = 3 * n
    idx_g = idx_e + n_e
    nv = idx_g + s_count * n
    idx_xi = nv
    if slack_penalty > 0:
        nv += n
    idx_zeta = nv
    idx_z = nv + (1 if cvar_beta > 0 else 0)
    if cvar_beta > 0:
        nv += 1 + s_count

    rows, cols, vals, rhs = [], [], [], []
    for k in range(n):
        r = len(rhs)
        if k < n_e:
            rows.append(r), cols.append(idx_e + k), vals.append(1.0)
        if 1 <= k <= n_e:
            rows.append(r), cols.append(idx_e + k - 1), vals.append(-1.0)
        rows.append(r), cols.append(idx_c + k), vals.append(-st.eta_c)
        rows.append(r), cols.append(idx_d + k), vals.append(1.0 / st.eta_d)
        if k == 0:
            rhs.append(soc_start)
        elif k == n - 1 and fixed_end:
            rhs.append(-float(soc_end))
        else:
            rhs.append(0.0)
    a_eq = _coo(rows, cols, vals, (n, nv))
    b_eq = np.array(rhs)

    rows, cols, vals, b_ub = [], [], [], []
    for s in range(s_count):
        for k in range(n):
            r = len(b_ub)
            rows.append(r), cols.append(idx_q + k), vals.append(-1.0)
            rows.append(r), cols.append(idx_c + k), vals.append(1.0)
            rows.append(r), cols.append(idx_d + k), vals.append(-1.0)
            rows.append(r), cols.append(idx_g + s * n + k), vals.append(-1.0)
            if slack_penalty > 0:
                rows.append(r), cols.append(idx_xi + k), vals.append(-1.0)
            b_ub.append(-net_scenarios[s, k] * st.dt)
    if cvar_beta > 0:
        for s in range(s_count):
            r = len(b_ub)
            for k in range(n):
                rows.append(r), cols.append(idx_q + k), vals.append(price[k])
                rows.append(r), cols.append(idx_g + s * n + k),
                vals.append(emergency_multiplier * price[k])
            rows.append(r), cols.append(idx_zeta), vals.append(-1.0)
            rows.append(r), cols.append(idx_z + s), vals.append(-1.0)
            b_ub.append(0.0)
    a_ub = _coo(rows, cols, vals, (max(len(b_ub), 1), nv))
    b_ub = np.array(b_ub) if b_ub else np.zeros(0)

    c_obj = np.zeros(nv)
    c_obj[idx_q:idx_q + n] = price
    if throughput_penalty:
        c_obj[idx_c:idx_c + n] += float(throughput_penalty)
        c_obj[idx_d:idx_d + n] += float(throughput_penalty)
    if not fixed_end and terminal_credit:
        c_obj[idx_e + n - 1] = -float(terminal_credit)
    for s in range(s_count):
        c_obj[idx_g + s * n: idx_g + (s + 1) * n] = (
            weights[s] * emergency_multiplier * price)
    if slack_penalty > 0:
        c_obj[idx_xi: idx_xi + n] = slack_penalty
    if cvar_beta > 0:
        c_obj *= (1.0 - cvar_beta)
        c_obj[idx_zeta] = cvar_beta
        c_obj[idx_z: idx_z + s_count] = cvar_beta * weights / (1.0 - cvar_alpha)

    bounds = [(0.0, None)] * n
    bounds += [(0.0, st.max_charge_energy)] * n
    bounds += [(0.0, st.max_discharge_energy)] * n
    bounds += [(st.soc_min, st.soc_max)] * n_e
    bounds += [(0.0, None)] * (s_count * n)
    if slack_penalty > 0:
        bounds += [(0.0, None)] * n
    if cvar_beta > 0:
        bounds += [(None, None)]
        bounds += [(0.0, None)] * s_count

    res = linprog(c_obj, A_ub=a_ub if a_ub.shape[0] else None, b_ub=b_ub if b_ub.size else None,
                  A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"two-stage LP failed: {res.message}")

    x = res.x
    q = x[idx_q: idx_q + n]
    charge = x[idx_c: idx_c + n]
    disch = x[idx_d: idx_d + n]
    g = x[idx_g: idx_g + s_count * n].reshape(s_count, n)
    soc = np.empty(n + 1)
    soc[0] = soc_start
    for k in range(n_e):
        soc[k + 1] = x[idx_e + k]
    soc[n] = float(soc_end) if fixed_end else float(x[idx_e + n - 1])
    credits = 0.0 if fixed_end else float(terminal_credit) * soc[n]
    scenario_cost = np.array([float(np.sum(price * q) + emergency_multiplier * np.sum(price * g[s])
                                    - credits) for s in range(s_count)])
    out = {
        "q": q, "c": charge, "d": disch, "soc": soc, "g": g,
        "cost": float(res.fun), "scenario_cost": scenario_cost,
        "expected_cost": float(scenario_cost @ weights),
        "q_energy": float(q.sum()), "charge_energy": float(charge.sum()),
        "discharge_energy": float(disch.sum()),
        "expected_emergency_energy": float(np.sum(g, axis=1) @ weights),
    }
    if cvar_beta > 0:
        out["cvar"] = float(x[idx_zeta] + np.sum(weights * x[idx_z: idx_z + s_count]) / (1 - cvar_alpha))
        out["zeta"] = float(x[idx_zeta])
    if slack_penalty > 0:
        out["slack_energy"] = float(x[idx_xi: idx_xi + n].sum())
    return out
