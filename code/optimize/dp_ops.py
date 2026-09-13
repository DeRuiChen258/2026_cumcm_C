"""储能电量的动态规划（min-plus 递推），用来和 LP 互相验证。

DP 把电量离散成网格、每个时段挑最优净动作；两边算出来对得上，说明 LP 和 DP 都没写错。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from code.optimize.lp_common import Storage


@dataclass
class DPResult:
    cost: float
    soc: np.ndarray                # n+1 levels
    charge_energy: np.ndarray      # kWh per slot
    discharge_energy: np.ndarray   # kWh per slot
    grid_energy: np.ndarray        # kWh purchased per slot
    step_kwh: float
    states: int


def solve_soc_dp(net_load: np.ndarray, price: np.ndarray, st: Storage, step_kwh: float,
                 soc_start: float | None = None, soc_end: float | None = None) -> DPResult:
    soc_start = st.soc_init if soc_start is None else float(soc_start)
    soc_end = soc_start if soc_end is None else float(soc_end)
    net_load = np.asarray(net_load, dtype=float)
    price = np.asarray(price, dtype=float)
    n = int(net_load.size)

    states = np.arange(st.soc_min, st.soc_max + step_kwh / 2.0, step_kwh)
    n_s = states.size
    s0 = int(round((soc_start - st.soc_min) / step_kwh))
    s_end = int(round((soc_end - st.soc_min) / step_kwh))
    if abs(states[s0] - soc_start) > 1e-9 or abs(states[s_end] - soc_end) > 1e-9:
        raise ValueError("soc_start/soc_end must lie on the discretisation grid")

    up_steps = int(np.floor(st.eta_c * st.max_charge_energy / step_kwh + 1e-9))
    down_steps = int(np.floor(st.max_discharge_energy / st.eta_d / step_kwh + 1e-9))
    actions = np.concatenate([np.arange(-down_steps, 0), np.arange(0, up_steps + 1)]) * step_kwh
    charge_kw = np.where(actions > 0, actions / st.eta_c, 0.0) / st.dt
    disch_kw = np.where(actions < 0, -actions * st.eta_d, 0.0) / st.dt
    shift = np.rint(actions / step_kwh).astype(int)

    cost_hist = np.empty((n + 1, n_s))
    cost_hist[0] = np.inf
    cost_hist[0, s0] = 0.0
    money_hist = np.empty((n, actions.size))
    for k in range(n):
        grid_kw = np.maximum(0.0, net_load[k] + charge_kw - disch_kw)
        money = price[k] * st.dt * grid_kw
        money_hist[k] = money
        dest = np.clip(np.arange(n_s)[:, None] + shift[None, :], 0, n_s - 1)
        ok = (np.arange(n_s)[:, None] + shift[None, :] >= 0) & (
            np.arange(n_s)[:, None] + shift[None, :] < n_s)
        cand = np.where(ok, cost_hist[k][:, None] + money[None, :], np.inf)
        new_cost = np.full(n_s, np.inf)
        np.minimum.at(new_cost, dest.ravel(), cand.ravel())
        cost_hist[k + 1] = new_cost

    if not np.isfinite(cost_hist[n, s_end]):
        raise RuntimeError("DP is infeasible for the requested terminal SoC")

    # backward reconstruction: pick an action whose transition matches the optimum
    soc_path = np.zeros(n + 1)
    soc_path[n] = states[s_end]
    acts = np.zeros(n)
    state = s_end
    for k in range(n - 1, -1, -1):
        target = cost_hist[k + 1, state]
        chosen = None
        for a in range(actions.size):
            prev = state - shift[a]
            if prev < 0 or prev >= n_s:
                continue
            if abs(cost_hist[k, prev] + money_hist[k, a] - target) < 1e-9:
                chosen = a
                break
        if chosen is None:
            raise RuntimeError("DP reconstruction failed (numerical tolerance)")
        acts[k] = actions[chosen]
        state = state - shift[chosen]
        soc_path[k] = states[state]

    charge_energy = np.where(acts > 0, acts / st.eta_c, 0.0)
    discharge_energy = np.where(acts < 0, -acts * st.eta_d, 0.0)
    grid_energy = np.maximum(0.0, net_load * st.dt + charge_energy - discharge_energy)
    return DPResult(cost=float(cost_hist[n, s_end]), soc=soc_path, charge_energy=charge_energy,
                    discharge_energy=discharge_energy, grid_energy=grid_energy,
                    step_kwh=float(step_kwh), states=int(n_s))
