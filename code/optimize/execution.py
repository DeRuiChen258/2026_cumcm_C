"""Q4 control-barrier safety filter and dynamic event-triggered execution.

Two layers sit after the economic MPC:

* Layer 3 (C4) - a control-barrier-function projection guaranteeing the state
  of charge stays inside the physical band even when the nominal command comes
  from a relaxed (soft-constrained) MPC;
* Layer 4 (C5) - a dynamic event-triggered controller deciding whether a safe
  command is worth sending to the PCS or whether the previous command is held.

Both layers operate on slot **energy** (kWh) so the workbook writer and the
settlement formulas need no unit conversion. Power is recovered with
``P = E / dt`` only where a power-valued threshold is required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class ExecutionConfig:
    soc_min: float = 1200.0
    soc_max: float = 10800.0
    power: float = 5000.0
    dt: float = 1.0 / 6.0
    eta_c: float = 0.9
    eta_d: float = 0.9
    gamma_cbf: float = 6.0       # h^-1, so gamma_cbf*dt = 1.0 at slot scale
    gamma_0: float = 0.01        # relative trigger threshold on ||u||^2
    sigma_eta: float = 1.0       # dynamic auxiliary decay

    @property
    def max_charge_energy(self) -> float:
        return self.power * self.dt

    @property
    def max_discharge_energy(self) -> float:
        return self.power * self.dt


def _barrier_bounds(soc_end: float, cfg: ExecutionConfig) -> tuple[float, float]:
    """Discrete-CBF feasible window for the next state (h >= (1-g*dt) h)."""
    shrink = max(0.0, 1.0 - cfg.gamma_cbf * cfg.dt)
    lower = cfg.soc_min + shrink * (soc_end - cfg.soc_min)
    upper = cfg.soc_max - shrink * (cfg.soc_max - soc_end)
    return lower, upper


def cbf_project(soc_end: float, charge_energy: float, discharge_energy: float,
                cfg: ExecutionConfig) -> tuple[float, float, bool]:
    """Minimal-perturbation CBF projection of one slot's charge/discharge.

    Solves ``min 0.5*||u - u_nom||^2`` subject to the two barrier conditions
    over ``u = [c, d] >= 0``. ``u = 0`` is always feasible, so the problem is
    bounded and is solved in closed form.

    The feasible set is the polytope formed by the two barrier half-planes and
    the box ``[0, P*dt]^2``.  For a two-variable problem the projection onto a
    convex polygon is the unconstrained projection when it is feasible, and
    otherwise the best of the candidate solutions on each face.  Candidates
    with both ``c > 0`` and ``d > 0`` are discarded: simultaneous charge and
    discharge is physically inadmissible (tasks/Q1.md proposition 1) and it
    also makes the perturbation cost artificially smaller.  Returns
    ``(c_safe, d_safe, bound_active)``.
    """
    lower, upper = _barrier_bounds(soc_end, cfg)
    v = cfg.eta_c * charge_energy - discharge_energy / cfg.eta_d
    lo_delta = lower - soc_end
    hi_delta = upper - soc_end
    cmax, dmax = cfg.max_charge_energy, cfg.max_discharge_energy

    def admissible(c: float, d: float) -> bool:
        if c < -1e-12 or d < -1e-12:
            return False
        if c > cmax + 1e-9 or d > dmax + 1e-9:
            return False
        return lo_delta - 1e-9 <= cfg.eta_c * c - d / cfg.eta_d <= hi_delta + 1e-9

    def cost(c: float, d: float) -> float:
        dc, dd = c - charge_energy, d - discharge_energy
        return 0.5 * (dc * dc + dd * dd)

    if admissible(charge_energy, discharge_energy) and min(charge_energy,
                                                           discharge_energy) <= 1e-9:
        return float(charge_energy), float(discharge_energy), False

    # Exact projection over the three structured cases {idle, charge-only,
    # discharge-only}.  Mutually exclusive operation is a physical requirement
    # (proposition 1), not an optional refinement, so the projection never
    # trades it away for a smaller perturbation.
    candidates: list[tuple[float, float]] = [(0.0, 0.0)]
    # Charge-only: best feasible c is the projection of u_nom onto [c_lo, c_hi].
    c_lo = max(0.0, lo_delta / cfg.eta_c)
    c_hi = min(cmax, max(0.0, hi_delta / cfg.eta_c))
    if c_lo <= c_hi + 1e-12:
        candidates.append((float(np.clip(charge_energy, c_lo, c_hi)), 0.0))
    # Discharge-only: -d/eta_d must lie inside [lo_delta, hi_delta].
    d_lo = max(0.0, -cfg.eta_d * hi_delta)
    d_hi = min(dmax, max(0.0, -cfg.eta_d * lo_delta))
    if d_lo <= d_hi + 1e-12:
        candidates.append((0.0, float(np.clip(discharge_energy, d_lo, d_hi))))
    feasible = [u for u in candidates if admissible(*u)]
    if not feasible:
        return 0.0, 0.0, True
    best = min(feasible, key=lambda u: cost(*u))
    return float(best[0]), float(best[1]), True


def enforce_hold_safety(soc_end: float, charge_held: float, discharge_held: float,
                        charge_safe: float, discharge_safe: float,
                        cfg: ExecutionConfig) -> tuple[float, float]:
    """Safety dominates the trigger decision: reject an unsafe hold."""
    lower, upper = _barrier_bounds(soc_end, cfg)
    held_next = soc_end + cfg.eta_c * charge_held - discharge_held / cfg.eta_d
    if lower - 1e-9 <= held_next <= upper + 1e-9:
        return charge_held, discharge_held
    return charge_safe, discharge_safe


@dataclass
class DetcState:
    """Dynamic event-triggered controller state (Layer 4)."""

    cfg: ExecutionConfig = field(default_factory=ExecutionConfig)
    last_charge: float = 0.0
    last_discharge: float = 0.0
    eta: float = 1.0
    triggers: int = 0
    holds: int = 0
    trigger_slots: list = field(default_factory=list)
    switches: int = 0

    def step(self, charge_safe: float, discharge_safe: float,
             slot: int, drift: float = 0.0, drift_limit: float = float("inf")) -> tuple[float, float, bool]:
        """Return ``(c_real, d_real, triggered)`` for one slot.

        ``drift`` is the state error relative to the nominal trajectory.  A
        hold is only acceptable while the drift stays small: without the guard
        the controller accumulates state error and the daily-cycle contract
        can only be restored by a large repair.  The drift guard therefore
        makes the event trigger *state-aware* rather than error-only.
        """
        power_new = np.array([charge_safe, discharge_safe]) / self.cfg.dt
        power_last = np.array([self.last_charge, self.last_discharge]) / self.cfg.dt
        error = power_new - power_last
        error_sq = float(error @ error)
        threshold = (self.cfg.gamma_0 * float(power_new @ power_new)
                     + self.eta / self.cfg.sigma_eta)
        triggered = bool(error_sq > threshold or abs(drift) > drift_limit)
        # eta_dot = -sigma*eta + ||e||^2 - gamma_0*||u||^2, then clipped at 0.
        self.eta = max(self.eta - self.cfg.sigma_eta * self.eta
                       + error_sq - self.cfg.gamma_0 * float(power_new @ power_new), 0.0)
        if triggered:
            if (self.last_charge > 1e-9) != (charge_safe > 1e-9):
                self.switches += 1
            elif (self.last_discharge > 1e-9) != (discharge_safe > 1e-9):
                self.switches += 1
            self.last_charge, self.last_discharge = charge_safe, discharge_safe
            self.triggers += 1
            self.trigger_slots.append(slot + 1)
        else:
            self.holds += 1
        return self.last_charge, self.last_discharge, triggered

    def summary(self) -> dict:
        total = max(self.triggers + self.holds, 1)
        return {
            "triggers": int(self.triggers),
            "holds": int(self.holds),
            "total_slots": int(total),
            "hold_rate": float(self.holds / total),
            "trigger_rate": float(self.triggers / total),
            "charge_discharge_switches": int(self.switches),
            "final_eta": float(self.eta),
        }


def run_execution_layer(charge: np.ndarray, discharge: np.ndarray, soc: np.ndarray,
                        cfg: ExecutionConfig, drift_limit: float = 800.0) -> dict:
    """Apply CBF then DETC over a 144-slot schedule.

    ``charge``/``discharge`` are nominal slot energies and ``soc`` the
    accompanying state trajectory (length 145). Returns the executed
    trajectory plus the safety and trigger statistics used by the ablations.

    ``drift_limit`` is the state-error threshold for the drift guard.  It is an
    empirical operating point, not a fitted parameter: the sweep in
    ``results/q4.json`` records the action-reduction / repair-cost trade-off,
    and the reported value is the smallest drift limit that reaches a 50%
    action reduction while keeping the daily-cycle contract satisfiable.
    """
    n = len(charge)
    c_real = np.zeros(n)
    d_real = np.zeros(n)
    soc_real = np.zeros(n + 1)
    soc_real[0] = soc[0]
    cbf_active = np.zeros(n, dtype=bool)
    triggered = np.zeros(n, dtype=bool)
    detc = DetcState(cfg=cfg)
    for k in range(n):
        c_safe, d_safe, active = cbf_project(soc_real[k], charge[k], discharge[k], cfg)
        cbf_active[k] = active
        drift = soc_real[k] - soc[k]
        c_out, d_out, hit = detc.step(c_safe, d_safe, k, drift=drift,
                                      drift_limit=drift_limit)
        c_out, d_out = enforce_hold_safety(soc_real[k], c_out, d_out, c_safe, d_safe, cfg)
        c_real[k], d_real[k] = c_out, d_out
        triggered[k] = hit
        soc_real[k + 1] = soc_real[k] + cfg.eta_c * c_real[k] - d_real[k] / cfg.eta_d
    return {
        "charge": c_real,
        "discharge": d_real,
        "soc": soc_real,
        "cbf_active_slots": int(cbf_active.sum()),
        "cbf_first_slot": int(np.argmax(cbf_active) + 1) if cbf_active.any() else 0,
        "soc_min_observed": float(soc_real.min()),
        "soc_max_observed": float(soc_real.max()),
        "soc_violation_slots": int(np.sum((soc_real < cfg.soc_min - 1e-6)
                                          | (soc_real > cfg.soc_max + 1e-6))),
        "detc": detc.summary(),
        "trigger_mask": triggered.tolist(),
    }


def throughput_schedule(charge: np.ndarray, discharge: np.ndarray) -> dict:
    """Time-driven reference (A4): every slot is dispatched, no event logic."""
    return {
        "charge": np.asarray(charge, dtype=float).copy(),
        "discharge": np.asarray(discharge, dtype=float).copy(),
        "triggers": int(np.sum(np.asarray(charge) > 1e-9)
                        + np.sum(np.asarray(discharge) > 1e-9)),
        "holds": 0,
    }


def command_switches(charge: np.ndarray, discharge: np.ndarray) -> int:
    """Number of charge/discharge mode changes (PCS switching proxy)."""
    mode = np.zeros(len(charge), dtype=int)
    mode[np.asarray(charge, dtype=float) > 1e-9] = 1
    mode[np.asarray(discharge, dtype=float) > 1e-9] = -1
    return int(np.sum(mode[1:] != mode[:-1]))


def terminal_reconcile(charge: np.ndarray, discharge: np.ndarray, soc: np.ndarray,
                       target: float, cfg: ExecutionConfig) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return the executed schedule to the terminal-SoC contract when possible.

    The barrier filter and the event trigger both perturb the tail of the
    trajectory, so the last slot is used to restore ``e_K = e_0``.  If the
    correction would violate the barrier window or the power limits, the
    schedule is left unchanged and the caller reports the residual gap instead
    of silently breaking the daily-cycle contract.
    """
    c = np.array(charge, dtype=float)
    d = np.array(discharge, dtype=float)
    gap = float(target - soc[-1])
    if abs(gap) <= 1e-6:
        return c, d, True
    k = len(c) - 1
    if gap > 0:                                     # need more charge
        headroom = cfg.max_charge_energy - c[k]
        add = min(gap / cfg.eta_c, headroom)
        if add > 1e-9:
            c[k] += add
            soc[-1] = soc[-1] + cfg.eta_c * add
    else:                                           # need more discharge
        headroom = cfg.max_discharge_energy - d[k]
        add = min(-gap * cfg.eta_d, headroom)
        if add > 1e-9:
            d[k] += add
            soc[-1] = soc[-1] - add / cfg.eta_d
    ok = abs(soc[-1] - target) <= 1e-6
    return c, d, ok


def close_cycle(charge: np.ndarray, discharge: np.ndarray, soc_start: float,
                target: float, cfg: ExecutionConfig,
                throughput_weight: float = 1e-3) -> tuple[np.ndarray, np.ndarray, bool, float]:
    """Minimal-norm repair restoring ``e_K = target`` after CBF/DETC.

    The barrier filter and the event trigger both perturb the trajectory, so
    the executed schedule can end away from the daily-cycle contract.  This is
    repaired with one small LP over the whole horizon:

        min  sum_k ||u_k - u_exec_k||_1 + w * sum_k (c_k + d_k)
        s.t. e_0 = target, e_{k+1} = e_k + eta_c c_k - d_k/eta_d,
             e_K = target, 0 <= c,d <= P*dt, e in [E_min, E_max]

    The throughput term keeps the repair from freezing into a degenerate
    charge/discharge pair.  Returns the repaired schedule, whether the
    contract is met, and the L1 repair distance.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    n = len(charge)
    n_e = n - 1
    idx_c, idx_d, idx_e = 0, n, 2 * n
    nv = 2 * n + n_e
    rows, cols, vals, rhs = [], [], [], []
    for k in range(n):
        r = len(rhs)
        if k < n_e:
            rows.append(r), cols.append(idx_e + k), vals.append(1.0)
        if 1 <= k <= n_e:
            rows.append(r), cols.append(idx_e + k - 1), vals.append(-1.0)
        rows.append(r), cols.append(idx_c + k), vals.append(-cfg.eta_c)
        rows.append(r), cols.append(idx_d + k), vals.append(1.0 / cfg.eta_d)
        if k == 0:
            rhs.append(soc_start)
        elif k == n - 1:
            rhs.append(-float(target))
        else:
            rhs.append(0.0)
    # |u - u_exec| linearised with two auxiliary variables.
    pos, neg = np.zeros(n), np.zeros(n)
    rows2, cols2, vals2, b_ub = [], [], [], []
    nv2 = nv + 2 * n
    a_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(n, nv2)).tocsr()
    c_obj = np.zeros(nv2)
    for k in range(n):
        for offset, sign in ((0, 1.0), (n, -1.0)):
            r = len(b_ub)
            rows2.append(r), cols2.append(idx_c + k), vals2.append(sign)
            rows2.append(r), cols2.append(nv + offset + k), vals2.append(-1.0)
            b_ub.append(sign * float(charge[k]))
            r = len(b_ub)
            rows2.append(r), cols2.append(idx_d + k), vals2.append(sign)
            rows2.append(r), cols2.append(nv + offset + k), vals2.append(-1.0)
            b_ub.append(sign * float(discharge[k]))
    a_ub = sparse.coo_matrix((vals2, (rows2, cols2)), shape=(len(b_ub), nv2)).tocsr()
    c_obj = np.zeros(nv2)
    c_obj[nv:nv + 2 * n] = 1.0
    c_obj[idx_c:idx_c + n] += throughput_weight
    c_obj[idx_d:idx_d + n] += throughput_weight
    bounds = ([(0.0, cfg.max_charge_energy)] * n + [(0.0, cfg.max_discharge_energy)] * n
              + [(cfg.soc_min, cfg.soc_max)] * n_e + [(0.0, None)] * (2 * n))
    result = linprog(c_obj, A_ub=a_ub, b_ub=np.array(b_ub), A_eq=a_eq, b_eq=np.array(rhs),
                     bounds=bounds, method="highs")
    if not result.success:
        return charge, discharge, False, float("nan")
    c = result.x[idx_c:idx_c + n]
    d = result.x[idx_d:idx_d + n]
    e = np.empty(n + 1)
    e[0] = soc_start
    for k in range(n_e):
        e[k + 1] = result.x[idx_e + k]
    e[n] = float(target)
    repair = float(np.sum(result.x[nv:nv + 2 * n]))
    return c, d, abs(e[-1] - target) <= 1e-6, repair
