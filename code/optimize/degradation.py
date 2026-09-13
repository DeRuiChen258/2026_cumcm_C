"""Battery degradation model: rainflow cycle counting + Wöhler cycle-life curve.

This replaces the earlier *throughput proxy* with an industry-standard engineering
model, so the third axis of the cost-risk-degradation Pareto (and the SOH column of
the ablation heatmap) is a real state-of-health estimate rather than an energy count.

Model
-----
1. The state-of-charge trajectory is reduced to its extrema (with a minimum swing
   threshold that removes numerical ripple) and processed with the ASTM E1049
   four-point rainflow algorithm, which decomposes the trajectory into cycles with
   depth-of-discharge DoD_i and half-cycles (counted as 0.5 each).
2. The cycle life follows the standard Wöhler/coffin-Manson form used in energy-system
   dispatch studies:      N(DoD) = N100 * DoD^(-k)     [cycles to failure]
3. Miner's linear damage accumulation gives the damage of the evaluation window
        D = sum_i  count_i / N(DoD_i)
   and the state of health after the window is  SOH = 1 - D  (capacity fade relative
   to a fresh cell).  The window is annualised (365/window_days) for reporting.
4. The damage is monetised with a linear depreciation:  C_deg = D * capex_pack, where
   capex_pack = unit_cost_cny_per_kwh * capacity_kwh.

Default parameters are typical LFP values (documented, not fitted to the data):
    N100        = 6000 full cycles at 100 % DoD (LFP, 25 degC)
    k (Wöhler)  = 1.1
    min swing   = 0.5 % of capacity (noise filter)
    pack cost   = 800 CNY/kWh  ->  12000 kWh pack = 9.6e6 CNY
All parameters are arguments, so the sensitivity of the conclusions to them can be
reported (see `results/degradation_model.json`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegradationParams:
    capacity_kwh: float = 12000.0
    n100_cycles: float = 6000.0          # cycles to failure at 100 % DoD
    wohler_k: float = 1.1                # cycle-life exponent
    min_swing_frac: float = 0.005        # ignore swings below 0.5 % of capacity
    capex_cny_per_kwh: float = 800.0     # pack capital cost for depreciation
    window_days: int = 334               # evaluation window (2025-02-01..12-31)

    @property
    def capex_pack(self) -> float:
        return self.capacity_kwh * self.capex_cny_per_kwh


@dataclass
class DegradationResult:
    equivalent_full_cycles: float        # throughput-equivalent cycles (reference)
    rainflow_cycles: float               # counted cycles (with half-cycles at 0.5)
    damage: float                        # Miner damage of the window
    annual_damage: float                 # window damage annualised
    soh_end: float                       # 1 - annual_damage (annualised capacity fade)
    annual_fade_pct: float
    degradation_cost_cny: float          # annualised, linear depreciation
    mean_dod: float
    max_dod: float


def _extrema(series: np.ndarray, min_swing: float) -> np.ndarray:
    """Keep the turning points of a SoC trajectory, dropping swings below min_swing."""
    if series.size == 0:
        return series
    points = [float(series[0])]
    direction = 0
    for value in series[1:]:
        if value == points[-1]:          # exact duplicates carry no cycle information
            continue
        delta = value - points[-1]
        if direction == 0:
            if abs(delta) > min_swing:   # strict: a zero threshold must not flip direction
                direction = 1 if delta > 0 else -1
                points.append(value)
            continue
        if direction > 0 and value > points[-1]:
            points[-1] = value
        elif direction < 0 and value < points[-1]:
            points[-1] = value
        elif (direction > 0 and points[-1] - value > min_swing) or \
             (direction < 0 and value - points[-1] > min_swing):
            points.append(value)
            direction = -direction
    return np.array(points)


def rainflow_cycles(soc: np.ndarray, min_swing: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """ASTM E1049 four-point rainflow decomposition of a SoC series.

    Returns (ranges, counts) where `ranges` are SoC swings (kWh) and `counts` are 1 for
    full cycles and 0.5 for half-cycles.
    """
    points = list(_extrema(np.asarray(soc, dtype=float), min_swing))
    ranges: list[float] = []
    counts: list[float] = []
    index = 0
    while index < len(points) - 2:
        a, b, c = points[index], points[index + 1], points[index + 2]
        if abs(b - a) <= abs(c - b):
            ranges.append(abs(b - a))
            counts.append(1.0)
            del points[index:index + 2]
            index = max(index - 2, 0)
        else:
            index += 1
    for i in range(len(points) - 1):
        ranges.append(abs(points[i + 1] - points[i]))
        counts.append(0.5)
    return np.array(ranges), np.array(counts)


def cycle_life(dod: np.ndarray, params: DegradationParams) -> np.ndarray:
    """Cycles to failure N(DoD) = N100 * DoD^(-k), clipped to the valid DoD range."""
    dod = np.clip(np.asarray(dod, dtype=float), 1e-6, 1.0)
    return params.n100_cycles * np.power(dod, -params.wohler_k)


def evaluate(soc_trajectory: np.ndarray, params: DegradationParams,
             throughput_cycles: float | None = None) -> DegradationResult:
    """Damage / SOH / degradation cost of one SoC trajectory (kWh, one day or window)."""
    ranges, counts = rainflow_cycles(np.asarray(soc_trajectory, dtype=float),
                                     params.min_swing_frac * params.capacity_kwh)
    dod = ranges / params.capacity_kwh
    damage = float(np.sum(counts / cycle_life(dod, params))) if dod.size else 0.0
    annual = damage * 365.0 / params.window_days
    if throughput_cycles is None:
        swings = np.abs(np.diff(np.asarray(soc_trajectory, dtype=float)))
        throughput_cycles = float(swings.sum() / (2.0 * params.capacity_kwh))
    return DegradationResult(
        equivalent_full_cycles=float(throughput_cycles),
        rainflow_cycles=float(counts.sum()),
        damage=damage,
        annual_damage=annual,
        soh_end=1.0 - annual,
        annual_fade_pct=100.0 * annual,
        degradation_cost_cny=annual * params.capex_pack,
        mean_dod=float(np.average(dod, weights=counts)) if dod.size else 0.0,
        max_dod=float(dod.max()) if dod.size else 0.0,
    )


def evaluate_panel(soc_matrix: np.ndarray, params: DegradationParams,
                   throughput_matrix: np.ndarray | None = None) -> DegradationResult:
    """Evaluate a multi-day SoC matrix (rows = days, columns = slot levels)."""
    ranges: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for row in np.asarray(soc_matrix, dtype=float):
        r, c = rainflow_cycles(row, params.min_swing_frac * params.capacity_kwh)
        ranges.append(r)
        counts.append(c)
    all_ranges = np.concatenate(ranges) if ranges else np.array([])
    all_counts = np.concatenate(counts) if counts else np.array([])
    dod = all_ranges / params.capacity_kwh
    damage = float(np.sum(all_counts / cycle_life(dod, params))) if dod.size else 0.0
    annual = damage * 365.0 / params.window_days
    if throughput_matrix is None:
        swings = np.abs(np.diff(np.asarray(soc_matrix, dtype=float), axis=1)).sum()
        throughput_cycles = float(swings / (2.0 * params.capacity_kwh))
    else:
        throughput_cycles = float(np.asarray(throughput_matrix).sum())
    return DegradationResult(
        equivalent_full_cycles=throughput_cycles,
        rainflow_cycles=float(all_counts.sum()),
        damage=damage,
        annual_damage=annual,
        soh_end=1.0 - annual,
        annual_fade_pct=100.0 * annual,
        degradation_cost_cny=annual * params.capex_pack,
        mean_dod=float(np.average(dod, weights=all_counts)) if dod.size else 0.0,
        max_dod=float(dod.max()) if dod.size else 0.0,
    )


def sensitivity(params: DegradationParams, soc_matrix: np.ndarray) -> dict:
    """SOH sensitivity to the two least certain parameters (N100 and the exponent)."""
    out = {}
    for n100 in (4000.0, 6000.0, 8000.0):
        for k in (0.9, 1.1, 1.3):
            variant = DegradationParams(**{**params.__dict__, "n100_cycles": n100,
                                           "wohler_k": k})
            res = evaluate_panel(soc_matrix, variant)
            out[f"N100={int(n100)}_k={k}"] = {
                "annual_fade_pct": res.annual_fade_pct,
                "degradation_cost_cny": res.degradation_cost_cny,
            }
    return out


def selftest() -> tuple[bool, list[str]]:
    """Golden cases for the rainflow decomposition and the damage arithmetic."""
    params = DegradationParams(capacity_kwh=12000.0, n100_cycles=6000.0, wohler_k=1.1,
                               min_swing_frac=0.0)
    logs: list[str] = []
    ok = True

    def check(name: str, condition: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and condition
        logs.append(f"[{'PASS' if condition else 'FAIL'}] {name} :: {detail}")

    # 1. one full cycle 6000 -> 10800 -> 6000 : DoD = 0.4, exactly one cycle
    soc = np.array([6000.0, 8000.0, 10800.0, 8000.0, 6000.0])
    ranges, counts = rainflow_cycles(soc)
    check("single full cycle", counts.sum() == 1.0 and abs(ranges.sum() - 4800.0) < 1e-9,
          f"cycles={counts.sum()}, range={ranges.sum():.1f} kWh")

    # 2. two equal swings -> two counted cycles (the rainflow pairing rule)
    soc = np.array([6000.0, 9000.0, 6000.0, 9000.0, 6000.0])
    ranges, counts = rainflow_cycles(soc)
    check("equal swings counted pairwise", abs(counts.sum() - 2.0) < 1e-9
          and abs(ranges.sum() - 6000.0) < 1e-9,
          f"cycles={counts.sum()}, range={ranges.sum():.1f} kWh")

    # 3. monotone charge -> one half cycle (count 0.5)
    soc = np.array([6000.0, 7000.0, 8000.0, 9000.0])
    ranges, counts = rainflow_cycles(soc)
    res = evaluate(soc, params)
    check("monotone ramp = half cycle", abs(counts.sum() - 0.5) < 1e-9,
          f"cycles={counts.sum()}, damage={res.damage:.3e}")

    # 4. flat trajectory -> no damage
    flat = np.full(144, 6000.0)
    res_flat = evaluate(flat, params)
    check("flat trajectory has no damage", res_flat.damage == 0.0 and res_flat.soh_end == 1.0,
          f"damage={res_flat.damage:.3e}")

    # 5. deeper cycles cause (strictly) more damage per unit throughput
    shallow = np.tile(np.array([6000.0, 6600.0, 6000.0]), 20)
    deep = np.tile(np.array([6000.0, 9000.0, 6000.0]), 20)
    d_shallow = evaluate(shallow, params).damage
    d_deep = evaluate(deep, params).damage
    check("depth-weighted damage", d_deep > d_shallow * 3,
          f"deep={d_deep:.3e} vs shallow={d_shallow:.3e}")

    # 6. DoD is defined on the *rated* capacity: the 1200-10800 window is DoD = 0.8 of
    #    the 12000 kWh pack, so its damage is exactly 1/N(0.8)
    full = np.array([1200.0, 10800.0, 1200.0])
    res_full = evaluate(full, DegradationParams(**{**params.__dict__, "min_swing_frac": 0.0}))
    expected = 1.0 / (6000.0 * (0.8 ** -1.1))
    check("usable-window swing = DoD 0.8", abs(res_full.damage - expected) < 1e-12,
          f"damage={res_full.damage:.6e} (expected {expected:.6e}, max DoD "
          f"{res_full.max_dod:.2f})")

    # 6b. with a pack sized to the usable window, a full sweep is exactly DoD 1.0 -> 1/N100
    tight = DegradationParams(**{**params.__dict__, "capacity_kwh": 9600.0,
                                 "min_swing_frac": 0.0})
    res_tight = evaluate(np.array([1200.0, 10800.0, 1200.0]), tight)
    check("DoD-1 sweep yields 1/N100", abs(res_tight.damage - 1.0 / 6000.0) < 1e-12,
          f"damage={res_tight.damage:.6e} (1/N100 = {1/6000:.6e})")

    # 7. noise filter: sub-threshold ripple must not create cycles
    ripple = 6000.0 + 20.0 * np.sin(np.linspace(0, 40 * np.pi, 144))
    res_ripple = evaluate(ripple, DegradationParams(**{**params.__dict__,
                                                      "min_swing_frac": 0.005}))
    check("ripple filtered by min swing", res_ripple.damage == 0.0,
          f"damage={res_ripple.damage:.3e}")
    return ok, logs


if __name__ == "__main__":
    passed, log = selftest()
    print("\n".join(log))
    print(f"[degradation] selftest {'ALL PASS' if passed else 'FAILED'}")
