"""共形预测的小工具：split/加权分位、ACI 反馈、场景削减。

夜里尺度过小会让归一化发散，所以 sigma 用 C++ 侧 NR8 给的 floor 兜底；
标定用的样本永远只看决策日之前的数据。
"""
from __future__ import annotations

import numpy as np


def weighted_quantile(sample: np.ndarray, q: float, weights: np.ndarray | None = None) -> float:
    """Weighted empirical quantile with the (q * sum w) convention."""
    sample = np.asarray(sample, dtype=float)
    if sample.size == 0:
        raise ValueError("empty sample")
    if weights is None:
        weights = np.ones_like(sample)
    order = np.argsort(sample, kind="mergesort")
    x = sample[order]
    w = np.asarray(weights, dtype=float)[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        raise ValueError("non-positive weight sum")
    idx = int(np.searchsorted(cw, q * total, side="left"))
    return float(x[min(max(idx, 0), x.size - 1)])


def conformal_level(level: float, n: int) -> float:
    """Split-conformal finite-sample correction: ceil((n+1) level)/n.

    Without this the empirical quantile of n residuals undershoots the nominal level
    by roughly (1-level)/n, which is visible as a systematic coverage deficit when the
    calibration window is only 30 days.
    """
    if n <= 0:
        return level
    return float(min(1.0, np.ceil((n + 1) * level) / n))


def conformal_quantile(residuals: np.ndarray, level: float, sigma: float | None = None) -> float:
    """Split-conformal quantile, optionally normalised by the NR8 scale sigma."""
    if sigma is None or sigma <= 0:
        return weighted_quantile(residuals, level)
    return float(sigma) * weighted_quantile(np.asarray(residuals, dtype=float) / sigma, level)


def aci_update(alpha: float, target_alpha: float, miss: float, gamma: float = 0.02) -> float:
    """Adaptive conformal inference: alpha_{t+1} = alpha_t + gamma (alpha* - miss)."""
    return float(np.clip(alpha + gamma * (target_alpha - miss), 0.001, 0.999))


def empirical_coverage(lower: np.ndarray, upper: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean((actual >= lower) & (actual <= upper)))


def scenario_selection(residuals: np.ndarray, n_scenarios: int) -> np.ndarray:
    """Deterministic stratified scenario reduction on the daily residual level.

    Rows are (block-correlated) daily residual vectors; we keep a quantile-spread
    subset, preserving the empirical tails without randomness so the pipeline stays
    byte-reproducible.
    """
    if residuals.shape[0] <= n_scenarios:
        return residuals
    level = residuals.sum(axis=1)
    order = np.argsort(level, kind="mergesort")
    idx = np.unique(np.linspace(0, residuals.shape[0] - 1, n_scenarios).round().astype(int))
    return residuals[order[idx]]


def hourly_forecast_to_slots(forecast_hourly: np.ndarray, vintage: int) -> np.ndarray:
    """Expand one vintage's 24 hourly PV forecasts onto the 144 ten-minute slots.

    Forecast column ``h`` (1-based) predicts the clock hour
    ``(vintage + h - 1, vintage + h]``.  Slot ``k`` (0-based) belongs to clock
    hour ``k // 6 + 1``, so its column is ``h = clock_hour - vintage``.  Slots
    at or before the vintage are already realised and therefore NaN.

    Lives here rather than in ``q3_mpc`` so Q2 can build its PV lower bound
    without importing the Q3 solver (which imports Q2).
    """
    slots = np.full(144, np.nan)
    for k in range(144):
        clock_hour = k // 6 + 1
        column = clock_hour - vintage
        if column < 1:
            continue
        slots[k] = forecast_hourly[column - 1]
    return slots
