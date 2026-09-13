"""几个轻量预测基线：同时段均值、谐波岭回归、技能评分。

只允许用决策日之前的数据，别让当天的实测值漏进 0:00 的计划里。
"""
from __future__ import annotations

import numpy as np


def history_slice(d: int, lookback: int, min_days: int = 7) -> slice:
    """Causal window [max(0, d-lookback), d), expanded when the day is early."""
    start = max(0, d - lookback)
    if d - start < min_days:
        start = max(0, d - min_days)
    return slice(start, d)


def slot_profile_forecast(values: np.ndarray, d: int, lookback: int = 30,
                          min_days: int = 7) -> np.ndarray:
    """Mean same-slot profile over the causal window (persistence/climatology blend)."""
    window = values[history_slice(d, lookback, min_days)]
    return window.mean(axis=0)


def hourly_features(slots: np.ndarray) -> np.ndarray:
    """Design matrix: const + daily and half-daily harmonics of the right endpoint."""
    hours = np.asarray(slots, dtype=float) / 6.0
    w = 2.0 * np.pi * hours / 24.0
    return np.column_stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])


def ridge_harmonic_forecast(values: np.ndarray, d: int, lookback: int = 30,
                            ridge: float = 1.0, min_days: int = 7) -> np.ndarray:
    """Per-slot ridge regression on harmonic time features (one solve for all slots)."""
    window = values[history_slice(d, lookback, min_days)]
    n_days, n_slots = window.shape
    x = hourly_features(np.arange(n_slots))
    design = np.repeat(x, n_days, axis=0)
    day_scale = np.tile(np.arange(n_days)[:, None], (1, n_slots)) / max(n_days - 1, 1)
    design = np.column_stack([design, day_scale.ravel()])
    y = window.T.reshape(-1)
    lhs = design.T @ design + ridge * np.eye(design.shape[1])
    coef = np.linalg.solve(lhs, design.T @ y)
    x_new = np.column_stack([x, np.ones(n_slots)])
    return x_new @ coef


def skill_score(actual: np.ndarray, forecast: np.ndarray, reference: np.ndarray) -> float:
    """1 - MSE(forecast)/MSE(reference); > 0 means better than the reference."""
    num = float(np.mean((actual - forecast) ** 2))
    den = float(np.mean((actual - reference) ** 2))
    return 1.0 - num / den if den > 0 else 0.0
