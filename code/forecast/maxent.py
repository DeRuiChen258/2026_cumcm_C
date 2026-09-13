"""Low-order maximum-entropy residual models used by the Q2 defense layer.

The model is deliberately small: a pooled standardized residual shape is fitted
with moments up to order four, while each ten-minute slot keeps its own robust
scale.  This supplies an analytic tail shape without pretending that a 30-day
window can identify 144 independent high-order distributions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MaxEntFit:
    """Serializable diagnostics and a quantile evaluator for one pooled fit."""

    moments: np.ndarray
    lambdas: np.ndarray
    objective: float
    gradient_inf: float
    hessian_min_eig: float
    quadrature_points: int
    bound: float
    grid: np.ndarray
    cdf: np.ndarray
    density: np.ndarray

    def quantile(self, level: float) -> float:
        level = float(np.clip(level, 1e-6, 1.0 - 1e-6))
        return float(np.interp(level, self.cdf, self.grid))

    def absolute_quantile(self, level: float) -> float:
        """Quantile of |X| under the fitted density (no symmetry assumed)."""
        level = float(np.clip(level, 1e-6, 1.0 - 1e-6))
        magnitude = np.abs(self.grid)
        order = np.argsort(magnitude, kind="mergesort")
        m = magnitude[order]
        mass = self.density[order]
        cdf = np.cumsum((mass[1:] + mass[:-1]) * 0.5 * np.diff(m))
        cdf = np.concatenate(([0.0], cdf))
        total = float(cdf[-1])
        if total <= 1e-300:
            return 0.0
        cdf = np.maximum.accumulate(np.clip(cdf / total, 0.0, 1.0))
        return float(np.interp(level, cdf, m)) * self.bound

    def diagnostics(self) -> dict:
        return {
            "moments": self.moments.tolist(),
            "lambdas": self.lambdas.tolist(),
            "objective": float(self.objective),
            "gradient_inf": float(self.gradient_inf),
            "hessian_min_eig": float(self.hessian_min_eig),
            "quadrature_points": int(self.quadrature_points),
            "bound": float(self.bound),
        }


def _quadrature_stats(lam: np.ndarray, nodes: np.ndarray, weights: np.ndarray,
                      features: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    log_density = -(features @ lam)
    shift = float(np.max(log_density))
    mass = weights * np.exp(log_density - shift)
    z = float(np.sum(mass))
    prob = mass / max(z, 1e-300)
    mean = prob @ features
    centered = features - mean
    hessian = (centered.T * prob) @ centered
    return float(np.log(z) + shift), mean, hessian


def fit_maxent(samples: np.ndarray, order: int = 4, quadrature_points: int = 96,
               grid_points: int = 2049) -> MaxEntFit:
    """Fit ``p(x) ∝ exp(-sum(lambda_k*x**k))`` on ``[-1, 1]``.

    The caller is responsible for scaling raw residuals.  Inputs are clipped to
    the support because the bounded MaxEnt model is a robust tail approximation,
    not an extrapolation beyond the observed support.
    """
    x = np.asarray(samples, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < max(8, order + 2):
        raise ValueError("MaxEnt requires more finite samples")
    bound = max(float(np.quantile(np.abs(x), 0.995)), 1e-6)
    x = np.clip(x / bound, -1.0, 1.0)
    # Fit in the monomial basis the model card specifies.  Raw power moments
    # and monomial coefficients are what the paper reports, so the diagnostics
    # must live in that same parameterisation.
    powers = np.arange(1, order + 1, dtype=float)
    moments = np.array([np.mean(x ** int(k)) for k in range(1, order + 1)], dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    features = nodes[:, None] ** powers[None, :]

    def objective(lam: np.ndarray):
        log_z, mean, hessian = _quadrature_stats(lam, nodes, weights, features)
        return (float(log_z + lam @ moments), (moments - mean), hessian)

    def fun(lam: np.ndarray):
        value, gradient, _ = objective(lam)
        return value, gradient

    def fun_hess(lam: np.ndarray):
        _, gradient, hessian = objective(lam)
        return gradient, hessian

    lam = np.zeros(order)
    try:
        result = minimize(fun, lam, jac=True, hess=fun_hess, method="trust-exact",
                          options={"gtol": 1e-12, "maxiter": 400})
        lam = np.asarray(result.x, dtype=float)
    except Exception:                                  # pragma: no cover - fallback
        result = minimize(fun, lam, jac=True, method="L-BFGS-B",
                          bounds=[(-40.0, 40.0)] * order,
                          options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 1000})
        lam = np.asarray(result.x, dtype=float)
    # A few damped Newton steps remove the residual gradient when the trust
    # region stops early because the Hessian approaches singularity.
    for _ in range(12):
        _, gradient, hessian = objective(lam)
        if np.max(np.abs(gradient)) <= 1e-10:
            break
        eigval, eigvec = np.linalg.eigh(hessian)
        damped = eigvec @ np.diag(1.0 / np.maximum(eigval, 1e-8)) @ eigvec.T
        step = damped @ gradient
        scale = min(1.0, 1.0 / max(np.max(np.abs(step)), 1e-12))
        lam = lam - scale * step
    log_z, mean, hessian = _quadrature_stats(lam, nodes, weights, features)
    gradient = moments - mean
    grid = np.linspace(-1.0, 1.0, int(grid_points))
    grid_features = grid[:, None] ** powers[None, :]
    log_density = -(grid_features @ lam)
    density = np.exp(log_density - np.max(log_density))
    cdf = np.cumsum((density[:-1] + density[1:]) * np.diff(grid) * 0.5)
    cdf = np.concatenate(([0.0], cdf))
    cdf /= max(float(cdf[-1]), 1e-300)
    # Numerical integration can create tiny non-monotonicity after normalization.
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    cdf[-1] = 1.0
    return MaxEntFit(
        moments=moments,
        lambdas=lam,
        objective=float(log_z + lam @ moments),
        gradient_inf=float(np.max(np.abs(gradient))),
        hessian_min_eig=float(np.min(np.linalg.eigvalsh(hessian))),
        quadrature_points=int(quadrature_points),
        bound=bound,
        grid=grid,
        cdf=cdf,
        density=density,
    )


def pooled_quantile(residuals: np.ndarray, level: float, order: int = 4,
                    scale_floor: float = 1.0) -> tuple[np.ndarray, MaxEntFit]:
    """Return per-slot MaxEnt quantiles with pooled shape and robust scales."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2:
        raise ValueError("residuals must have shape (history_days, slots)")
    center = np.nanmean(values, axis=0)
    centered = values - center[None, :]
    scale = np.nanquantile(np.abs(centered), 0.9, axis=0)
    scale = np.maximum(np.nan_to_num(scale, nan=scale_floor), float(scale_floor))
    normalized = centered / scale[None, :]
    fit = fit_maxent(normalized, order=order)
    return center + scale * fit.quantile(level) * fit.bound, fit


def absolute_upper_quantile(residuals: np.ndarray, level: float, order: int = 4,
                            scale_floor: float = 1.0) -> tuple[np.ndarray, MaxEntFit]:
    """Return nonnegative per-slot upper magnitudes for a PV lower bound."""
    magnitude = np.abs(np.asarray(residuals, dtype=float))
    quantile, fit = pooled_quantile(magnitude, level, order=order,
                                    scale_floor=scale_floor)
    return np.maximum(0.0, quantile), fit
