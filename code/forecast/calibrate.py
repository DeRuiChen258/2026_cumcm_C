"""共形标定：三档分位锚点（0.80 / 0.25 / 0.70）的覆盖率和 ACI 反馈。

每天拿最近 30 天的因果残差算共形分位（带有限样本修正），再和实测净负荷比覆盖率；
顺带用费用曲线反推报童最优分位，给这几个锚点一个经济上的说法。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import aci_update, conformal_level
from code.forecast.ts_light import ridge_harmonic_forecast, skill_score, slot_profile_forecast
from code.optimize.lp_common import Storage, solve_stage_lp
from code.optimize.q3_mpc import VINTAGE_HOUR, hourly_forecast_to_slots

RESULTS = ROOT / "results"
FIRST_DAY = 31
LOOKBACK = 30
LEVELS = {"q2_plan": 0.80, "q3_plan": 0.25, "q3_adjust": 0.70}


def _required(panel, d: int, st: Storage, j: int | None):
    """Conformal score sample and point prediction of the net-load requirement.

    j = None -> Q2 slot-profile forecast of N = L - PV;  j in 0..3 -> 附件3 vintage
    forecast, where the load of the decision day is treated as known (Q3 assumption).
    Returns (actual requirement, score sample [n_window, 144], covered-slot mask).
    """
    net = panel.net_load()
    win = slice(max(0, d - LOOKBACK), d)
    if j is None:
        base = net[win].mean(axis=0)
        scores = (net[win] - base) * st.dt
        predicted = base * st.dt
        mask = np.ones(144, dtype=bool)
    else:
        pv = panel.pv_filled()
        fc_hist = np.array([hourly_forecast_to_slots(panel.pv_forecast[day, j], VINTAGE_HOUR[j])
                            for day in range(win.start, win.stop)])
        fc_now = hourly_forecast_to_slots(panel.pv_forecast[d, j], VINTAGE_HOUR[j])
        scores = (np.nan_to_num(fc_hist, nan=0.0) - pv[win]) * st.dt
        predicted = (panel.load[d] - np.nan_to_num(fc_now, nan=0.0)) * st.dt
        mask = ~np.isnan(fc_now)
    return net[d] * st.dt, predicted, scores, mask


def coverage_report(panel, aci_gamma: float = 0.05) -> dict:
    st = Storage.from_config(panel.config)
    price = panel.day1[:, 0]
    out: dict = {"anchors": {}, "lookback_days": LOOKBACK, "aci_gamma": aci_gamma}
    for name, level in LEVELS.items():
        j = None if name == "q2_plan" else (0 if name == "q3_plan" else 2)
        alpha = 1.0 - level
        hits, misses, widths, trace = [], [], [], []
        day_hits = []
        for d in range(FIRST_DAY, panel.load.shape[0]):
            required, predicted, sample, mask = _required(panel, d, st, j)
            valid = ~np.isnan(sample).any(axis=1)
            sample = sample[valid]
            if sample.shape[0] < 5:
                continue
            lvl = conformal_level(1.0 - alpha, sample.shape[0])
            quantile = predicted + np.quantile(sample, lvl, axis=0)
            k_rise = int(panel.windows["k_rise"][d]) - 1
            k_set = int(panel.windows["k_set"][d]) - 1
            daylight = np.zeros(144, dtype=bool)
            daylight[max(k_rise, 0):k_set + 1] = True
            covered = mask & daylight
            if covered.sum() == 0:
                continue
            miss_slots = (required > quantile) & covered
            hit = 1.0 - float(np.mean(miss_slots[covered]))
            hits.append(hit)
            day_hits.append(hit)
            misses.append(float(np.mean(miss_slots[covered])))
            widths.append(float(np.mean(quantile - required)))
            trace.append({"date": panel.dates[d], "level": lvl, "alpha": alpha,
                          "coverage": hit})
            alpha = aci_update(alpha, 1.0 - level, float(np.mean(miss_slots[covered])),
                               gamma=aci_gamma)
        coverage = float(np.mean(hits))
        out["anchors"][name] = {
            "nominal": level,
            "coverage": coverage,
            "deviation": coverage - level,
            "mean_margin_kwh": float(np.mean(widths)),
            "final_alpha": float(alpha),
            "days": len(hits),
            "daylight_coverage": float(np.mean(day_hits)) if day_hits else float("nan"),
            "daylight_deviation": (float(np.mean(day_hits)) - level) if day_hits else float("nan"),
            "statistic": "coverage over the informative daylight slots of each day",
            "aci_trace": trace,
        }
    return out


def newsvendor_level(panel, n_days: int = 120) -> dict:
    """Empirical optimum of the Q2 quantile level (cost vs coverage trade-off)."""
    st = Storage.from_config(panel.config)
    price = panel.day1[:, 0]
    emergency = float(panel.config.get("tariff", {}).get("emergency_multiplier", 5.0))
    net = panel.net_load()
    days = list(range(FIRST_DAY, panel.load.shape[0], 3))[:n_days]
    grid = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    costs, coverages = [], []
    for level in grid:
        total, hit = 0.0, []
        for d in days:
            win = net[max(0, d - LOOKBACK):d]
            base = win.mean(axis=0)
            det = solve_stage_lp(base, price, st)
            sample = (win - base) * st.dt
            lvl = conformal_level(level, sample.shape[0])
            required_sample = base[None, :] * st.dt + det["c"] - det["d"] + sample
            q = np.quantile(required_sample, lvl, axis=0)
            required = net[d] * st.dt + det["c"] - det["d"]
            g = np.maximum(0.0, required - q)
            total += float(np.sum(price * q) + emergency * np.sum(price * g))
            hit.append(float(np.mean(required <= q)))
        costs.append(total)
        coverages.append(float(np.mean(hit)))
    best = int(np.argmin(costs))
    return {"levels": grid, "cost_cny": costs, "coverage": coverages,
            "argmin_level": grid[best], "days": len(days),
            "note": "sampled every 3rd day; storage schedule fixed by the mean forecast"}


def forecast_skill(panel, n_days: int = 90) -> dict:
    """Skill score of the ridge-harmonic predictor against the persistence baseline."""
    load, pv = panel.load, panel.pv_filled()
    days = list(range(FIRST_DAY, panel.load.shape[0], 4))[:n_days]
    out = {}
    for name, values in (("load", load), ("pv", pv)):
        ridge_err, pers_err = [], []
        for d in days:
            ridge = ridge_harmonic_forecast(values, d, lookback=LOOKBACK)
            pers = slot_profile_forecast(values, d, lookback=LOOKBACK)
            ridge_err.append(float(np.mean((values[d] - ridge) ** 2)))
            pers_err.append(float(np.mean((values[d] - pers) ** 2)))
        out[name] = {
            "mse_ridge": float(np.mean(ridge_err)),
            "mse_persistence": float(np.mean(pers_err)),
            "skill_score_vs_persistence": 1.0 - float(np.mean(ridge_err)) / float(np.mean(pers_err)),
            "days": len(days),
        }
    return out


def main() -> None:
    panel = load_panel()
    report = {
        "coverage": coverage_report(panel),
        "newsvendor_level": newsvendor_level(panel),
        "forecast_skill": forecast_skill(panel),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "p2_conformal.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    for name, stats in report["coverage"]["anchors"].items():
        print(f"[P2] {name}: nominal={stats['nominal']:.2f} coverage={stats['coverage']:.4f} "
              f"deviation={stats['deviation']*100:+.2f}pp days={stats['days']}")
    nv = report["newsvendor_level"]
    print(f"[P2] empirical optimal Q2 level = {nv['argmin_level']:.2f} "
          f"(grid {nv['levels']}, {nv['days']} sample days)")
    for name, stats in report["forecast_skill"].items():
        print(f"[P2] {name}: skill vs persistence = {stats['skill_score_vs_persistence']:+.3f}")


if __name__ == "__main__":
    main()
