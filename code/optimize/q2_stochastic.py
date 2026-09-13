"""问题2：负荷和光伏都不确定时，每天 0:00 定当天的计划购电量。

按天解两阶段随机规划：第一阶段定计划购电量和充放电，第二阶段按场景付 5 倍应急购电费。
报童结构算出来的临界分位是 1-1/5=0.80。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import (aci_update, conformal_level,
                                     hourly_forecast_to_slots, scenario_selection,
                                     weighted_quantile)
from code.forecast.maxent import fit_maxent
from code.optimize.lp_common import Storage, solve_stage_lp, solve_two_stage_lp

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
FIRST_DAY = 31          # 2025-02-01 (January is the conformal calibration period)
TARGET_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def merge_blocks(energy: np.ndarray, threshold_kwh: float = 1e-6) -> list[tuple[int, int, float]]:
    """Merge consecutive slots with non-zero energy into (first_slot, last_slot, energy)."""
    blocks: list[tuple[int, int, float]] = []
    start = None
    total = 0.0
    for k, value in enumerate(energy):
        if value > threshold_kwh:
            if start is None:
                start = k
            total += value
        elif start is not None:
            blocks.append((start, k - 1, total))
            start, total = None, 0.0
    if start is not None:
        blocks.append((start, len(energy) - 1, total))
    return blocks


def slot_time_label(slot_index: int) -> str:
    """Data-side label of slot k (1-based, right endpoint 10k minutes)."""
    minutes = slot_index * 10
    return f"{minutes // 60}:{minutes % 60:02d}"


def block_label(first: int, last: int) -> str:
    return f"{slot_time_label(first)}-{slot_time_label(last + 1)}"


def day_window_slice(d: int, lookback: int) -> slice:
    start = max(0, d - lookback)
    return slice(start, d)


def maxent_tail_scenario(residuals: np.ndarray, level: float, order: int = 4,
                         scale_floor: float = 1.0) -> tuple[np.ndarray, dict]:
    """MaxEnt residual shape -> one deterministic tail scenario (Q2-1).

    The MaxEnt density supplies the analytic tail shape; the block bootstrap
    still supplies the temporal dependence.  Both are needed: a 30-day window
    cannot identify 144 independent high-order distributions, but it can
    identify one pooled shape plus per-slot robust scales.
    """
    center, scale, fit = fit_pooled_shape(residuals, order=order, scale_floor=scale_floor)
    signed = center + scale * (fit.quantile(level) * fit.bound)
    magnitude = scale * fit.absolute_quantile(level)
    return signed, {
        **fit.diagnostics(),
        "level": float(level),
        # Per-slot magnitudes: a scalar would push a daytime-sized margin
        # into every night slot and destroy the PV lower bound.
        "upper_magnitude_per_slot": magnitude.tolist(),
        "median_upper_magnitude": float(np.median(magnitude)),
        "max_upper_magnitude": float(np.max(magnitude)),
        "scenario": "deterministic MaxEnt tail scenario alongside block-bootstrap SAA",
    }


def fit_pooled_shape(residuals: np.ndarray, order: int = 4,
                     scale_floor: float = 1.0):
    """Pooled MaxEnt shape plus per-slot robust scale (tasks/Q2.md Q2-1)."""
    values = np.asarray(residuals, dtype=float)
    center = np.nanmean(values, axis=0)
    centered = values - center[None, :]
    scale = np.nanquantile(np.abs(centered), 0.9, axis=0)
    scale = np.maximum(np.nan_to_num(scale, nan=scale_floor), float(scale_floor))
    fit = fit_maxent((centered / scale[None, :]).reshape(-1), order=order)
    return center, scale, fit


def pv_lower_bound(panel, d: int, win: slice, level: float) -> tuple[np.ndarray, dict]:
    """Physical PV lower bound for the day, from causal forecast errors (Q2-4).

    ``P_{k,L} = max(0, forecast_k - r_k)`` with ``r_k`` the MaxEnt conformal
    magnitude.  Mapping this bound back to net load gives ``N_hat + r_k``;
    the two readings are algebraically the same constraint, and the model
    uses only the net-load reading to avoid double counting.
    """
    pv = panel.pv_filled()
    history = np.array([hourly_forecast_to_slots(panel.pv_forecast[day, 0], 0)
                        for day in range(win.start, win.stop)])
    history = np.nan_to_num(history, nan=0.0)
    errors = pv[win] - history
    # NR8-style floor: night slots must not produce a zero scale that would
    # later divide by zero and fabricate a spurious margin.
    scale_floor = float(max(1.0, 0.005 * np.max(pv)))
    center, scale, fit = fit_pooled_shape(errors, scale_floor=scale_floor)
    magnitude = scale * fit.absolute_quantile(level)
    forecast = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[d, 0], 0), nan=0.0)
    lower = np.maximum(0.0, forecast - center - magnitude)
    # Night mask: core-night PV is identically zero, so the bound is 0 there.
    k_rise = int(panel.windows["k_rise"][d]) - 1
    k_set = int(panel.windows["k_set"][d]) - 1
    night = np.ones(144, dtype=bool)
    night[max(k_rise - 1, 0): k_set + 2] = False
    lower[night] = 0.0
    return lower, {
        **fit.diagnostics(),
        "level": float(level),
        "upper_magnitude_per_slot": magnitude.tolist(),
        "median_upper_magnitude": float(np.median(magnitude)),
        "max_upper_magnitude": float(np.max(magnitude)),
        "night_slots_masked": int(night.sum()),
        "pv_lower_kwh": float(lower.sum() * (1.0 / 6.0)),
    }


def solve_q2(panel, price: np.ndarray, tag: str = "q2", days: range | None = None,
             lookback: int = 30, n_scenarios: int = 30, cvar_beta: float = 0.0,
             cvar_alpha: float = 0.9, soft_penalty: float = 0.0,
             seed: int = 20250910, aci_gamma: float = 0.02,
             throughput_penalty: float = 0.0) -> dict:
    st = Storage.from_config(panel.config)
    tariff = panel.config.get("tariff", {})
    emergency = float(tariff.get("emergency_multiplier", 5.0))
    quantile = float(panel.config.get("quantiles", {}).get("q2_plan", 0.8))
    net_actual = panel.net_load()
    days = days or range(FIRST_DAY, panel.load.shape[0])
    del seed  # determinism: the pipeline never samples randomly

    q_plan = np.zeros((len(days), 144))
    charge = np.zeros_like(q_plan)
    disch = np.zeros_like(q_plan)
    soc = np.zeros((len(days), 145))
    emerg = np.zeros_like(q_plan)
    curtail = np.zeros_like(q_plan)
    pv_lower = np.zeros_like(q_plan)
    theta_trace = np.zeros((len(days), 2))     # (alpha, residual magnitude) per day
    records = []
    controls = {"C1_no_margin": [], "C2_analytic_newsvendor": [], "oracle_bound": [],
                "C3_terminal_value": []}
    coverage_hits = []
    missing_hits = []
    alpha = 1.0 - quantile          # ACI state (target miss rate)
    aci_trace = []
    maxent_trace = []

    for i, d in enumerate(days):
        win = day_window_slice(d, lookback)
        hist = net_actual[win]
        n_hat = hist.mean(axis=0)
        eps = hist - n_hat
        # Keep the LP size identical to the pre-v2 pipeline: the MaxEnt tail
        # scenario replaces one bootstrap draw rather than enlarging the model.
        scen_bootstrap = scenario_selection(eps, max(1, n_scenarios - 1))
        # Q2-1/Q2-4: the MaxEnt tail scenario supplies the analytic pessimistic
        # shape; the block bootstrap still carries the temporal dependence.
        tail_scenario, tail_info = maxent_tail_scenario(eps, 1.0 - alpha)
        scen = np.vstack([scen_bootstrap, tail_scenario[None, :]])
        lower, lower_info = pv_lower_bound(panel, d, win, 1.0 - alpha)
        pv_lower[i] = lower
        theta_trace[i] = (alpha, float(lower_info["median_upper_magnitude"]))
        # Store scalars only: the per-slot magnitude vector is 144 numbers per
        # day and would inflate q2.json by an order of magnitude.
        maxent_trace.append({
            "date": panel.dates[d],
            "gradient_inf": float(tail_info["gradient_inf"]),
            "hessian_min_eig": float(tail_info["hessian_min_eig"]),
            "objective": float(tail_info["objective"]),
            "bootstrap_extreme": float(np.max(np.abs(eps))),
            "tail_energy_ratio": float(
                np.sum(np.abs(tail_scenario)) / max(np.sum(np.abs(eps)), 1e-9)),
            "median_upper_magnitude": float(lower_info["median_upper_magnitude"]),
            "max_upper_magnitude": float(lower_info["max_upper_magnitude"]),
            "night_slots_masked": int(lower_info["night_slots_masked"]),
            "pv_lower_kwh": float(lower_info["pv_lower_kwh"]),
        })
        scen_net = n_hat[None, :] + scen
        p_row = price[d]

        lp = solve_two_stage_lp(scen_net, p_row, st, emergency_multiplier=emergency,
                                cvar_beta=cvar_beta, cvar_alpha=cvar_alpha,
                                slack_penalty=soft_penalty,
                                throughput_penalty=throughput_penalty)
        q_plan[i] = lp["q"]
        charge[i] = lp["c"]
        disch[i] = lp["d"]
        soc[i] = lp["soc"]

        n_act_energy = net_actual[d] * st.dt
        g = np.maximum(0.0, n_act_energy + lp["c"] - lp["d"] - lp["q"])
        emerg[i] = g
        curtail[i] = np.maximum(0.0, -n_act_energy - lp["c"] + lp["d"])
        cost = float(np.sum(p_row * lp["q"]) + emergency * np.sum(p_row * g))

        # --- control 1: deterministic plan on the mean profile -----------------
        det = solve_stage_lp(n_hat, p_row, st)
        g1 = np.maximum(0.0, n_act_energy + det["c"] - det["d"] - det["q"])
        controls["C1_no_margin"].append(float(np.sum(p_row * det["q"]) + emergency * np.sum(p_row * g1)))

        # --- control 2: analytic newsvendor quantile on the same storage path ---
        required = scen_net * st.dt + det["c"] - det["d"]
        level = conformal_level(1.0 - alpha, required.shape[0])
        q_nv = np.array([max(0.0, weighted_quantile(required[:, k], level))
                         for k in range(144)])
        g2 = np.maximum(0.0, n_act_energy + det["c"] - det["d"] - q_nv)
        controls["C2_analytic_newsvendor"].append(
            float(np.sum(p_row * q_nv) + emergency * np.sum(p_row * g2)))
        per_slot_required = n_hat * st.dt + det["c"] - det["d"]
        per_slot_quantile = np.array([weighted_quantile(required[:, k], level)
                                      for k in range(144)])
        miss = float(np.mean(n_act_energy > per_slot_quantile))
        coverage_hits.append(1.0 - miss)
        missing_hits.append(miss)
        aci_trace.append({"date": panel.dates[d], "level": level, "alpha": alpha, "miss": miss})
        alpha = aci_update(alpha, 1.0 - quantile, miss, gamma=aci_gamma)

        # --- oracle bound (perfect foresight, labelled as a bound only) --------
        oracle = solve_stage_lp(net_actual[d], p_row, st)
        controls["oracle_bound"].append(float(np.sum(p_row * oracle["q"])))

        # --- control 3: cross-day coupling (identical information, terminal value) --
        credit = float(np.mean(p_row[-12:]))
        free = solve_two_stage_lp(scen_net, p_row, st, emergency_multiplier=emergency,
                                  soc_end=None, terminal_credit=credit)
        controls["C3_terminal_value"].append(float(free["expected_cost"]))
        controls.setdefault("C3_terminal_value_cycle_expected", []).append(
            float(lp["expected_cost"]))

        blocks = merge_blocks(g)
        records.append({
            "date": panel.dates[d],
            "cost_cny": cost,
            "planned_purchase_kwh": float(lp["q"].sum()),
            "planned_cost_cny": float(np.sum(p_row * lp["q"])),
            "emergency_energy_kwh": float(g.sum()),
            "emergency_cost_cny": float(emergency * np.sum(p_row * g)),
            "emergency_slots": int(np.sum(g > 1e-6)),
            "emergency_blocks": len(blocks),
            "charge_kwh": float(lp["c"].sum()),
            "discharge_kwh": float(lp["d"].sum()),
            "soc_0_kwh": float(lp["soc"][0]),
            "soc_24_kwh": float(lp["soc"][-1]),
            "soc_min_kwh": float(lp["soc"].min()),
            "soc_max_kwh": float(lp["soc"].max()),
            "curtail_kwh": float(curtail[i].sum()),
            "expected_cost_cny": float(lp["expected_cost"]),
            "blocks": [{"first_slot": b[0] + 1, "last_slot": b[1] + 1,
                        "label": block_label(b[0], b[1]), "energy_kwh": b[2]} for b in blocks],
        })

    summary = {
        "tag": tag,
        "model": "two-stage stochastic LP (newsvendor structure, p*=0.80)",
        "days": len(records),
        "total_cost_cny": float(sum(r["cost_cny"] for r in records)),
        "total_emergency_cost_cny": float(sum(r["emergency_cost_cny"] for r in records)),
        "total_emergency_energy_kwh": float(sum(r["emergency_energy_kwh"] for r in records)),
        "total_purchase_kwh": float(sum(r["planned_purchase_kwh"] for r in records)),
        "emergency_slot_rate": float(np.mean([r["emergency_slots"] / 144.0 for r in records])),
        "emergency_purchase_share": float(
            sum(r["emergency_energy_kwh"] for r in records)
            / max(sum(r["planned_purchase_kwh"] + r["emergency_energy_kwh"] for r in records), 1e-9)),
        "days_with_emergency": int(sum(1 for r in records if r["emergency_slots"] > 0)),
        "avg_emergency_blocks_per_day": float(np.mean([r["emergency_blocks"] for r in records])),
        "controls_total_cost_cny": {k: float(np.sum(v)) for k, v in controls.items()},
        "quantile_coverage": float(np.mean(coverage_hits)),
        "quantile_coverage_nominal": quantile,
        "quantile_miss_rate": float(np.mean(missing_hits)),
        "aci_gamma": aci_gamma,
        "aci_final_alpha": float(alpha),
        "maxent": {
            "order": 4,
            "days_fitted": len(maxent_trace),
            "max_gradient_inf": float(np.max([t["gradient_inf"] for t in maxent_trace])),
            "min_hessian_eig": float(np.min([t["hessian_min_eig"] for t in maxent_trace])),
            "median_upper_magnitude_kw": float(np.median([t["median_upper_magnitude"]
                                                          for t in maxent_trace])),
            "max_upper_magnitude_kw": float(np.max([t["max_upper_magnitude"]
                                                    for t in maxent_trace])),
            "mean_pv_lower_kwh": float(np.mean([t["pv_lower_kwh"] for t in maxent_trace])),
            "role": "analytic tail shape; block bootstrap keeps temporal dependence",
        },
        "cvar_beta": cvar_beta,
        "soft_penalty": soft_penalty,
        "aci_path_identity": {
            "gamma": aci_gamma,
            "bound": float((1.0 + (0.999 - 0.001)) / max(len(days) * aci_gamma, 1e-9)),
            "note": "|mean(I) - alpha*| <= (|alpha_1 - alpha_T| + range)/ (T*gamma)",
        },
    }
    return {
        "summary": summary,
        "records": records,
        "aci_trace": aci_trace,
        "maxent_trace": maxent_trace,
        "arrays": {"q_plan": q_plan, "charge": charge, "discharge": disch, "soc": soc,
                   "emergency": emerg, "curtail": curtail,
                   "pv_lower": pv_lower, "theta": theta_trace},
    }


def save(out: dict, tag: str, dates: list[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    ARRAYS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{tag}.json").write_text(
        json.dumps({"summary": out["summary"], "records": out["records"],
                    "aci_trace": out.get("aci_trace", []),
                    "maxent_trace": out.get("maxent_trace", [])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for name, arr in out["arrays"].items():
        np.save(ARRAYS / f"{tag}_{name}.npy", arr)
    header = ("date,cost_cny,planned_purchase_kwh,planned_cost_cny,emergency_energy_kwh,"
              "emergency_cost_cny,emergency_slots,emergency_blocks,charge_kwh,discharge_kwh,"
              "soc_0_kwh,soc_24_kwh,soc_min_kwh,soc_max_kwh,curtail_kwh,expected_cost_cny")
    lines = [header]
    for r in out["records"]:
        lines.append(",".join([
            r["date"], f"{r['cost_cny']:.6f}", f"{r['planned_purchase_kwh']:.6f}",
            f"{r['planned_cost_cny']:.6f}", f"{r['emergency_energy_kwh']:.6f}",
            f"{r['emergency_cost_cny']:.6f}", str(r["emergency_slots"]), str(r["emergency_blocks"]),
            f"{r['charge_kwh']:.6f}", f"{r['discharge_kwh']:.6f}", f"{r['soc_0_kwh']:.6f}",
            f"{r['soc_24_kwh']:.6f}", f"{r['soc_min_kwh']:.6f}", f"{r['soc_max_kwh']:.6f}",
            f"{r['curtail_kwh']:.6f}", f"{r['expected_cost_cny']:.6f}",
        ]))
    (RESULTS / f"{tag}_days.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_panel()
    # Q2 uses the representative-day tariff of 附件1 replicated for every day; 附件4 is
    # reserved for Q4 (problem statement: Q1-Q3 have a constant daily price profile).
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    out = solve_q2(panel, price, tag="q2")
    save(out, "q2", list(panel.dates))
    s = out["summary"]
    print(f"[Q2] days={s['days']} total={s['total_cost_cny']:.2f} CNY "
          f"(avg/day {s['total_cost_cny']/s['days']:.2f})")
    print(f"[Q2] emergency energy={s['total_emergency_energy_kwh']:.1f} kWh "
          f"slot-rate={s['emergency_slot_rate']*100:.2f}% days-with-emergency={s['days_with_emergency']}")
    print(f"[Q2] quantile coverage={s['quantile_coverage']:.4f} (nominal {s['quantile_coverage_nominal']})")
    for k, v in s["controls_total_cost_cny"].items():
        print(f"[Q2] control {k}: {v:.2f} CNY (delta {v - s['total_cost_cny']:+.2f})")


if __name__ == "__main__":
    main()
