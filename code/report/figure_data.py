"""Data preparation for the paper figure suite required by `图片.md`.

Only the *visualisation* layer changes here: every array is either read from
`results/` or recomputed by calling the existing model functions unchanged
(`code.optimize` / `code.forecast` are imported, never edited).  Expensive products
are cached in `results/paper_fig_data.json` so the figures are reproducible without
re-solving.

Figure map (图片.md):
  Fig.0 framework        -> results 直接绘制（无新数据）
  Fig.1 dispatch+SOC     -> q1.json / q1_slots.csv
  Fig.2 shadow price     -> q1_slots.csv（μ_t）+ KKT 阈值判定
  Fig.3 prediction fan   -> conformal bands（本模块计算）
  Fig.4 reliability/sharpness -> 本模块计算（4 种方法 × 多名义覆盖率）
  Fig.5 VOI boundary     -> q3_days.csv（S0–S4）+ q3 plan/adjust 数组
  Fig.6 rolling update   -> q3 arrays
  Fig.7 RT control + SOC -> q4_2 / q4_3 arrays
  Fig.8 cost-risk-degradation Pareto -> 本模块运行 β 扫描
  Fig.9 ablation heatmap -> M0–M4 指标矩阵（本模块计算 + 已有结果）
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import conformal_level, weighted_quantile
from code.optimize.degradation import DegradationParams, evaluate_panel, sensitivity as deg_sens
from code.optimize.lp_common import Storage, solve_stage_lp, solve_two_stage_lp
from code.optimize.q3_mpc import VINTAGE_HOUR, hourly_forecast_to_slots

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
CACHE = RESULTS / "paper_fig_data.json"
FIRST_DAY = 31
LOOKBACK = 30


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arr(name: str) -> np.ndarray:
    return np.load(ARRAYS / name)


def read_days_csv(tag: str) -> list[dict]:
    with (RESULTS / f"{tag}_days.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _cache() -> dict:
    return load_json(CACHE) if CACHE.exists() else {}


def _update(**blocks) -> dict:
    data = _cache()
    data.update(blocks)
    CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# Fig.2 shadow price / KKT coupling
# ---------------------------------------------------------------------------

def shadow_price_coupling(panel) -> dict:
    """Verified marginal-value statements for the Q1 optimum (KKT interpretation).

    Facts checked here (all verifiable from the delivered solution bundle):
      1. complementary slackness of the purchase variable: q_t > 0 ⇒ μ_t = p_t
         (and μ_t ≤ p_t everywhere);
      2. every discharging slot satisfies p_t ≥ μ_t (the market price is above the
         marginal value of stored energy);
      3. charging concentrates in the cheapest hours (median market price well below
         the discharging median).
    The *full* storage KKT also contains the SoC dual λ, which is not part of the
    delivered bundle, so the figure reports the verified statements only.
    """
    q1 = load_json(RESULTS / "q1.json")
    st = Storage.from_config(panel.config)
    price = panel.day1[:, 0]
    q = np.array(q1["lp"]["q"])
    charge = np.array(q1["lp"]["c"])
    disch = np.array(q1["lp"]["d"])
    mu = np.array(q1["lp"]["mu"])
    tol = 1e-6 * float(np.max(price))
    purchase = q > 1e-9
    ch = charge > 1e-6
    dis = disch > 1e-6
    mu_eq_p = int(np.sum(np.abs(mu[purchase] - price[purchase]) <= tol))
    mu_violations = int(np.sum(mu > price + tol))
    dis_above = int(np.sum(price[dis] >= mu[dis] - tol))
    return {
        "price": price.tolist(), "mu": mu.tolist(),
        "charge": charge.tolist(), "discharge": disch.tolist(),
        "purchase_slots": int(purchase.sum()),
        "mu_eq_price_slots": mu_eq_p,
        "mu_above_price_violations": mu_violations,
        "charge_slots": int(ch.sum()),
        "discharge_slots": int(dis.sum()),
        "discharge_price_above_mu": dis_above,
        "charge_price_median": float(np.median(price[ch])) if ch.any() else float("nan"),
        "discharge_price_median": float(np.median(price[dis])) if dis.any() else float("nan"),
        "discharge_ratio_median": float(np.median(price[dis] / mu[dis])) if dis.any() else float("nan"),
        "charge_ratio_median": float(np.median(price[ch] / mu[ch])) if ch.any() else float("nan"),
        "eta_c": st.eta_c, "eta_d": st.eta_d,
    }


# ---------------------------------------------------------------------------
# Fig.3 prediction fan chart with ACI bands
# ---------------------------------------------------------------------------

def prediction_fan(panel, day_iso: str = "2025-06-21", level: float = 0.90,
                   bands=(0.50, 0.80, 0.90, 0.95)) -> dict:
    """PV prediction intervals for one day (0:00 vintage) + ACI coverage trace."""
    pv = panel.pv_filled()
    d = panel.dates.index(day_iso)
    forecast = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[d, 0], VINTAGE_HOUR[0]),
                             nan=0.0)
    win = slice(max(0, d - LOOKBACK), d)
    fc_hist = np.array([np.nan_to_num(
        hourly_forecast_to_slots(panel.pv_forecast[t, 0], VINTAGE_HOUR[0]), nan=0.0)
        for t in range(win.start, win.stop)])
    residuals = pv[win] - fc_hist

    # causal ACI state for the monitored level, evaluated on every day
    alpha = 1.0 - level
    aci_days, aci_miss, aci_level = [], [], []
    for day in range(FIRST_DAY, 365):
        w = slice(max(0, day - LOOKBACK), day)
        fch = np.array([np.nan_to_num(
            hourly_forecast_to_slots(panel.pv_forecast[t, 0], VINTAGE_HOUR[0]), nan=0.0)
            for t in range(w.start, w.stop)])
        res = pv[w] - fch
        lvl = conformal_level(1.0 - alpha, res.shape[0])
        upper = np.quantile(res, lvl, axis=0)
        k_rise = int(panel.windows["k_rise"][day]) - 1
        k_set = int(panel.windows["k_set"][day]) - 1
        mask = np.zeros(144, dtype=bool)
        mask[max(k_rise, 0):k_set + 1] = True
        fcn = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[day, 0],
                                                     VINTAGE_HOUR[0]), nan=0.0)
        miss = float(np.mean((pv[day] - fcn > upper)[mask]))
        aci_days.append(panel.dates[day])
        aci_miss.append(miss)
        aci_level.append(lvl)
        alpha = float(np.clip(alpha + 0.05 * ((1.0 - level) - miss), 0.001, 0.999))

    bands_out = {}
    for b in bands:
        lvl = conformal_level(b, residuals.shape[0])
        bands_out[f"q{int(b*100)}"] = {
            "lower": (forecast + np.quantile(residuals, 1.0 - lvl, axis=0)).tolist(),
            "upper": (forecast + np.quantile(residuals, lvl, axis=0)).tolist(),
        }
    lvl_aci = aci_level[panel.dates.index(day_iso) - FIRST_DAY] if d >= FIRST_DAY else 0.9
    upper_aci = forecast + np.quantile(residuals, lvl_aci, axis=0)
    lower_aci = forecast + np.quantile(residuals, 1.0 - lvl_aci, axis=0)
    return {
        "date": day_iso, "forecast": forecast.tolist(), "actual": pv[d].tolist(),
        "bands": bands_out,
        "aci": {"level": float(lvl_aci), "lower": lower_aci.tolist(),
                "upper": upper_aci.tolist(),
                "violations": (pv[d] > upper_aci).tolist()},
        "aci_trace": {"dates": aci_days, "miss": aci_miss, "level": aci_level},
        "achieved_coverage": float(1.0 - np.mean(aci_miss)),
        "mean_miss_rate": float(np.mean(aci_miss)),
        "target_miss_rate": float(1.0 - level),
        "days_miss_below_target": float(np.mean(np.array(aci_miss) <= 1.0 - level)),
    }


# ---------------------------------------------------------------------------
# Fig.4 reliability diagram + sharpness
# ---------------------------------------------------------------------------

def reliability_sharpness(panel, levels=(0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95),
                          n_boot: int = 200, seed: int = 20250911) -> dict:
    """Empirical coverage (reliability) and mean interval width (sharpness) per method."""
    pv = panel.pv_filled()
    rng = np.random.default_rng(seed)
    method_names = ["经验分位（无修正）", "split 共形（有限样本修正）", "ACI 自适应",
                    "Bootstrap 分位均值"]
    coverage = {name: [] for name in method_names}
    widths = {name: [] for name in method_names}
    windows = []
    for day in range(FIRST_DAY, 365):
        w = slice(max(0, day - LOOKBACK), day)
        fch = np.array([np.nan_to_num(
            hourly_forecast_to_slots(panel.pv_forecast[t, 0], VINTAGE_HOUR[0]), nan=0.0)
            for t in range(w.start, w.stop)])
        fcn = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[day, 0],
                                                    VINTAGE_HOUR[0]), nan=0.0)
        k_rise = int(panel.windows["k_rise"][day]) - 1
        k_set = int(panel.windows["k_set"][day]) - 1
        mask = np.zeros(144, dtype=bool)
        mask[max(k_rise, 0):k_set + 1] = True
        if mask.sum() == 0:
            continue
        windows.append((pv[w] - fch, pv[day] - fcn, mask))

    n_bag = 15          # bagged (bootstrap-averaged) quantiles per day
    for level in levels:
        alpha = 1.0 - level
        cov = {name: [] for name in method_names}
        wid = {name: [] for name in method_names}
        for res, actual_res, mask in windows:
            n = res.shape[0]
            split_lvl = conformal_level(level, n)
            aci_lvl = conformal_level(1.0 - alpha, n)
            bounds = {
                "经验分位（无修正）": np.quantile(res, level, axis=0),
                "split 共形（有限样本修正）": np.quantile(res, split_lvl, axis=0),
                "ACI 自适应": np.quantile(res, aci_lvl, axis=0),
            }
            for name, upper in bounds.items():
                lower = np.quantile(res, 1.0 - conformal_level(level, n), axis=0)
                cov[name].append(float(np.mean((actual_res <= upper)[mask])))
                wid[name].append(float(np.mean((upper - lower)[mask])))
            # bootstrap: bagged (bootstrap-averaged) quantile of the causal window
            draws = np.empty((n_bag, 144))
            for b in range(n_bag):
                idx = rng.integers(0, res.shape[0], res.shape[0])
                draws[b] = np.quantile(res[idx], conformal_level(level, res.shape[0]), axis=0)
            boot_upper = draws.mean(axis=0)
            boot_lower = np.quantile(res, 1.0 - conformal_level(level, res.shape[0]), axis=0)
            cov["Bootstrap 分位均值"].append(float(np.mean((actual_res <= boot_upper)[mask])))
            wid["Bootstrap 分位均值"].append(float(np.mean((boot_upper - boot_lower)[mask])))
            miss = 1.0 - cov["ACI 自适应"][-1]
            alpha = float(np.clip(alpha + 0.05 * ((1.0 - level) - miss), 0.001, 0.999))
        for name in method_names:
            coverage[name].append(float(np.mean(cov[name])))
            widths[name].append(float(np.mean(wid[name])))
    return {"levels": list(levels), "methods": coverage, "widths": widths}


# ---------------------------------------------------------------------------
# Fig.5/6 VOI decision boundary and rolling update trajectory
# ---------------------------------------------------------------------------

def voi_boundary(panel) -> dict:
    """VOI_k vs adjustment penalty C_adjust,k for the three update vintages."""
    rows = read_days_csv("q3")
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    plan, adj = arr("q3_q_plan.npy"), arr("q3_q_adjust.npy")
    segments = {6: (36, 72), 12: (72, 108), 18: (108, 144)}
    out = {6: [], 12: [], 18: []}
    for i, row in enumerate(rows):
        s0, s1, s2, s3 = (float(row[k]) for k in ("S0", "S1", "S2", "S3"))
        voi = {6: s0 - s1, 12: s1 - s2, 18: s2 - s3}
        p_row = price[FIRST_DAY + i]
        for stage, (a, b) in segments.items():
            penalty = float(0.5 * np.sum(p_row[a:b] * np.maximum(0.0, plan[i][a:b] - adj[i][a:b]))
                            + 1.5 * np.sum(p_row[a:b] * np.maximum(0.0, adj[i][a:b] - plan[i][a:b])))
            out[stage].append({"date": row["date"], "voi": voi[stage], "penalty": penalty,
                               "net": voi[stage] - penalty})
    summary = {}
    for stage, items in out.items():
        voi = np.array([it["voi"] for it in items])
        pen = np.array([it["penalty"] for it in items])
        trigger = voi > pen
        summary[stage] = {
            "voi_mean": float(voi.mean()), "penalty_mean": float(pen.mean()),
            "trigger_rate": float(trigger.mean()),
            "net_mean": float((voi - pen).mean()),
            "voi_total": float(voi.sum()), "penalty_total": float(pen.sum()),
        }
    return {"stages": {str(k): v for k, v in out.items()}, "summary": summary}


def rolling_update_example(panel, day_iso: str = "") -> dict:
    """Plan / updated / realised trajectories with trigger annotations for one day."""
    rows = read_days_csv("q3")
    dates = [r["date"] for r in rows]
    if not day_iso:
        voi = np.array([float(r["S0"]) - float(r["S3"]) for r in rows])
        day_iso = dates[int(np.argmax(voi))]
    i = dates.index(day_iso)
    plan, adj, g = arr("q3_q_plan.npy")[i], arr("q3_q_adjust.npy")[i], arr("q3_emergency.npy")[i]
    charge, disch = arr("q3_charge.npy")[i], arr("q3_discharge.npy")[i]
    soc = arr("q3_soc.npy")[i]
    need = panel.net_load()[FIRST_DAY + i] * (1 / 6) + charge - disch
    s0, s1, s2, s3 = (float(rows[i][k]) for k in ("S0", "S1", "S2", "S3"))
    return {
        "date": day_iso,
        "plan": plan.tolist(), "adjusted": adj.tolist(), "realised": (adj + g).tolist(),
        "demand": need.tolist(), "soc": soc.tolist(),
        "triggers": [{"slot": 36, "label": "6:00", "voi": s0 - s1},
                     {"slot": 72, "label": "12:00", "voi": s1 - s2},
                     {"slot": 108, "label": "18:00", "voi": s2 - s3}],
        "day_subset_costs": {"S0": s0, "S1": s1, "S2": s2, "S3": s3},
    }


# ---------------------------------------------------------------------------
# Fig.7 real-time control vs committed baseline (Q4-2 vs Q4-3)
# ---------------------------------------------------------------------------

def realtime_control(panel, day_iso: str = "") -> dict:
    rows = read_days_csv("q4_3")
    dates = [r["date"] for r in rows]
    if not day_iso:
        cost = np.array([float(r["cost_cny"]) for r in rows])
        day_iso = dates[int(np.argmax(cost))]
    i = dates.index(day_iso)
    base_c, base_d = arr("q4_2_charge.npy")[i], arr("q4_2_discharge.npy")[i]
    mpc_c, mpc_d = arr("q4_3_charge.npy")[i], arr("q4_3_discharge.npy")[i]
    soc_base, soc_mpc = arr("q4_2_soc.npy")[i], arr("q4_3_soc.npy")[i]

    def total_variation(x):
        return float(np.abs(np.diff(x)).sum())

    return {
        "date": day_iso,
        "price": panel.price[FIRST_DAY + i].tolist(),
        "load": panel.load[FIRST_DAY + i].tolist(),
        "pv": panel.pv_filled()[FIRST_DAY + i].tolist(),
        "baseline": {"charge": base_c.tolist(), "discharge": base_d.tolist(),
                     "net": (base_c - base_d).tolist(), "soc": soc_base.tolist(),
                     "tv": total_variation(base_c - base_d)},
        "mpc": {"charge": mpc_c.tolist(), "discharge": mpc_d.tolist(),
                "net": (mpc_c - mpc_d).tolist(), "soc": soc_mpc.tolist(),
                "tv": total_variation(mpc_c - mpc_d)},
        "soc_limits": [float(panel.config["storage"]["soc_min_kwh"]),
                       float(panel.config["storage"]["soc_max_kwh"])],
    }


# ---------------------------------------------------------------------------
# Fig.8 cost - risk - degradation Pareto
# ---------------------------------------------------------------------------

def pareto_cost_risk_degradation(panel, betas=(0.0, 0.25, 0.5, 0.75, 0.9),
                                 step: int = 5) -> dict:
    """β sweep under the 附件4 tariff with rainflow-based SOH (real degradation model)."""
    from code.optimize.q2_stochastic import solve_q2

    st = Storage.from_config(panel.config)
    days = list(range(FIRST_DAY, 365, step))
    params = DegradationParams(capacity_kwh=st.capacity, window_days=len(days))
    rows = []
    for beta in betas:
        run = solve_q2(panel, panel.price, tag=f"pfig_beta{beta}", days=days,
                       cvar_beta=beta, cvar_alpha=0.95)
        daily = np.array([r["cost_cny"] for r in run["records"]])
        charge = run["arrays"]["charge"]
        disch = run["arrays"]["discharge"]
        cycles = float((charge.sum() + disch.sum()) / (2.0 * st.capacity))
        deg = evaluate_panel(run["arrays"]["soc"], params)
        tail = int(max(1, np.ceil(0.05 * daily.size)))
        rows.append({
            "beta": beta, "strategy": "CVaR" if beta > 0 else "风险中性",
            "cost_cny": float(daily.sum()),
            "cvar95_cny": float(np.sort(daily)[-tail:].mean()),
            "degradation_cycles": cycles,
            "annual_fade_pct": deg.annual_fade_pct,
            "soh_end_pct": 100.0 * deg.soh_end,
            "degradation_cost_cny": deg.degradation_cost_cny,
            "rainflow_cycles": deg.rainflow_cycles,
            "mean_dod": deg.mean_dod,
            "emergency_kwh": float(sum(r["emergency_energy_kwh"] for r in run["records"])),
            "days": len(days),
        })
    return {"rows": rows, "days": len(days), "step": step}


def degradation_priced(panel, prices=(0.0, 0.05, 0.1, 0.2, 0.4), step: int = 5) -> dict:
    """Degradation-aware strategies: price battery wear, then measure the real SOH.

    `prices` are CNY per kWh cycled, added to the LP objective; the resulting schedules
    are evaluated with the rainflow/ Wöhler model, so this is the genuine
    cost-vs-degradation frontier (not a proxy).
    """
    from code.optimize.q2_stochastic import solve_q2

    st = Storage.from_config(panel.config)
    days = list(range(FIRST_DAY, 365, step))
    params = DegradationParams(capacity_kwh=st.capacity, window_days=len(days))
    rows = []
    for rho in prices:
        run = solve_q2(panel, panel.price, tag=f"pfig_deg{rho}", days=days,
                       throughput_penalty=rho)
        daily = np.array([r["cost_cny"] for r in run["records"]])
        deg = evaluate_panel(run["arrays"]["soc"], params)
        tail = int(max(1, np.ceil(0.05 * daily.size)))
        rows.append({
            "rho_cny_per_kwh": rho,
            "electricity_cost_cny": float(daily.sum()),
            "cvar95_cny": float(np.sort(daily)[-tail:].mean()),
            "annual_fade_pct": deg.annual_fade_pct,
            "degradation_cost_cny": deg.degradation_cost_cny,
            "total_cost_cny": float(daily.sum()) + deg.degradation_cost_cny,
            "rainflow_cycles": deg.rainflow_cycles,
            "throughput_kwh": float(run["arrays"]["charge"].sum() + run["arrays"]["discharge"].sum()),
            "emergency_kwh": float(sum(r["emergency_energy_kwh"] for r in run["records"])),
            "days": len(days),
        })
    return {"rows": rows, "days": len(days), "step": step,
            "note": ("ρ 为单位吞吐的退化价格；年衰减由雨流计数 + Wöhler 寿命曲线给出，"
                     "退化成本 = 年衰减比例 × 电池组资本成本（800 元/kWh × 12000 kWh）")}


def degradation_sensitivity(panel, step: int = 5) -> dict:
    """SOH sensitivity to N100 and the Wöhler exponent on a representative trajectory."""
    st = Storage.from_config(panel.config)
    days = list(range(FIRST_DAY, 365, step))
    soc = np.load(ARRAYS / "q4_3_soc.npy") if (ARRAYS / "q4_3_soc.npy").exists() else None
    if soc is None:
        soc = np.load(ARRAYS / "q3_soc.npy")
    if soc.shape[0] != len(days):
        idx = np.linspace(0, soc.shape[0] - 1, len(days)).round().astype(int)
        soc = soc[idx]
    params = DegradationParams(capacity_kwh=st.capacity, window_days=len(days))
    base = evaluate_panel(soc, params)
    return {"base": {"annual_fade_pct": base.annual_fade_pct,
                     "soh_end_pct": 100.0 * base.soh_end,
                     "degradation_cost_cny": base.degradation_cost_cny,
                     "rainflow_cycles": base.rainflow_cycles,
                     "mean_dod": base.mean_dod, "max_dod": base.max_dod},
            "grid": deg_sens(params, soc), "days": len(days),
            "source": "q4_3 rolling-MPC SoC trajectory (sub-sampled to the Pareto sample)"}


# ---------------------------------------------------------------------------
# Fig.9 ablation matrix M0-M4
# ---------------------------------------------------------------------------

def ablation_matrix(panel, cvar_beta: float = 0.5) -> dict:
    """Module ladder under one tariff: deterministic → conformal → SAA → rolling → CVaR."""
    from code.optimize.q2_stochastic import solve_q2

    st = Storage.from_config(panel.config)
    tariff = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    emergency = float(panel.config["tariff"]["emergency_multiplier"])
    net = panel.net_load()
    bench = load_json(RESULTS / "compute_benchmarks.json") if (RESULTS / "compute_benchmarks.json").exists() else {}
    stage_time = {s["stage"]: s["seconds"] for s in bench.get("pipeline", [])}

    def metrics(daily_cost, charge, disch, g, curtail):
        daily = np.asarray(daily_cost, dtype=float)
        tail = int(max(1, np.ceil(0.05 * daily.size)))
        return {
            "cost_wan": float(daily.sum()) / 1e4,
            "cvar95_cny": float(np.sort(daily)[-tail:].mean()),
            "emergency_wan_kwh": float(np.sum(g)) / 1e4,
            "degradation_cycles": float((np.sum(charge) + np.sum(disch)) / (2.0 * st.capacity)),
            "curtail_wan_kwh": float(np.sum(curtail)) / 1e4,
        }

    deg_params = DegradationParams(capacity_kwh=st.capacity,
                                   window_days=365 - FIRST_DAY)

    import time
    # M0 deterministic mean plan; M1 adds the conformal margin *inside* the LP, so the
    # storage schedule adapts as well (otherwise the throughput column would be identical).
    t0 = time.perf_counter()
    m0 = {"cost": [], "charge": [], "disch": [], "g": [], "curtail": []}
    m1 = {k: [] for k in m0}
    for d in range(FIRST_DAY, 365):
        win = net[max(0, d - LOOKBACK):d]
        base = win.mean(axis=0)
        det = solve_stage_lp(base, tariff[d], st)
        sample = base[None, :] * st.dt + det["c"] - det["d"] + (win - base) * st.dt
        need = net[d] * st.dt + det["c"] - det["d"]
        lvl = conformal_level(0.80, sample.shape[0])
        margin = np.quantile(sample, lvl, axis=0) - (base * st.dt + det["c"] - det["d"])
        # M1 plans against the *conformal upper* requirement, so the storage schedule is
        # re-optimised on base + margin (kW): the margin raises the requirement.
        det_margin = solve_stage_lp(base + margin / st.dt, tariff[d], st)
        for bucket, schedule in ((m0, det), (m1, det_margin)):
            q = np.maximum(0.0, schedule["q"])
            need_sched = net[d] * st.dt + schedule["c"] - schedule["d"]
            g = np.maximum(0.0, need_sched - q)
            bucket["cost"].append(float(np.sum(tariff[d] * q) + emergency * np.sum(tariff[d] * g)))
            bucket["charge"].append(schedule["c"])
            bucket["disch"].append(schedule["d"])
            bucket.setdefault("soc", []).append(schedule["soc"])
            bucket["g"].append(g)
            bucket["curtail"].append(np.maximum(0.0, schedule["d"] - schedule["c"]
                                                - net[d] * st.dt))
    cheap_seconds = time.perf_counter() - t0

    # M2 = SAA two-stage (stored result), M3 = rolling MPC (stored result)
    q2_rows = read_days_csv("q2")
    q3_rows = read_days_csv("q3")
    m2 = metrics([float(r["cost_cny"]) for r in q2_rows], arr("q2_charge.npy"),
                 arr("q2_discharge.npy"), arr("q2_emergency.npy"), arr("q2_curtail.npy"))
    m3 = metrics([float(r["cost_cny"]) for r in q3_rows], arr("q3_charge.npy"),
                 arr("q3_discharge.npy"), arr("q3_emergency.npy"), arr("q3_curtail.npy"))

    # M4 = CVaR risk layer (same tariff as M0-M3)
    t4 = time.perf_counter()
    run4 = solve_q2(panel, tariff, tag="pfig_ablation_cvar", days=range(FIRST_DAY, 365),
                    cvar_beta=cvar_beta, cvar_alpha=0.90)
    m4_seconds = time.perf_counter() - t4
    m4 = metrics([r["cost_cny"] for r in run4["records"]], run4["arrays"]["charge"],
                 run4["arrays"]["discharge"], run4["arrays"]["emergency"],
                 run4["arrays"]["curtail"])

    rows = {"M0": metrics(m0["cost"], np.array(m0["charge"]), np.array(m0["disch"]),
                          np.array(m0["g"]), np.array(m0["curtail"])),
            "M1": metrics(m1["cost"], np.array(m1["charge"]), np.array(m1["disch"]),
                          np.array(m1["g"]), np.array(m1["curtail"])),
            "M2": m2, "M3": m3, "M4": m4}
    soc_matrices = {"M0": np.array(m0["soc"]), "M1": np.array(m1["soc"]),
                    "M2": arr("q2_soc.npy"), "M3": arr("q3_soc.npy"),
                    "M4": run4["arrays"]["soc"]}
    for key, soc in soc_matrices.items():
        deg = evaluate_panel(soc, deg_params)
        rows[key]["annual_fade_pct"] = deg.annual_fade_pct
        rows[key]["soh_end_pct"] = 100.0 * deg.soh_end
        rows[key]["degradation_cost_cny"] = deg.degradation_cost_cny
        rows[key]["rainflow_cycles"] = deg.rainflow_cycles
    times = {"M0": cheap_seconds / 2, "M1": cheap_seconds / 2,
             "M2": stage_time.get("Q2 两阶段随机规划", float("nan")),
             "M3": stage_time.get("Q3 滚动 MPC + S0–S5", float("nan")),
             "M4": m4_seconds}
    for key in rows:
        rows[key]["solve_seconds"] = float(times[key])
    return {"rows": rows,
            "labels": {"M0": "M0 确定性均值计划", "M1": "M1 +共形裕度(p*=0.80)",
                       "M2": "M2 +场景SAA(两阶段)", "M3": "M3 +多时次滚动MPC",
                       "M4": f"M4 = M2+CVaR(β={cvar_beta}) 风险分支"},
            "ladder_note": ("M0→M1→M2→M3 为主链（物理建模→共形裕度→场景随机规划→滚动更新）；"
                            "M4 由 M2 分支而来，用于单独度量 CVaR 风险层的贡献"
                            "（当前滚动 MPC 实现不含 CVaR 旋钮，故不做 M3+CVaR）"),
            "metric_labels": {"cost_wan": "费用\n(万元)", "cvar95_cny": "CVaR95\n(元/日)",
                              "emergency_wan_kwh": "紧急购电\n(万kWh)",
                              "annual_fade_pct": "SOH 年衰减\n(%)",
                              "curtail_wan_kwh": "弃光\n(万kWh)",
                              "solve_seconds": "求解耗时\n(s)"}}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def build(force: bool = False, quick: bool = True, verbose: bool = True) -> dict:
    """Prepare every dataset the paper figures need (cached in results/)."""
    panel = load_panel()
    data = _cache()
    steps = {
        "shadow_price": lambda: shadow_price_coupling(panel),
        "fan_chart": lambda: prediction_fan(panel),
        "reliability": lambda: reliability_sharpness(panel),
        "voi": lambda: voi_boundary(panel),
        "rolling": lambda: rolling_update_example(panel),
        "realtime": lambda: realtime_control(panel),
        "pareto": lambda: pareto_cost_risk_degradation(panel),
        "ablation": lambda: ablation_matrix(panel),
        "degradation_priced": lambda: degradation_priced(panel),
        "degradation_sensitivity": lambda: degradation_sensitivity(panel),
    }
    for key, fn in steps.items():
        if force or key not in data:
            data = _update(**{key: fn()})
            if verbose:
                print(f"[figure-data] prepared {key}")
        elif verbose:
            print(f"[figure-data] cached {key}")
    return data


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="prepare datasets for 图片.md figures")
    parser.add_argument("--force", action="store_true", help="recompute every dataset")
    args = parser.parse_args()
    build(force=args.force)
    print(f"[figure-data] cache -> {CACHE}")


if __name__ == "__main__":
    main()
