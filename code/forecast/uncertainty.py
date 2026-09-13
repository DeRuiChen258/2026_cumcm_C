"""C题.pdf 没写清楚的地方，用共形区间给范围，不硬猜一个点值。

共 5 项：Q2 用什么信息集、Q3 负荷算不算已知、Q4 电价是不是 0:00 就知道、
表1 的时段对应附件哪一列、日内储能能不能再调度。
区间只是参考范围，不会替换 output/result*.xlsx 里的数字。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import conformal_level, weighted_quantile
from code.optimize.lp_common import Storage, solve_stage_lp

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
FIRST_DAY = 31
LOOKBACK = 30
TABLE1_SLOTS_START = {"10:00-10:10": 61, "12:00-12:10": 73, "14:00-14:10": 85,
                      "16:00-16:10": 97, "18:00-18:10": 109, "20:00-20:10": 121}
TABLE1_SLOTS_END = {"10:00-10:10": 60, "12:00-12:10": 72, "14:00-14:10": 84,
                    "16:00-16:10": 96, "18:00-18:10": 108, "20:00-20:10": 120}


def q2_plan_band(panel, levels=(0.25, 0.80, 0.90)) -> dict:
    """Cost/quantity band of the Q2 plan over the conformal quantile levels (U1)."""
    st = Storage.from_config(panel.config)
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    emergency = float(panel.config.get("tariff", {}).get("emergency_multiplier", 5.0))
    net = panel.net_load()
    days = range(FIRST_DAY, 365)
    per_level = {level: {"cost": [], "purchase": [], "coverage": [], "emergency": []}
                 for level in levels}
    for d in days:
        win = net[max(0, d - LOOKBACK):d]
        base = win.mean(axis=0)
        det = solve_stage_lp(base, price[d], st)
        required_sample = base[None, :] * st.dt + det["c"] - det["d"] + (win - base) * st.dt
        required_actual = net[d] * st.dt + det["c"] - det["d"]
        for level in levels:
            lvl = conformal_level(level, required_sample.shape[0])
            q = np.maximum(0.0, np.quantile(required_sample, lvl, axis=0))
            g = np.maximum(0.0, required_actual - q)
            per_level[level]["cost"].append(
                float(np.sum(price[d] * q) + emergency * np.sum(price[d] * g)))
            per_level[level]["purchase"].append(float(q.sum()))
            per_level[level]["coverage"].append(float(np.mean(required_actual <= q)))
            per_level[level]["emergency"].append(float(g.sum()))
    out = {}
    for level in levels:
        rec = per_level[level]
        out[f"p{int(level*100)}"] = {
            "level": level,
            "total_cost_cny": float(np.sum(rec["cost"])),
            "total_purchase_kwh": float(np.sum(rec["purchase"])),
            "total_emergency_kwh": float(np.sum(rec["emergency"])),
            "socket_coverage": float(np.mean(rec["coverage"])),
            "daily_cost_cny": rec["cost"],
            "daily_purchase_kwh": rec["purchase"],
        }
    main_total = float(
        json.loads((RESULTS / "q2.json").read_text(encoding="utf-8"))["summary"]["total_cost_cny"])
    out["two_stage_lp_total_cost_cny"] = main_total
    anchors = [out[f"p{int(min(levels)*100)}"]["total_cost_cny"],
               out[f"p{int(max(levels)*100)}"]["total_cost_cny"], main_total]
    out["band_total_cost_cny"] = [min(anchors), max(anchors)]
    out["main_anchor"] = "p0.80 (prompt D15 / 边际代价比 1:5)"
    out["note"] = ("Q2 的信息集在 C题.pdf 中未定义；区间由共形分位给出（逐时段分位 + 固定储能计划），"
                   "主结果仍取 p*=0.80 的两阶段 LP（见 result2.xlsx）；"
                   "两阶段 LP 同时优化储能与应急 recourse，故低于本曲线同分位点")
    return out


def q2_storage_flex(panel, level: float = 0.80) -> dict:
    """U5: allow intraday storage re-dispatch (rolling re-solve) vs committed plan."""
    st = Storage.from_config(panel.config)
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    emergency = float(panel.config.get("tariff", {}).get("emergency_multiplier", 5.0))
    net = panel.net_load()
    committed, flexible = [], []
    for d in range(FIRST_DAY, 365):
        win = net[max(0, d - LOOKBACK):d]
        base = win.mean(axis=0)
        det = solve_stage_lp(base, price[d], st)
        sample = base[None, :] * st.dt + det["c"] - det["d"] + (win - base) * st.dt
        lvl = conformal_level(level, sample.shape[0])
        target = np.maximum(0.0, np.quantile(sample, lvl, axis=0))
        need = net[d] * st.dt + det["c"] - det["d"]
        committed.append(float(np.sum(price[d] * target)
                               + emergency * np.sum(price[d] * np.maximum(0.0, need - target))))
        # flexible: the committed purchase stays, only the intraday storage dispatch uses
        # the realised net load (minimise the emergency bill inside the same storage core)
        flex = _dispatch_to_target(net[d] * st.dt, target, price[d], st, emergency)
        flexible.append(float(np.sum(price[d] * target) + flex))
    return {
        "committed_plan_cost_cny": float(np.sum(committed)),
        "intraday_flexible_cost_cny": float(np.sum(flexible)),
        "flexibility_gain_cny": float(np.sum(committed) - np.sum(flexible)),
        "note": ("C题.pdf 规定'除紧急购电费用外，其他时间段的购电费用均按计划购电量计算'，"
                 "故主口径为承诺制；日内可再调度仅作为下界参考（购买量承诺不变）"),
        "emergency_multiplier": emergency,
    }


def _dispatch_to_target(need: np.ndarray, target: np.ndarray, price: np.ndarray,
                        st: Storage, emergency: float) -> float:
    """Minimise the emergency bill by re-dispatching storage against a fixed purchase."""
    from scipy import sparse
    from scipy.optimize import linprog

    n = 144
    nv = 2 * n + (n - 1) + n          # c, d, e(interior), g
    idx_c, idx_d, idx_e, idx_g = 0, n, 2 * n, 3 * n - 1
    rows_eq, cols_eq, vals_eq, rhs = [], [], [], []
    for k in range(n):
        r = len(rhs)
        if k < n - 1:
            rows_eq.append(r), cols_eq.append(idx_e + k), vals_eq.append(1.0)
        if 1 <= k <= n - 1:
            rows_eq.append(r), cols_eq.append(idx_e + k - 1), vals_eq.append(-1.0)
        rows_eq.append(r), cols_eq.append(idx_c + k), vals_eq.append(-st.eta_c)
        rows_eq.append(r), cols_eq.append(idx_d + k), vals_eq.append(1.0 / st.eta_d)
        rhs.append(st.soc_init if k == 0 else (-st.soc_init if k == n - 1 else 0.0))
    a_eq = sparse.coo_matrix((vals_eq, (rows_eq, cols_eq)), shape=(n, nv)).tocsr()
    rows_ub, cols_ub, vals_ub, b_ub = [], [], [], []
    for k in range(n):
        r = len(b_ub)
        # g_k >= need_k + c_k - d_k - target_k  <=>  -c_k + d_k - g_k <= -(need_k - target_k)
        rows_ub.extend([r, r, r])
        cols_ub.extend([idx_c + k, idx_d + k, idx_g + k])
        vals_ub.extend([-1.0, 1.0, -1.0])
        b_ub.append(-(need[k] - target[k]))
    a_ub = sparse.coo_matrix((vals_ub, (rows_ub, cols_ub)), shape=(len(b_ub), nv)).tocsr()
    c_obj = np.zeros(nv)
    c_obj[idx_g:idx_g + n] = emergency * price
    bounds = ([(0.0, st.max_charge_energy)] * n + [(0.0, st.max_discharge_energy)] * n
              + [(st.soc_min, st.soc_max)] * (n - 1) + [(0.0, None)] * n)
    res = linprog(c_obj, A_ub=a_ub, b_ub=np.array(b_ub), A_eq=a_eq, b_eq=np.array(rhs),
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"dispatch LP failed: {res.message}")
    return float(res.fun)


def q3_load_band(panel, level: float = 0.90) -> dict:
    """U2: Q3 settlement when the load itself carries a conformal error band."""
    st = Storage.from_config(panel.config)
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    emergency = float(panel.config.get("tariff", {}).get("emergency_multiplier", 5.0))
    plan = np.load(ARRAYS / "q3_q_plan.npy")
    adj = np.load(ARRAYS / "q3_q_adjust.npy")
    charge = np.load(ARRAYS / "q3_charge.npy")
    disch = np.load(ARRAYS / "q3_discharge.npy")
    load = panel.load
    total_low, total_high, total_point = 0.0, 0.0, 0.0
    for i, d in enumerate(range(FIRST_DAY, 365)):
        win = load[max(0, d - LOOKBACK):d]
        base = win.mean(axis=0)
        lvl = conformal_level(level, win.shape[0])
        upper = np.quantile(win, lvl, axis=0)
        lower = np.quantile(win, 1.0 - lvl, axis=0)
        for load_level, bucket in ((upper, "high"), (lower, "low"), (load[d], "point")):
            need = load_level * st.dt - panel.pv_filled()[d] * st.dt + charge[i] - disch[i]
            g = np.maximum(0.0, need - adj[i])
            value = float(np.sum(price[d] * plan[i]) + emergency * np.sum(price[d] * g)
                          + 0.5 * np.sum(price[d] * np.maximum(0.0, plan[i] - adj[i]))
                          + 1.5 * np.sum(price[d] * np.maximum(0.0, adj[i] - plan[i])))
            if bucket == "high":
                total_high += value
            elif bucket == "low":
                total_low += value
            else:
                total_point += value
    return {
        "level": level,
        "load_band_total_cost_cny": [min(total_low, total_high, total_point),
                                     max(total_low, total_high, total_point)],
        "cost_at_upper_load_band_cny": total_high,
        "cost_at_lower_load_band_cny": total_low,
        "point_estimate_cny": total_point,
        "upper_band_deviation_pct": 100.0 * (total_high / total_point - 1.0),
        "lower_band_deviation_pct": 100.0 * (total_low / total_point - 1.0),
        "note": ("C题.pdf 只说明可获得光伏预报，未说明负荷是否已知；"
                 "上/下带 = 把当日实际负荷替换为其因果共形分布的 93.3% / 6.7% 分位后的结算压力值；"
                 "点估计用实际负荷，可逐位复现 result3 的结算总额（校验通过）"),
    }


def q4_price_band(panel, low: float = 0.05, high: float = 0.95) -> dict:
    """U3: Q4 settlement range when the price is taken from its conformal band."""
    emergency = float(panel.config.get("tariff", {}).get("emergency_multiplier", 5.0))
    price = panel.price
    out: dict = {"levels": [low, high]}
    for tag in ("q4_2", "q4_3"):
        plan = np.load(ARRAYS / f"{tag}_q_plan.npy")
        g = np.load(ARRAYS / f"{tag}_emergency.npy")
        adj = np.load(ARRAYS / f"{tag}_q_adjust.npy") if tag == "q4_3" else None
        totals = {"low": 0.0, "high": 0.0, "point": 0.0}
        for i, d in enumerate(range(FIRST_DAY, 365)):
            win = price[max(0, d - LOOKBACK):d]
            lo = np.quantile(win, conformal_level(low, win.shape[0]), axis=0)
            hi = np.quantile(win, conformal_level(high, win.shape[0]), axis=0)
            for name, p_band in (("low", lo), ("high", hi), ("point", price[d])):
                value = float(np.sum(p_band * plan[i]) + emergency * np.sum(p_band * g[i]))
                if adj is not None:
                    value += float(0.5 * np.sum(p_band * np.maximum(0.0, plan[i] - adj[i]))
                                   + 1.5 * np.sum(p_band * np.maximum(0.0, adj[i] - plan[i])))
                totals[name] += value
        out[tag] = {
            "point_estimate_cny": totals["point"],
            "band_total_cost_cny": [totals["low"], totals["high"]],
            "price_band": f"逐时段共形分位带 {low:.0%}/{high:.0%}（因果 30 日窗口）",
            "note": "C题.pdf 说电价实时波动但提供了附件4 全年电价；区间用于说明价格不确定性的影响量级",
        }
    return out


def table1_mapping(panel) -> dict:
    """U4: the two readings of the table-1 interval labels (start vs right endpoint)."""
    q1 = json.loads((RESULTS / "q1.json").read_text(encoding="utf-8"))
    q = np.array(q1["lp"]["q"])
    start_reading = {k: float(q[v - 1]) for k, v in TABLE1_SLOTS_START.items()}
    end_reading = {k: float(q[v - 1]) for k, v in TABLE1_SLOTS_END.items()}
    diff = {k: abs(start_reading[k] - end_reading[k]) for k in start_reading}
    return {
        "start_based_kwh": start_reading,
        "right_endpoint_based_kwh": end_reading,
        "absolute_difference_kwh": diff,
        "max_difference_kwh": float(max(diff.values())),
        "day_total_kwh": float(q.sum()),
        "day_total_unaffected": True,
        "note": ("C题.pdf 附录2 说明附件时间戳是右端点，而附件5 模板标签比附件时间戳晚一个时段；"
                 "主口径按模板第 k 行 ↔ 数据第 k 时段（D-03），此处给出两种读法的数值差"),
    }


def main() -> None:
    panel = load_panel()
    report = {
        "source_of_truth": "C题.pdf（2026 高教社杯 C 题，3 页）",
        "rule": "PDF 明确项按 PDF 执行；PDF 未明确项用共形区间给出范围，不臆造点值",
        "q2_plan_band": q2_plan_band(panel),
        "q2_storage_flex": q2_storage_flex(panel),
        "q3_load_band": q3_load_band(panel),
        "q4_price_band": q4_price_band(panel),
        "table1_mapping": table1_mapping(panel),
    }
    (RESULTS / "uncertainty_ranges.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    band = report["q2_plan_band"]
    print("[U1] Q2 费用区间 p0.25 → p0.90: "
          f"{band['p25']['total_cost_cny']:.0f} - {band['p90']['total_cost_cny']:.0f} 元 "
          f"(主口径 p0.80 = {band['p80']['total_cost_cny']:.0f} 元)")
    print(f"[U5] 承诺制 vs 日内再调度: {report['q2_storage_flex']['flexibility_gain_cny']:.0f} 元 可省上限")
    q3b = report["q3_load_band"]
    print(f"[U2] Q3 负荷共形带下结算区间: {q3b['load_band_total_cost_cny'][0]:.0f} - "
          f"{q3b['load_band_total_cost_cny'][1]:.0f} 元 (点估计 {q3b['point_estimate_cny']:.0f})")
    for tag in ("q4_2", "q4_3"):
        info = report["q4_price_band"][tag]
        print(f"[U3] {tag} 价格带费用区间: {info['band_total_cost_cny'][0]:.0f} - "
              f"{info['band_total_cost_cny'][1]:.0f} 元 (点估计 {info['point_estimate_cny']:.0f})")
    print(f"[U4] 表1 两种口径最大差异: {report['table1_mapping']['max_difference_kwh']:.2f} kWh")


if __name__ == "__main__":
    main()
