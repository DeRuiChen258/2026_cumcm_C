"""为 figures/CAPTIONS_3.md 的 25 张图准备数据证据包。

只做只读读取与派生计算：所有数字都来自 results/、clean/ 与求解器模块的
**纯函数复用**（不重新求解任何优化问题、不写回任何交付物）。派生的
figures/science_data.json 里逐项记录来源，供图注与审查追溯。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import conformal_level
from code.optimize.execution import ExecutionConfig, run_execution_layer
from code.optimize.lp_common import Storage

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
OUT_DIR = ROOT / "figures"
FIRST_DAY = 31


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hourly(values_energy: np.ndarray) -> np.ndarray:
    """144 个 10 分钟时段 → 24 小时合计（输入已是 kWh 能量口径）。"""
    return np.asarray(values_energy, dtype=float).reshape(24, 6).sum(axis=1)


def build(sample_day: int = FIRST_DAY) -> dict:
    panel = load_panel()
    st = Storage.from_config(panel.config)
    q1 = _json(RESULTS / "q1.json")
    q2 = _json(RESULTS / "q2.json")
    q3 = _json(RESULTS / "q3.json")
    q4 = _json(RESULTS / "q4.json")
    p2 = _json(RESULTS / "p2_conformal.json")
    dt = st.dt

    price = panel.day1[:, 0]
    load = panel.day1[:, 1]
    pv = panel.day1[:, 2]
    c = np.array(q1["lp"]["c"], dtype=float)
    d = np.array(q1["lp"]["d"], dtype=float)
    q = np.array(q1["lp"]["q"], dtype=float)
    soc = np.array(q1["lp"]["soc"], dtype=float)
    mu = np.array(q1["lp"]["mu"], dtype=float)
    lam = np.array(q1["lp"]["lambda"], dtype=float)

    # ---- 派生：Q1 日循环恒等式与 KKT 触发的数值核验 ----
    identity_lhs = float(q.sum())
    curtail_total = float(np.sum(q1["lp"]["curtail"]))
    identity_rhs = (float((load - pv).sum() * dt) + curtail_total
                    + (1.0 - st.eta_c * st.eta_d) * float(c.sum()))

    out: dict = {
        "meta": {
            "source": "results/q1..q4.json, p2_conformal.json, arrays/*.npy, clean/",
            "sample_day_index": int(sample_day),
            "sample_day_date": panel.dates[sample_day],
            "dt_hours": float(dt),
            "note": "no optimization is re-run here; solver functions are reused read-only",
        },
        "storage": {
            "capacity_kwh": float(st.capacity), "soc_min_kwh": float(st.soc_min),
            "soc_max_kwh": float(st.soc_max), "soc_init_kwh": float(st.soc_init),
            "power_kw": float(st.power), "eta_c": float(st.eta_c), "eta_d": float(st.eta_d),
        },
        "q1": {
            "price_10min": price.tolist(),
            "load_kw_10min": load.tolist(),
            "pv_kw_10min": pv.tolist(),
            "price_hourly": _hourly(price * dt).tolist(),
            "load_hourly_kwh": _hourly(load * dt).tolist(),
            "pv_hourly_kwh": _hourly(pv * dt).tolist(),
            "charge_hourly_kwh": _hourly(c).tolist(),
            "discharge_hourly_kwh": _hourly(d).tolist(),
            "purchase_hourly_kwh": _hourly(q).tolist(),
            # 0:00 … 23:00 共 24 个整点（24:00 的终值单列，避免与 0:00 重复）
            "soc_hourly": soc[:145:6][:24].tolist(),
            "soc_final_kwh": float(soc[-1]),
            "soc_10min": soc.tolist(),
            "mu_hourly": _hourly(mu * dt).tolist(),
            "lambda_hourly": _hourly(lam * dt).tolist(),
            "charge_threshold_10min": q1["kkt_deadband"]["charge_threshold"],
            "discharge_threshold_10min": q1["kkt_deadband"]["discharge_threshold"],
            "cost_cny": float(q1["cost_cny"]),
            "baselines": q1["baselines"],
            "energy": q1["energy"],
            "night_statistics": q1["night_statistics"],
            "energy_identity": q1["energy_identity"],
            "energy_identity_recomputed": {
                "lhs_kwh": identity_lhs, "rhs_kwh": identity_rhs,
                "residual_kwh": identity_lhs - identity_rhs,
            },
            "kkt_summary": {k: v for k, v in q1["kkt_deadband"].items()
                            if not isinstance(v, list)},
            "simultaneous_charge_discharge_kw": q1["simultaneous_charge_discharge_kw"],
            "soc_terminal_kwh": q1["soc_terminal_kwh"],
            "dp": q1["dp"],
        },
        "q2": {
            "summary": q2["summary"],
            # 只保留均值尺度与样例行，避免把 334x144 数组灌进 JSON（图直接读 .npy）
            "theta_first_rows": np.load(ARRAYS / "q2_theta.npy")[:5].tolist(),
            "theta_mean_alpha": float(np.load(ARRAYS / "q2_theta.npy")[:, 0].mean()),
            "theta_mean_margin_kw": float(np.load(ARRAYS / "q2_theta.npy")[:, 1].mean()),
            "pv_lower_sample_rows": np.load(ARRAYS / "q2_pv_lower.npy")[:3].tolist(),
            "pv_lower_mean_kwh_per_day": float(
                np.load(ARRAYS / "q2_pv_lower.npy").sum(axis=1).mean() * dt),
            "emergency_kwh_daily": [r["emergency_energy_kwh"] for r in q2["records"]],
            "cost_daily": [r["cost_cny"] for r in q2["records"]],
            "dates": [r["date"] for r in q2["records"]],
            "maxent_trace": q2.get("maxent_trace", []),
        },
        "q2_conformal": {
            "anchors": p2["coverage"]["anchors"],
            "newsvendor_level": p2["newsvendor_level"],
            "forecast_skill": p2["forecast_skill"],
        },
        "q3": {
            "summary": q3["summary"],
            "nav_trace": q3["nav_trace"],
            "uri_trace": q3["uri_trace"],
            "deadband_trace": q3["deadband_trace"],
            # 轨迹图直接读 arrays/*.npy，不在此重复存储
        },
        "q4": {
            "summary": q4,
            "cvar_pareto": q4["cvar_pareto"],
            "student_t": q4.get("student_t_copula", {}),
            "execution_ablation": q4.get("execution_ablation", {}),
            "spike_conformity": q4["spike_conformity"],
            "copula_value": q4["copula_value"],
            "soft_penalty": q4["soft_penalty_evidence"],
            "emergency_kwh_daily": np.load(ARRAYS / "q4_2_emergency.npy").sum(axis=1).tolist(),
        },
    }
    return out


def _lead_time_curve() -> list[dict]:
    """从 q3_lead_time.csv 读取最后一个评估窗口的逐时次曲线。"""
    path = RESULTS / "q3_lead_time.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "vintage_hour": int(float(row["vintage_hour"])),
                "slot": int(row["slot"]),
                "hour_end": int(row["hour_end"]),
                "lead_hours": float(row["lead_hours"]),
                "sigma_kw": float(row["sigma_kw"]),
                "quantile_kw": float(row["quantile_kw"]),
                "samples": int(row["samples"]),
            })
    return rows


def execution_traces(sample_day: int = FIRST_DAY) -> dict:
    """复用 Q4 执行层纯函数，得到 CBF/DETC 的逐时段轨迹（只读、确定性）。"""
    panel = load_panel()
    st = Storage.from_config(panel.config)
    cfg = ExecutionConfig(soc_min=st.soc_min, soc_max=st.soc_max, power=st.power,
                          dt=st.dt, eta_c=st.eta_c, eta_d=st.eta_d)
    day = q4_day_index(sample_day)
    charge = np.load(ARRAYS / "q4_2_charge.npy")[day]
    discharge = np.load(ARRAYS / "q4_2_discharge.npy")[day]
    soc = np.load(ARRAYS / "q4_2_soc.npy")[day]
    executed = run_execution_layer(charge, discharge, soc, cfg)
    return {
        "day_index": int(day),
        "nominal_charge": charge.tolist(),
        "nominal_discharge": discharge.tolist(),
        "nominal_soc": soc.tolist(),
        "cbf_active_slots": int(executed["cbf_active_slots"]),
        "soc_violation_slots": int(executed["soc_violation_slots"]),
        "detc": executed["detc"],
        "trigger_mask": executed["trigger_mask"],
    }


def q4_day_index(sample_day: int = FIRST_DAY) -> int:
    """Q4 数组是 2025-02-01 起的 334 天切片，需把全局日序换算为切片下标。"""
    q4_days = np.load(ARRAYS / "q4_2_q_plan.npy").shape[0]
    return int(np.clip(sample_day - FIRST_DAY, 0, q4_days - 1))


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = build()
    data["lead_time_curve"] = _lead_time_curve()
    data["execution"] = execution_traces()
    (OUT_DIR / "science_data.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[science_data] wrote {OUT_DIR / 'science_data.json'}")
    print(f"[science_data] sample day {data['meta']['sample_day_date']} "
          f"(index {data['meta']['sample_day_index']})")


if __name__ == "__main__":
    main()
