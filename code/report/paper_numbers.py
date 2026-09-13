"""汇总论文要用的数字，顺带排好表1–表4。

论文里的数字一律从这儿取，每个值都带单位、来源文件和生成命令，不手抄。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.optimize.q2_stochastic import block_label, merge_blocks

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
PAPER = ROOT / "paper"
TABLE1_SLOTS = {"10:00-10:10": 60, "12:00-12:10": 72, "14:00-14:10": 84,
                "16:00-16:10": 96, "18:00-18:10": 108, "20:00-20:10": 120}
TARGET_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")
BLOCK_LABELS = [f"{4 * i}:00-{4 * i + 4}:00" for i in range(6)]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def entry(value, unit, source, command, note=""):
    return {"value": value, "unit": unit, "source": source, "command": command, "note": note}


def build() -> dict:
    panel = load_panel()
    q1 = load_json(RESULTS / "q1.json")
    q2 = load_json(RESULTS / "q2.json")
    q3 = load_json(RESULTS / "q3.json")
    q4 = load_json(RESULTS / "q4.json")
    p2 = load_json(RESULTS / "p2_conformal.json")
    cross = load_json(ROOT / "clean" / "cross_check.json")
    manifest = load_json(ROOT / "clean" / "manifest.json")
    meta = load_json(ROOT / "clean" / "meta.json")

    numbers = {
        "cleaning": {
            "audit_records": entry(manifest["audit_records"], "count", "clean/manifest.json",
                                   "cleaning_cpp clean", "NR1..NR12 全部改动条数"),
            "quarantine_records": entry(manifest["quarantine_records"], "count",
                                        "clean/manifest.json", "cleaning_cpp clean",
                                        "核夜间 >5 kW 硬异常，置 NaN"),
            "core_night_zero_before": entry(cross["att2_core_night_zero_raw"], "count",
                                            "clean/cross_check.json", "cleaning_cpp verify"),
            "core_night_zero_after": entry(cross["att2_core_night_zero_after"], "count",
                                           "clean/cross_check.json", "cleaning_cpp verify"),
            "daytime_zero_0800_1600": entry(cross["att2_daytime_0800_1600_zeros"], "count",
                                            "clean/cross_check.json", "cleaning_cpp verify",
                                            "日间无零值不变式 I2"),
            "near_zero_price_cells": entry(cross["att4_near_zero_price_cells"], "count",
                                           "clean/cross_check.json", "cleaning_cpp verify",
                                           "电价 <0.1 元/kWh"),
            "runtime_seconds": entry(meta["wall_seconds"], "s", "clean/meta.json",
                                     "cleaning_cpp clean"),
            "peak_rss_kb": entry(meta["peak_rss_kb"], "kB", "clean/meta.json",
                                 "cleaning_cpp clean"),
        },
        "q1": {
            "cost_cny": entry(q1["cost_cny"], "元", "results/q1.json", "python -m code.optimize.q1_lp"),
            "baseline_b0": entry(q1["baselines"]["B0_all_purchase_cny"], "元", "results/q1.json",
                                 "python -m code.optimize.q1_lp", "全额外购"),
            "baseline_b1": entry(q1["baselines"]["B1_pv_netting_cny"], "元", "results/q1.json",
                                 "python -m code.optimize.q1_lp", "光伏抵扣（含余量折价）"),
            "baseline_b2": entry(q1["baselines"]["B2_pv_selfuse_curtail_no_storage_cny"], "元",
                                 "results/q1.json", "python -m code.optimize.q1_lp",
                                 "自用+弃光，无储能（验收锚点 48,052）"),
            "purchase_kwh": entry(q1["energy"]["purchase_kwh"], "kWh", "results/q1.json",
                                  "python -m code.optimize.q1_lp"),
            "charge_kwh": entry(q1["energy"]["charge_kwh"], "kWh", "results/q1.json",
                                "python -m code.optimize.q1_lp"),
            "discharge_kwh": entry(q1["energy"]["discharge_kwh"], "kWh", "results/q1.json",
                                   "python -m code.optimize.q1_lp"),
            "saving_vs_b2_pct": entry(
                100.0 * (1 - q1["cost_cny"] / q1["baselines"]["B2_pv_selfuse_curtail_no_storage_cny"]),
                "%", "results/q1.json", "python -m code.optimize.q1_lp"),
            "dp_table": entry(q1["dp"], "list", "results/q1.json", "python -m code.optimize.q1_lp",
                              "步长收敛序列"),
            "night_charge_share": entry(q1["night_statistics"]["charge_share_core_night"], "ratio",
                                        "results/q1.json", "python -m code.optimize.q1_lp"),
            "night_discharge_share": entry(q1["night_statistics"]["discharge_share_core_night"],
                                           "ratio", "results/q1.json",
                                           "python -m code.optimize.q1_lp"),
        },
        "q2": {
            "total_cost_cny": entry(q2["summary"]["total_cost_cny"], "元", "results/q2.json",
                                    "python -m code.optimize.q2_stochastic"),
            "avg_daily_cost_cny": entry(q2["summary"]["total_cost_cny"] / q2["summary"]["days"],
                                        "元/日", "results/q2.json",
                                        "python -m code.optimize.q2_stochastic"),
            "emergency_energy_kwh": entry(q2["summary"]["total_emergency_energy_kwh"], "kWh",
                                          "results/q2.json", "python -m code.optimize.q2_stochastic"),
            "emergency_slot_rate": entry(q2["summary"]["emergency_slot_rate"], "ratio",
                                         "results/q2.json", "python -m code.optimize.q2_stochastic",
                                         "报童结构 p*=0.80 下的结构性缺额比例"),
            "controls": entry(q2["summary"]["controls_total_cost_cny"], "元", "results/q2.json",
                              "python -m code.optimize.q2_stochastic"),
            "quantile_coverage": entry(q2["summary"]["quantile_coverage"], "ratio",
                                       "results/q2.json", "python -m code.optimize.q2_stochastic"),
        },
        "q3": {
            "total_cost_cny": entry(q3["summary"]["total_cost_cny"], "元", "results/q3.json",
                                    "python -m code.optimize.q3_mpc"),
            "emergency_energy_kwh": entry(q3["summary"]["total_emergency_energy_kwh"], "kWh",
                                          "results/q3.json", "python -m code.optimize.q3_mpc"),
            "over_plan_penalty_cny": entry(q3["summary"]["total_over_plan_penalty_cny"], "元",
                                           "results/q3.json", "python -m code.optimize.q3_mpc"),
            "under_plan_penalty_cny": entry(q3["summary"]["total_under_plan_penalty_cny"], "元",
                                            "results/q3.json", "python -m code.optimize.q3_mpc"),
            "subset_costs": entry(q3["summary"]["subset_total_cost_cny"], "元", "results/q3.json",
                                  "python -m code.optimize.q3_mpc"),
            "subset_marginal_value": entry(q3["summary"]["subset_marginal_value_cny"], "元",
                                           "results/q3.json", "python -m code.optimize.q3_mpc"),
            "value_split": entry(q3["summary"]["subset_value_split_cny"], "元", "results/q3.json",
                                 "python -m code.optimize.q3_mpc", "夜间/日间分口径"),
        },
        "q4": {
            "q4_2_total_cost_cny": entry(q4["q4_2"]["total_cost_cny"], "元", "results/q4.json",
                                         "python -m code.optimize.q4_price"),
            "q4_3_total_cost_cny": entry(q4["q4_3"]["total_cost_cny"], "元", "results/q4.json",
                                         "python -m code.optimize.q4_price"),
            "spike_coverage": entry(q4["spike_conformity"]["coverage"], "ratio", "results/q4.json",
                                    "python -m code.optimize.q4_price", "门禁 ≥ 90%"),
            "cvar_pareto": entry(q4["cvar_pareto"], "list", "results/q4.json",
                                 "python -m code.optimize.q4_price"),
            "copula_value": entry(q4["copula_value"], "元", "results/q4.json",
                                  "python -m code.optimize.q4_price"),
            "soft_penalty": entry(q4["soft_penalty_evidence"], "list", "results/q4.json",
                                  "python -m code.optimize.q4_price"),
        },
        "p2": {
            "coverage": entry({k: v["coverage"] for k, v in p2["coverage"]["anchors"].items()},
                              "ratio", "results/p2_conformal.json", "python -m code.forecast.calibrate"),
            "deviations_pp": entry({k: 100 * v["deviation"]
                                    for k, v in p2["coverage"]["anchors"].items()},
                                   "pp", "results/p2_conformal.json",
                                   "python -m code.forecast.calibrate"),
            "newsvendor_level": entry(p2["newsvendor_level"], "list", "results/p2_conformal.json",
                                      "python -m code.forecast.calibrate"),
            "forecast_skill": entry(p2["forecast_skill"], "dict", "results/p2_conformal.json",
                                    "python -m code.forecast.calibrate"),
        },
        "storage": {
            "capacity_kwh": entry(12000.0, "kWh", "附录1", "-"),
            "power_kw": entry(5000.0, "kW", "附录1", "-"),
            "efficiency": entry(0.9, "ratio", "附录1", "-"),
            "soc_band_kwh": entry([1200.0, 10800.0], "kWh", "附录1", "-"),
        },
    }
    return numbers


def table1_q1(q1: dict) -> str:
    rows = ["| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |",
            "| --- | --- | --- | --- | --- | --- |"]
    labels = list(TABLE1_SLOTS)
    values = [q1["table1"]["periods"][k] for k in labels]
    for i in range(3):
        a, b = labels[2 * i], labels[2 * i + 1]
        rows.append(f"| {a} | {values[2*i]:.2f} | {b} | {values[2*i+1]:.2f} |  |  |")
    rows.append(f"| 全天购电量 | {q1['table1']['day_purchase_kwh']:.2f} | "
                f"全天购电费 | {q1['table1']['day_cost_cny']:.2f} |  |  |")
    return "\n".join(rows)


def table2_q1(q1: dict) -> str:
    charge = q1["table2"]["charge_kwh"]
    disch = q1["table2"]["discharge_kwh"]
    rows = ["| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for i in range(3):
        a, b = BLOCK_LABELS[2 * i], BLOCK_LABELS[2 * i + 1]
        rows.append(f"| {a} | {charge[2*i]:.2f} | {disch[2*i]:.2f} | "
                    f"{b} | {charge[2*i+1]:.2f} | {disch[2*i+1]:.2f} |")
    rows.append(f"| 0:00 储电量 | {q1['table2']['soc_0_00_kwh']:.2f} | 24:00 储电量 | "
                f"{q1['table2']['soc_24_00_kwh']:.2f} |  |  |")
    return "\n".join(rows)


def table3_q2(panel, q2: dict, tag: str = "q2") -> str:
    dates = [panel.dates[d] for d in range(31, 365)]
    q_plan = np.load(ARRAYS / f"{tag}_q_plan.npy")
    charge = np.load(ARRAYS / f"{tag}_charge.npy")
    disch = np.load(ARRAYS / f"{tag}_discharge.npy")
    emerg = np.load(ARRAYS / f"{tag}_emergency.npy")
    blocks = []
    for iso in TARGET_DATES:
        i = dates.index(iso)
        p_block = [f"{q_plan[i][k - 1]:.2f}" for k in TABLE1_SLOTS.values()]
        ch = charge[i].reshape(6, 24).sum(axis=1)
        di = disch[i].reshape(6, 24).sum(axis=1)
        eb = merge_blocks(emerg[i])
        eb_txt = "；".join(f"{block_label(b[0], b[1])} {b[2]:.2f} kWh" for b in eb) or "无"
        blocks.append((iso, p_block, ch, di, eb_txt))
    out = ["#### 表3 四个指定日期的计划购电量（指定时段，kWh）", "",
           "| 日期 | 10:00-10:10 | 12:00-12:10 | 14:00-14:10 | 16:00-16:10 | 18:00-18:10 | 20:00-20:10 |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for iso, p_block, _, _, _ in blocks:
        out.append("| " + iso + " | " + " | ".join(p_block) + " |")
    out += ["", "#### 表3（续）四个指定日期的充放电量（4 小时块，kWh）与 0:00/24:00 储电量", "",
            "| 日期 | 0:00-4:00 | 4:00-8:00 | 8:00-12:00 | 12:00-16:00 | 16:00-20:00 | 20:00-24:00 | 0:00 储电量 | 24:00 储电量 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    soc = np.load(ARRAYS / f"{tag}_soc.npy")
    for iso, _, ch, di, _ in blocks:
        i = dates.index(iso)
        ch_txt = "/".join(f"{v:.0f}" for v in ch)
        di_txt = "/".join(f"{v:.0f}" for v in di)
        out.append(f"| {iso} | {ch_txt}（充）| {di_txt}（放）|  |  |  |  | {soc[i][0]:.0f} | {soc[i][-1]:.0f} |")
    out += ["", "#### 表3（续）四个指定日期的紧急购电量与时段", "",
            "| 日期 | 紧急购电时段与电量 |", "| --- | --- |"]
    for iso, _, _, _, eb_txt in blocks:
        out.append(f"| {iso} | {eb_txt} |")
    return "\n".join(out)


def table4_q2(panel, tag: str = "q2", sample_dates=("2025-03-01", "2025-03-20", "2025-06-21")) -> str:
    dates = [panel.dates[d] for d in range(31, 365)]
    emerg = np.load(ARRAYS / f"{tag}_emergency.npy")
    out = ["#### 表4 紧急购电量填报（示例，与 result 文件“紧急购电量”工作表同构）", "",
           "| 日期 | 紧急购电时间段 | 紧急购电量(kWh) |", "| --- | --- | --- |"]
    for iso in sample_dates:
        i = dates.index(iso)
        for b in merge_blocks(emerg[i]):
            out.append(f"| {iso} | {block_label(b[0], b[1])} | {b[2]:.2f} |")
    return "\n".join(out)


def main() -> None:
    PAPER.mkdir(exist_ok=True)
    panel = load_panel()
    numbers = build()
    (RESULTS / "paper_numbers.json").write_text(json.dumps(numbers, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    q1 = load_json(RESULTS / "q1.json")
    q2 = load_json(RESULTS / "q2.json")
    q3 = load_json(RESULTS / "q3.json")
    q4 = load_json(RESULTS / "q4.json")
    p2 = load_json(RESULTS / "p2_conformal.json")
    body = [
        "# 论文表 1–表 4（数字全部来自 results/paper_numbers.json）", "",
        "#### 表1 微网在指定时间段的购电量及全天购电量和购电费（问题1，附件1 代表性日）", "",
        table1_q1(q1), "",
        "#### 表2 储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量（问题1）", "",
        table2_q1(q1), "",
        table3_q2(panel, q2),
        "",
        table4_q2(panel),
        "",
        "---",
        "## 关键数字（供摘要与正文引用）",
        f"- 问题1：最优费用 **{q1['cost_cny']:.2f} 元**（B2 基线 "
        f"{q1['baselines']['B2_pv_selfuse_curtail_no_storage_cny']:.2f} 元，降幅 "
        f"{100*(1-q1['cost_cny']/q1['baselines']['B2_pv_selfuse_curtail_no_storage_cny']):.2f}%）；"
        f"核夜间充电占比 {q1['night_statistics']['charge_share_core_night']*100:.1f}%、"
        f"放电占比 {q1['night_statistics']['discharge_share_core_night']*100:.1f}%。",
        f"- 问题2：全年费用 **{q2['summary']['total_cost_cny']:.2f} 元**（均值 "
        f"{q2['summary']['total_cost_cny']/q2['summary']['days']:.2f} 元/日）；紧急购电 "
        f"{q2['summary']['total_emergency_energy_kwh']:.2f} kWh，时段触发率 "
        f"{q2['summary']['emergency_slot_rate']*100:.2f}%；较“无裕度”计划节省 "
        f"{q2['summary']['controls_total_cost_cny']['C1_no_margin'] - q2['summary']['total_cost_cny']:.2f} 元。",
        f"- 问题3：结算费用 **{q3['summary']['total_cost_cny']:.2f} 元**；引入 6:00 预报价值 "
        f"{q3['summary']['subset_marginal_value_cny']['value_of_0600']:.2f} 元，12:00 价值 "
        f"{q3['summary']['subset_marginal_value_cny']['value_of_1200']:.2f} 元，18:00 价值 "
        f"{q3['summary']['subset_marginal_value_cny']['value_of_1800']:.2f} 元；"
        f"完美预报下界 {q3['summary']['subset_total_cost_cny']['S5']:.2f} 元（仅作“界”）。",
        f"- 问题4：实时电价下 Q4-2 **{q4['q4_2']['total_cost_cny']:.2f} 元**、Q4-3 "
        f"**{q4['q4_3']['total_cost_cny']:.2f} 元**；尖峰共形覆盖率 "
        f"{q4['spike_conformity']['coverage']*100:.2f}%；CVaR 风险–费用前沿见 results/q4.json。",
        f"- 共形层：计划分位覆盖 = "
        + "、".join(f"{k} {v['coverage']:.4f}（偏差 {100*v['deviation']:+.2f}pp）"
                    for k, v in p2["coverage"]["anchors"].items()) + "。",
        "",
        "> 说明：所有数字由 `python -m code.report.paper_numbers` 生成，禁止手抄。",
    ]
    (PAPER / "表1-表4.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"[paper] wrote {PAPER/'表1-表4.md'} and results/paper_numbers.json")
    print(f"[paper] Q1 cost={q1['cost_cny']:.2f} | Q2={q2['summary']['total_cost_cny']:.2f} | "
          f"Q3={q3['summary']['total_cost_cny']:.2f} | Q4-2={q4['q4_2']['total_cost_cny']:.2f} | "
          f"Q4-3={q4['q4_3']['total_cost_cny']:.2f}")


if __name__ == "__main__":
    main()
