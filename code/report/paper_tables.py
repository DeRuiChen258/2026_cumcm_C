"""论文结果表：严格按 `C题.pdf` 原题要求的表1 / 表2 / 表3 格式输出。

题面要求（原文）：
  问题1 —— “在论文中以表 1 的格式给出指定时间段的购电量（单位：kWh）及全天的购电量
  （单位：kWh）和购电费（单位：元），以表 2 的格式给出储能设备在指定时间段的充放电量
  及 0:00 和 24:00 的储电量（单位：kWh）”。
  问题2/3/4 —— “在论文中按表 1 和表 2 的格式给出表 3 中指定日期的结果，按表 3 的格式
  给出紧急购电的结果”（指定日期 = 2025.3.20 / 2025.6.21 / 2025.9.23 / 2025.12.21）。

本模块只读取已生成的 `results/` 与 `output/`，不改变任何模型或结果数值：
  * 表1：6 个指定时段（10:00-10:10 … 20:00-20:10）+ 全天购电量 + 全天购电费；
  * 表2：6 个 4 小时块的充电量/放电量 + 0:00 储电量 + 24:00 储电量；
  * 表3：指定日期的紧急购电时间段与购电量（连续时段合并成块，与表4 填写方法一致）。
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
DT = 1.0 / 6.0
FIRST_DAY = 31

# 指定时段 ↔ 时段序号（模板第 k 行 ↔ 数据第 k 时段，见 D-03）
PERIODS = [("10:00-10:10", 60), ("12:00-12:10", 72), ("14:00-14:10", 84),
           ("16:00-16:10", 96), ("18:00-18:10", 108), ("20:00-20:10", 120)]
BLOCKS = [("0:00-4:00", 0), ("4:00-8:00", 1), ("8:00-12:00", 2),
          ("12:00-16:00", 3), ("16:00-20:00", 4), ("20:00-24:00", 5)]
TARGET_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def _tags() -> dict:
    return {
        "Q2": {"tag": "q2", "plan_price": "att1", "title": "问题 2（附件1 电价 + 附件2 数据，result2.xlsx）"},
        "Q3": {"tag": "q3", "plan_price": "att1",
               "title": "问题 3（附件1 电价 + 附件2/附件3 数据，result3.xlsx）"},
        "Q4-2": {"tag": "q4_2", "plan_price": "att4",
                 "title": "问题 4 对应问题 2（附件4 波动电价，result4-2.xlsx）"},
        "Q4-3": {"tag": "q4_3", "plan_price": "att4",
                 "title": "问题 4 对应问题 3（附件4 波动电价，result4-3.xlsx）"},
    }


def _price_matrix(panel, kind: str) -> np.ndarray:
    if kind == "att4":
        return panel.price
    return np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))


def _tables_for(panel, tag: str, price_kind: str) -> dict:
    plan = np.load(ARRAYS / f"{tag}_q_plan.npy")
    adj = np.load(ARRAYS / f"{tag}_q_adjust.npy") if (ARRAYS / f"{tag}_q_adjust.npy").exists() else None
    charge = np.load(ARRAYS / f"{tag}_charge.npy")
    disch = np.load(ARRAYS / f"{tag}_discharge.npy")
    soc = np.load(ARRAYS / f"{tag}_soc.npy")
    emerg = np.load(ARRAYS / f"{tag}_emergency.npy")
    price = _price_matrix(panel, price_kind)
    dates = [panel.dates[d] for d in range(FIRST_DAY, 365)]
    out: dict = {"table1_plan": {}, "table1_adjust": {}, "table2": {}, "table3": {}}

    for iso in TARGET_DATES:
        i = dates.index(iso)
        row_price = price[FIRST_DAY + i]

        def purchase_block(values: np.ndarray) -> dict:
            periods = {label: float(values[k - 1]) for label, k in PERIODS}
            return {"periods": periods,
                    "day_energy_kwh": float(values.sum()),
                    "day_cost_cny": float(np.sum(row_price * values))}

        out["table1_plan"][iso] = purchase_block(plan[i])
        if adj is not None:
            out["table1_adjust"][iso] = purchase_block(adj[i])
        blocks_ch = charge[i].reshape(6, 24).sum(axis=1)
        blocks_dis = disch[i].reshape(6, 24).sum(axis=1)
        out["table2"][iso] = {
            "blocks": {label: {"charge_kwh": float(blocks_ch[b]),
                               "discharge_kwh": float(blocks_dis[b])}
                       for label, b in BLOCKS},
            "soc_0_kwh": float(soc[i][0]),
            "soc_24_kwh": float(soc[i][-1]),
        }
        blocks_emerg = merge_blocks(emerg[i])
        out["table3"][iso] = [{"label": block_label(a, b), "energy_kwh": float(value)}
                              for a, b, value in blocks_emerg]
    return out


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def table1_md(title: str, data: dict, unit_note: str = "") -> list[str]:
    rows = [f"**{title}**", "",
            "| 时间段 | 购电量 | 时间段 | 购电量 | 时间段 | 购电量 |",
            "| --- | ---: | --- | ---: | --- | ---: |"]
    periods = data["periods"]
    labels = [label for label, _ in PERIODS]
    for i in range(3):
        a, b = labels[2 * i], labels[2 * i + 1]
        rows.append(f"| {a} | {_fmt(periods[a])} | {b} | {_fmt(periods[b])} |  |  |")
    rows.append(f"| 全天购电量 | {_fmt(data['day_energy_kwh'])} | 全天购电费 | "
                f"{_fmt(data['day_cost_cny'])} |  |  |")
    if unit_note:
        rows += ["", f"*{unit_note}*"]
    rows.append("")
    return rows


def table2_md(title: str, data: dict) -> list[str]:
    rows = [f"**{title}**", "",
            "| 时间段 | 充电量 | 放电量 | 时间段 | 充电量 | 放电量 |",
            "| --- | ---: | ---: | --- | ---: | ---: |"]
    for i in range(3):
        (la, _), (lb, _) = BLOCKS[2 * i], BLOCKS[2 * i + 1]
        a, b = data["blocks"][la], data["blocks"][lb]
        rows.append(f"| {la} | {_fmt(a['charge_kwh'])} | {_fmt(a['discharge_kwh'])} | "
                    f"{lb} | {_fmt(b['charge_kwh'])} | {_fmt(b['discharge_kwh'])} |")
    rows.append(f"| 0:00 储电量 | {_fmt(data['soc_0_kwh'])} | 24:00 储电量 | "
                f"{_fmt(data['soc_24_kwh'])} |  |  |")
    rows.append("")
    return rows


def table3_md(title: str, per_date: dict) -> list[str]:
    rows = [f"**{title}**", "",
            "| 日期 | 紧急购电时间段 | 紧急购电量 |", "| --- | --- | ---: |"]
    for iso in TARGET_DATES:
        blocks = per_date.get(iso, [])
        if not blocks:
            rows.append(f"| {iso.replace('-', '.')} | 无 | 0.00 |")
            continue
        for j, block in enumerate(blocks):
            date_cell = iso.replace("-", ".") if j == 0 else ""
            rows.append(f"| {date_cell} | {block['label']} | {_fmt(block['energy_kwh'])} |")
    rows.append("")
    return rows


def main() -> None:
    panel = load_panel()
    PAPER.mkdir(exist_ok=True)
    q1 = json.loads((RESULTS / "q1.json").read_text(encoding="utf-8"))
    lines = ["# 论文结果表（严格按 `C题.pdf` 表1 / 表2 / 表3 格式）", "",
             "> 单位：购电量 / 充放电量 / 储电量 = kWh，购电费 = 元。",
             "> 指定日期 = 2025.3.20 / 2025.6.21 / 2025.9.23 / 2025.12.21。",
             "> 数值来源：`output/result1.xlsx` … `result4-3.xlsx`（本表由其数组逐位生成，未改动任何数值）。",
             ""]
    machine: dict = {}

    # ---------------- 问题 1 ----------------
    price1 = panel.day1[:, 0]
    q1_plan = np.array(q1["lp"]["q"])
    lines += ["## 一、问题 1（附件1 代表性日，result1.xlsx）", ""]
    t1_q1 = {"periods": {label: float(q1_plan[k - 1]) for label, k in PERIODS},
             "day_energy_kwh": float(q1_plan.sum()),
             "day_cost_cny": float(np.sum(price1 * q1_plan))}
    lines += table1_md("表1　微网在指定时间段的购电量及全天购电量和购电费", t1_q1)
    charge1 = np.array(q1["lp"]["c"]).reshape(6, 24).sum(axis=1)
    disch1 = np.array(q1["lp"]["d"]).reshape(6, 24).sum(axis=1)
    soc1 = np.array(q1["lp"]["soc"])
    t2_q1 = {"blocks": {label: {"charge_kwh": float(charge1[b]),
                                "discharge_kwh": float(disch1[b])} for label, b in BLOCKS},
             "soc_0_kwh": float(soc1[0]), "soc_24_kwh": float(soc1[-1])}
    lines += table2_md("表2　储能设备在指定时间段的充放电量及 0:00 和 24:00 的储电量", t2_q1)
    machine["Q1"] = {"table1": t1_q1, "table2": t2_q1}

    # ---------------- 问题 2 / 3 / 4 ----------------
    section = {"Q2": "二", "Q3": "三", "Q4-2": "四", "Q4-3": "五"}
    for key, meta in _tags().items():
        data = _tables_for(panel, meta["tag"], meta["plan_price"])
        machine[key] = data
        lines += [f"## {section[key]}、{meta['title']}", ""]
        lines += ["### 表1 格式（指定日期的购电量，kWh）", ""]
        for iso in TARGET_DATES:
            lines += table1_md(f"{iso.replace('-', '.')}　计划购电量", data["table1_plan"][iso])
        if data["table1_adjust"]:
            lines += ["### 表1 格式（指定日期的调整购电量，kWh）", ""]
            for iso in TARGET_DATES:
                lines += table1_md(f"{iso.replace('-', '.')}　调整购电量", data["table1_adjust"][iso])
        lines += ["### 表2 格式（指定日期的充放电量与 0:00 / 24:00 储电量，kWh）", ""]
        for iso in TARGET_DATES:
            lines += table2_md(f"{iso.replace('-', '.')}", data["table2"][iso])
        lines += ["### 表3 格式（指定日期的紧急购电量）", ""]
        lines += table3_md("表3　微网在指定日期的紧急购电量", data["table3"])

    (PAPER / "表1-表3（原题格式）.md").write_text("\n".join(lines).rstrip() + "\n",
                                                   encoding="utf-8")
    (RESULTS / "paper_tables_format.json").write_text(
        json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[tables] wrote paper/表1-表3（原题格式）.md")
    for key in ("Q2", "Q3", "Q4-2", "Q4-3"):
        n_emerg = sum(len(v) for v in machine[key]["table3"].values())
        print(f"[tables] {key}: 指定日期 {len(machine[key]['table1_plan'])} 天，"
              f"紧急购电块 {n_emerg} 个")


if __name__ == "__main__":
    main()
