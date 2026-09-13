"""照着附件5 模板把五个结果文件写出来。

模板怎么写就怎么写：工作表名和表头照抄（连 `0:00+1-0:10+1` 这种写法也不改）、
日期保持日期格式、`⁝` 行删掉后补齐 334 天，样式字体全用模板自带的。
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.common.xlsx_template_io import Workbook, idx_to_col, write_workbook

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
TEMPLATES = ROOT / "templates" / "附件5"
OUTPUT = ROOT / "output"
FIRST_DAY = 31          # result2/3/4-* 覆盖 2025-02-01 起（1 月为共形校准期）
EPOCH = dt.date(1899, 12, 30)
BLOCK_LABELS = [f"{4 * i}:00-{4 * i + 4}:00" for i in range(6)]


def date_serial(iso: str) -> int:
    y, m, d = (int(part) for part in iso.split("-"))
    return (dt.date(y, m, d) - EPOCH).days


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def style_of(wb: Workbook, sheet: str, row: int, col: int, fallback_row: int = 1) -> str | None:
    style = wb.style(sheet, row, col)
    return style if style is not None else wb.style(sheet, fallback_row, col)


def header_row(wb: Workbook, sheet: str) -> list[tuple]:
    cells = wb.sheets[sheet].rows.get(1, {})
    out = []
    for col in sorted(cells):
        text = wb.text(sheet, 1, col)
        out.append((col, text, "text", wb.style(sheet, 1, col)))
    return out


def wide_table_rows(wb: Workbook, sheet: str, dates: list[str], values: np.ndarray,
                    price: np.ndarray, first_day: int = 31) -> list[list[tuple]]:
    """Wide-table rows: date + 144 slot values + daily energy + daily cost.

    `price` is the full-year price matrix, so the daily cost of row i must use the
    price row of the same calendar day (`first_day + i`).  Using `price[i]` instead
    would silently shift the tariff by the length of the calibration month — invisible
    for Q2/Q3 (constant daily tariff) but wrong for Q4, where the price changes daily.
    """
    rows = [header_row(wb, sheet)]
    for i, iso in enumerate(dates):
        row = [(0, date_serial(iso), "number", style_of(wb, sheet, 2, 0))]
        for k in range(144):
            row.append((1 + k, float(values[i, k]), "number", style_of(wb, sheet, 2, 1 + k)))
        row.append((145, float(values[i].sum()), "number", style_of(wb, sheet, 2, 145)))
        day_index = first_day + i
        row.append((146, float(np.sum(price[day_index] * values[i])), "number",
                    style_of(wb, sheet, 2, 146)))
        rows.append(row)
    return rows


def charge_discharge_rows(wb: Workbook, sheet: str, dates: list[str], charge: np.ndarray,
                          discharge: np.ndarray, soc: np.ndarray,
                          with_date: bool = True) -> list[list[tuple]]:
    rows = [header_row(wb, sheet)]
    for i, iso in enumerate(dates):
        blocks_charge = charge[i].reshape(6, 24).sum(axis=1)
        blocks_discharge = discharge[i].reshape(6, 24).sum(axis=1)
        for b in range(6):
            template_row = 2 + b
            first = b == 0
            second = b == 1
            row: list[tuple] = []
            if with_date:
                row.append((0, date_serial(iso) if first else None, "number",
                            style_of(wb, sheet, template_row, 0)))
                offset = 1
            else:
                offset = 0
            row.append((offset + 0, BLOCK_LABELS[b], "text",
                        style_of(wb, sheet, template_row, offset + 0)))
            row.append((offset + 1, float(blocks_charge[b]), "number",
                        style_of(wb, sheet, template_row, offset + 1)))
            row.append((offset + 2, float(blocks_discharge[b]), "number",
                        style_of(wb, sheet, template_row, offset + 2)))
            if first:
                row.append((offset + 3, 0.0, "number", style_of(wb, sheet, template_row, offset + 3)))
                row.append((offset + 4, float(soc[i][0]), "number",
                            style_of(wb, sheet, template_row, offset + 4)))
            elif second:
                row.append((offset + 3, "24:00", "text", style_of(wb, sheet, template_row, offset + 3)))
                row.append((offset + 4, float(soc[i][-1]), "number",
                            style_of(wb, sheet, template_row, offset + 4)))
            rows.append(row)
    return rows


def emergency_rows(wb: Workbook, sheet: str, dates: list[str],
                   records: list[dict]) -> list[list[tuple]]:
    rows = [header_row(wb, sheet)]
    by_date = {r["date"]: r for r in records}
    for iso in dates:
        record = by_date.get(iso, {"blocks": []})
        blocks = record.get("blocks", [])
        if not blocks:
            rows.append([(0, date_serial(iso), "number", style_of(wb, sheet, 2, 0)),
                         (1, None, "empty", style_of(wb, sheet, 2, 1)),
                         (2, None, "empty", style_of(wb, sheet, 2, 2))])
            continue
        for j, block in enumerate(blocks):
            row = [(0, date_serial(iso) if j == 0 else None, "number", style_of(wb, sheet, 2, 0)),
                   (1, block["label"], "text", style_of(wb, sheet, 2, 1)),
                   (2, float(block["energy_kwh"]), "number", style_of(wb, sheet, 2, 2))]
            rows.append(row)
    return rows


def write_result1(panel, out_dir: Path) -> dict:
    q1 = load_json(RESULTS / "q1.json")
    wb = Workbook(TEMPLATES / "result1.xlsx")
    slots_rows = [header_row(wb, "计划购电量")]
    for k in range(144):
        slots_rows.append([(0, wb.text("计划购电量", k + 2, 0), "text", wb.style("计划购电量", k + 2, 0)),
                           (1, float(q1["lp"]["q"][k]), "number", style_of(wb, "计划购电量", 2, 1))])
    cd_rows = charge_discharge_rows(wb, "充放电量", [panel.dates[0]],
                                    np.array([q1["lp"]["c"]]), np.array([q1["lp"]["d"]]),
                                    np.array([q1["lp"]["soc"]]), with_date=False)
    write_workbook(wb, out_dir / "result1.xlsx",
                   {"计划购电量": slots_rows, "充放电量": cd_rows})
    wb.close()
    return {"file": "result1.xlsx", "sheets": {"计划购电量": len(slots_rows),
                                               "充放电量": len(cd_rows)}}


def write_wide_result(template_name: str, out_name: str, panel, tags: list[str],
                      kinds: list[str]) -> dict:
    """tags/kinds: [('q2', 'plan')] etc. — 计划购电量 always comes first."""
    wb = Workbook(TEMPLATES / template_name)
    plan_tag, plan_kind = tags[0], kinds[0]
    plan = np.load(ARRAYS / f"{plan_tag}_{plan_kind}.npy")
    plan_cost_price = price_matrix_for(plan_tag, panel)
    sheets: dict[str, list[list[tuple]]] = {
        "计划购电量": wide_table_rows(wb, "计划购电量", list(panel.dates[FIRST_DAY:365]), plan,
                                      plan_cost_price, FIRST_DAY)
    }
    if len(tags) > 1:
        adj_tag, adj_kind = tags[1], kinds[1]
        adj = np.load(ARRAYS / f"{adj_tag}_{adj_kind}.npy")
        sheets["调整购电量"] = wide_table_rows(wb, "调整购电量",
                                               list(panel.dates[FIRST_DAY:365]), adj,
                                               price_matrix_for(adj_tag, panel), FIRST_DAY)
    summary = load_json(RESULTS / f"{plan_tag}.json")["summary"]
    records = load_json(RESULTS / f"{plan_tag}.json")["records"]
    charge = np.load(ARRAYS / f"{plan_tag}_charge.npy")
    discharge = np.load(ARRAYS / f"{plan_tag}_discharge.npy")
    soc = np.load(ARRAYS / f"{plan_tag}_soc.npy")
    sheets["充放电量"] = charge_discharge_rows(wb, "充放电量", list(panel.dates[FIRST_DAY:365]),
                                             charge, discharge, soc)
    sheets["紧急购电量"] = emergency_rows(wb, "紧急购电量", list(panel.dates[FIRST_DAY:365]), records)
    write_workbook(wb, OUTPUT / out_name, sheets)
    wb.close()
    return {"file": out_name, "rows": {k: len(v) for k, v in sheets.items()},
            "summary": summary}


def price_matrix_for(tag: str, panel) -> np.ndarray:
    """Q2/Q3 settle against the 附件1 tariff; Q4 against 附件4."""
    if tag.startswith("q4"):
        return panel.price
    return np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    panel = load_panel()
    manifest = {"files": []}
    manifest["files"].append(write_result1(panel, OUTPUT))
    manifest["files"].append(write_wide_result("result2.xlsx", "result2.xlsx", panel,
                                              ["q2"], ["q_plan"]))
    manifest["files"].append(write_wide_result("result3.xlsx", "result3.xlsx", panel,
                                              ["q3", "q3"], ["q_plan", "q_adjust"]))
    manifest["files"].append(write_wide_result("result4-2.xlsx", "result4-2.xlsx", panel,
                                              ["q4_2"], ["q_plan"]))
    manifest["files"].append(write_wide_result("result4-3.xlsx", "result4-3.xlsx", panel,
                                              ["q4_3", "q4_3"], ["q_plan", "q_adjust"]))
    (RESULTS / "result_files.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for entry in manifest["files"]:
        print(f"[write] {entry['file']}: " +
              ", ".join(f"{k}={v}" for k, v in entry["rows"].items()) if "rows" in entry
              else f"[write] {entry['file']}: " + str(entry["sheets"]))


if __name__ == "__main__":
    main()
