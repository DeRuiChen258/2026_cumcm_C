"""论文正文用的灵敏度复合图（中文标注，两个面板）。

内容与数据来源：

* (a) 问题 1 全天费用对（单程效率 η、最大充放电功率 P_max）的响应面 —— 由
  `results/sensitivity_grids.json` 的 `q1_surface` 提供，该网格由生产用
  `solve_stage_lp` 逐点求解，基准点即 `results/q1.json` 的 35,126.95 元。
* (b) 问题 3 全年结算费用对（超额违约系数、缺额违约系数）的热力图 —— 由
  `q3_grid` 提供，纯结算式重算，基准单元即 `results/q3.json` 的 14,705,624.88 元。
* 单因素 ±10% 龙卷风数据（`tornado`）同时导出为 `results/sensitivity_paper.json`，
  供论文表格引用。

不重跑优化、不修改任何求解器；只读 results/ 并写出图与一张数字表。

用法：
    python3 -m code.report.sensitivity_figs
"""
from __future__ import annotations

import json

import numpy as np

from code.common.load_clean import ROOT
from code.report import viz_style as vs

RESULTS = ROOT / "results"


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    grids = load_json("sensitivity_grids.json")
    q1 = load_json("q1.json")
    q3 = load_json("q3.json")["summary"]
    q1_grid, q3_grid, tornado = grids["q1_surface"], grids["q3_grid"], grids["tornado"]

    import matplotlib.pyplot as plt
    vs.setup()
    fig = plt.figure(figsize=(9.2, 3.8))

    # ------------------------------------------------------------- (a) Q1 响应面
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    etas = np.array(q1_grid["etas"])
    powers = np.array(q1_grid["powers"])
    Z = np.array(q1_grid["cost_cny"])
    X, Y = np.meshgrid(etas, powers, indexing="ij")
    surf = ax.plot_surface(X, Y, Z, cmap=vs.SEQ_CMAP, edgecolor="none", alpha=0.93,
                           rstride=1, cstride=1, antialiased=True)
    ax.contourf(X, Y, Z, zdir="z", offset=Z.min() - 0.12 * (Z.max() - Z.min()),
                cmap=vs.SEQ_CMAP, alpha=0.5)
    ax.scatter([0.9], [5000.0], [q1["cost_cny"]], color=vs.RED, s=42, depthshade=False)
    ax.text(0.9, 5000.0, q1["cost_cny"],
            f"  基准 (0.90, 5000)\n  {q1['cost_cny']:,.0f} 元", color=vs.RED, fontsize=8.0)
    ax.set_xlabel("单程效率 $\\eta$", labelpad=5)
    ax.set_ylabel("最大充放电功率 (kW)", labelpad=5)
    ax.set_zlabel("问题 1 全天费用 (元)", labelpad=5)
    ax.set_xticks(etas)
    ax.set_yticks(powers[::2])
    ax.view_init(elev=22, azim=-128)
    ax.set_title("(a) 问题 1 费用对 $(\\eta,\\ P_{max})$ 的响应面", fontsize=9.6, pad=2)
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10, label="费用 (元)")

    # ------------------------------------------------------------- (b) Q3 热力图
    ax2 = fig.add_subplot(1, 2, 2)
    overs = [f"{v:g}" for v in q3_grid["overs"]]
    unders = [f"{v:g}" for v in q3_grid["unders"]]
    W = np.array(q3_grid["cost_cny"]) / 1e4
    im = ax2.imshow(W, cmap=vs.DIV_CMAP, aspect="auto")
    ax2.set_xticks(range(len(unders)), unders)
    ax2.set_yticks(range(len(overs)), overs)
    ax2.set_xlabel("缺额违约系数 $\\theta_{under}$")
    ax2.set_ylabel("超额违约系数 $\\theta_{over}$")
    mid = float(W.mean())
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax2.text(j, i, f"{W[i, j]:.0f}", ha="center", va="center", fontsize=8.4,
                     color="white" if abs(W[i, j] - mid) > 0.35 * (W.max() - W.min())
                     else vs.BLACK)
    i0 = list(q3_grid["overs"]).index(0.5)
    j0 = list(q3_grid["unders"]).index(1.5)
    ax2.scatter([j0], [i0], marker="s", s=240, facecolors="none", edgecolors=vs.GREEN, lw=2.0)
    cb = fig.colorbar(im, ax=ax2, pad=0.02)
    cb.set_label("问题 3 全年结算费用 (万元)")
    ax2.grid(False)
    ax2.set_title("(b) 问题 3 结算费用对违约系数的热力图", fontsize=9.6, pad=6)

    fig.tight_layout(rect=(0, 0.06, 0.97, 1))
    fig.text(0.60, 0.005,
             f"绿框 = 题目给定 $({q3_grid['overs'][i0]},\\ {q3_grid['unders'][j0]})$，"
             f"对应全年结算费用 {q3['total_cost_cny']:,.2f} 元（与 result3.xlsx 一致）",
             fontsize=8.0, color=vs.GREEN, ha="center", va="bottom")
    vs.save(fig, "FigS_sensitivity_bundle",
            "多参数耦合灵敏度。(a) 问题 1 全天费用随储能单程效率 η 与最大充放电功率的变化，"
            "效率的影响远大于功率（5000 kW 附近功率约束已不紧）；"
            "(b) 问题 3 全年结算费用随超额/缺额违约系数的变化，绿框为题目给定取值 (0.5, 1.5)。")
    vs.flush_captions()

    # ------------------------------------------------------- 龙卷风表（论文表格引用）
    rows = []
    for row in tornado["rows"]:
        base = float(row["base"])
        rows.append({"label": row["label"], "base": base,
                     "low": float(row["low"]), "high": float(row["high"]),
                     "low_pct": (float(row["low"]) - base) / base * 100.0,
                     "high_pct": (float(row["high"]) - base) / base * 100.0})
    out = {"rows": rows,
           "note": "单因素 ±10% 扰动；Q1 为全天费用，Q2/Q3 为全年费用；"
                   "基准值与 output/result*.xlsx 一致。"}
    (RESULTS / "sensitivity_paper.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"  {row['label']}: {row['low_pct']:+.2f}% / {row['high_pct']:+.2f}%"
              f"  (基准 {row['base']:,.2f})")
    print("[sens] FigS_sensitivity_bundle.pdf + results/sensitivity_paper.json 已生成")


if __name__ == "__main__":
    main()
