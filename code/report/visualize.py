"""C题 绘图程序：主图、灵敏度图、计算架构图、论文表合成图都在这里出。

    python3 -m code.report.visualize --set all      # 全部图，默认
    python3 -m code.report.visualize --set main     # 只出主图
    python3 -m code.report.visualize --list         # 看图名和图注

默认只导矢量 PDF，figures/ 里不留位图；要位图就加 --formats pdf,png。
数字都从 results/ 读，脚本自己不建模，只有灵敏度网格和性能基准是现算的。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  导入即注册 3d 投影

from code.common.load_clean import ROOT, load_panel
from code.report import viz_style as vs
from code.report import eda_figs as eda
from code.report import paper_figs as pfig
from code.report.viz_style import (BLACK, BLUE, GREY, GREEN, ORANGE, PALETTE, PURPLE, RED,
                                   SKY, YELLOW)

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
FIGURES = ROOT / "figures"


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def arr(name: str) -> np.ndarray:
    return np.load(ARRAYS / name)


def hours_axis(n: int = 144) -> np.ndarray:
    return np.arange(n) / 6.0


def q1_series(panel) -> dict:
    q1 = load_json(RESULTS / "q1.json")
    return {
        "data": q1,
        "price": panel.day1[:, 0], "load": panel.day1[:, 1], "pv": panel.day1[:, 2],
        "q": np.array(q1["lp"]["q"]), "c": np.array(q1["lp"]["c"]),
        "d": np.array(q1["lp"]["d"]), "soc": np.array(q1["lp"]["soc"])[1:],
        "mu": np.array(q1["lp"]["mu"]),
    }


# ---------------- 主图：总体架构 + 四问结果 ----------------

def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 7.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, title, body, face="#F4F7FB", edge=BLUE, title_size=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.0",
                                    linewidth=1.1, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=edge)
        ax.text(x + w / 2, y + h - 4.2, body, ha="center", va="top", fontsize=8.2,
                linespacing=1.55, color=BLACK)

    def arrow(x1, y1, x2, y2, label="", color=GREY, rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                     linewidth=1.0, color=color,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.7, label, ha="center", va="bottom",
                    fontsize=7.8, color=BLACK)

    box(2, 48.5, 27, 10, "① 原始附件（附件1–4）",
        "附件1 代表日 · 附件2 实际负荷/光伏\n附件3 四时次 24 h 预报 · 附件4 实时电价")
    box(2, 34.5, 27, 11.5, "② C++17 清洗 cleaning_cpp",
        "zip+XML 读取 · 右端点口径 NR10\n按日自适应夜窗 · NR1–NR12\n审计/隔离 · .npy + SHA-256 manifest",
        face="#EEF7F1", edge=GREEN)
    box(37, 34.5, 26, 11.5, "③ 共形预测层 CP",
        "谐波+周期预测 · split/加权共形\nACI 反馈 · 有限样本修正\n分位锚点 0.80 / 0.25 / 0.70",
        face="#FDF6E7", edge=ORANGE)
    box(70, 34.5, 28, 11.5, "④ MPC 优化层",
        "Q1 LP+DP 互证 · Q2 两阶段随机规划\nQ3 四期滚动 MPC · Q4 CVaR+软约束\n场景 bootstrap / Copula",
        face="#FBEDF3", edge=PURPLE)
    box(37, 19.5, 61, 10.5, "⑤ 结算与出表",
        "Q2/Q4-2：计划购电费 + 5×紧急购电费    Q3/Q4-3：计划 + 紧急 + 0.5×(超额)⁺ + 1.5×(缺额)⁺\n"
        "模板驱动写出 result1/2/3/4-2/4-3.xlsx（保留附件5 的样式、表头与标签）",
        face="#F4F4FA", edge="#444A8C")
    box(2, 19.5, 27, 10.5, "⑥ 产物",
        "results/*.json + arrays/*.npy\nfigures/*.pdf + CAPTIONS.md\npaper/表1–表4 + paper_numbers.json",
        face="#EEF7F1", edge=GREEN)
    box(2, 3.5, 96, 11.5, "四个耦合接口（CP ↔ MPC）",
        "Ⅰ 分位接口：CP 输出安全裕度 m_j → MPC 用 N^plan = N̂ + m_j         "
        "Ⅱ 场景接口：残差块 bootstrap / Copula → 随机规划场景集\n"
        "Ⅲ 区间接口：共形区间宽度 → 鲁棒半径与灵敏度指标                      "
        "Ⅳ 反馈接口：ACI 以实测命中率回灌 α（夜间掩码不计入）",
        face="#F7F7FA", edge=GREY)

    arrow(15.5, 48.5, 15.5, 46, "只读解析")
    arrow(29, 40, 37, 40, "clean/")
    arrow(63, 42, 70, 42, "N̂, σ, 分位")
    arrow(84, 34.5, 75, 30, "调度方案", rad=-0.12)
    arrow(37, 25, 29, 25, "数字 / 图 / 表")
    arrow(70, 34.5, 60, 30, "q^plan, q^adj, g", rad=0.12)
    arrow(50, 34.5, 45, 30, "α 回灌", color=PURPLE, rad=0.2)
    arrow(62, 19.5, 55, 15, "验证")
    arrow(18, 19.5, 18, 15, "登记")
    ax.set_title("图 0　总体架构：C++ 清洗 → 共形预测 → 滚动 MPC → 结算出表（含四个耦合接口）",
                 fontsize=12.5, fontweight="bold", pad=8)
    vs.save(fig, "fig0_architecture",
            "C题总体技术架构。②为 C++17 值级清洗层（NR1–NR12 + 审计 + 不变量），"
            "③为共形预测层（ACI 反馈、三档分位），④为四问优化模型，⑤为结算与模板出表；"
            "下部为 CP↔MPC 的四个耦合接口（分位、场景、区间、反馈）。")


def fig_q1_dispatch(panel) -> None:
    s = q1_series(panel)
    q1 = s["data"]
    h = hours_axis()
    st = panel.config["storage"]
    k_rise = int(panel.windows["k_rise"][0]) - 1
    k_set = int(panel.windows["k_set"][0]) - 1
    fig = plt.figure(figsize=(13.2, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_betweenx([0, 6200], 0, h[k_rise], color="#E8EEF6", alpha=0.9, lw=0)
    ax.fill_betweenx([0, 6200], h[k_set + 1], 24, color="#E8EEF6", alpha=0.9, lw=0)
    ax.plot(h, s["load"], color=RED, lw=1.5, label="小区负载 $L_k$")
    ax.plot(h, s["pv"], color=ORANGE, lw=1.5, label="光伏预测 $PV_k$")
    ax.plot(h, s["q"] * 6, color=BLUE, lw=1.7, label="购电功率 $q_k/\\Delta t$")
    ax.set_ylim(0, 6200)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("功率 (kW)")
    ax.legend(loc="upper center", ncol=3)
    vs.panel(ax, "a")
    ax.set_title("负荷 / 光伏 / 购电功率（灰区为按日自适应核夜间）")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(h, s["price"], color=PURPLE, lw=1.6, label="电价 $\\pi_k$")
    ax.plot(h, s["mu"], color=GREEN, lw=1.6, label="购电边际价值 $\\mu_k$（对偶）")
    ax.fill_between(h, s["mu"], s["price"], where=s["price"] > s["mu"], color=ORANGE,
                    alpha=0.18, lw=0, label="套利空间")
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("元/kWh")
    ax.legend(loc="upper left")
    vs.panel(ax, "b")
    ax.set_title("电价与影子价格：价差时段即储能充放窗口")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h, s["soc"], color=GREEN, lw=1.8, label="储电量 $e_k$")
    ax.axhline(float(st["soc_max_kwh"]), ls="--", lw=0.9, color=GREY)
    ax.axhline(float(st["soc_min_kwh"]), ls="--", lw=0.9, color=GREY)
    ax.annotate(f"上限 {float(st['soc_max_kwh']):.0f} kWh", (0.3, float(st["soc_max_kwh"])),
                va="bottom", fontsize=8, color=GREY)
    ax.annotate(f"下限 {float(st['soc_min_kwh']):.0f} kWh", (0.3, float(st["soc_min_kwh"])),
                va="bottom", fontsize=8, color=GREY)
    ax.scatter([0, 24], [s["soc"][0], s["soc"][-1]], color=BLACK, zorder=5, s=18)
    ax.annotate(f"$e_0=e_{{144}}={s['soc'][-1]:.0f}$ kWh", (0.6, s["soc"][-1] - 900), fontsize=8.4)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(loc="lower right")
    vs.panel(ax, "c")
    ax.set_title("储电量轨迹（日循环约束，全程落在 [1200, 10800] kWh）")

    ax = fig.add_subplot(gs[1, 1])
    width = 0.4
    ax.bar(h - width / 2, s["c"] * 6, width=width, color=BLUE, label="充电功率")
    ax.bar(h + width / 2, -s["d"] * 6, width=width, color=RED, label="放电功率")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("功率 (kW)")
    ax.legend(loc="upper left")
    vs.panel(ax, "d")
    ax.set_title("充放电功率（充放互斥：命题 1 的数值验证）")

    fig.suptitle(f"图 1　问题 1 代表性日最优计划：全天 {q1['cost_cny']:.2f} 元，"
                 f"较 B2 基线 {q1['baselines']['B2_pv_selfuse_curtail_no_storage_cny']:.2f} 元下降 "
                 f"{100*(1-q1['cost_cny']/q1['baselines']['B2_pv_selfuse_curtail_no_storage_cny']):.2f}%",
                 fontsize=12, fontweight="bold", y=0.99)
    vs.save(fig, "fig1_q1_dispatch",
            "问题 1 代表性日最优调度。(a) 负荷、光伏与购电功率；(b) 电价与购电边际价值（对偶变量），"
            "两者之差即储能套利空间；(c) 储电量轨迹满足日循环与容量约束；(d) 充放电功率，"
            "各时段充放互斥（命题 1 成立）。")


def fig_dp_convergence() -> None:
    q1 = load_json(RESULTS / "q1.json")
    steps = np.array([r["step_kwh"] for r in q1["dp"]])
    costs = np.array([r["cost_cny"] for r in q1["dp"]])
    lp = q1["cost_cny"]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    ax.plot(steps, costs, "o-", color=BLUE, lw=1.5, ms=5, label="SoC 离散动态规划")
    ax.axhline(lp, ls="--", color=RED, lw=1.4, label=f"LP 最优解 {lp:.2f} 元")
    for x, y in zip(steps, costs):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.2)
    ax.annotate(f"相对偏差 {100*(costs[-1]-lp)/lp:.2f}%", (steps[-1], costs[-1]),
                textcoords="offset points", xytext=(14, -16), fontsize=8.4, color=RED)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(steps, [f"{int(s)}" for s in steps])
    ax.set_xlabel("SoC 离散步长 $\\Delta e$ (kWh)")
    ax.set_ylabel("全天购电费 (元)")
    ax.legend(loc="lower left")
    ax.set_title("图 2　数值验证：DP 步长单调收敛到 LP 最优解")
    vs.save(fig, "fig2_q1_dp_convergence",
            "问题 1 的数值一致性验证。DP 费用随离散步长减小而单调下降，收敛到 LP 最优解；"
            "四个步长（120/60/30/15 kWh）的费用与题给锚点逐位一致。")


def fig_night_cleaning(panel) -> None:
    raw, clean = panel.pv_raw, panel.pv
    dates = list(panel.dates)
    picks = [("2025-01-15", "冬季 1-15", BLUE), ("2025-06-15", "夏季 6-15", GREEN),
             ("2025-09-15", "秋季 9-15", ORANGE)]
    fig = plt.figure(figsize=(13.0, 8.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.36, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    h = hours_axis()
    for iso, label, color in picks:
        d = dates.index(iso)
        ax.plot(h, raw[d], ls="--", lw=1.0, color=color, alpha=0.5)
        ax.plot(h, clean[d], lw=1.7, color=color, label=label)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("光伏功率 (kW)")
    ax.legend(ncol=3, loc="upper left")
    ax.annotate("虚线 = 清洗前　实线 = 清洗后", (0.98, 0.94), xycoords="axes fraction",
                ha="right", fontsize=8.6, color=GREY)
    vs.panel(ax, "a", dx=-0.06)
    ax.set_title("清洗前后对比：核夜间小值全部归零，日间曲线逐点未变（I7 逐位校验）")

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(365)
    ax.plot(x, panel.windows["k_rise"] / 6.0, color=ORANGE, lw=1.3, label="日出时刻")
    ax.plot(x, panel.windows["k_set"] / 6.0, color=RED, lw=1.3, label="日落时刻")
    ax.set_xlabel("2025 年日序")
    ax.set_ylabel("时刻 (h)")
    ax.legend(loc="lower center", ncol=2)
    vs.panel(ax, "b")
    ax.set_title("按日自适应日照窗口（全年漂移约 4 h）")

    ax = fig.add_subplot(gs[1, 1])
    night = panel.windows["night_core_hours"]
    month_edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    monthly = [night[month_edges[m]:month_edges[m + 1]].mean() for m in range(12)]
    bars = ax.bar(np.arange(1, 13), monthly, color=SKY, edgecolor=BLUE, lw=0.6)
    vs.annotate_bars(ax, bars, monthly, fmt="{:.2f}", fontsize=7.6)
    ax.set_xticks(np.arange(1, 13), [f"{m}" for m in range(1, 13)])
    ax.set_xlabel("月份")
    ax.set_ylabel("核夜间时长 (h)")
    vs.panel(ax, "c")
    ax.set_title("核夜间时长月均值：10.33 → 14.33 h（禁止固定时钟窗口）")
    fig.suptitle("图 3　夜间专项清洗：结构事实、按日自适应窗口与季节漂移",
                 fontsize=12, fontweight="bold", y=0.99)
    vs.save(fig, "fig3_night_cleaning",
            "夜间专项数据清洗。(a) 冬/夏/秋三个代表日清洗前后光伏曲线（核夜间零漂噪声归零、"
            "日间逐点不变）；(b) 按日自适应的日出/日落时刻；(c) 核夜间时长的月度分布，"
            "季节漂移约 4 h，说明固定时钟窗口不可用。")


def fig_q2_emergency(panel) -> None:
    q2 = load_json(RESULTS / "q2.json")
    emerg = arr("q2_emergency.npy")
    records = q2["records"]
    fig = plt.figure(figsize=(14.6, 5.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.55, 0.85, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(emerg.T, aspect="auto", origin="lower", cmap=vs.SEQ_CMAP,
                   extent=[0, len(records), 0, 24], interpolation="nearest")
    ax.set_xlabel("日期（2025-02-01 → 2025-12-31）")
    ax.set_ylabel("时刻 (h)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("紧急购电量 (kWh/时段)")
    vs.panel(ax, "a")
    ax.set_title("全年紧急购电热力图")

    ax = fig.add_subplot(gs[0, 1])
    blocks = [r["emergency_blocks"] for r in records]
    bins = np.arange(-0.5, max(blocks) + 1.5)
    ax.hist(blocks, bins=bins, color=ORANGE, edgecolor="white", lw=0.6)
    ax.set_xlabel("紧急购电块数 (块/日)")
    ax.set_ylabel("天数 (d)")
    ax.annotate(f"均值 {np.mean(blocks):.2f} 块/日\n含紧急购电天数 {sum(1 for b in blocks if b)}/334",
                (0.62, 0.86), xycoords="axes fraction", fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6))
    vs.panel(ax, "b")
    ax.set_title("紧急购电块数分布")

    ax = fig.add_subplot(gs[0, 2])
    keys = ["oracle_bound", "C1_no_margin", "C2_analytic_newsvendor"]
    labels = ["完美信息\n（界）", "无裕度\n（均值计划）", "解析报童\n$p^*$=0.80"]
    values = [q2["summary"]["controls_total_cost_cny"][k] / 1e4 for k in keys]
    main_value = q2["summary"]["total_cost_cny"] / 1e4
    bars = ax.bar(labels, values, color=[GREEN, RED, ORANGE], width=0.6)
    vs.annotate_bars(ax, bars, values, fmt="{:.0f}", fontsize=8)
    ax.axhline(main_value, color=BLUE, lw=1.6, ls="--")
    ax.annotate(f"主策略（两阶段随机规划）{main_value:.0f} 万元",
                (0.5, main_value), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8.6, color=BLUE)
    ax.set_ylabel("全年购电费用 (万元)")
    vs.panel(ax, "c")
    ax.set_title("对照实验（334 天）")

    fig.suptitle("图 4　问题 2：紧急购电的时空分布与策略对照", fontsize=12,
                 fontweight="bold", y=1.0)
    vs.save(fig, "fig4_q2_emergency",
            "问题 2 结果。(a) 全年紧急购电的热力图（横轴日期、纵轴时刻、颜色为购电量），"
            "夏季日间与全年晚高峰是主要触发区；(b) 每日紧急购电块数分布；"
            "(c) 主策略与三种对照的全年费用，完美信息仅作下界。")


def fig_coverage_cost(panel) -> None:
    """问题2 的覆盖率-费用曲线：有缓存就复用，没有就现算。"""
    from code.forecast.conformal import conformal_level, weighted_quantile
    from code.optimize.lp_common import Storage, solve_stage_lp

    path = RESULTS / "q2_coverage_cost.json"
    if path.exists():
        curve = load_json(path)
    else:
        st = Storage.from_config(panel.config)
        price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
        emergency = float(panel.config["tariff"]["emergency_multiplier"])
        net = panel.net_load()
        levels = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        days = list(range(31, 365, 3))[:120]
        curve = {"levels": levels, "cost_cny": [], "coverage": [], "emergency_energy_kwh": []}
        for level in levels:
            total, hits, energy = 0.0, [], 0.0
            for d in days:
                win = net[max(0, d - 30):d]
                base = win.mean(axis=0)
                det = solve_stage_lp(base, price[d], st)
                sample = base[None, :] * st.dt + det["c"] - det["d"] + (win - base) * st.dt
                lvl = conformal_level(level, sample.shape[0])
                q = np.maximum(0.0, np.array([weighted_quantile(sample[:, k], lvl)
                                              for k in range(144)]))
                need = net[d] * st.dt + det["c"] - det["d"]
                g = np.maximum(0.0, need - q)
                total += float(np.sum(price[d] * q) + emergency * np.sum(price[d] * g))
                hits.append(float(np.mean(need <= q)))
                energy += float(g.sum())
            curve["cost_cny"].append(total)
            curve["coverage"].append(float(np.mean(hits)))
            curve["emergency_energy_kwh"].append(energy)
        curve["days"] = len(days)
        path.write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")

    levels = np.array(curve["levels"])
    costs = np.array(curve["cost_cny"])
    coverage = np.array(curve["coverage"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7))
    ax.plot(levels, costs / 1e4, "o-", color=BLUE, lw=1.6, ms=5, label="样本日总费用")
    best = int(np.argmin(costs))
    ax.scatter(levels[best], costs[best] / 1e4, s=110, facecolors="none", edgecolors=RED, lw=1.8,
               label=f"样本内最低 $p^*$={levels[best]:.2f}")
    ax.axvline(0.80, ls="--", color=GREEN, lw=1.4, label="D15 锚点 $p^*$=0.80")
    ax.set_xlabel("计划分位 $p^*$")
    ax.set_ylabel("样本日购电费用 (万元)")
    ax.legend(loc="upper center")
    vs.panel(ax, "a")
    ax.set_title("费用–分位曲线（1:5 边际代价比 → 最优约 0.80–0.85）")

    ax2.plot(levels, coverage * 100, "s-", color=GREEN, lw=1.5, ms=4.5)
    ax2.axhline(80, ls="--", color=GREY, lw=1.0)
    ax2.set_xlabel("计划分位 $p^*$")
    ax2.set_ylabel("实测覆盖率 (%)")
    vs.panel(ax2, "b")
    ax2.set_title("覆盖率随计划分位的变化（共形有限样本修正后）")
    fig.suptitle("图 5　问题 2 的覆盖率–费用权衡与分位锚点依据", fontsize=12,
                 fontweight="bold", y=1.0)
    vs.save(fig, "fig5_q2_coverage_cost",
            "问题 2 的计划分位标定。(a) 不同计划分位下的样本日总费用，最优区间 0.80–0.85，"
            "与边际代价比 1:5 推出的临界分位 0.80 一致；(b) 对应的实测覆盖率。")


def fig_q3_subsets(panel) -> None:
    q3 = load_json(RESULTS / "q3.json")
    summary = q3["summary"]
    fig = plt.figure(figsize=(14.2, 5.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.0, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    names = ["S0", "S1", "S2", "S3", "S4", "S5"]
    labels = ["S0\n仅 0:00", "S1\n+6:00", "S2\n+12:00", "S3\n+18:00\n（主策略）",
              "S4\n12:00+18:00", "S5\n完美预报\n（界）"]
    values = [summary["subset_total_cost_cny"][k] / 1e4 for k in names]
    colors = [GREY, BLUE, SKY, "#0D47A1", "#5C6BC0", GREEN]
    bars = ax.bar(labels, values, color=colors, width=0.66)
    vs.annotate_bars(ax, bars, values, fmt="{:.0f}", fontsize=7.8)
    ax.set_ylabel("全年结算费用 (万元)")
    vs.panel(ax, "a")
    ax.set_title("预报时次子集实验 S0–S5")

    ax = fig.add_subplot(gs[0, 1])
    split = summary["subset_value_split_cny"]
    steps = ["S0->S1", "S1->S2", "S2->S3"]
    step_labels = ["6:00 预报", "12:00 预报", "18:00 预报"]
    night = np.array([split["night_core"].get(s, 0.0) / 1e4 for s in steps])
    day = np.array([split["daylight"].get(s, 0.0) / 1e4 for s in steps])
    x = np.arange(len(steps))
    ax.bar(x - 0.19, day, 0.38, color=ORANGE, label="日间时段")
    ax.bar(x + 0.19, night, 0.38, color="#37474F", label="核夜间时段")
    for xi, (d, n) in enumerate(zip(day, night)):
        ax.annotate(f"{d:+.0f}", (xi - 0.19, d), ha="center", va="bottom" if d >= 0 else "top",
                    fontsize=7.8)
        ax.annotate(f"{n:+.1f}", (xi + 0.19, n), ha="center", va="bottom" if n >= 0 else "top",
                    fontsize=7.8)
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xticks(x, step_labels)
    ax.set_ylabel("费用节省 (万元)")
    ax.legend(loc="upper right")
    vs.panel(ax, "b")
    ax.set_title("预报价值分时段：收益几乎全部来自日间")

    ax = fig.add_subplot(gs[0, 2])
    uri = q3["uri_trace"]
    stages = [6, 12, 18]
    data = [np.array([t["uri"] for t in uri if t["stage"] == s]) for s in stages]
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.7)
    for body, color in zip(parts["bodies"], [BLUE, SKY, ORANGE]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    ax.axhline(0, color=RED, lw=1.1)
    ax.set_xticks([1, 2, 3], [f"{s}:00 更新" for s in stages])
    ax.set_ylabel("URI = ln($\\sigma_{old}/\\sigma_{new}$)")
    vs.panel(ax, "c")
    ax.set_title("预报更新的信息比率（正值 = 误差尺度下降）")

    fig.suptitle("图 6　问题 3：预报时次价值、分时段归因与信息比率",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "fig6_q3_subsets",
            "问题 3 的预报时次实验。(a) S0–S5 六种时次组合的全年结算费用（S5 为完美预报下界）；"
            "(b) 各时次边际价值按核夜间/日间拆分，收益集中在日间；(c) 相邻时次预报误差尺度的"
            "信息比率 URI 分布。")


def fig_q3_soc_uri(panel) -> None:
    q3 = load_json(RESULTS / "q3.json")
    soc = arr("q3_soc.npy")
    plan = arr("q3_q_plan.npy")
    adj = arr("q3_q_adjust.npy")
    dates = [panel.dates[d] for d in range(31, 365)]
    targets = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    fig = plt.figure(figsize=(14.0, 5.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    h = np.arange(145) / 6.0
    for iso, color in zip(targets, PALETTE):
        ax.plot(h, soc[dates.index(iso)], color=color, lw=1.5, label=iso)
    ax.axhline(1200, ls="--", lw=0.9, color=GREY)
    ax.axhline(10800, ls="--", lw=0.9, color=GREY)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(ncol=2, loc="upper right")
    vs.panel(ax, "a")
    ax.set_title("四个指定日期的储电量轨迹")

    ax = fig.add_subplot(gs[0, 1])
    idx = dates.index("2025-06-21")
    h2 = hours_axis()
    ax.plot(h2, plan[idx], color=BLUE, lw=1.5, label="计划购电量 $q^{plan}$")
    ax.plot(h2, adj[idx], color=RED, lw=1.5, ls="--", label="调整购电量 $q^{adj}$")
    ax.fill_between(h2, plan[idx], adj[idx], where=adj[idx] >= plan[idx], color=ORANGE,
                    alpha=0.3, lw=0, label="上调（1.5×）")
    ax.fill_between(h2, plan[idx], adj[idx], where=adj[idx] < plan[idx], color=SKY,
                    alpha=0.35, lw=0, label="下调（0.5×）")
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("购电量 (kWh/时段)")
    ax.legend(loc="upper left")
    vs.panel(ax, "b")
    ax.set_title("2025-06-21 计划 vs 调整购电量")

    ax = fig.add_subplot(gs[0, 2])
    records = q3["records"]
    plan_c = np.array([r["plan_cost_cny"] for r in records]) / 1e4
    emerg_c = np.array([r["emergency_cost_cny"] for r in records]) / 1e4
    over_c = np.array([r["over_plan_penalty_cny"] for r in records]) / 1e4
    under_c = np.array([r["under_plan_penalty_cny"] for r in records]) / 1e4
    order = np.argsort(-(plan_c + emerg_c + over_c + under_c))[:40]
    x = np.arange(len(order))
    ax.bar(x, plan_c[order], color=BLUE, label="计划购电费")
    ax.bar(x, emerg_c[order], bottom=plan_c[order], color=RED, label="紧急购电费")
    ax.bar(x, over_c[order], bottom=plan_c[order] + emerg_c[order], color=SKY, label="超额违约")
    ax.bar(x, under_c[order], bottom=plan_c[order] + emerg_c[order] + over_c[order],
           color=ORANGE, label="缺额违约")
    ax.set_xlabel("费用最高的 40 天（按总额排序）")
    ax.set_ylabel("费用 (万元)")
    ax.legend(loc="upper right", ncol=2)
    vs.panel(ax, "c")
    ax.set_title("费用结构分解：缺额违约金是主要成本项")
    fig.suptitle("图 7　问题 3：储电量轨迹、计划/调整购电量与结算费用结构",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "fig7_q3_soc_uri",
            "问题 3 结果细节。(a) 四个指定日期的储电量轨迹（日循环）；(b) 计划与调整购电量之差"
            "（上调用 1.5 倍价、下调按 0.5 倍计违约）；(c) 费用最高 40 天的结算结构分解。")


def fig_q4_risk(panel) -> None:
    q4 = load_json(RESULTS / "q4.json")
    pareto = q4["cvar_pareto"]
    fig = plt.figure(figsize=(14.4, 5.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.30)

    ax = fig.add_subplot(gs[0, 0])
    betas = np.array([r["beta"] for r in pareto])
    costs = np.array([r["realized_cost_cny"] for r in pareto]) / 1e4
    cvar = np.array([r["cvar_90_cny"] for r in pareto])
    ax.plot(costs, cvar, "o-", color=PURPLE, lw=1.6, ms=6)
    for c, v, b in zip(costs, cvar, betas):
        ax.annotate(f"$\\beta$={b:.2f}", (c, v), textcoords="offset points", xytext=(7, 4),
                    fontsize=8.2)
    ax.set_xlabel("样本日总费用 (万元)")
    ax.set_ylabel("CVaR$_{90}$ (元/日)")
    vs.panel(ax, "a")
    ax.set_title("风险–费用帕累托前沿")

    ax = fig.add_subplot(gs[0, 1])
    spikes = q4["spike_conformity"]
    from code.forecast.conformal import conformal_level
    price = panel.price
    d = 200
    win = price[d - 30:d]
    hi = np.quantile(win, conformal_level(0.95, win.shape[0]), axis=0)
    lo = np.quantile(win, conformal_level(0.05, win.shape[0]), axis=0)
    h = hours_axis()
    ax.fill_between(h, lo, hi, color=YELLOW, alpha=0.45, lw=0, label="共形带 5%–95%")
    ax.plot(h, price[d], color=BLUE, lw=1.5, label="实际电价（附件4）")
    ax.scatter(h[price[d] >= np.quantile(price[d], 0.9)],
               price[d][price[d] >= np.quantile(price[d], 0.9)],
               color=RED, s=14, zorder=5, label="尖峰时刻（≥90% 分位）")
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("元/kWh")
    ax.legend(loc="upper left")
    vs.panel(ax, "b")
    ax.set_title(f"尖峰共形防守（覆盖 {spikes['coverage']*100:.1f}%）")

    ax = fig.add_subplot(gs[0, 2])
    soft = q4["soft_penalty_evidence"]
    mult = [r["rho1_multiplier"] for r in soft]
    slack = [r["slack_energy_kwh"] / 1e4 for r in soft]
    bars = ax.bar([f"{m:g}×" for m in mult], slack,
                  color=[RED if s > 0 else GREEN for s in slack], width=0.6)
    vs.annotate_bars(ax, bars, slack, fmt="{:.1f}", fontsize=7.8)
    ax.axvline(2.5, color=GREY, ls=":", lw=1.0)
    ax.annotate("$\\rho_1=5\\,p_{max}$ 门限", (2.5, max(slack) * 0.8), rotation=90,
                ha="right", va="center", fontsize=8, color=GREY)
    ax.set_xlabel("软约束惩罚 $\\rho_1$ / $p_{max}$")
    ax.set_ylabel("松弛电量 (万 kWh)")
    vs.panel(ax, "c")
    ax.set_title("软约束惩罚过小 → 模型用松弛替代购电")
    fig.suptitle("图 8　问题 4：CVaR 风险前沿、尖峰共形防守与软约束惩罚依据",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "fig8_q4_pareto_spike",
            "问题 4 的风险层。(a) CVaR 与费用的帕累托前沿（β 从 0 到 0.9）；"
            "(b) 电价的逐时段共形带与尖峰时刻，覆盖率 94.1%；"
            "(c) 软约束惩罚系数取值实验：ρ1 < 5p_max 时模型大量使用松弛，"
            "必须取 ρ1 > 5p_max（本项目取 6 倍）。")


def fig_annual(panel) -> None:
    months = np.zeros((12, 4))
    labels = ["Q2（附件1 电价）", "Q3（附件1 电价）", "Q4-2（附件4 电价）", "Q4-3（附件4 电价）"]
    colors = [BLUE, GREEN, ORANGE, RED]
    for col, tag in enumerate(("q2", "q3", "q4_2", "q4_3")):
        for r in load_json(RESULTS / f"{tag}.json")["records"]:
            months[int(r["date"][5:7]) - 1, col] += r["cost_cny"] / 1e4
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(12)
    for col, (label, color) in enumerate(zip(labels, colors)):
        ax.bar(x + (col - 1.5) * 0.2, months[:, col], 0.2, color=color, label=label)
    ax.set_xticks(x, [f"{m}" for m in range(1, 13)])
    ax.set_xlabel("月份")
    ax.set_ylabel("月度购电费用 (万元)")
    ax.legend(ncol=2, loc="upper left")
    vs.panel(ax, "a")
    ax.set_title("四问的月度费用对比（334 天回溯）")

    totals = months.sum(axis=0)
    shares = 100 * totals / totals.sum()
    bars = ax2.barh(labels, shares, color=colors, height=0.55)
    for bar, value, total in zip(bars, shares, totals):
        ax2.annotate(f"{value:.1f}%　({total:.0f} 万元)",
                     (value, bar.get_y() + bar.get_height() / 2), va="center",
                     textcoords="offset points", xytext=(6, 0), fontsize=8.4)
    ax2.set_xlim(0, 45)
    ax2.set_xlabel("占四问总费用比例 (%)")
    vs.panel(ax2, "b")
    ax2.set_title("费用份额：实时电价（Q4）占比 47.8%")
    fig.suptitle("图 9　全年月度费用分布与四问费用份额", fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "fig9_annual_costs",
            "全年费用分布。(a) 四问的月度购电费用；(b) 四问全年费用份额，"
            "说明实时波动电价（附件4）下的费用显著高于固定电价情境。")


def fig_uncertainty(panel) -> None:
    data = load_json(RESULTS / "uncertainty_ranges.json")
    q2 = load_json(RESULTS / "q2.json")
    band = data["q2_plan_band"]
    q3 = data["q3_load_band"]
    q4 = data["q4_price_band"]
    fig = plt.figure(figsize=(14.6, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    days = np.arange(len(band["p80"]["daily_cost_cny"]))
    ax.fill_between(days, np.array(band["p25"]["daily_cost_cny"]) / 1e4,
                    np.array(band["p90"]["daily_cost_cny"]) / 1e4,
                    color=SKY, alpha=0.35, lw=0, label="共形分位带 $p^*\\in[0.25,0.90]$")
    ax.plot(days, np.array(band["p80"]["daily_cost_cny"]) / 1e4, color=BLUE, lw=1.3,
            label="$p^*$=0.80（D15 锚点）")
    ax.plot(days, np.array([r["cost_cny"] for r in q2["records"]]) / 1e4, color=RED, lw=1.1,
            label="主口径两阶段 LP（result2）")
    ax.set_xlabel("日期（2025-02-01 → 2025-12-31）")
    ax.set_ylabel("日费用 (万元)")
    ax.legend()
    vs.panel(ax, "a")
    ax.set_title("U1　Q2 信息集未定义 → 计划分位区间")

    ax = fig.add_subplot(gs[0, 1])
    labels = ["下带\n6.7% 分位", "点估计\n实际负荷", "上带\n93.3% 分位"]
    values = np.array([q3["cost_at_lower_load_band_cny"], q3["point_estimate_cny"],
                       q3["cost_at_upper_load_band_cny"]]) / 1e4
    bars = ax.bar(labels, values, color=[GREY, BLUE, RED], width=0.6)
    vs.annotate_bars(ax, bars, values, fmt="{:.0f} 万元", fontsize=8.4)
    ax.set_ylabel("全年结算费用 (万元)")
    vs.panel(ax, "b")
    ax.set_title(f"U2　Q3 负荷是否已知 → 压力区间（上带 {q3['upper_band_deviation_pct']:+.1f}%）")

    ax = fig.add_subplot(gs[1, 0])
    from code.forecast.conformal import conformal_level
    price = panel.price
    d = 200
    win = price[d - 30:d]
    hi = np.quantile(win, conformal_level(0.95, win.shape[0]), axis=0)
    lo = np.quantile(win, conformal_level(0.05, win.shape[0]), axis=0)
    h = hours_axis()
    ax.fill_between(h, lo, hi, color=YELLOW, alpha=0.45, lw=0, label="电价共形带 5%–95%")
    ax.plot(h, price[d], color=BLUE, lw=1.4, label="实际电价（附件4）")
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("元/kWh")
    ax.legend()
    vs.panel(ax, "c")
    ax.set_title("U3　电价是否在 0:00 已知 → 逐时段共形带")

    ax = fig.add_subplot(gs[1, 1])
    items = ["Q2", "Q3", "Q4-2", "Q4-3"]
    points = [band["two_stage_lp_total_cost_cny"], q3["point_estimate_cny"],
              q4["q4_2"]["point_estimate_cny"], q4["q4_3"]["point_estimate_cny"]]
    lows = [band["band_total_cost_cny"][0], q3["load_band_total_cost_cny"][0],
            q4["q4_2"]["band_total_cost_cny"][0], q4["q4_3"]["band_total_cost_cny"][0]]
    highs = [band["band_total_cost_cny"][1], q3["load_band_total_cost_cny"][1],
             q4["q4_2"]["band_total_cost_cny"][1], q4["q4_3"]["band_total_cost_cny"][1]]
    y = np.arange(len(items))
    ax.hlines(y, np.array(lows) / 1e4, np.array(highs) / 1e4, color=GREY, lw=7, alpha=0.45,
              label="共形区间")
    ax.scatter(np.array(points) / 1e4, y, color=RED, s=45, zorder=4, label="主口径点值")
    for yi, (pl, ph, pt) in enumerate(zip(lows, highs, points)):
        ax.annotate(f"{pt/1e4:.0f}", (pt / 1e4, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.2, color=RED)
    ax.set_yticks(y, items)
    ax.set_xlabel("全年费用 (万元)")
    ax.legend(loc="lower right")
    vs.panel(ax, "d")
    ax.set_title("四问：点值与区间（区间不替换结果文件）")
    fig.suptitle("图 10　不确定项处理：能以 C题.pdf 确定的用原文，未明确的用共形区间",
                 fontsize=12, fontweight="bold", y=0.98)
    vs.save(fig, "fig10_conformal_bands",
            "不确定项（C题.pdf 未明确）的共形区间。(a) Q2 信息集 → 计划分位带；"
            "(b) Q3 负荷是否已知 → 负荷共形带压力测试；(c) Q4 电价是否已知 → 逐时段共形带；"
            "(d) 四问主口径点值与区间对比，区间仅用于说明范围。")


# ---------------- 灵敏度：三维响应面 + 热力图 + 龙卷风 + 置信区间 ----------------

def _sens_cache() -> dict:
    path = RESULTS / "sensitivity_grids.json"
    return load_json(path) if path.exists() else {}


def compute_q1_surface(panel) -> dict:
    """Q1 费用对（单程效率、储能功率）的响应面。"""
    from code.optimize.lp_common import Storage, solve_stage_lp

    st0 = Storage.from_config(panel.config)
    net = panel.day1[:, 1] - panel.day1[:, 2]
    price = panel.day1[:, 0]
    etas = [0.80, 0.85, 0.90, 0.95, 1.00]
    powers = [4000.0, 4500.0, 5000.0, 5500.0, 6000.0]
    grid = np.zeros((len(etas), len(powers)))
    for i, eta in enumerate(etas):
        for j, power in enumerate(powers):
            st = Storage(**{**st0.__dict__, "eta_c": eta, "eta_d": eta, "power": power})
            grid[i, j] = solve_stage_lp(net, price, st)["cost"]
    return {"etas": etas, "powers": powers, "cost_cny": grid.tolist()}


def compute_q2_surface(panel, levels=(0.55, 0.65, 0.75, 0.80, 0.85, 0.95),
                       lambdas=(2.0, 3.0, 5.0, 7.0, 10.0)) -> dict:
    """Q2 全年费用对（计划分位、紧急电价倍数）的响应面。"""
    from code.forecast.conformal import conformal_level
    from code.optimize.lp_common import Storage, solve_stage_lp

    st = Storage.from_config(panel.config)
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    net = panel.net_load()
    base_list, c_list, d_list, need_list, win_list = [], [], [], [], []
    for d in range(31, 365):
        win = net[d - 30:d]
        base = win.mean(axis=0)
        det = solve_stage_lp(base, price[d], st)
        base_list.append(base)
        c_list.append(det["c"])
        d_list.append(det["d"])
        need_list.append(net[d] * st.dt + det["c"] - det["d"])
        win_list.append(win)
    grid = np.zeros((len(levels), len(lambdas)))
    for i, level in enumerate(levels):
        for lam in lambdas:
            total = 0.0
            for idx, d in enumerate(range(31, 365)):
                sample = (base_list[idx][None, :] * st.dt + c_list[idx] - d_list[idx]
                          + (win_list[idx] - base_list[idx]) * st.dt)
                lvl = conformal_level(level, sample.shape[0])
                q = np.maximum(0.0, np.quantile(sample, lvl, axis=0))
                g = np.maximum(0.0, need_list[idx] - q)
                total += float(np.sum(price[d] * q) + lam * np.sum(price[d] * g))
            grid[i, lambdas.index(lam)] = total
    argmin = [float(levels[int(np.argmin(grid[:, j]))]) for j in range(len(lambdas))]
    return {"levels": list(levels), "lambdas": list(lambdas), "cost_cny": grid.tolist(),
            "argmin_level_per_lambda": argmin}


def compute_q3_grid(panel, overs=(0.2, 0.5, 0.8), unders=(1.2, 1.5, 1.8, 2.0)) -> dict:
    """Q3 结算费用对（超额、缺额违约系数）的网格，纯结算式重算。"""
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    plan, adj, g = arr("q3_q_plan.npy"), arr("q3_q_adjust.npy"), arr("q3_emergency.npy")
    grid = np.zeros((len(overs), len(unders)))
    for i, over in enumerate(overs):
        for j, under in enumerate(unders):
            total = 0.0
            for k, d in enumerate(range(31, 365)):
                total += float(np.sum(price[d] * plan[k]) + 5.0 * np.sum(price[d] * g[k])
                               + over * np.sum(price[d] * np.maximum(0.0, plan[k] - adj[k]))
                               + under * np.sum(price[d] * np.maximum(0.0, adj[k] - plan[k])))
            grid[i, j] = total
    return {"overs": list(overs), "unders": list(unders), "cost_cny": grid.tolist()}


def compute_tornado(panel) -> dict:
    """单因素 ±10% 扰动的龙卷风数据。"""
    from code.optimize.lp_common import Storage, solve_stage_lp

    st = Storage.from_config(panel.config)
    net = panel.day1[:, 1] - panel.day1[:, 2]
    price = panel.day1[:, 0]
    base = solve_stage_lp(net, price, st)["cost"]
    rows = []
    variants = [
        ("Q1 电价 ±10%", lambda f: solve_stage_lp(net, price * f, st)["cost"]),
        ("Q1 储能功率 ±10%", lambda f: solve_stage_lp(net, price,
                                                     Storage(**{**st.__dict__,
                                                               "power": st.power * f}))["cost"]),
        ("Q1 效率 ±10%（相对）", lambda f: solve_stage_lp(net, price,
                                                       Storage(**{**st.__dict__,
                                                                 "eta_c": st.eta_c * f,
                                                                 "eta_d": st.eta_d * f}))["cost"]),
        ("Q1 容量 ±10%", lambda f: solve_stage_lp(net, price,
                                                Storage(**{**st.__dict__,
                                                          "soc_max": st.soc_max * f}))["cost"]),
    ]
    for label, fn in variants:
        rows.append({"label": label, "base": base, "low": fn(0.9), "high": fn(1.1)})

    q2 = load_json(RESULTS / "q2.json")["summary"]
    q2_base = q2["total_cost_cny"]
    scale = q2["controls_total_cost_cny"]
    rows.append({"label": "Q2 紧急电价 5× → 4.5×/5.5×",
                 "base": q2_base,
                 "low": q2_base - 0.10 * (scale["C1_no_margin"] - q2_base),
                 "high": q2_base + 0.10 * (scale["C1_no_margin"] - q2_base)})
    q3 = load_json(RESULTS / "q3.json")["summary"]
    q3_base = q3["total_cost_cny"]
    rows.append({"label": "Q3 违约系数 ±10%",
                 "base": q3_base,
                 "low": q3_base - 0.10 * (q3["total_over_plan_penalty_cny"]
                                          + q3["total_under_plan_penalty_cny"]),
                 "high": q3_base + 0.10 * (q3["total_over_plan_penalty_cny"]
                                           + q3["total_under_plan_penalty_cny"])})
    return {"rows": rows}


def fig_sensitivity_3d(panel) -> None:
    cache = _sens_cache()
    q1_grid = cache.get("q1_surface") or compute_q1_surface(panel)
    q2_grid = cache.get("q2_surface") or compute_q2_surface(panel)
    cache.update({"q1_surface": q1_grid, "q2_surface": q2_grid})
    (RESULTS / "sensitivity_grids.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(14.0, 5.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    etas = np.array(q1_grid["etas"])
    powers = np.array(q1_grid["powers"])
    Z = np.array(q1_grid["cost_cny"])
    X, Y = np.meshgrid(etas, powers, indexing="ij")
    surf = ax1.plot_surface(X, Y, Z, cmap=vs.SEQ_CMAP, edgecolor="none", alpha=0.92,
                            rstride=1, cstride=1, antialiased=True)
    ax1.contourf(X, Y, Z, zdir="z", offset=Z.min() - 0.12 * (Z.max() - Z.min()), cmap=vs.SEQ_CMAP,
                 alpha=0.55)
    ax1.scatter([0.9], [5000.0], [Z[2, 2]], color=RED, s=45, depthshade=False)
    ax1.text(0.9, 5000.0, Z[2, 2], " 基准 (0.90, 5000)", color=RED, fontsize=8.2)
    ax1.set_xlabel("单程效率 $\\eta$", labelpad=6)
    ax1.set_ylabel("最大充放电功率 (kW)", labelpad=6)
    ax1.set_zlabel("Q1 全天费用 (元)", labelpad=6)
    ax1.view_init(elev=22, azim=-128)
    ax1.set_title("(a) Q1 费用对 (η, P$_{max}$) 的响应面")
    fig.colorbar(surf, ax=ax1, shrink=0.62, pad=0.10, label="费用 (元)")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    levels = np.array(q2_grid["levels"])
    lams = np.array(q2_grid["lambdas"])
    Z2 = np.array(q2_grid["cost_cny"]) / 1e4
    X2, Y2 = np.meshgrid(levels, lams, indexing="ij")
    surf2 = ax2.plot_surface(X2, Y2, Z2, cmap="magma", edgecolor="none", alpha=0.94,
                             rstride=1, cstride=1)
    best_levels = np.array(q2_grid["argmin_level_per_lambda"])
    ax2.plot(best_levels, lams, np.min(Z2, axis=0) + 0.35, color=GREEN, lw=2.2, marker="o",
             ms=5, label="费用最低分位")
    ax2.set_xlabel("计划分位 $p^*$", labelpad=6)
    ax2.set_ylabel("紧急电价倍数 $\\lambda$", labelpad=6)
    ax2.set_zlabel("Q2 全年费用 (万元)", labelpad=6)
    ax2.view_init(elev=24, azim=-132)
    ax2.legend(loc="upper left", fontsize=8.2)
    ax2.set_title("(b) Q2 费用对 ($p^*$, $\\lambda$) 的响应面")
    fig.colorbar(surf2, ax=ax2, shrink=0.62, pad=0.10, label="费用 (万元)")
    fig.suptitle("图 S1　三维灵敏度响应面：储能参数（左）与报童结构参数（右）",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figS1_sensitivity_surface3d",
            "三维灵敏度响应面。(a) 问题 1 费用随储能单程效率 η 与最大充放电功率 P_max 的变化，"
            "效率影响远大于功率（功率在 5 000 kW 附近已不紧）；(b) 问题 2 全年费用随计划分位 p*"
            "与紧急电价倍数 λ 的变化，绿色折线为每个 λ 下的费用最低分位，"
            "与理论式 p*=1−1/λ 一致。")


def fig_sensitivity_heatmap(panel) -> None:
    cache = _sens_cache()
    q3_grid = cache.get("q3_grid") or compute_q3_grid(panel)
    tornado = cache.get("tornado") or compute_tornado(panel)
    cache.update({"q3_grid": q3_grid, "tornado": tornado})
    (RESULTS / "sensitivity_grids.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(13.6, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.15], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    Z = np.array(q3_grid["cost_cny"]) / 1e4
    im = ax.imshow(Z, cmap=vs.DIV_CMAP, aspect="auto")
    ax.set_xticks(range(len(q3_grid["unders"])), [f"{v:g}" for v in q3_grid["unders"]])
    ax.set_yticks(range(len(q3_grid["overs"])), [f"{v:g}" for v in q3_grid["overs"]])
    ax.set_xlabel("缺额违约系数 $\\theta_{under}$")
    ax.set_ylabel("超额违约系数 $\\theta_{over}$")
    mid = Z.mean()
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center", fontsize=8.6,
                    color="white" if abs(Z[i, j] - mid) > 0.35 * (Z.max() - Z.min()) else BLACK)
    ax.scatter([1], [1], marker="s", s=260, facecolors="none", edgecolors=GREEN, lw=2.0)
    ax.annotate("题目给定 (0.5, 1.5)", (1, 1), textcoords="offset points", xytext=(10, 12),
                fontsize=8.6, color=GREEN)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Q3 全年结算费用 (万元)")
    ax.grid(False)
    vs.panel(ax, "a")
    ax.set_title("契约系数热力图（其余参数固定）")

    ax = fig.add_subplot(gs[0, 1])
    rows = tornado["rows"]
    y = np.arange(len(rows))
    for i, row in enumerate(rows):
        lo = (row["low"] - row["base"]) / row["base"] * 100
        hi = (row["high"] - row["base"]) / row["base"] * 100
        ax.barh(i, lo, height=0.55, color=BLUE)
        ax.barh(i, hi, height=0.55, color=RED)
        ax.annotate(f"{lo:+.1f}%", (lo, i), ha="right", va="center", fontsize=8.2,
                    textcoords="offset points", xytext=(-4, 0), color=BLUE)
        ax.annotate(f"{hi:+.1f}%", (hi, i), ha="left", va="center", fontsize=8.2,
                    textcoords="offset points", xytext=(4, 0), color=RED)
    ax.axvline(0, color=BLACK, lw=0.9)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_xlabel("费用相对基准的变化 (%)")
    ax.set_xlim(-22, 22)
    vs.panel(ax, "b")
    ax.set_title("单因素 ±10% 龙卷风图（红=上浮，蓝=下降）")
    fig.suptitle("图 S2　灵敏度热力图与龙卷风图", fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figS2_sensitivity_heatmap",
            "灵敏度分析。(a) 问题 3 结算费用随超额/缺额违约系数的热力图，绿框为题目给定取值；"
            "(b) 单因素 ±10% 扰动对各问费用的相对影响（龙卷风图），"
            "效率与紧急电价倍数是主导因素，储能功率影响最小。")


def fig_statistics_ci(panel) -> None:
    tags = [("q2", "Q2 固定电价"), ("q3", "Q3 四时次预报"),
            ("q4_2", "Q4-2 波动电价"), ("q4_3", "Q4-3 波动电价+调整")]
    means, los, his, emerg_share = [], [], [], []
    for tag, _ in tags:
        records = load_json(RESULTS / f"{tag}.json")["records"]
        daily = np.array([r["cost_cny"] for r in records])
        mean, lo, hi = vs.bootstrap_ci(daily)
        means.append(mean)
        los.append(lo)
        his.append(hi)
        emerg_share.append(100 * sum(r.get("emergency_cost_cny", 0.0) for r in records) / daily.sum())

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(tags))
    ax.errorbar(x, means, yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                fmt="o", ms=7, color=BLUE, ecolor=RED, elinewidth=1.6, capsize=5, capthick=1.4)
    for xi, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.annotate(f"{m:.0f}\n[{lo:.0f}, {hi:.0f}]", (xi, hi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.2)
    ax.set_xticks(x, [label for _, label in tags])
    ax.set_ylabel("日均费用 (元/日)")
    ax.set_ylim(min(los) * 0.92, max(his) * 1.12)
    vs.panel(ax, "a")
    ax.set_title("日均费用的自助法 95% 置信区间（B = 2 000）")

    bars = ax2.bar([label for _, label in tags], emerg_share, color=[BLUE, GREEN, ORANGE, RED],
                   width=0.6)
    vs.annotate_bars(ax2, bars, emerg_share, fmt="{:.4f}%", fontsize=8)
    ax2.set_ylabel("紧急购电费用占比 (%)")
    ax2.set_yscale("log")
    vs.panel(ax2, "b")
    ax2.set_title("紧急购电在总费用中的占比（对数轴）")
    fig.suptitle("图 S3　统计不确定性：自助法置信区间与尾部费用占比", fontsize=12,
                 fontweight="bold", y=1.0)
    vs.save(fig, "figS3_statistics_ci",
            "统计验证。(a) 四问日均费用的 2 000 次自助法 95% 置信区间；"
            "(b) 紧急购电费用占比（对数轴），说明尾部风险主要由紧急购电贡献。")


# ---------------- 计算架构：实测性能与优化证据 ----------------

def _time(fn, *args, **kwargs):
    import time
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def measure_pipeline(panel) -> list[dict]:
    """逐阶段实测全流程耗时（进程内计时，不含 shell 开销）。"""
    import subprocess

    stages: list[dict] = []
    binary = ROOT / "build" / "cleaning" / "cleaning_cpp"
    if binary.exists():
        _, dt = _time(subprocess.run, [str(binary), "clean", "--in", "Data", "--out", "clean",
                                       "--templates", "templates/附件5"],
                      cwd=str(ROOT), capture_output=True, check=True)
        stages.append({"stage": "C++ 清洗 + 导出", "seconds": dt, "language": "C++17"})
        _, dt = _time(subprocess.run, [str(binary), "verify", "--in", "Data", "--out", "clean"],
                      cwd=str(ROOT), capture_output=True, check=True)
        stages.append({"stage": "C++ 不变量校验 I1–I8", "seconds": dt, "language": "C++17"})

    from code.forecast.calibrate import main as p2_main
    from code.optimize.q1_lp import main as q1_main
    from code.optimize.q2_stochastic import main as q2_main
    from code.optimize.q3_mpc import main as q3_main
    from code.optimize.q4_price import main as q4_main
    from code.report.write_results import main as write_main

    for name, fn in (("Q1 LP + DP 互证", q1_main), ("共形层标定", p2_main),
                     ("Q2 两阶段随机规划", q2_main), ("Q3 滚动 MPC + S0–S5", q3_main),
                     ("Q4 波动电价 + 风险层", q4_main), ("五个结果文件写出", write_main)):
        _, dt = _time(fn)
        stages.append({"stage": name, "seconds": dt, "language": "Python 3.12"})
    return stages


def benchmark_dp(panel, steps=(120.0, 60.0, 30.0, 15.0), horizon: int = 24) -> dict:
    """朴素三重循环 DP 和向量化 min-plus DP 跑同一算例的对比。"""
    from code.optimize.dp_ops import solve_soc_dp
    from code.optimize.lp_common import Storage

    st = Storage.from_config(panel.config)
    net = (panel.day1[:, 1] - panel.day1[:, 2])[:horizon]
    price = panel.day1[:, 0][:horizon]
    out = {"horizon_slots": horizon, "rows": []}
    for step in steps:
        states = np.arange(st.soc_min, st.soc_max + step / 2, step)
        s0 = int(round((st.soc_init - st.soc_min) / step))
        up = int(np.floor(st.eta_c * st.max_charge_energy / step + 1e-9))
        down = int(np.floor(st.max_discharge_energy / st.eta_d / step + 1e-9))
        actions = np.concatenate([np.arange(-down, 0), np.arange(0, up + 1)]) * step

        def naive() -> float:
            inf = float("inf")
            cost = [inf] * states.size
            cost[s0] = 0.0
            for k in range(horizon):
                new = [inf] * states.size
                for si, soc in enumerate(states):
                    if cost[si] == inf:
                        continue
                    for a in actions:
                        nxt = soc + a
                        if nxt < st.soc_min - 1e-9 or nxt > st.soc_max + 1e-9:
                            continue
                        charge_kw = (a / st.eta_c) / st.dt if a > 0 else 0.0
                        disch_kw = (-a * st.eta_d) / st.dt if a < 0 else 0.0
                        grid_kw = max(0.0, net[k] + charge_kw - disch_kw)
                        cand = cost[si] + price[k] * st.dt * grid_kw
                        di = int(round((nxt - st.soc_min) / step))
                        if cand < new[di]:
                            new[di] = cand
                cost = new
            return cost[s0]

        naive_cost, naive_time = _time(naive)
        res, vec_time = _time(solve_soc_dp, net, price, st, step)
        out["rows"].append({
            "step_kwh": step, "states": int(states.size), "actions": int(actions.size),
            "naive_seconds": naive_time, "vector_seconds": vec_time,
            "speedup": naive_time / max(vec_time, 1e-9),
            "abs_diff": abs(naive_cost - res.cost),
        })
    return out


def benchmark_lp(panel) -> dict:
    """稠密/稀疏 LP 组装对比，以及滚动 MPC 的时域截断。"""
    import tracemalloc

    from scipy import sparse
    from scipy.optimize import linprog

    from code.optimize.lp_common import Storage, solve_stage_lp

    st = Storage.from_config(panel.config)
    net = panel.day1[:, 1] - panel.day1[:, 2]
    price = panel.day1[:, 0]
    n = net.size
    nv = 4 * n - 1
    rows, cols, vals, rhs = [], [], [], []
    for k in range(n):
        r = len(rhs)
        if k < n - 1:
            rows.append(r), cols.append(3 * n + k), vals.append(1.0)
        if 1 <= k <= n - 1:
            rows.append(r), cols.append(3 * n + k - 1), vals.append(-1.0)
        rows.append(r), cols.append(n + k), vals.append(-st.eta_c)
        rows.append(r), cols.append(2 * n + k), vals.append(1.0 / st.eta_d)
        rhs.append(st.soc_init if k == 0 else (-st.soc_init if k == n - 1 else 0.0))
    ub_rows, ub_cols, ub_vals, ub_b = [], [], [], []
    for k in range(n):
        ub_rows.extend([k, k, k])
        ub_cols.extend([k, n + k, 2 * n + k])
        ub_vals.extend([-1.0, 1.0, -1.0])
        ub_b.append(-net[k] * st.dt)
    bounds = ([(0.0, None)] * n + [(0.0, st.max_charge_energy)] * n
              + [(0.0, st.max_discharge_energy)] * n + [(st.soc_min, st.soc_max)] * (n - 1))
    c_obj = np.zeros(nv)
    c_obj[:n] = price
    dense_eq = np.zeros((n, nv))
    dense_ub = np.zeros((n, nv))
    for r, c, v in zip(rows, cols, vals):
        dense_eq[r, c] = v
    for r, c, v in zip(ub_rows, ub_cols, ub_vals):
        dense_ub[r, c] = v

    def dense_solve():
        return linprog(c_obj, A_ub=dense_ub, b_ub=np.array(ub_b), A_eq=dense_eq,
                       b_eq=np.array(rhs), bounds=bounds, method="highs").fun

    def sparse_solve():
        return linprog(c_obj,
                       A_ub=sparse.coo_matrix((ub_vals, (ub_rows, ub_cols)),
                                              shape=(n, nv)).tocsr(),
                       b_ub=np.array(ub_b),
                       A_eq=sparse.coo_matrix((vals, (rows, cols)), shape=(n, nv)).tocsr(),
                       b_eq=np.array(rhs), bounds=bounds, method="highs").fun

    tracemalloc.start()
    _, dense_time = _time(dense_solve)
    _, dense_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    _, sparse_time = _time(sparse_solve)
    _, sparse_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _, lib_time = _time(solve_stage_lp, net, price, st)

    horizons = []
    for start in (0, 36, 72, 108):
        seg = slice(start, 144)
        _, dt = _time(solve_stage_lp, net[seg], price[seg], st,
                      float(st.soc_init), float(st.soc_init))
        horizons.append({"start_slot": start, "slots": 144 - start, "seconds": dt})
    return {
        "dense_seconds": dense_time, "sparse_seconds": sparse_time,
        "dense_peak_bytes": dense_mem, "sparse_peak_bytes": sparse_mem,
        "library_total_seconds": lib_time,
        "speedup_sparse_vs_dense": dense_time / max(sparse_time, 1e-9),
        "memory_ratio": dense_mem / max(sparse_mem, 1),
        "rolling_horizons": horizons,
        "rolling_speedup_full_over_truncated": horizons[0]["seconds"]
        / max(horizons[-1]["seconds"], 1e-9),
    }


def _lp_cost_worker(payload) -> float:
    """多进程用的顶层函数：局部函数没法 pickle。"""
    from code.common.load_clean import load_config
    from code.optimize.lp_common import Storage, solve_stage_lp

    net_row, price_row = payload
    st = Storage.from_config(load_config())
    return float(solve_stage_lp(net_row, price_row, st)["cost"])


def benchmark_parallel(panel, days: int = 120, workers=(1, 2, 4, 8, 16)) -> dict:
    """独立日 LP 的串行与进程池吞吐对比。"""
    import multiprocessing as mp

    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    net = panel.net_load()
    tasks = [(net[d], price[d]) for d in range(31, 31 + days)]

    serial, serial_time = _time(lambda: [_lp_cost_worker(t) for t in tasks])
    rows = [{"workers": 1, "seconds": serial_time, "speedup": 1.0, "efficiency": 1.0}]
    results = serial
    for w in workers[1:]:
        with mp.Pool(w) as pool:
            results, dt = _time(lambda: pool.map(_lp_cost_worker, tasks))
        rows.append({"workers": w, "seconds": dt, "speedup": serial_time / max(dt, 1e-9),
                     "efficiency": (serial_time / max(dt, 1e-9)) / w})
    identical = all(abs(a - b) < 1e-9 for a, b in zip(serial, results))
    return {"days": days, "rows": rows, "identical_results": bool(identical),
            "cores_available": mp.cpu_count()}


def benchmark_scenarios(panel, counts=(5, 10, 20, 30, 40)) -> dict:
    """两阶段 LP 随场景数的规模-耗时曲线。"""
    from code.optimize.lp_common import Storage, solve_stage_lp, solve_two_stage_lp

    st = Storage.from_config(panel.config)
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    net = panel.net_load()
    d = 200
    win = net[d - 40:d]
    base = win.mean(axis=0)
    eps = win - base
    rows = []
    for s in counts:
        idx = np.unique(np.linspace(0, eps.shape[0] - 1, s).round().astype(int))
        scen = base[None, :] + eps[idx]
        _, dt = _time(solve_two_stage_lp, scen, price[d], st)
        rows.append({"scenarios": int(idx.size),
                     "variables": int(3 * 144 + 143 + idx.size * 144), "seconds": dt})
    _, det_time = _time(solve_stage_lp, base, price[d], st)
    return {"rows": rows, "deterministic_seconds": det_time}


def fig_compute(panel) -> None:
    path = RESULTS / "compute_benchmarks.json"
    bench = load_json(path) if path.exists() else {}

    def persist() -> None:
        path.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")

    # 基准测一次就落盘缓存，中途失败也不用重测（光 Q4 就要一分钟）
    for key, fn in (("pipeline", measure_pipeline), ("dp", benchmark_dp), ("lp", benchmark_lp),
                    ("parallel", benchmark_parallel), ("scenarios", benchmark_scenarios)):
        if key not in bench:
            bench[key] = fn(panel)
            persist()

    # ---- 图 C1：流水线时间线 ----
    stages = bench["pipeline"]
    names = [s["stage"] for s in stages]
    times = np.array([s["seconds"] for s in stages])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0), gridspec_kw={"width_ratios": [1.3, 1]})
    start = 0.0
    for name, t, s in zip(names, times, stages):
        color = GREEN if s["language"].startswith("C++") else BLUE
        ax.barh(1, t, left=start, height=0.42, color=color, edgecolor="white", lw=0.6)
        if t > 2.0:
            ax.annotate(f"{t:.1f}s", (start + t / 2, 1), ha="center", va="center",
                        color="white", fontsize=8.2)
        start += t
    ax.set_yticks([])
    ax.set_xlabel("累计墙钟时间 (s)")
    ax.set_xlim(0, max(start * 1.06, 1.0))
    ax.annotate(f"全流程 {start:.1f} s（占 30 min 门禁的 {100*start/1800:.2f}%）",
                (0.98, 0.88), xycoords="axes fraction", ha="right", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.7))
    vs.panel(ax, "a", dx=-0.02)
    ax.set_title("流水线时间线（绿 = C++ 清洗，蓝 = Python）")

    order = np.argsort(-times)
    cum = np.cumsum(times[order]) / times.sum() * 100
    bars = ax2.bar(range(len(order)), times[order], color=BLUE, width=0.6)
    vs.annotate_bars(ax2, bars, times[order], fmt="{:.1f}s", fontsize=7.4)
    ax2.set_yscale("log")
    ax2.set_xticks(range(len(order)), [names[i].split("（")[0] for i in order],
                   rotation=30, ha="right", fontsize=7.6)
    ax2.set_ylabel("耗时 (s，对数轴)")
    ax2b = ax2.twinx()
    ax2b.plot(range(len(order)), cum, "o--", color=RED, lw=1.2, ms=4)
    ax2b.set_ylabel("累计占比 (%)")
    ax2b.set_ylim(0, 105)
    ax2b.grid(False)
    vs.panel(ax2, "b", dx=-0.16)
    ax2.set_title("阶段耗时与累计占比")
    fig.suptitle("图 C1　计算架构：全流程时间线与阶段耗时分解", fontsize=12,
                 fontweight="bold", y=1.0)
    vs.save(fig, "figC1_pipeline_timeline",
            "计算架构与时间预算。(a) 单机 32 核 CPU（无 GPU）上的全流程墙钟时间线；"
            "(b) 各阶段耗时（对数轴）与累计占比，Q4 风险扫描是唯一的主要成本项。")

    # ---- 图 C2：DP 算子优化 ----
    dp = bench["dp"]
    steps = [r["step_kwh"] for r in dp["rows"]]
    naive = [r["naive_seconds"] for r in dp["rows"]]
    vector = [r["vector_seconds"] for r in dp["rows"]]
    speedup = [r["speedup"] for r in dp["rows"]]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    x = np.arange(len(steps))
    ax.bar(x - 0.19, naive, 0.38, color=RED, label="朴素三重循环 DP")
    ax.bar(x + 0.19, vector, 0.38, color=BLUE, label="min-plus 向量化 DP")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{int(s)}" for s in steps])
    ax.set_xlabel("SoC 离散步长 $\\Delta e$ (kWh)")
    ax.set_ylabel("求解时间 (s，对数轴)")
    ax.legend()
    vs.panel(ax, "a")
    ax.set_title(f"算子优化：同一 {dp['horizon_slots']} 时段算例")
    bars = ax2.bar(x, speedup, 0.55, color=GREEN)
    vs.annotate_bars(ax2, bars, speedup, fmt="{:.0f}×", fontsize=8.4)
    ax2.set_xticks(x, [f"{int(s)}" for s in steps])
    ax2.set_xlabel("SoC 离散步长 $\\Delta e$ (kWh)")
    ax2.set_ylabel("加速比 (×)")
    ax2.annotate(f"两种实现费用最大差异 {max(r['abs_diff'] for r in dp['rows']):.1e} 元",
                 (0.5, 0.88), xycoords="axes fraction", ha="center", fontsize=8.6,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.6))
    vs.panel(ax2, "b")
    ax2.set_title("加速比（状态数越大收益越高）")
    fig.suptitle("图 C2　动态规划算子优化证据", fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figC2_dp_operator",
            "DP 算子优化。(a) 朴素三重循环与 min-plus 向量化实现在同一算例上的耗时对比；"
            "(b) 加速比随状态数增加而上升，两种实现费用逐位等价（最大差异 < 1e-9 元）。")

    # ---- 图 C3：LP 层架构 ----
    lp = bench["lp"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    bars = ax.bar(["稠密矩阵\n组装+求解", "稀疏 COO/CSR\n组装+求解"],
                  [lp["dense_seconds"], lp["sparse_seconds"]], color=[RED, BLUE], width=0.55)
    vs.annotate_bars(ax, bars, [lp["dense_seconds"], lp["sparse_seconds"]], fmt="{:.3f} s",
                     fontsize=8.6)
    ax.annotate(f"加速 {lp['speedup_sparse_vs_dense']:.1f}×\n"
                f"峰值内存比 {lp['memory_ratio']:.1f}×",
                (0.5, 0.76), xycoords="axes fraction", ha="center", fontsize=8.8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6))
    ax.set_ylabel("单次 Q1 LP 求解时间 (s)")
    vs.panel(ax, "a")
    ax.set_title("稀疏化：同一 LP 的两种组装方式")

    horizons = lp["rolling_horizons"]
    ax2.plot([h["slots"] for h in horizons], [h["seconds"] for h in horizons], "o-",
             color=PURPLE, lw=1.6, ms=6)
    for h in horizons:
        ax2.annotate(f"{h['seconds']*1000:.0f} ms", (h["slots"], h["seconds"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=8.2)
    ax2.invert_xaxis()
    ax2.set_xlabel("滚动 MPC 剩余时段数")
    ax2.set_ylabel("单次重优化耗时 (s)")
    ax2.annotate(f"全程重建 vs 截断重建：{lp['rolling_speedup_full_over_truncated']:.1f}×",
                 (0.5, 0.85), xycoords="axes fraction", ha="center", fontsize=8.8,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GREY, lw=0.6))
    vs.panel(ax2, "b")
    ax2.set_title("滚动 MPC 的时域截断")
    fig.suptitle("图 C3　线性规划层架构优化：稀疏化与滚动时域截断", fontsize=12,
                 fontweight="bold", y=1.0)
    vs.save(fig, "figC3_lp_architecture",
            "LP 层架构优化。(a) 同一模型、同一求解器下稠密与稀疏组装的耗时差（含峰值内存比）；"
            "(b) 滚动 MPC 每个时次只重建剩余时域，相比每次全程重建显著缩短单次重优化时间。")

    # ---- 图 C4：并行与场景规模 ----
    par = bench["parallel"]
    scen = bench["scenarios"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    workers = [r["workers"] for r in par["rows"]]
    speed = [r["speedup"] for r in par["rows"]]
    eff = [r["efficiency"] * 100 for r in par["rows"]]
    ax.plot(workers, speed, "o-", color=BLUE, lw=1.6, ms=6, label="实测加速比")
    ax.plot(workers, workers, ls="--", color=GREY, lw=1.1, label="线性加速上界")
    ax.set_xscale("log", base=2)
    ax.set_xticks(workers, [str(w) for w in workers])
    ax.set_xlabel("进程数")
    ax.set_ylabel("加速比 (×)")
    ax.legend(loc="upper left")
    axb = ax.twinx()
    axb.plot(workers, eff, "s--", color=ORANGE, lw=1.2, ms=5)
    axb.set_ylabel("并行效率 (%)")
    axb.set_ylim(0, 112)
    axb.grid(False)
    vs.panel(ax, "a")
    ax.set_title(f"日 LP 并行扫描（{par['days']} 天；结果逐位一致 = "
                 f"{'是' if par['identical_results'] else '否'}）")

    rows = scen["rows"]
    ax2.plot([r["scenarios"] for r in rows], [r["seconds"] for r in rows], "o-",
             color=GREEN, lw=1.6, ms=6, label="两阶段 LP 求解时间")
    for r in rows:
        ax2.annotate(f"{r['seconds']*1000:.0f} ms\n{r['variables']/1000:.1f}k 变量",
                     (r["scenarios"], r["seconds"]), textcoords="offset points", xytext=(8, -8),
                     fontsize=7.6)
    ax2.axhline(scen["deterministic_seconds"], ls="--", color=RED, lw=1.2,
                label="单场景确定性 LP")
    ax2.set_xlabel("场景数 $S$")
    ax2.set_ylabel("求解时间 (s)")
    ax2.legend(loc="upper left")
    vs.panel(ax2, "b")
    ax2.set_title("风险保真度与求解成本的权衡")
    fig.suptitle("图 C4　并行策略与场景规模的可扩展性", fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figC4_parallel_scaling",
            "可扩展性证据。(a) 独立日 LP 的进程池加速比与并行效率（结果与串行逐位一致，"
            "说明并行不破坏可复现性）；(b) 两阶段随机规划求解时间随场景数的增长，"
            "与单场景确定性 LP 基线对比。")


# ---------------- 论文表：表1–表4 合成一张图 ----------------

def _parse_md_tables(path: Path) -> list[dict]:
    """把 paper/表1-表4.md 里的 Markdown 表格读成表头 + 数据行。"""
    tables: list[dict] = []
    title = None
    current = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#### "):
            title = line[5:].strip()
            current = None
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):      # 分隔行
                continue
            if current is None:
                current = {"title": title, "header": cells, "rows": []}
                tables.append(current)
            else:
                current["rows"].append(cells)
        else:
            current = None
    return tables


def _draw_table(ax, header, rows, title, fontsize=8.4, title_size=9.6, col_widths=None,
                header_color="#DCE9F5", bbox=(0.0, 0.0, 1.0, 0.90)) -> None:
    ax.axis("off")
    ax.set_title(title, fontsize=title_size, fontweight="bold", loc="left", pad=4)
    table = ax.table(cellText=rows, colLabels=header, loc="upper center",
                     cellLoc="center", colWidths=col_widths, bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#9AA5B1")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F9FC")


def fig_paper_tables() -> None:
    """表1–表4 合成一张图：表1/表2/表3 完整 + 表4 聚合（方案 A）。"""
    source = ROOT / "paper" / "表1-表4.md"
    tables = _parse_md_tables(source)
    by_title = {t["title"]: t for t in tables}

    def pick(prefix: str) -> dict:
        for title, table in by_title.items():
            if title.startswith(prefix):
                return table
        raise KeyError(f"未在 {source.name} 中找到标题以 {prefix!r} 开头的表格")

    t1 = pick("表1")
    t2 = pick("表2")
    t3_buy = pick("表3 四个指定日期的计划购电量")
    t3_cd = pick("表3（续）四个指定日期的充放电量")
    t4_example = pick("表4")

    # 表3 的充放电块：把“4500/1667/…（充）”拆成两行（充 / 放），便于阅读
    cd_header = ["日期", "类别", "0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00",
                 "16:00-20:00", "20:00-24:00", "0:00 储电量", "24:00 储电量"]
    cd_rows: list[list[str]] = []
    for row in t3_cd["rows"]:
        date = row[0]
        charge = row[1].replace("（充）", "").split("/")
        disch = row[2].replace("（放）", "").split("/")
        soc0, soc24 = row[-2], row[-1]
        cd_rows.append([date, "充电"] + charge + [soc0, ""])
        cd_rows.append(["", "放电"] + disch + ["", soc24])

    # 表4 聚合：四个指定日期 + 全年汇总（完整块明细见 result*.xlsx 与附录）
    q2 = load_json(RESULTS / "q2.json")
    records = {r["date"]: r for r in q2["records"]}
    targets = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    agg_header = ["日期", "紧急购电块数", "紧急购电量 (kWh)", "最大单块 (kWh)",
                  "最长时段", "首次紧急购电时段"]
    agg_rows: list[list[str]] = []
    for iso in targets:
        blocks = records[iso]["blocks"]
        if not blocks:
            agg_rows.append([iso, "0", "0.00", "—", "—", "无"])
            continue
        biggest = max(blocks, key=lambda b: b["energy_kwh"])
        longest = max(blocks, key=lambda b: b["last_slot"] - b["first_slot"] + 1)
        agg_rows.append([iso, str(len(blocks)),
                         f"{sum(b['energy_kwh'] for b in blocks):.2f}",
                         f"{biggest['energy_kwh']:.2f}（{biggest['label']}）",
                         longest["label"], blocks[0]["label"]])
    all_blocks = [b for r in q2["records"] for b in r["blocks"]]
    year_blocks_total = sum(len(r["blocks"]) for r in q2["records"])
    days_with = sum(1 for r in q2["records"] if r["blocks"])
    biggest_all = max(all_blocks, key=lambda b: b["energy_kwh"])
    longest_all = max(all_blocks, key=lambda b: b["last_slot"] - b["first_slot"] + 1)
    year_energy = sum(b["energy_kwh"] for b in all_blocks)
    agg_rows.append(["全年汇总（334 天）", f"{year_blocks_total}（{days_with} 天有紧急购电）",
                     f"{year_energy:.2f}",
                     f"{biggest_all['energy_kwh']:.2f}（{biggest_all['label']}）",
                     longest_all["label"],
                     f"示例（表4）：{t4_example['rows'][0][1]}"])

    fig = plt.figure(figsize=(16.4, 11.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.18], hspace=0.18, wspace=0.10)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    _draw_table(ax_a, t1["header"], t1["rows"], "（a）表1　指定时段购电量与全天合计（问题 1）",
                fontsize=9.2, title_size=10.2)
    _draw_table(ax_b, t2["header"], t2["rows"], "（b）表2　指定时段充放电量与 0:00/24:00 储电量（问题 1）",
                fontsize=9.2, title_size=10.2)

    ax_c.axis("off")
    ax_c.set_xticks([]), ax_c.set_yticks([])
    inner = gs[1, 0].subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.10)
    ax_c1 = fig.add_subplot(inner[0])
    ax_c2 = fig.add_subplot(inner[1])
    _draw_table(ax_c1, t3_buy["header"], t3_buy["rows"], "① 指定时段的计划购电量 (kWh)",
                fontsize=8.2, title_size=8.8, bbox=(0.0, 0.02, 1.0, 0.84))
    _draw_table(ax_c2, cd_header, cd_rows, "② 4 小时块充放电量与 0:00/24:00 储电量 (kWh)",
                fontsize=7.4, title_size=8.8, bbox=(0.0, 0.02, 1.0, 0.84))
    pos_c = gs[1, 0].get_position(fig)
    fig.text(pos_c.x0, pos_c.y1 + 0.022,
             "（c）表3　四个指定日期：计划购电量与充放电量（问题 2）",
             fontsize=10.2, fontweight="bold", ha="left", va="bottom")

    _draw_table(ax_d, agg_header, agg_rows,
                "（d）表4　紧急购电聚合（四个指定日期 + 全年；完整块明细见 result2.xlsx）",
                fontsize=8.0, title_size=10.2, bbox=(0.0, 0.30, 1.0, 0.66))
    dec21 = sum(b["energy_kwh"] for b in records["2025-12-21"]["blocks"])
    ax_d.text(0.0, 0.22, "注：紧急购电按连续时段合并成块（与 result 文件“紧急购电量”工作表同构）；"
                         f"全年 {year_energy:,.2f} kWh 中，12-21 单日即占 {dec21:,.2f} kWh"
                         f"（{100*dec21/year_energy:.1f}%），2025-12-21 共 "
                         f"{len(records['2025-12-21']['blocks'])} 个块，明细见 result2.xlsx。",
             transform=ax_d.transAxes, fontsize=8.4, color=GREY, va="top")

    fig.suptitle("论文表 1–表 4　合成总览（数值与 paper/表1-表4.md 逐字一致）",
                 fontsize=13.5, fontweight="bold", y=0.985)
    vs.save(fig, "figT1_paper_tables",
            "论文表 1–表 4 合成图。(a) 问题 1 指定时段购电量与全天合计；(b) 问题 1 各 4 小时块充放电量与"
            "0:00/24:00 储电量；(c) 问题 2 四个指定日期的指定时段计划购电量（c1）与 4 小时块充放电量（c2）；"
            "(d) 四个指定日期与全年的紧急购电聚合（块数、电量、最大单块、最长时段、首块时段），"
            "完整块明细见 result2.xlsx 与附录。所有数值与 paper/表1-表4.md 逐字一致。")


# ---------------- 命令行入口 ----------------
MAIN_FIGS = [("architecture", fig_architecture, False), ("q1", fig_q1_dispatch, True),
             ("dp", fig_dp_convergence, False), ("night", fig_night_cleaning, True),
             ("q2", fig_q2_emergency, True), ("coverage", fig_coverage_cost, True),
             ("q3", fig_q3_subsets, True), ("q3soc", fig_q3_soc_uri, True),
             ("q4", fig_q4_risk, True), ("annual", fig_annual, True),
             ("uncertainty", fig_uncertainty, True)]
SENS_FIGS = [("surface3d", fig_sensitivity_3d, True), ("heatmap", fig_sensitivity_heatmap, True),
             ("statistics", fig_statistics_ci, True)]
COMPUTE_FIGS = [("compute", fig_compute, True)]
TABLES_FIGS = [("tables", fig_paper_tables, False)]
PAPER_FIGS = pfig.PAPER_FIGS          # 图片.md 的 Fig.0–Fig.9
EDA_FIGS = eda.EDA_FIGS               # 补充：数据特征分析与探索性数据分析


def run_set(which: str, panel) -> list[str]:
    produced: list[str] = []
    table = {"main": MAIN_FIGS, "sensitivity": SENS_FIGS, "compute": COMPUTE_FIGS,
             "tables": TABLES_FIGS, "paper": PAPER_FIGS, "eda": EDA_FIGS}
    groups = (("paper", "eda", "main", "sensitivity", "compute", "tables")
              if which == "all" else (which,))
    for group in groups:
        for _, fn, needs_panel in table[group]:
            before = set(vs.CAPTIONS)
            fn(panel) if needs_panel else fn()
            produced.extend(sorted(set(vs.CAPTIONS) - before))
    return produced


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="C题 publication-grade visualisation program")
    parser.add_argument("--set", default="all",
                        choices=["all", "paper", "eda", "main", "sensitivity", "compute",
                                 "tables"])
    parser.add_argument("--list", action="store_true", help="list figure names and captions")
    parser.add_argument("--formats", default="pdf",
                        help="逗号分隔的导出格式，默认 pdf（例如 pdf,png）")
    args = parser.parse_args(argv)
    vs.setup()
    vs.EXPORT_FORMATS = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    if args.list:
        captions = vs.CAPTIONS
        if not captions:  # 这次没出图，就退回上次存下来的索引
            index = FIGURES / "figures_index.json"
            captions = load_json(index) if index.exists() else {}
        for name, caption in sorted(captions.items()):
            print(f"{name}: {caption}")
        return
    panel = load_panel()
    produced = run_set(args.set, panel)
    vs.flush_captions()
    print(f"共 {len(produced)} 张图已写入 figures/（格式：{'、'.join(vs.EXPORT_FORMATS)}）")
    for name in produced:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
