"""Paper figure suite Fig.0-Fig.9 required by `图片.md`.

Each figure answers exactly one mathematical/engineering question and follows the
standard visual grammar of the energy-systems / probabilistic-forecasting / MPC
literature.  Data comes from `code.report.figure_data` (cached in
`results/paper_fig_data.json`); no model code is touched.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from code.common.load_clean import load_panel
from code.report import figure_data as fd
from code.report import viz_style as vs
from code.report.viz_style import BLACK, BLUE, GREY, GREEN, ORANGE, PALETTE, PURPLE, RED, SKY, YELLOW

RESULTS = fd.RESULTS

_DATA: dict | None = None


def data() -> dict:
    """Cached figure datasets (built once per process, no re-solving)."""
    global _DATA
    if _DATA is None:
        _DATA = fd.build(verbose=False)
    return _DATA


def _hours(n: int = 144) -> np.ndarray:
    return np.arange(n) / 6.0


# ---------------------------------------------------------------------------
# Fig.0 methodological framework
# ---------------------------------------------------------------------------

def fig0_framework() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def layer(y, title, operator, io, color, width=64, x=18):
        ax.add_patch(FancyBboxPatch((x, y), width, 7.6,
                                    boxstyle="round,pad=0.45,rounding_size=1.0",
                                    linewidth=1.15, edgecolor=color, facecolor="white"))
        ax.text(x + width / 2, y + 5.9, title, ha="center", va="center", fontsize=10.6,
                fontweight="bold", color=color)
        ax.text(x + width / 2, y + 3.1, operator, ha="center", va="center", fontsize=8.6,
                color=BLACK)
        ax.text(x + width / 2, y + 1.1, io, ha="center", va="center", fontsize=7.8,
                color=GREY)

    layer(53.0, "物理层 Physical layer", "PV · 储能 · 小区负荷 · 外网（不可倒送、日循环）",
          "input：附件1–4　→　output：标准面板 clean/*.npy（C++17 清洗 + 审计）", GREY)
    layer(42.6, "Q1　图-LP + KKT", "min Σp_t q_t   s.t. 节点平衡 · SOC 动态 · e₀=e_T",
          "operator：网络流等价 LP + 对偶影子价格 μ_t　→　output：计划购电/充放电 + μ_t", BLUE)
    layer(32.2, "Q2　MaxEnt 场景 + ACI 共形 + CVaR", "min Σp q + λ·E[紧急购电]   s.t. 场景 recourse",
          "operator：残差块 bootstrap + 共形分位(p*=0.80)　→　output：q_plan、覆盖区间", ORANGE)
    layer(21.8, "Q3　VOI + Trigger-MPC", "min 结算式（计划+紧急+0.5/1.5 违约）",
          "operator：四时次滚动更新 + NAV/URI 门控　→　output：q_plan/q_adj/紧急购电", GREEN)
    layer(11.4, "Q4　Soft-MPC + Repair", "min (1−β)E[C] + β CVaR₉₅   s.t. 软约束 always-feasible",
          "operator：波动电价 + 共形尖峰界 + 软松弛惩罚 ρ₁>5p_max　→　output：实时控制轨迹", PURPLE)
    layer(1.0, "真实运行 Real operation", "储能充放电执行 · SOC 安全域 [1200, 10800] kWh",
          "feedback：实测回灌（ACI/场景更新），形成闭环", RED)

    for y0, y1 in ((53.0, 50.2), (42.6, 39.8), (32.2, 29.4), (21.8, 19.0), (11.4, 8.6)):
        ax.add_patch(FancyArrowPatch((50, y0), (50, y1), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.3, color=GREY))
    ax.add_patch(FancyArrowPatch((84, 5), (95, 5), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.1, color=RED))
    ax.add_patch(FancyArrowPatch((95, 5), (95, 56.8), arrowstyle="-", mutation_scale=12,
                                 linewidth=1.1, color=RED, linestyle="--"))
    ax.add_patch(FancyArrowPatch((95, 56.8), (52, 56.8), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.1, color=RED, linestyle="--"))
    ax.text(96.5, 31, "反馈闭环\n(feedback)", rotation=90, va="center", ha="center",
            fontsize=8.4, color=RED)
    ax.text(2, 57.5, "(a) 四层模型递进", fontsize=10.5, fontweight="bold")
    ax.text(2, 30, "每一层：\nInput\n→ Operator\n→ Output", fontsize=8.6, color=GREY,
            va="center")
    ax.set_title("Fig.0　总体方法框架：物理层 → Q1 → Q2 → Q3 → Q4 → 真实运行（含反馈闭环）",
                 fontsize=12.4, fontweight="bold", pad=10)
    vs.save(fig, "pfig0_framework",
            "总体方法框架。每一层给出 Input → Mathematical Operator → Output；"
            "右侧虚线为实测回灌的闭环通道（ACI 覆盖率与场景集更新）。")


# ---------------------------------------------------------------------------
# Fig.1 dispatch + SOC
# ---------------------------------------------------------------------------

def fig1_dispatch_soc(panel) -> None:
    q1 = fd.load_json(RESULTS / "q1.json")
    st = fd.Storage.from_config(panel.config)
    price, load, pv = panel.day1[:, 0], panel.day1[:, 1], panel.day1[:, 2]
    q = np.array(q1["lp"]["q"]) / st.dt          # kWh -> kW
    c = np.array(q1["lp"]["c"]) / st.dt
    d = np.array(q1["lp"]["d"]) / st.dt
    soc = np.array(q1["lp"]["soc"])[1:]
    grid = q  # 购电功率
    h = _hours()

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.45, 1.0], hspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h, pv, color=ORANGE, lw=1.7, label="光伏 $P^{PV}_t$")
    ax.plot(h, load, color=RED, lw=1.7, label="负荷 $P^{Load}_t$")
    ax.plot(h, grid, color=BLUE, lw=1.9, label="购电 $P^{Grid}_t$")
    ax.bar(h, c, width=0.14, color=SKY, label="充电 $P^{ch}_t$")
    ax.bar(h, -d, width=0.14, color=GREEN, label="放电 $P^{dis}_t$")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(h, price, color=PURPLE, lw=1.1, ls="--", alpha=0.85, label="电价 $p_t$")
    ax2.set_ylabel("电价 (元/kWh)")
    ax2.grid(False)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("功率 (kW)")
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(handles + h2, labels + l2, ncol=3, loc="upper left")
    vs.panel(ax, "a", dx=-0.06)
    ax.set_title("能量调度轨迹：负荷由光伏、储能与外网购电共同满足（144×10 min）")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h, soc, color=GREEN, lw=2.0, label="储电量 $E_t$")
    ax.axhline(float(st.soc_max), ls="--", lw=1.0, color=GREY)
    ax.axhline(float(st.soc_min), ls="--", lw=1.0, color=GREY)
    ax.fill_between(h, float(st.soc_min), soc, color=GREEN, alpha=0.10, lw=0)
    ax.scatter([0, 24], [soc[0], soc[-1]], color=BLACK, zorder=5, s=22)
    ax.annotate(f"$E_0=E_{{144}}={soc[-1]:.0f}$ kWh", (0.4, soc[-1] - 1250), fontsize=8.8)
    ax.annotate(f"上限 {float(st.soc_max):.0f} kWh", (23.6, float(st.soc_max)), ha="right",
                va="bottom", fontsize=8.2, color=GREY)
    ax.annotate(f"下限 {float(st.soc_min):.0f} kWh", (23.6, float(st.soc_min)), ha="right",
                va="bottom", fontsize=8.2, color=GREY)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12000)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(loc="lower right")
    vs.panel(ax, "b", dx=-0.06)
    ax.set_title("储电量轨迹：全程落在安全域内，日循环约束 $E_0=E_{144}$ 成立")
    fig.suptitle(f"Fig.1　Q1 全天能量调度与 SOC 轨迹（全天费用 {q1['cost_cny']:.2f} 元）",
                 fontsize=12.2, fontweight="bold", y=0.98)
    vs.save(fig, "pfig1_dispatch_soc",
            "问题 1 的物理运行机制。(a) 负荷、光伏、购电与充放电功率（右轴为电价）；"
            "(b) 储电量轨迹，满足容量上下限与日循环。")


# ---------------------------------------------------------------------------
# Fig.2 price - shadow price - SOC coupling
# ---------------------------------------------------------------------------

def fig2_shadow_price(panel) -> None:
    payload = data()["shadow_price"]
    price = np.array(payload["price"])
    mu = np.array(payload["mu"])
    charge = np.array(payload["charge"]) / (1 / 6)
    disch = np.array(payload["discharge"]) / (1 / 6)
    soc = np.array(fd.load_json(RESULTS / "q1.json")["lp"]["soc"])[1:]
    h = _hours()
    st = fd.Storage.from_config(panel.config)

    fig = plt.figure(figsize=(12.8, 8.2))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.05, 0.95], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h, price, color=PURPLE, lw=1.7, label="市场电价 $p_t$")
    ax.scatter(h[charge > 1e-6], np.full(int((charge > 1e-6).sum()), price.max() * 0.97),
               marker="^", color=BLUE, s=20, label="充电时段")
    ax.scatter(h[disch > 1e-6], np.full(int((disch > 1e-6).sum()), price.max() * 0.90),
               marker="v", color=RED, s=20, label="放电时段")
    ax.set_ylabel("电价 (元/kWh)")
    ax.set_xlim(0, 24)
    ax.legend(ncol=3, loc="upper left", fontsize=8.4)
    vs.panel(ax, "a", dx=-0.06)
    ax.set_title(f"① 电价轨迹：充电时段中位价 {payload['charge_price_median']:.3f}、"
                 f"放电时段中位价 {payload['discharge_price_median']:.3f} 元/kWh")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(h, price, color=PURPLE, lw=1.6, label="市场电价 $p_t$")
    ax.plot(h, mu, color=GREEN, lw=1.8, label="影子价格 $\\mu_t$（能量平衡对偶）")
    ax.fill_between(h, mu, price, where=price > mu, color=RED, alpha=0.20, lw=0,
                    label="价高于边际价值：放电/少购")
    ax.fill_between(h, price, mu, where=price <= mu, color=BLUE, alpha=0.16, lw=0,
                    label="价低于边际价值：充电/多购")
    ax.set_ylabel("元/kWh")
    ax.set_xlim(0, 24)
    ax.legend(ncol=2, loc="upper left", fontsize=8.2)
    ax.annotate(f"互补松弛校验：{payload['purchase_slots']} 个购电时段全部满足 $\\mu_t=p_t$（精确成立），"
                f"全程 $\\mu_t\\leqq p_t$ 违反 {payload['mu_above_price_violations']} 次；"
                f"放电时段 {payload['discharge_price_above_mu']}/{payload['discharge_slots']} 满足 $p_t\\geq\\mu_t$",
                (0.5, 0.035), xycoords="axes fraction", ha="center", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=GREY, lw=0.6))
    vs.panel(ax, "b", dx=-0.06)
    ax.set_title("② 影子价格耦合：储能动作的经济解释（购电边际价值 = 市场价的时段才购电）")

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(h, soc, color=GREEN, lw=1.8, label="储电量 $E_t$")
    ax.axhline(st.soc_max, ls="--", lw=1.0, color=GREY)
    ax.axhline(st.soc_min, ls="--", lw=1.0, color=GREY)
    ax.scatter(h[charge > 1e-6], np.full(int((charge > 1e-6).sum()), st.soc_max + 250),
               marker="^", color=BLUE, s=20, label="充电时段")
    ax.scatter(h[disch > 1e-6], np.full(int((disch > 1e-6).sum()), st.soc_min - 250),
               marker="v", color=RED, s=20, label="放电时段")
    ax.set_ylim(st.soc_min - 800, st.soc_max + 900)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(ncol=3, loc="lower right", fontsize=8.2)
    vs.panel(ax, "c", dx=-0.06)
    ax.set_title("③ SOC 轨迹与充放电时段：低价充电、高价放电跨时段搬移能量")
    fig.suptitle("Fig.2　电价–影子价格–SOC 耦合图（KKT 的经济解释）",
                 fontsize=12.2, fontweight="bold", y=0.98)
    vs.save(fig, "pfig2_shadow_price",
            "对偶影子价格的经济解释。① 电价与充/放电时段；② 电价与影子价格 μ_t 的关系，"
            "购电时段 μ_t=p_t 精确成立（互补松弛），放电时段全部满足 p_t≥μ_t；"
            "③ SOC 轨迹。说明储能的充放动作可由边际价值关系解释，而非求解器巧合。")


# ---------------------------------------------------------------------------
# Fig.3 probabilistic forecast fan chart
# ---------------------------------------------------------------------------

def fig3_prediction_fan(panel) -> None:
    payload = data()["fan_chart"]
    h = _hours()
    fig = plt.figure(figsize=(13.0, 6.4))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    colors = {"q50": SKY, "q80": BLUE, "q90": ORANGE, "q95": RED}
    order = ["q95", "q90", "q80", "q50"]
    for key in order:
        band = payload["bands"][key]
        ax.fill_between(h, band["lower"], band["upper"], color=colors[key],
                        alpha=0.18 if key != "q50" else 0.35, lw=0,
                        label=f"{int(key[1:])}% 区间")
    ax.plot(h, payload["forecast"], color=BLACK, lw=1.6, ls="-.", label="点预测 $\\hat{PV}_t$（0:00 发布）")
    ax.plot(h, payload["actual"], color=GREEN, lw=1.9, label="实际 $PV_t$")
    viol = np.array(payload["aci"]["violations"])
    ax.scatter(h[viol], np.array(payload["actual"])[viol], color=RED, s=26, zorder=6,
               label=f"超出 ACI 上界（{int(viol.sum())} 个时段）")
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("光伏功率 (kW)")
    ax.legend(ncol=3, loc="upper left", fontsize=8.4)
    vs.panel(ax, "a", dx=-0.05)
    ax.set_title(f"预测扇形图：{payload['date']}　多层分位区间 + ACI 自适应上下界"
                 f"（当日 ACI 水平 {payload['aci']['level']:.3f}，"
                 f"全年实测覆盖率 {payload['achieved_coverage']*100:.1f}%"
                 f" vs 目标 {100*(1-payload['target_miss_rate']):.0f}%）")

    ax = fig.add_subplot(gs[1, 0])
    trace = payload["aci_trace"]
    x = np.arange(len(trace["miss"]))
    ax.plot(x, np.array(trace["miss"]) * 100, color=BLUE, lw=1.0, alpha=0.9,
            label="实测越界率 (%%), 目标 10%")
    ax.axhline(10.0, color=RED, ls="--", lw=1.3, label="目标越界率 10%")
    window = 30
    if x.size > window:
        kernel = np.ones(window) / window
        ax.plot(x[window - 1:], np.convolve(np.array(trace["miss"]) * 100, kernel,
                                            mode="valid"), color=ORANGE, lw=1.6,
                label=f"{window} 日滑动平均")
    ax.set_xlabel("2025 年日序（自 2 月 1 日起）")
    ax.set_ylabel("ACI 越界率 (%)")
    ax.legend(ncol=3, loc="upper right", fontsize=8.2)
    vs.panel(ax, "b", dx=-0.05)
    ax.set_title("ACI 在线反馈：越界率被逐步拉回名义水平（可靠性诊断）")
    fig.suptitle("Fig.3　Q2 概率预测扇形图与 ACI 覆盖率轨迹",
                 fontsize=12.2, fontweight="bold", y=0.98)
    vs.save(fig, "pfig3_prediction_fan",
            "概率预测表达。(a) 光伏实际值、点预测与 50/80/90/95% 共形分位区间，"
            "红点为穿透 90% ACI 上界的时段；(b) 全年 ACI 越界率及其滑动平均，"
            "显示自适应机制把覆盖率拉回名义水平。")


# ---------------------------------------------------------------------------
# Fig.4 reliability - sharpness - cost/risk
# ---------------------------------------------------------------------------

def fig4_reliability_sharpness(panel) -> None:
    rel = data()["reliability"]
    levels = np.array(rel["levels"])
    fig = plt.figure(figsize=(13.4, 4.9))
    gs = GridSpec(1, 3, figure=fig, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0.45, 1.0], [0.45, 1.0], color=GREY, ls="--", lw=1.1, label="理想校准线")
    for (name, cov), color in zip(rel["methods"].items(), PALETTE):
        ax.plot(levels, cov, "o-", color=color, lw=1.4, ms=4.5, label=name)
    ax.set_xlabel("名义覆盖率")
    ax.set_ylabel("实测覆盖率")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower right", fontsize=7.8)
    vs.panel(ax, "a")
    ax.set_title("Reliability 可靠性图")

    ax = fig.add_subplot(gs[0, 1])
    for (name, cov), color in zip(rel["methods"].items(), PALETTE):
        widths = rel["widths"][name]
        ax.plot(cov, widths, "s-", color=color, lw=1.4, ms=4.2, label=name)
    ax.set_xlabel("实测覆盖率")
    ax.set_ylabel("平均区间宽度 (kW)")
    ax.legend(fontsize=7.6)
    vs.panel(ax, "b")
    ax.set_title("Sharpness：覆盖率↑但宽度未失控膨胀")

    ax = fig.add_subplot(gs[0, 2])
    pareto = data()["pareto"]["rows"]
    cost = np.array([r["cost_cny"] for r in pareto]) / 1e4
    cvar = np.array([r["cvar95_cny"] for r in pareto])
    ax.plot(cost, cvar, "o-", color=PURPLE, lw=1.5, ms=6)
    for c, v, b in zip(cost, cvar, [r["beta"] for r in pareto]):
        ax.annotate(f"$\\beta$={b:.2f}", (c, v), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.8)
    ax.set_xlabel("样本期总费用 (万元)")
    ax.set_ylabel("CVaR$_{95}$ (元/日)")
    vs.panel(ax, "c")
    ax.set_title("经济–风险权衡（同一信息集）")
    fig.suptitle("Fig.4　Q2 概率标定与经济权衡：Reliability / Sharpness / Cost–Risk",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "pfig4_reliability_sharpness",
            "概率预测诊断。(a) Reliability 图：四种方法的名义覆盖率与实测覆盖率；"
            "(b) Sharpness：覆盖率提升时区间宽度的代价；"
            "(c) 同一信息集下的费用–CVaR 权衡，说明风险下降不是靠无限放宽区间。")


# ---------------------------------------------------------------------------
# Fig.5 VOI - penalty decision boundary
# ---------------------------------------------------------------------------

def fig5_voi_boundary(panel) -> None:
    voi = data()["voi"]
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    stage_style = {6: (BLUE, "o", "6:00 更新"), 12: (ORANGE, "s", "12:00 更新"),
                   18: (GREEN, "^", "18:00 更新")}
    limit = 0
    for stage, (color, marker, label) in stage_style.items():
        items = voi["stages"][str(stage)]
        x = np.array([it["penalty"] for it in items])
        y = np.array([it["voi"] for it in items])
        ax.scatter(x, y, s=16, alpha=0.55, color=color, marker=marker, label=label,
                   edgecolors="none")
        limit = max(limit, float(np.quantile(x, 0.995)), float(np.quantile(y, 0.995)))
    ax.plot([0, limit], [0, limit], color=RED, ls="--", lw=1.5, label="触发边界 VOI = $C_{adjust}$")
    ax.fill_between([0, limit], [0, limit], limit, color=GREEN, alpha=0.06, lw=0)
    ax.text(limit * 0.62, limit * 0.88, "VOI > 调整成本\n→ 值得更新", color=GREEN, fontsize=9,
            ha="center")
    ax.text(limit * 0.72, limit * 0.20, "VOI ≤ 调整成本\n→ 保持原计划", color=RED, fontsize=9,
            ha="center")
    ax.set_xlim(0, limit)
    ax.set_ylim(-limit * 0.25, limit)
    ax.set_xlabel("调整成本 $C_{adjust,k}$ (元)")
    ax.set_ylabel("信息价值 $VOI_k$ (元)")
    ax.legend(loc="upper left", fontsize=8.4)
    vs.panel(ax, "a")
    ax.set_title("VOI–调整成本决策平面（每个点 = 某日的某次更新）")

    ax = fig.add_subplot(gs[0, 1])
    summary = voi["summary"]
    stages = [6, 12, 18]
    voi_mean = [summary[str(s)]["voi_mean"] / 1e4 for s in stages]
    pen_mean = [summary[str(s)]["penalty_mean"] / 1e4 for s in stages]
    x = np.arange(len(stages))
    ax.bar(x - 0.19, voi_mean, 0.38, color=BLUE, label="平均 VOI")
    ax.bar(x + 0.19, pen_mean, 0.38, color=RED, label="平均调整成本")
    for xi, (v, p) in enumerate(zip(voi_mean, pen_mean)):
        ax.annotate(f"{v:.2f}", (xi - 0.19, v), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.2)
        ax.annotate(f"{p:.2f}", (xi + 0.19, p), ha="center", va="bottom" if p >= 0 else "top",
                    fontsize=8.2)
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xticks(x, [f"{s}:00\n触发率 {summary[str(s)]['trigger_rate']*100:.0f}%"
                      for s in stages])
    ax.set_ylabel("万元/日")
    ax.legend(loc="upper right")
    vs.panel(ax, "b")
    ax.set_title("各更新时次的平均信息价值与调整成本")
    fig.suptitle("Fig.5　Q3 的 VOI–惩罚决策边界（什么时候值得更新预报）",
                 fontsize=12.2, fontweight="bold", y=1.0)
    vs.save(fig, "pfig5_voi_boundary",
            "信息价值决策边界。(a) 每个点代表某日某次更新，横轴为调整成本、纵轴为信息价值，"
            "红色虚线 VOI = C_adjust 即触发边界；(b) 三个时次的平均 VOI、平均调整成本与触发率，"
            "说明 6:00/12:00 的更新几乎总是划算、而 18:00 的更新价值趋近于零。")


# ---------------------------------------------------------------------------
# Fig.6 rolling-horizon update trajectory
# ---------------------------------------------------------------------------

def fig6_rolling_update(panel) -> None:
    roll = data()["rolling"]
    h = _hours()
    plan = np.array(roll["plan"])
    adj = np.array(roll["adjusted"])
    realised = np.array(roll["realised"])
    soc = np.array(roll["soc"])
    fig = plt.figure(figsize=(13.0, 6.6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1.0], hspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h, plan, color=GREY, lw=1.7, ls="--", label="DA 计划 $q^{plan}_t$（0:00）")
    ax.step(h, adj, where="post", color=BLUE, lw=1.7, label="更新后计划 $q^{adj}_t$")
    ax.plot(h, realised, color=RED, lw=1.0, alpha=0.85, label="实际采购 $q^{adj}_t+g_t$")
    for tr in roll["triggers"]:
        ax.axvline(tr["slot"] / 6.0, color=GREEN, ls=":", lw=1.2)
        ax.annotate(f"{tr['label']}\nVOI {tr['voi']/1e4:.2f} 万元",
                    (tr["slot"] / 6.0, ax.get_ylim()[1] * 0.98), ha="center", va="top",
                    fontsize=8.2, color=GREEN,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=GREEN, lw=0.7))
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("购电量 (kWh/10 min)")
    ax.legend(loc="lower left", fontsize=8.6)
    vs.panel(ax, "a", dx=-0.05)
    ax.set_title(f"滚动更新轨迹：{roll['date']}（新信息 → VOI → 触发 → MPC 重优化 → 新计划）")

    ax = fig.add_subplot(gs[1, 0])
    h_soc = np.arange(145) / 6.0
    ax.plot(h_soc, soc, color=GREEN, lw=1.8, label="最终储电量轨迹")
    st_cfg = fd.Storage.from_config(panel.config)
    ax.axhline(st_cfg.soc_max, ls="--", lw=1.0, color=GREY)
    ax.axhline(st_cfg.soc_min, ls="--", lw=1.0, color=GREY)
    for tr in roll["triggers"]:
        ax.axvline(tr["slot"] / 6.0, color=GREEN, ls=":", lw=1.0)
    ax.set_xlim(0, 24)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(loc="lower right", fontsize=8.6)
    vs.panel(ax, "b", dx=-0.05)
    ax.set_title(f"对应 SOC 轨迹（当日子集费用：S0={roll['day_subset_costs']['S0']/1e4:.2f} "
                 f"→ S3={roll['day_subset_costs']['S3']/1e4:.2f} 万元）")
    fig.suptitle("Fig.6　Q3 滚动时域闭环：计划 → 触发 → 更新 → 执行",
                 fontsize=12.2, fontweight="bold", y=0.98)
    vs.save(fig, "pfig6_rolling_update",
            "滚动时域闭环证据。(a) 0:00 计划、更新后计划与实际采购量，绿虚线为三个更新时次"
            "（标注当日该时次的信息价值 VOI）；(b) 对应的 SOC 轨迹与安全域。")


# ---------------------------------------------------------------------------
# Fig.7 real-time control + SOC safety envelope
# ---------------------------------------------------------------------------

def fig7_realtime_control(panel) -> None:
    rt = data()["realtime"]
    h = _hours()
    price = np.array(rt["price"])
    load = np.array(rt["load"])
    pv = np.array(rt["pv"])
    net_base = np.array(rt["baseline"]["net"]) / (1 / 6)
    net_mpc = np.array(rt["mpc"]["net"]) / (1 / 6)
    soc_base = np.array(rt["baseline"]["soc"])
    soc_mpc = np.array(rt["mpc"]["soc"])
    lim = rt["soc_limits"]

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.9, 1.15, 1.0], hspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(h, price, color=PURPLE, lw=1.6, label="实时电价 $p^{RT}_t$")
    ax.set_ylabel("元/kWh")
    ax.set_xlim(0, 24)
    ax.legend(loc="upper left", fontsize=8.4)
    vs.panel(ax, "a", dx=-0.05)
    ax.set_title(f"实时电价与净负荷（{rt['date']}）")
    ax2 = ax.twinx()
    ax2.plot(h, load - pv, color=GREY, lw=1.0, ls="--", label="净负荷")
    ax2.set_ylabel("净负荷 (kW)")
    ax2.grid(False)
    ax2.legend(loc="upper right", fontsize=8.4)

    ax = fig.add_subplot(gs[1, 0])
    ax.step(h, net_base, where="post", color=GREY, lw=1.4, label="承诺基线 $u^{base}_t$")
    ax.step(h, net_mpc, where="post", color=BLUE, lw=1.8, label="MPC 控制 $u^{MPC}_t$")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xlim(0, 24)
    ax.set_ylabel("控制动作 $u_t$ (kW)")
    ax.legend(loc="upper left", fontsize=8.6)
    vs.panel(ax, "b", dx=-0.05)
    ax.set_title(f"控制轨迹对比：MPC 通过滚动重优化改变充放动作"
                 f"（动作总变差 {rt['baseline']['tv']/1e4:.2f} → {rt['mpc']['tv']/1e4:.2f} 万 kW）")

    ax = fig.add_subplot(gs[2, 0])
    h_soc = np.arange(145) / 6.0
    ax.plot(h_soc, soc_base, color=GREY, lw=1.3, ls="--", label="基线 SOC")
    ax.plot(h_soc, soc_mpc, color=GREEN, lw=1.9, label="MPC SOC")
    ax.fill_between([0, 24], lim[0], lim[1], color=GREEN, alpha=0.05, lw=0)
    ax.axhline(lim[0], color=RED, ls="--", lw=1.1)
    ax.axhline(lim[1], color=RED, ls="--", lw=1.1)
    ax.annotate(f"安全下限 {lim[0]:.0f} kWh", (0.2, lim[0]), va="bottom", fontsize=8.2, color=RED)
    ax.annotate(f"安全上限 {lim[1]:.0f} kWh", (0.2, lim[1]), va="bottom", fontsize=8.2, color=RED)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12000)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("储电量 (kWh)")
    ax.legend(loc="lower right", fontsize=8.6)
    vs.panel(ax, "c", dx=-0.05)
    ax.set_title("SOC 安全域：MPC 轨迹全程不越界（越界时段数 = "
                 f"{int(((soc_mpc < lim[0]) | (soc_mpc > lim[1])).sum())}）")
    fig.suptitle("Fig.7　Q4 实时控制轨迹与 SOC 安全域",
                 fontsize=12.2, fontweight="bold", y=0.98)
    vs.save(fig, "pfig7_rt_control_soc",
            "实时控制与安全约束。(a) 波动电价与净负荷；(b) 承诺基线与 MPC 的控制动作对比，"
            "标注动作总变差；(c) 两种策略的 SOC 轨迹与安全域，MPC 全程不越界。")


# ---------------------------------------------------------------------------
# Fig.8 cost - risk - degradation Pareto
# ---------------------------------------------------------------------------

def fig8_pareto_cost_risk_degradation(panel) -> None:
    rows = data()["pareto"]["rows"]
    priced = data()["degradation_priced"]["rows"]
    sens = data()["degradation_sensitivity"]
    cost = np.array([r["cost_cny"] for r in rows]) / 1e4
    cvar = np.array([r["cvar95_cny"] for r in rows])
    fade = np.array([r["annual_fade_pct"] for r in rows])
    beta = np.array([r["beta"] for r in rows])

    fig = plt.figure(figsize=(15.4, 5.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.0], wspace=0.26)
    ax = fig.add_subplot(gs[0, 0])
    sizes = 60 + 300 * (fade - fade.min()) / max(fade.max() - fade.min(), 1e-9)
    sc = ax.scatter(cost, cvar, s=sizes, c=fade, cmap="viridis", edgecolors=BLACK, linewidths=0.7,
                    zorder=4)
    ax.plot(cost, cvar, color=GREY, lw=1.0, ls="--", zorder=2)
    for c, v, b in zip(cost, cvar, beta):
        ax.annotate(f"$\\beta$={b:.2f}", (c, v), textcoords="offset points", xytext=(8, -12),
                    fontsize=8.0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("雨流 SOH 年衰减 (%)")
    ax.set_xlabel("样本期电费 (万元)")
    ax.set_ylabel("CVaR$_{95}$ (元/日)")
    vs.panel(ax, "a")
    ax.set_title("风险层 Pareto（气泡/颜色 = 真实 SOH 衰减）")

    ax = fig.add_subplot(gs[0, 1])
    rho = np.array([r["rho_cny_per_kwh"] for r in priced])
    elec = np.array([r["electricity_cost_cny"] for r in priced]) / 1e4
    total = np.array([r["total_cost_cny"] for r in priced]) / 1e4
    pfade = np.array([r["annual_fade_pct"] for r in priced])
    ax.plot(pfade, total, "o-", color=BLUE, lw=1.7, ms=6, label="总成本（电费 + 退化）")
    ax.plot(pfade, elec, "s--", color=ORANGE, lw=1.3, ms=5, label="仅电费")
    best = int(np.argmin(total))
    ax.scatter([pfade[best]], [total[best]], s=220, facecolors="none", edgecolors=RED, lw=1.7,
               zorder=5)
    ax.annotate(f"最优退化定价 $\\rho$={rho[best]:.2f} 元/kWh\n"
                f"衰减 {pfade[best]:.1f}%、总成本 {total[best]:.0f} 万元",
                (pfade[best], total[best]), textcoords="offset points", xytext=(10, 14),
                fontsize=8.4, color=RED)
    for x, y, r in zip(pfade, total, rho):
        ax.annotate(f"$\\rho$={r:.2f}", (x, y), textcoords="offset points", xytext=(6, -12),
                    fontsize=7.8, color=BLUE)
    ax.set_xlabel("雨流 SOH 年衰减 (%)")
    ax.set_ylabel("样本期成本 (万元)")
    ax.legend(loc="upper right", fontsize=8.4)
    vs.panel(ax, "b")
    ax.set_title("退化定价前沿：忽略退化并非经济最优")

    ax3 = fig.add_subplot(gs[0, 2], projection="3d")
    p3 = ax3.scatter(cost, cvar, fade, s=70, c=beta, cmap="plasma", depthshade=False,
                     edgecolors=BLACK, linewidths=0.5)
    ax3.plot(cost, cvar, fade, color=GREY, lw=1.0, ls="--")
    ax3.set_xlabel("电费 (万元)", labelpad=6)
    ax3.set_ylabel("CVaR$_{95}$ (元/日)", labelpad=6)
    ax3.set_zlabel("SOH 年衰减 (%)", labelpad=6)
    ax3.view_init(elev=22, azim=-125)
    ax3.set_title("三维目标空间（雨流 SOH）")
    fig.suptitle("Fig.8　成本–风险–退化三重权衡：雨流计数 + Wöhler 寿命曲线的真实 SOH",
                 fontsize=12.2, fontweight="bold", y=1.0)
    vs.save(fig, "pfig8_pareto_cost_risk_deg",
            "多目标权衡（退化由雨流计数 + Wöhler 寿命曲线给出，非吞吐代理）。(a) 费用–CVaR 平面，"
            "气泡大小与颜色为年容量衰减；(b) 退化定价前沿：横轴为年衰减、纵轴为成本，"
            "把退化定价 ρ 增大可同时降低衰减与总成本（最优约 ρ=0.20 元/kWh）；"
            "(c) 费用–CVaR–SOH 三维目标空间。参数敏感性 "
            f"N100∈[4000, 8000]、k∈[0.9, 1.3] 时年衰减区间为 "
            f"{min(v['annual_fade_pct'] for v in sens['grid'].values()):.1f}%–"
            f"{max(v['annual_fade_pct'] for v in sens['grid'].values()):.1f}%。")


# ---------------------------------------------------------------------------
# Fig.9 ablation performance heatmap
# ---------------------------------------------------------------------------

def fig9_ablation_heatmap(panel) -> None:
    ablation = data()["ablation"]
    rows, labels = ablation["rows"], ablation["labels"]
    metric_labels = ablation["metric_labels"]
    variants = [k for k in ("M0", "M1", "M2", "M3", "M4") if k in rows]
    metrics = list(metric_labels)
    raw = np.array([[rows[v][m] for m in metrics] for v in variants], dtype=float)
    norm = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        span = col.max() - col.min()
        norm[:, j] = 0.5 if span <= 0 else (col - col.min()) / span

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                                  gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.26})
    im = ax.imshow(norm, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), [metric_labels[m] for m in metrics], fontsize=8.2)
    ax.set_yticks(range(len(variants)), [labels[v] for v in variants], fontsize=8.6)
    for i in range(len(variants)):
        for j, m in enumerate(metrics):
            value = raw[i, j]
            text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:,.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.0,
                    color="white" if norm[i, j] > 0.75 or norm[i, j] < 0.12 else BLACK)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("列内归一化（0 = 该列最优，1 = 最差）")
    ax.grid(False)
    vs.panel(ax, "a", dx=-0.12)
    ax.set_title("消融性能热力图：每个模块解决了什么问题")

    base = raw[0]
    x = np.arange(len(metrics))
    for i, v in enumerate(variants[1:], start=1):
        delta = 100 * (raw[i] - base) / np.where(np.abs(base) < 1e-9, 1, np.abs(base))
        ax2.plot(x, delta, "o-", color=PALETTE[(i - 1) % len(PALETTE)], lw=1.4, ms=4.4,
                 label=labels[v])
    ax2.axhline(0, color=BLACK, lw=0.9)
    ax2.set_xticks(x, [metric_labels[m] for m in metrics], fontsize=8.0)
    ax2.set_ylabel("相对 M0 的变化 (%)")
    ax2.set_yscale("symlog", linthresh=10)
    ax2.legend(fontsize=7.8, loc="lower left")
    vs.panel(ax2, "b", dx=-0.14)
    ax2.set_title("逐模块增量（相对 M0，symlog 轴）")
    fig.suptitle("Fig.9　全局消融：M0→M4 各模块对费用/风险/寿命/弃光的贡献",
                 fontsize=12.2, fontweight="bold", y=1.0)
    vs.save(fig, "pfig9_ablation_heatmap",
            "模块消融证据。(a) M0–M4 × 六项指标的性能热力图（列内归一化，同时标注原始值）；"
            "(b) 各模块相对 M0 的增量，说明物理建模、共形裕度、场景随机规划、滚动更新与"
            "CVaR 风险层各自的贡献。")


PAPER_FIGS = [
    ("framework", fig0_framework, False),
    ("dispatch", fig1_dispatch_soc, True),
    ("shadow", fig2_shadow_price, True),
    ("fan", fig3_prediction_fan, True),
    ("reliability", fig4_reliability_sharpness, True),
    ("voi", fig5_voi_boundary, True),
    ("rolling", fig6_rolling_update, True),
    ("realtime", fig7_realtime_control, True),
    ("pareto", fig8_pareto_cost_risk_degradation, True),
    ("ablation", fig9_ablation_heatmap, True),
]
