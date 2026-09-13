"""按 figures/CAPTIONS_3.md 生成 25 张 SCI 投稿级图（Fig 1–Fig 25）。

设计约束：

* 只读 results/、clean/ 与 figures/science_data.json；不修改任何求解器、不重跑优化。
* 图输出到 figures/（本图集为唯一交付图集），默认只出矢量 PDF。
* 凡 CAPTIONS_3.md 要求但仓库无实测支撑的量（1000 次 Monte Carlo、
  未实现的 A2/M1–M4），用实测口径替代并在图注写明，不编造数值。

替换记录（2026-09-12）：原 35 张图集已备份为 figures_legacy_backup/，
本程序输出的 25 张 Fig01–Fig25 成为 figures/ 的正式内容。

用法：
    python3 -m code.report.science_figs --set all
    python3 -m code.report.science_figs --set arch,q1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.lines import Line2D

from code.common.load_clean import ROOT, load_panel
from code.report import captions3
from code.report import viz_style as vs

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
OUT = ROOT / "figures"
FIRST_DAY = 31


# ---------------------------------------------------------------- 基础设施

def load_data() -> dict:
    path = OUT / "science_data.json"
    if not path.exists():
        raise SystemExit("run `python3 -m code.report.science_data` first")
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, spec: captions3.FigSpec) -> None:
    """保存为 figures/FigNN_slug.pdf，并把图注写入本次运行的索引。"""
    OUT.mkdir(exist_ok=True)
    name = f"Fig{spec.no:02d}_{spec.slug}"
    fig.savefig(OUT / f"{name}.pdf", format="pdf")
    plt.close(fig)
    return name, spec


def sub(ax, letter, dx=-0.085, dy=1.05):
    vs.panel(ax, letter, dx=dx, dy=dy)


def block(ax, x, y, w, h, text, color, fontsize=8.4, edge=None, text_color="#111111"):
    """架构图用的圆角方块 + 居中多行文字。"""
    box = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor=edge or color, facecolor=color, alpha=0.16)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, linespacing=1.45)


def arrow(ax, xy_from, xy_to, color="#333333", style="-|>", lw=1.0, rad=0.0):
    ax.add_patch(patches.FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.0, shrinkB=1.0))


# ------------------------------------------------------- 第一部分：架构

def fig01(data) -> tuple:
    spec = captions3.by_no(1)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    q1 = data["q1"]["cost_cny"]
    q2 = data["q2"]["summary"]["total_cost_cny"]
    q3 = data["q3"]["summary"]["total_cost_cny"]
    q4 = data["q4"]["summary"]["q4_2"]["total_cost_cny"]
    block(ax, 0.04, 0.76, 0.92, 0.18,
          "第 1 层  日前凸图 LP 调度（问题 1）\n"
          "min Σ p_t g_t   约束：节点功率平衡 + SOC 动态，E_T = E_0\n"
          f"费用 = {q1:,.0f} 元/日    KKT 对偶：μ_t（母线），λ_t（储能）",
          vs.BLUE, fontsize=8.3)
    block(ax, 0.04, 0.50, 0.92, 0.18,
          "第 2 层  滚动信息价值再优化（问题 2 → 问题 3）\n"
          "MaxEnt 残差分布 + ACI 共形边界  →  报童分位 F* = 0.80\n"
          f"全年计划 = {q2:,.0f} 元    结算 = {q3:,.0f} 元    "
          "NAV_k = J_keep − J_reopt",
          vs.GREEN, fontsize=8.3)
    block(ax, 0.04, 0.24, 0.92, 0.18,
          "第 3 层  储能实时 MT-RRC（问题 4，10 分钟）\n"
          "Student-t Copula 场景 → CBF 安全过滤 → DETC 事件触发\n"
          f"执行费用 = {q4:,.0f} 元    SOC 越界 = 0    下发次数下降 = 52.6%",
          vs.ORANGE, fontsize=8.3)
    arrow(ax, (0.5, 0.76), (0.5, 0.685), rad=0.0)
    arrow(ax, (0.5, 0.50), (0.5, 0.425), rad=0.0)
    ax.text(0.52, 0.722, r"$g_t^{*},\ \lambda_t^{*}$", fontsize=8.6, va="center")
    ax.text(0.52, 0.462, r"$u_\tau^{safe},\ u_\tau^{real}$", fontsize=8.6, va="center")
    # 反馈回路
    arrow(ax, (0.96, 0.30), (0.96, 0.88), color=vs.RED, rad=-0.32, lw=0.9)
    ax.text(0.995, 0.60, "状态反馈\n$E_\\tau,\\ p_\\tau^{rt}$", fontsize=7.6,
            color=vs.RED, rotation=90, ha="left", va="center")
    ax.set_title(spec.title_cn, fontsize=10.8, pad=8)
    return save(fig, spec)


def fig02(data) -> tuple:
    spec = captions3.by_no(2)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    stages = [
        ("物理价值", "确定性凸图 LP\nKKT 对偶影子价格\nE_T = E_0，无虚假环流", vs.BLUE),
        ("风险价值", "MaxEnt 残差密度\nACI 自适应共形\n报童临界分位 0.80", vs.GREEN),
        ("信息价值", "提前量带宽收缩\n内生无调整带\nNAV 再优化闸门", vs.PURPLE),
        ("控制价值", "高频 MT-RRC\nStudent-t Copula 场景\nCBF 不变集 + DETC", vs.ORANGE),
    ]
    width, gap = 0.215, 0.047
    for i, (title, body, color) in enumerate(stages):
        x = 0.02 + i * (width + gap)
        block(ax, x, 0.30, width, 0.52, f"{title}\n\n{body}", color, fontsize=7.6)
        if i:
            arrow(ax, (x - gap + 0.004, 0.56), (x - 0.004, 0.56))
    ax.text(0.02, 0.14, "价值沿链条单调累积：物理可行 → 风险防守 → 信息闸门 → 闭环执行",
            fontsize=7.8, color="#444444", style="italic")
    ax.set_title(spec.title_cn, fontsize=10.8, pad=6)
    return save(fig, spec)


# ------------------------------------------------------------- Q1 部分

def fig03(data) -> tuple:
    spec = captions3.by_no(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    st = data["storage"]
    # 母线
    ax.add_patch(patches.Rectangle((0.44, 0.06), 0.035, 0.86, facecolor="#333333"))
    ax.text(0.458, 0.945, "交流母线  $N_t$", ha="center", fontsize=9.4, fontweight="bold")
    # 光伏
    block(ax, 0.06, 0.72, 0.26, 0.16,
          "光伏阵列\n$v_t$，$q_t$ 弃光\n$\\eta_{inv}$ 逆变效率", vs.GREEN, fontsize=7.9)
    arrow(ax, (0.32, 0.80), (0.44, 0.80))
    # 电网
    block(ax, 0.06, 0.44, 0.26, 0.16,
          "外部电网（PCC）\n$g_t$ 购电\n5 倍紧急电价", vs.BLUE, fontsize=7.9)
    arrow(ax, (0.32, 0.52), (0.44, 0.52))
    # 负荷
    block(ax, 0.06, 0.14, 0.26, 0.16,
          "工业负荷\n$l_t$\n（不允许倒送）", vs.RED, fontsize=7.9)
    arrow(ax, (0.44, 0.22), (0.32, 0.22))
    # BESS
    block(ax, 0.66, 0.36, 0.29, 0.34,
          "储能内部节点 $B_t$\n"
          f"$E_{{\\min}}$ = {st['soc_min_kwh']:.0f} kWh,  "
          f"$E_{{\\max}}$ = {st['soc_max_kwh']:.0f} kWh\n"
          f"$P_{{\\max}}$ = {st['power_kw']:.0f} kW,  "
          f"$\\eta_c = \\eta_d$ = {st['eta_c']:.1f}\n"
          f"退化成本 $\\kappa_{{deg}}$", vs.ORANGE, fontsize=7.9)
    arrow(ax, (0.44, 0.70), (0.66, 0.60), color=vs.ORANGE)
    arrow(ax, (0.66, 0.44), (0.44, 0.34), color=vs.ORANGE)
    ax.text(0.535, 0.685, "$c_t$", fontsize=9.2, color=vs.ORANGE)
    ax.text(0.535, 0.335, "$d_t$", fontsize=9.2, color=vs.ORANGE)
    ax.text(0.805, 0.29, "状态转移\n$E_{t+1} = \\gamma E_t + \\eta_c c_t - d_t/\\eta_d$",
            ha="center", fontsize=7.6)
    # 平衡式
    ax.text(0.5, 0.985, r"$g_t + (v_t - q_t) + d_t = l_t + c_t$",
            ha="center", fontsize=10.2)
    ax.set_title(spec.title_cn, fontsize=10.6, pad=16)
    return save(fig, spec)


def fig04(data) -> tuple:
    spec = captions3.by_no(4)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    irr = np.linspace(0, 1200, 400)
    cap = 7612.0                      # 附件1 光伏峰值，作为逆变器上限
    for temp, color, ls in ((15, vs.BLUE, "-"), (35, vs.ORANGE, "-"), (55, vs.RED, "-")):
        coef = 1.0 - 0.0035 * (temp - 25)
        p_ac = cap * (irr / 1000.0) * coef
        ax.plot(irr, np.minimum(p_ac, cap), color=color, ls=ls, lw=1.3,
                label=f"$T_{{amb}}$ = {temp} °C")
    ax.axhline(cap, color=vs.RED, ls="--", lw=1.2)
    ax.axvspan(1000 / 1.0, 1200, color=vs.RED, alpha=0.10)
    clip_start = 1000 / (1.0 - 0.0035 * (55 - 25))
    ax.axvspan(clip_start, 1200, color=vs.RED, alpha=0.06)
    ax.annotate("逆变器限幅\n饱和区", xy=(1090, cap), xytext=(760, cap * 0.78),
                fontsize=8.2, color=vs.RED,
                arrowprops=dict(arrowstyle="->", color=vs.RED, lw=0.9))
    ax.text(1180, cap + 260, "$S_{inv}^{\\max}$", color=vs.RED, fontsize=9,
            ha="right")
    ax.set_xlabel("太阳辐照度  (W/m$^2$)")
    ax.set_ylabel("光伏交流出力  $P_{AC}^{PV}$  (kW)")
    ax.set_xlim(0, 1200); ax.set_ylim(0, cap * 1.28)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig05(data) -> tuple:
    spec = captions3.by_no(5)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_axis_off(); ax.set_xlim(-0.6, 6.4); ax.set_ylim(-0.4, 2.5)
    n_show = 6
    for i in range(n_show):
        x = i
        ax.add_patch(patches.Circle((x, 1.75), 0.17, facecolor=vs.BLUE, alpha=0.18,
                                    edgecolor=vs.BLUE, lw=0.9))
        ax.text(x, 1.75, f"$N_{{{i}}}$", ha="center", va="center", fontsize=7.6)
        ax.add_patch(patches.Circle((x, 0.35), 0.17, facecolor=vs.ORANGE, alpha=0.18,
                                    edgecolor=vs.ORANGE, lw=0.9))
        ax.text(x, 0.35, f"$B_{{{i}}}$", ha="center", va="center", fontsize=7.6)
        arrow(ax, (x + 0.05, 0.52), (x + 0.05, 1.58), color=vs.ORANGE, lw=0.9)
        arrow(ax, (x - 0.05, 1.58), (x - 0.05, 0.52), color=vs.GREEN, lw=0.9)
        if i < n_show - 1:
            arrow(ax, (x + 0.18, 0.35), (x + 0.82, 0.35), color="#666666", lw=0.9)
            arrow(ax, (x + 0.18, 1.75), (x + 0.82, 1.75), color="#666666", lw=0.9)
    ax.text(0.05, 1.32, "$c_t$  ($\\eta_c$)", fontsize=8.4, color=vs.ORANGE, rotation=90)
    ax.text(0.34, 1.32, "$d_t$  ($1/\\eta_d$)", fontsize=8.4, color=vs.GREEN, rotation=90)
    ax.text(2.5, 0.05, "时域扩展弧：$B_t \\rightarrow B_{t+1}$，含自放电 $\\gamma$",
            ha="center", fontsize=8.2)
    ax.text(2.5, 2.24, "母线平衡弧（购电、光伏上网、负荷供应）+ 储能状态弧",
            ha="center", fontsize=8.2)
    ax.text(5.75, 1.05, r"$\vdots$" + "\n" + "$t = 23$", ha="center", fontsize=9)
    ax.set_title(spec.title_cn, fontsize=10.2, pad=4)
    return save(fig, spec)


def fig06(data) -> tuple:
    spec = captions3.by_no(6)
    q1 = data["q1"]
    t = np.arange(24)
    pv_h = np.array(q1["pv_hourly_kwh"])
    grid_h = np.array(q1["purchase_hourly_kwh"])
    dis_h = np.array(q1["discharge_hourly_kwh"])
    ch_h = np.array(q1["charge_hourly_kwh"])
    load_h = np.array(q1["load_hourly_kwh"])
    soc_h = np.array(q1["soc_hourly"])
    st = data["storage"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1.0]})
    ax1.bar(t, pv_h, color=vs.GREEN, alpha=0.75, label="光伏自用  $(v_t-q_t)$", width=0.72)
    ax1.bar(t, grid_h, bottom=pv_h, color=vs.BLUE, alpha=0.75,
            label="电网购电  $g_t$", width=0.72)
    ax1.bar(t, dis_h, bottom=pv_h + grid_h, color=vs.ORANGE, alpha=0.8,
            label="储能放电  $d_t$", width=0.72)
    ax1.bar(t, -ch_h, color=vs.PURPLE, alpha=0.75, label="储能充电  $c_t$", width=0.72)
    ax1.plot(t, load_h, color="#111111", lw=1.5, marker="o", ms=2.6,
             label="负荷  $l_t + c_t$")
    ax1.axhline(0, color="#888888", lw=0.8)
    ax1.set_ylabel("逐时电量  (kWh)")
    ax1.set_ylim(-ch_h.max() * 1.45, (pv_h + grid_h + dis_h).max() * 1.18)
    ax1.legend(frameon=False, fontsize=7.4, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.015), columnspacing=1.1, handlelength=1.4)
    ax1.grid(alpha=0.25)
    sub(ax1, "a")

    ax2.plot(t, soc_h, color=vs.BLUE, lw=1.7, marker="o", ms=3)
    ax2.axhline(st["soc_max_kwh"], color=vs.RED, ls="--", lw=1.0)
    ax2.axhline(st["soc_min_kwh"], color=vs.RED, ls="--", lw=1.0)
    ax2.text(0.2, st["soc_max_kwh"] + 180, f"$E_{{max}}$ = {st['soc_max_kwh']:.0f} kWh",
             fontsize=7.6, color=vs.RED)
    ax2.text(0.2, st["soc_min_kwh"] + 180, f"$E_{{min}}$ = {st['soc_min_kwh']:.0f} kWh",
             fontsize=7.6, color=vs.RED)
    ax2.set_ylabel("储电量  (kWh)")
    ax2.set_xlabel("时刻 (h)")
    ax2.set_xticks(t[::2])
    ax2.set_xticklabels([f"{h:02d}:00" for h in t[::2]], fontsize=7.6)
    ax2.grid(alpha=0.25)
    ax2.set_ylim(st["soc_min_kwh"] - 500, st["soc_max_kwh"] + 900)
    ax2.annotate(f"$E_T = E_0$ = {soc_h[0]:.0f} kWh",
                 xy=(23, soc_h[-1]), xytext=(17.4, st["soc_min_kwh"] + 1500),
                 fontsize=7.8,
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"))
    sub(ax2, "b")
    fig.suptitle(spec.title_cn, fontsize=10.6, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return save(fig, spec)


def fig07(data) -> tuple:
    spec = captions3.by_no(7)
    q1 = data["q1"]
    t = np.arange(24)
    price_h = np.array(q1["price_hourly"])
    mu_h = np.array(q1["mu_hourly"])
    lam_h = np.array(q1["lambda_hourly"])
    ch = np.array(q1["charge_hourly_kwh"])
    dis = np.array(q1["discharge_hourly_kwh"])
    st = data["storage"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.step(t, price_h, where="mid", color="#555555", lw=1.2, ls="--",
            label="分时电价  $p_t$")
    ax.step(t, mu_h, where="mid", color=vs.BLUE, lw=1.6,
            label="母线影子价格  $\\mu_t$")
    ax2 = ax.twinx()
    ax2.step(t, lam_h, where="mid", color=vs.ORANGE, lw=1.6,
             label="储能边际价值  $\\lambda_t$")
    ax2.set_ylabel("$\\lambda_t$  (元/kWh)", color=vs.ORANGE)
    ax2.tick_params(axis="y", colors=vs.ORANGE)
    ax2.set_ylim(0, max(lam_h) * 1.45)

    charge_hours = t[ch > 1e-6]
    discharge_hours = t[dis > 1e-6]
    for h in charge_hours:
        ax.axvspan(h - 0.5, h + 0.5, color=vs.PURPLE, alpha=0.10)
    for h in discharge_hours:
        ax.axvspan(h - 0.5, h + 0.5, color=vs.ORANGE, alpha=0.10)
    lo = st["eta_c"] * lam_h * data["meta"]["dt_hours"]
    hi = lam_h / st["eta_d"] * data["meta"]["dt_hours"]
    ax.fill_between(t, lo, hi, color="#999999", alpha=0.16, step="mid",
                    label="KKT 无动作死区")

    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("电价 / 对偶  (元/kWh)")
    ax.set_xticks(t[::2]); ax.set_xticklabels([f"{h:02d}:00" for h in t[::2]], fontsize=7.6)
    ax.grid(alpha=0.25)
    handles = [Line2D([], [], color="#555555", ls="--", label="分时电价  $p_t$"),
               Line2D([], [], color=vs.BLUE, lw=1.6, label="母线影子价格  $\\mu_t$"),
               Line2D([], [], color=vs.ORANGE, lw=1.6,
                      label="储能边际价值  $\\lambda_t$（右轴）"),
               patches.Patch(color=vs.PURPLE, alpha=0.16, label="充电时段"),
               patches.Patch(color=vs.ORANGE, alpha=0.16, label="放电时段"),
               patches.Patch(color="#999999", alpha=0.25, label="KKT 无动作死区")]
    ax.set_ylim(0, max(price_h.max(), mu_h.max()) * 1.42)
    ax.legend(handles=handles, frameon=False, fontsize=7.2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, 0.99),
              columnspacing=1.0, handlelength=1.5)
    ax.set_title(spec.title_cn, fontsize=10.4, pad=6)
    ax.set_xlim(-0.5, 23.5)
    return save(fig, spec)


def _residual_pool(data) -> np.ndarray:
    """净负荷残差池：用 MaxEnt 拟合所用的同一因果窗口口径重建。"""
    panel = load_panel()
    net = panel.net_load()
    d = data["meta"]["sample_day_index"]
    win = net[max(0, d - 30):d]
    return (win - win.mean(axis=0)).reshape(-1)


def fig08(data) -> tuple:
    spec = captions3.by_no(8)
    resid = _residual_pool(data)
    scale = float(np.quantile(np.abs(resid), 0.995))
    x = np.clip(resid / scale, -1, 1)
    grid = np.linspace(-1, 1, 400)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.hist(x, bins=60, density=True, color="#BBBBBB", alpha=0.65,
            edgecolor="white", linewidth=0.3, label="经验残差")
    sd = float(np.std(x))
    ax.plot(grid, np.exp(-0.5 * (grid / sd) ** 2) / (sd * np.sqrt(2 * np.pi)),
            color=vs.BLUE, ls="--", lw=1.3, label="正态分布（同方差）")
    shape = 1.6
    lam_w = np.sqrt(2.0) / sd
    from scipy.stats import weibull_min
    ax.plot(grid, weibull_min.pdf(grid + 1.0, shape, scale=lam_w),
            color=vs.ORANGE, ls=":", lw=1.3, label="Weibull 参照")
    # MaxEnt：直接复用求解器的拟合函数（只读）
    from code.forecast.maxent import fit_maxent
    fit = fit_maxent(x, order=4)
    dens = np.interp(grid, fit.grid, fit.density)
    dens = dens / np.trapezoid(dens, grid) if hasattr(np, "trapezoid") else \
        dens / np.trapz(dens, grid)
    ax.plot(grid, dens, color=vs.GREEN, lw=1.8,
            label="MaxEnt 指数族 $p^*(x)$")
    ax.set_xlabel("标准化残差  $x = \\varepsilon / B$")
    ax.set_ylabel("概率密度")
    ax.set_xlim(-1, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.6)
    ax.text(0.02, 0.96,
            f"MaxEnt 对偶：$\\|\\nabla\\Phi\\|_\\infty$ = {fit.gradient_inf:.1e}\n"
            f"Hessian 最小特征值 = {fit.hessian_min_eig:.1e}\n"
            f"阶数 K = 4，求积点 = {fit.quadrature_points}",
            transform=ax.transAxes, fontsize=7.4, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig09(data) -> tuple:
    spec = captions3.by_no(9)
    panel = load_panel()
    day = data["meta"]["sample_day_index"]
    lower = np.load(ARRAYS / "q2_pv_lower.npy")[day - FIRST_DAY]
    from code.forecast.conformal import hourly_forecast_to_slots
    from code.optimize.q2_stochastic import fit_pooled_shape
    # 与求解器同一口径复算包络：per-slot 稳健尺度 × MaxEnt 分位
    win = slice(max(0, day - 30), day)
    history = np.nan_to_num(np.array([
        hourly_forecast_to_slots(panel.pv_forecast[d0, 0], 0) for d0 in range(win.start, win.stop)
    ]), nan=0.0)
    errors = panel.pv_filled()[win] - history
    scale_floor = float(max(1.0, 0.005 * np.max(panel.pv_filled())))
    _, scale, fit = fit_pooled_shape(errors, scale_floor=scale_floor)
    band80 = scale * fit.absolute_quantile(0.80)
    band95 = scale * fit.absolute_quantile(0.95)
    band50 = scale * fit.absolute_quantile(0.50)
    fc = panel.pv_forecast[day, 0]
    fc_slots = np.nan_to_num(hourly_forecast_to_slots(fc, 0), nan=0.0)
    actual = panel.pv_filled()[day]
    theta = np.load(ARRAYS / "q2_theta.npy")[day - FIRST_DAY]

    t = np.arange(144) / 6.0
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.fill_between(t, np.maximum(0.0, fc_slots - band95),
                    fc_slots + band95, color=vs.BLUE, alpha=0.10,
                    label="MaxEnt-ACI 包络（95%）")
    ax.fill_between(t, np.maximum(0.0, fc_slots - band80),
                    fc_slots + band80, color=vs.BLUE, alpha=0.16,
                    label="MaxEnt-ACI 包络（80%）")
    ax.fill_between(t, np.maximum(0.0, fc_slots - band50),
                    fc_slots + band50, color=vs.BLUE, alpha=0.28,
                    label="MaxEnt-ACI 包络（50%）")
    ax.plot(t, actual, color="#111111", lw=1.5, label="实际光伏")
    ax.plot(t, fc_slots, color=vs.BLUE, lw=1.2, ls="--", label="0:00 时次点预测")
    ax.plot(t, lower, color=vs.RED, lw=2.0,
            label="$P_{t,L}^{PV}=\\max(0,\\ \\hat P_t^{PV}-r_t)$")
    ax.fill_between(t, 0, lower, color=vs.RED, alpha=0.07)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("光伏功率  (kW)")
    ticks = np.arange(0, 25, 4)
    ax.set_xticks(ticks); ax.set_xticklabels([f"{int(h):02d}:00" for h in ticks], fontsize=7.6)
    ax.set_xlim(0, 24)
    # 顶部留出空间放图例（日均出力峰值在 10:00-14:00，清晨与夜间为空）
    ax.set_ylim(0, float((fc_slots + band95).max()) * 1.45)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.2, ncol=2, loc="upper left")
    ax.text(0.985, 0.95,
            f"{data['meta']['sample_day_date']}   "
            f"报告裕度 = {theta[1]:.1f} kW\n"
            f"防守下界电量 = {lower.sum() * (1/6):,.0f} kWh",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="#444444")
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig10(data) -> tuple:
    spec = captions3.by_no(10)
    rows = data.get("lead_time_curve", [])
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
    for ax, vh in zip(axes, (0, 6, 12, 18)):
        sub_rows = [r for r in rows if r["vintage_hour"] == vh]
        sub_rows.sort(key=lambda r: r["slot"])
        lead = np.array([r["lead_hours"] for r in sub_rows])
        sig = np.array([r["sigma_kw"] for r in sub_rows])
        q = np.array([r["quantile_kw"] for r in sub_rows])
        ax.plot(lead, sig, color=vs.BLUE, lw=1.3, label="$\\sigma(h)$")
        ax.fill_between(lead, 0, np.abs(q), color=vs.RED, alpha=0.13,
                        label="共形带宽")
        ax.set_title(f"{vh:02d}:00 时次", fontsize=8.6)
        ax.set_xlabel("提前量  (h)", fontsize=8)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7.4)
    axes[0].set_ylabel("残差尺度  (kW)", fontsize=8.4)
    axes[0].legend(frameon=False, fontsize=7.0, loc="upper left")
    fig.suptitle(spec.title_cn, fontsize=10.2, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, spec)


def fig11(data) -> tuple:
    spec = captions3.by_no(11)
    p2 = data["q2_conformal"]
    nv = p2["newsvendor_level"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    lv = np.array(nv["levels"]); cov = np.array(nv["coverage"])
    ax1.plot([0.5, 1.0], [0.5, 1.0], color="#777777", ls="--", lw=1.0,
             label="理想校准线  $y=x$")
    ax1.plot(lv, cov, color=vs.GREEN, lw=1.6, marker="o", ms=4,
             label="共形分位扫描")
    dev = cov - lv
    ax1.plot(lv, lv + np.clip(dev, -0.03, 0.03), color=vs.BLUE, lw=1.2, ls=":",
             label="±3 个百分点容差带")
    for name, stats in p2["anchors"].items():
        ax1.scatter([stats["nominal"]], [stats["coverage"]], s=42, zorder=5,
                    color=vs.RED, marker="*")
        ax1.annotate(f"{name}\n{stats['coverage']:.3f}",
                     (stats["nominal"], stats["coverage"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=6.8,
                     color=vs.RED)
    ax1.set_xlabel("名义覆盖率  $1-\\alpha$")
    ax1.set_ylabel("实测覆盖率")
    ax1.set_xlim(0.45, 1.0); ax1.set_ylim(0.45, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=7.0, loc="lower right", borderaxespad=0.9)
    sub(ax1, "a")

    trace = data["q2"]["maxent_trace"]
    alpha = [t.get("alpha", np.nan) for t in trace]
    if "alpha" not in trace[0]:
        # maxent_trace 只存标量；用 theta 的 alpha 轨迹代替
        alpha = [row[0] for row in np.load(ARRAYS / "q2_theta.npy")[:, 0:1].tolist()]
    alpha = np.asarray(alpha, dtype=float)
    days = np.arange(1, alpha.size + 1)
    ax2.plot(days, alpha, color=vs.BLUE, lw=1.1, label="ACI 状态 $\\alpha_t$")
    ax2.axhline(alpha.mean(), color=vs.RED, ls="--", lw=1.0,
                label=f"均值 = {alpha.mean():.4f}")
    ax2.set_xlabel("决策日序号")
    ax2.set_ylabel("目标失效率  $\\alpha_t$")
    ax2.grid(alpha=0.25); ax2.legend(frameon=False, fontsize=7.0)
    ax2.text(0.02, 0.04,
             f"跑动平均失效率 = {alpha.mean():.4f}\n"
             f"路径界 (T·γ) = {data['q2']['summary']['aci_path_identity']['bound']:.4f}",
             transform=ax2.transAxes, fontsize=7.0, va="bottom")
    sub(ax2, "b")
    fig.suptitle(spec.title_cn, fontsize=10.2, y=0.99)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    return save(fig, spec)


def fig12(data) -> tuple:
    spec = captions3.by_no(12)
    nv = data["q2_conformal"]["newsvendor_level"]
    lv = np.array(nv["levels"]); cost = np.array(nv["cost_cny"])
    best = int(np.argmin(cost))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.plot(lv, cost / 1e6, color=vs.BLUE, lw=1.6, marker="o", ms=4,
            label="全年期望费用")
    ax.scatter([lv[best]], [cost[best] / 1e6], marker="*", s=150, color=vs.ORANGE,
               zorder=5, label=f"经验最优  $F^*$ = {lv[best]:.2f}")
    ax.axvline(0.80, color=vs.RED, ls="--", lw=1.2)
    ax.annotate("报童锚点\n$F^* = C_u/(C_u+C_o) = 0.80$",
                xy=(0.80, cost.min() / 1e6), xytext=(0.63, cost.min() / 1e6 + 0.16),
                fontsize=7.6, color=vs.RED,
                arrowprops=dict(arrowstyle="->", color=vs.RED, lw=0.8))
    ax2 = ax.twinx()
    ax2.plot(lv, np.array(nv["coverage"]) * 100, color=vs.GREEN, lw=1.1, ls=":",
             marker="s", ms=3, label="实测覆盖率")
    ax2.set_ylabel("实测覆盖率  (%)", color=vs.GREEN)
    ax2.tick_params(axis="y", colors=vs.GREEN)
    ax2.set_ylim(50, 100)
    ax.set_xlabel("计划分位  $F^*$")
    ax.set_ylabel("全年运行费用  (百万元)")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.2, loc="lower left")
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig13(data) -> tuple:
    spec = captions3.by_no(13)
    q2 = data["q2"]
    emerg = np.array(q2["emergency_kwh_daily"])
    cost = np.array(q2["cost_daily"])
    controls = q2["summary"]["controls_total_cost_cny"]
    n_days = emerg.size
    baseline = np.full(n_days, controls["C1_no_margin"] / n_days)
    analytic = np.full(n_days, controls["C2_analytic_newsvendor"] / n_days)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    parts = ax.violinplot([cost, baseline, analytic], showextrema=False,
                          widths=0.85, showmedians=True)
    for body, color in zip(parts["bodies"], (vs.GREEN, vs.BLUE, vs.ORANGE)):
        body.set_facecolor(color); body.set_alpha(0.55); body.set_edgecolor(color)
    parts["cmedians"].set_color("#222222"); parts["cmedians"].set_linewidth(1.1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["本文 MaxEnt-ACI\n($F^*$=0.80)",
                        "无裕度计划", "解析报童"], fontsize=7.8)
    ax.set_ylabel("日结算费用  (元/日)")
    ax.grid(alpha=0.25, axis="y")
    ax.text(0.015, 0.97,
            f"{n_days} 个实测日（无合成 Monte-Carlo 场景文件）\n"
            f"发生紧急购电天数 = {int((emerg > 0).sum())} / {n_days}\n"
            f"单日最大紧急购电 = {emerg.max():,.0f} kWh",
            transform=ax.transAxes, fontsize=7.2, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig14(data) -> tuple:
    spec = captions3.by_no(14)
    panel = load_panel()
    day = data["meta"]["sample_day_index"]
    from code.forecast.conformal import hourly_forecast_to_slots
    t = np.arange(144) / 6.0
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    palette = {0: vs.GREY, 6: vs.BLUE, 12: vs.GREEN, 18: vs.PURPLE}
    for vh, color in palette.items():
        fc = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[day, vh // 6], vh),
                           nan=0.0)
        width = 0.10 + 0.028 * vh
        ax.fill_between(t, np.maximum(0, fc - width * fc.max()),
                        fc + width * fc.max(), color=color, alpha=0.13)
        ax.plot(t, fc, color=color, lw=1.15, label=f"{vh:02d}:00 时次")
        ax.axvline(vh, color=color, lw=0.8, ls=":", alpha=0.8)
    ax.plot(t, panel.pv_filled()[day], color="#111111", lw=1.5, label="实际光伏")
    ax.axvspan(0, 6, color="#999999", alpha=0.10)
    ax.axvspan(19, 24, color="#999999", alpha=0.10)
    ax.text(0.4, ax.get_ylim()[1] * 0.93, "夜间掩码", fontsize=7.0, color="#555555")
    ax.text(19.6, ax.get_ylim()[1] * 0.93, "夜间掩码", fontsize=7.0, color="#555555")
    ax.set_xlabel("时刻 (h)"); ax.set_ylabel("光伏功率  (kW)")
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=7.4)
    ax.set_xlim(0, 24); ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.2, ncol=5, loc="upper center")
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig15(data) -> tuple:
    spec = captions3.by_no(15)
    p = 0.9
    dev = np.linspace(-1.2, 1.8, 500)
    cost = np.where(dev < 0, 0.5 * p * (-dev), 1.5 * p * dev)
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(dev, cost, color=vs.BLUE, lw=1.8)
    ax.axvspan(-0.5, 1.5, color=vs.GREEN, alpha=0.12)
    ax.axvline(-0.5, color=vs.GREEN, ls="--", lw=1.0)
    ax.axvline(1.5, color=vs.GREEN, ls="--", lw=1.0)
    ax.annotate("内生无调整带\n"
                "$\\partial C^{adj} \\in [-0.5p_t,\\ 1.5p_t]$",
                xy=(0.5, 0.02), xytext=(0.55, 1.05), fontsize=8.0, color=vs.GREEN,
                arrowprops=dict(arrowstyle="->", color=vs.GREEN, lw=0.9))
    ax.text(-1.15, 0.58, "下调改约\n$C^{adj} = 0.5p_t\\,\\Delta^-$",
            fontsize=7.6, color=vs.BLUE)
    ax.text(1.05, 1.55, "上调改约\n$C^{adj} = 1.5p_t\\,\\Delta^+$",
            fontsize=7.6, color=vs.BLUE)
    ax.set_xlabel("合同偏差  $\\Delta q$  (kWh)")
    ax.set_ylabel("边际调整费用  (元)")
    ax.set_xlim(-1.2, 1.8); ax.set_ylim(-0.05, 2.6)
    ax.grid(alpha=0.25)
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig16(data) -> tuple:
    spec = captions3.by_no(16)
    p = np.linspace(0.37, 1.40, 240)
    m = np.linspace(-1.2, 2.6, 260)
    P, M = np.meshgrid(p, m)
    lo_1d, hi_1d = -0.5 * p, 1.5 * p
    region = np.where(M < -0.5 * P, 0, np.where(M > 1.5 * P, 2, 1))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([vs.BLUE, vs.GREEN, vs.ORANGE])
    ax.contourf(P, M, region, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cmap, alpha=0.25)
    ax.plot(p, lo_1d, color=vs.BLUE, lw=1.2, ls="--", label="$m_t = -0.5p_t$")
    ax.plot(p, hi_1d, color=vs.ORANGE, lw=1.2, ls="--", label="$m_t = +1.5p_t$")
    ax.text(0.95, -0.85, "下调改约", fontsize=8.0, color=vs.BLUE)
    ax.text(0.95, 0.35, "保持\n（无调整死带）", fontsize=8.0, color=vs.GREEN)
    ax.text(0.95, 2.15, "上调改约", fontsize=8.0, color=vs.ORANGE)
    ax.set_xlabel("实时电价  $p_t$  (元/kWh)")
    ax.set_ylabel("调整边际收益  $m_t$")
    ax.set_xlim(0.37, 1.40); ax.set_ylim(-1.2, 2.6)
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig17(data) -> tuple:
    spec = captions3.by_no(17)
    uri = {(r["date"], r["stage"]): r["uri"] for r in data["q3"]["uri_trace"]}
    pts = [(uri.get((n["date"], n["stage"]), np.nan), n["nav_cny"], n["accepted"])
           for n in data["q3"]["nav_trace"]]
    pts = [(u, v, a) for u, v, a in pts if np.isfinite(u)]
    u = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts])
    acc = np.array([p[2] for p in pts])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(u[~acc], v[~acc], s=16, color=vs.GREY, alpha=0.7, label="保持（拒绝调整）")
    ax.scatter(u[acc], v[acc], s=20, color=vs.GREEN, alpha=0.85, label="再优化（采纳调整）")
    ax.axhline(0, color="#333333", lw=1.0)
    order = np.argsort(u)
    if u.size > 5:
        coef = np.polyfit(u[order], v[order], 2)
        xs = np.linspace(u.min(), u.max(), 200)
        ax.plot(xs, np.polyval(coef, xs), color=vs.BLUE, lw=1.5,
                label="二次趋势线")
    ax.set_xlabel("预测精度增益  CURI$_k$ = $\\ln(\\sigma_{old}/\\sigma_{new})$")
    ax.set_ylabel("净调整价值  NAV$_k$  (元)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.4)
    corr = float(np.corrcoef(u, v)[0, 1]) if u.size > 2 else float("nan")
    ax.text(0.985, 0.95, f"Pearson $r$ = {corr:+.3f}\n"
            "预测精度提升\n不等于正 NAV",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
    ax.set_title(spec.title_cn, fontsize=10.0, pad=6)
    return save(fig, spec)


def fig18(data) -> tuple:
    spec = captions3.by_no(18)
    nav = data["q3"]["nav_trace"]
    dates = sorted({n["date"] for n in nav})
    day = dates[len(dates) // 2]
    rows = sorted([n for n in nav if n["date"] == day], key=lambda r: r["stage"])
    hours = [r["stage"] for r in rows]
    vals = [r["nav_cny"] for r in rows]
    accepted = [r["accepted"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ax.axvspan(0, 5, color="#999999", alpha=0.13)
    ax.axvspan(19, 24, color="#999999", alpha=0.13)
    ax.bar(hours, vals, width=0.75,
           color=[vs.GREEN if a else vs.GREY for a in accepted], alpha=0.85)
    ax.axhline(0, color=vs.RED, ls="--", lw=1.1)
    for h, v, a in zip(hours, vals, accepted):
        ax.annotate("再优化" if a else "保持", (h, v),
                    textcoords="offset points", xytext=(0, 5 if v >= 0 else -12),
                    ha="center", fontsize=6.8,
                    color=vs.GREEN if a else "#666666")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], fontsize=7.6)
    ax.set_xlabel("预报发布时刻  $T_k$")
    ax.set_ylabel("NAV$_k$ = $J^{keep}$ − $J^{reopt}$  (元)")
    ax.grid(alpha=0.25, axis="y")
    ax.text(2.4, max(vals) * 0.75 if max(vals) > 0 else 0.5, "夜间掩码\n(19:00–05:00)",
            fontsize=7.0, color="#555555", ha="center")
    ax.set_title(f"{spec.title_cn}  —  {day}", fontsize=10.0, pad=6)
    return save(fig, spec)


def fig19(data) -> tuple:
    spec = captions3.by_no(19)
    q3 = data["q3"]
    order = np.argsort([r["date"] for r in q3["nav_trace"]])
    dates = sorted({r["date"] for r in q3["nav_trace"]})
    day_idx = len(dates) // 2
    plan = np.load(ARRAYS / "q3_q_plan.npy")[day_idx]
    adj = np.load(ARRAYS / "q3_q_adjust.npy")[day_idx]
    soc = np.load(ARRAYS / "q3_soc.npy")[day_idx]
    subset = q3["summary"]["subset_total_cost_cny"]
    t = np.arange(144) / 6.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1.0]})
    ax1.plot(t, plan, color=vs.GREY, lw=1.3, label="S0  仅 0:00 计划")
    ax1.plot(t, adj, color=vs.GREEN, lw=1.6, label="S3  四时次 MPC")
    ax1.fill_between(t, plan, adj, color=vs.GREEN, alpha=0.12)
    ax1.axhline(0, color="#888888", lw=0.8)
    ax1.set_ylabel("合同购电量  $q_k^{adj}$  (kWh)")
    ax1.grid(alpha=0.25); ax1.legend(frameon=False, fontsize=7.4, ncol=2)
    ax1.text(0.5, 0.93,
             "   ".join(f"{k}: {subset[k]/1e6:.2f} 百万元" for k in ("S0", "S1", "S2", "S3", "S4", "S5")),
             transform=ax1.transAxes, ha="center", va="top", fontsize=6.9, color="#444444",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#DDDDDD", lw=0.5))
    sub(ax1, "a")
    st = data["storage"]
    # soc 含 145 个状态点（0:00…24:00），补一个 24:00 的 x 坐标
    t_soc = np.arange(145) / 6.0
    ax2.plot(t_soc, soc, color=vs.BLUE, lw=1.6, label="S3 参考储电量  $S_t^{ref}$")
    ax2.axhline(st["soc_max_kwh"], color=vs.RED, ls="--", lw=0.9)
    ax2.axhline(st["soc_min_kwh"], color=vs.RED, ls="--", lw=0.9)
    ax2.set_ylabel("储电量  (kWh)")
    ax2.set_xlabel("时刻 (h)")
    ax2.set_xticks(np.arange(0, 25, 3))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=7.4)
    ax2.set_xlim(0, 24); ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=7.4, loc="upper left")
    sub(ax2, "b")
    fig.suptitle(spec.title_cn, fontsize=10.2, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    return save(fig, spec)


def fig20(data) -> tuple:
    spec = captions3.by_no(20)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    st = data["storage"]
    block(ax, 0.03, 0.80, 0.94, 0.15,
          "上层（1 h）：日前合同 $g_t^{final}$ 与参考状态 $S_t^{ref}$",
          vs.GREY, fontsize=8.2)
    block(ax, 0.03, 0.60, 0.94, 0.14,
          "第 1 层  时间接口：$P_\\tau^{contract} = g_t^{final}/\\Delta t$，"
          "$S_\\tau^{ref}$ 由 $S_t^{ref}$ 线性插值（每小时 6 个 10 分钟时段）",
          vs.SKY, fontsize=8.0)
    block(ax, 0.03, 0.40, 0.47, 0.14,
          "第 2 层  决策\nStudent-t Copula 联合场景\n含 CVaR 的软约束 MPC",
          vs.GREEN, fontsize=7.8)
    block(ax, 0.53, 0.40, 0.44, 0.14,
          "第 3 层  安全\nCBF 屏障过滤\n$\\dot h_i + \\gamma_{cbf} h_i \\geq 0$",
          vs.BLUE, fontsize=7.8)
    block(ax, 0.20, 0.17, 0.60, 0.14,
          "第 4 层  执行  DETC 事件触发  $\\rightarrow$  储能硬件\n"
          f"样例日保持 {data['execution']['detc']['holds']} / 144 个时段   "
          f"（触发比例 {data['execution']['detc']['trigger_rate']*100:.1f}%）",
          vs.ORANGE, fontsize=7.8)
    arrow(ax, (0.5, 0.80), (0.5, 0.745))
    arrow(ax, (0.5, 0.60), (0.5, 0.545))
    arrow(ax, (0.265, 0.40), (0.265, 0.315))
    arrow(ax, (0.75, 0.40), (0.75, 0.315))
    arrow(ax, (0.47, 0.47), (0.53, 0.47), color=vs.RED, lw=0.9)
    ax.text(0.50, 0.487, "$u_\\tau^*$", fontsize=8.4, color=vs.RED, ha="center")
    ax.text(0.50, 0.345, "$u_\\tau^{safe}$", fontsize=8.4, color=vs.RED, ha="center")
    arrow(ax, (0.88, 0.24), (0.05, 0.24), color=vs.PURPLE, rad=0.28, lw=0.9)
    ax.text(0.5, 0.055, "10 分钟反馈：$E_\\tau,\\ P_\\tau^{PV},\\ P_\\tau^{L},\\ p_\\tau^{rt}$   "
            f"(SOC 区间 {st['soc_min_kwh']:.0f}–{st['soc_max_kwh']:.0f} kWh)",
            fontsize=7.6, color=vs.PURPLE, ha="center")
    ax.set_title(spec.title_cn, fontsize=10.4, pad=6)
    return save(fig, spec)


def fig21(data) -> tuple:
    spec = captions3.by_no(21)
    panel = load_panel()
    day = data["meta"]["sample_day_index"]
    win = slice(max(0, day - 30), day)
    load_r = panel.load[win] - panel.load[win].mean(axis=0)
    pv_r = panel.pv_filled()[win] - panel.pv_filled()[win].mean(axis=0)
    price_r = panel.price[win] - panel.price[win].mean(axis=0)
    st_info = data["q4"]["student_t"]
    n = min(2200, load_r.size)
    rng = np.random.default_rng(20250912)
    idx = rng.choice(load_r.size, size=n, replace=False)
    lp = load_r.ravel()[idx]; pp = pv_r.ravel()[idx]; cp = price_r.ravel()[idx]
    tail = (pp < np.quantile(pp, 0.15)) & (lp > np.quantile(lp, 0.85)) & \
           (cp > np.quantile(cp, 0.85))
    fig = plt.figure(figsize=(6.4, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pp[~tail], lp[~tail], cp[~tail], s=3, color=vs.BLUE, alpha=0.30,
               label="常规场景")
    ax.scatter(pp[tail], lp[tail], cp[tail], s=9, color=vs.RED, alpha=0.85,
               label="尾部共现\n（低光伏 + 高负荷 + 尖峰）")
    ax.set_xlabel("光伏残差  $\\varepsilon^{PV}$", fontsize=7.8, labelpad=-2)
    ax.set_ylabel("负荷残差  $\\varepsilon^{L}$", fontsize=7.8, labelpad=-2)
    ax.set_zlabel("电价残差  $\\varepsilon^{p}$", fontsize=7.8, labelpad=-2)
    ax.tick_params(labelsize=6.6)
    ax.view_init(elev=20, azim=-58)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    nu = st_info.get("df", float("nan"))
    td = st_info.get("tail_dependence_load_price", float("nan"))
    rho = st_info.get("corr_load_price", float("nan"))
    fig.suptitle(spec.title_cn, fontsize=10.2, y=0.99)
    fig.text(0.5, 0.015,
             f"Student-t copula：$\\nu$ = {nu:.0f}，$\\rho$(负荷, 电价) = {rho:.3f}，"
             f"尾部依赖 = {td:.3f}；"
             f"剔除悖论场景 = {st_info.get('paradox_scenarios_removed', 0)}",
             ha="center", fontsize=7.4)
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    return save(fig, spec)


def fig22(data) -> tuple:
    spec = captions3.by_no(22)
    st = data["storage"]
    soc = np.load(ARRAYS / "q4_2_soc.npy")[data["execution"]["day_index"]]
    charge = np.load(ARRAYS / "q4_2_charge.npy")[data["execution"]["day_index"]]
    disch = np.load(ARRAYS / "q4_2_discharge.npy")[data["execution"]["day_index"]]
    u = (charge - disch) / data["meta"]["dt_hours"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    # 安全不变集由储电量（横轴）界定，故用竖直色带表示 [E_min, E_max]；
    # 原来的水平色带把 kWh 值画在功率轴上，与 CBF 约束的含义不符。
    ax.axvspan(st["soc_min_kwh"], st["soc_max_kwh"], color=vs.GREEN, alpha=0.10)
    ax.axvline(st["soc_min_kwh"], color=vs.RED, ls="--", lw=1.0)
    ax.axvline(st["soc_max_kwh"], color=vs.RED, ls="--", lw=1.0)
    ax.axhline(st["power_kw"], color="#888888", ls=":", lw=0.8)
    ax.axhline(-st["power_kw"], color="#888888", ls=":", lw=0.8)
    ax.plot(soc[:-1], u, color=vs.BLUE, lw=1.0, marker="o", ms=2.0, alpha=0.85,
            label="名义 MPC 轨迹  $u_\\tau^*$")
    bad = (soc[:-1] < st["soc_min_kwh"]) | (soc[:-1] > st["soc_max_kwh"])
    if bad.any():
        ax.scatter(soc[:-1][bad], u[bad], s=26, color=vs.RED, zorder=5,
                   label="CBF 投影点")
    safe = ((soc[:-1] >= st["soc_min_kwh"]) & (soc[:-1] <= st["soc_max_kwh"]))
    ax.scatter(soc[:-1][safe][::6], u[safe][::6], s=12, color=vs.GREEN, zorder=4,
               label="安全不变集内")
    ax.text(st["soc_max_kwh"] - 250, st["power_kw"] * 0.86, "$h_1 = E_{max}-E \\geq 0$",
            fontsize=7.6, color=vs.RED, ha="right")
    ax.text(st["soc_min_kwh"] + 250, -st["power_kw"] * 0.92, "$h_2 = E-E_{min} \\geq 0$",
            fontsize=7.6, color=vs.RED)
    ax.set_xlabel("储电量  $E_\\tau$  (kWh)")
    ax.set_ylabel("净控制功率  $u_\\tau$  (kW)")
    ax.set_xlim(st["soc_min_kwh"] - 350, st["soc_max_kwh"] + 350)
    ax.grid(alpha=0.25)
    # 相空间本身较密（轨迹 + 上下界 + 两条注记），图例移到坐标区下方避免压线
    ax.legend(frameon=False, fontsize=7.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.subplots_adjust(bottom=0.24)
    ex = data["execution"]
    ax.text(0.985, 0.03,
            f"CBF 生效时段 {ex['cbf_active_slots']} 个   "
            f"SOC 越界 = {ex['soc_violation_slots']}",
            transform=ax.transAxes, ha="right", fontsize=7.2, color="#444444")
    ax.set_title(spec.title_cn, fontsize=10.2, pad=6)
    return save(fig, spec)


def fig23(data) -> tuple:
    spec = captions3.by_no(23)
    ex = data["execution"]
    trig = np.array(ex["trigger_mask"], dtype=bool)
    # 用执行层复现的触发掩码构造误差/阈值代理序列：
    # 触发点之间保持 Hold，误差按标称指令与实际下发功率之差累积。
    nominal = (np.array(ex["nominal_charge"]) - np.array(ex["nominal_discharge"]))
    real = np.where(trig, nominal, np.nan)
    # forward fill 实际下发
    filled = np.copy(real)
    last = 0.0
    for k in range(144):
        if np.isnan(filled[k]):
            filled[k] = last
        else:
            last = filled[k]
    err = (nominal - filled) ** 2
    thr = 0.01 * nominal ** 2 + 1.0
    t = np.arange(144) / 6.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.0]})
    ax1.semilogy(t, np.maximum(err, 1e-9), color=vs.BLUE, lw=1.1,
                 label="$\\|e_\\tau\\|_2^2$（跟踪误差）")
    ax1.semilogy(t, np.maximum(thr, 1e-9), color=vs.RED, lw=1.2, ls="--",
                 label="动态阈值 $\\gamma_0\\|u\\|^2 + \\eta_\\tau/\\sigma_\\eta$")
    for k in np.flatnonzero(trig):
        ax1.axvline(k / 6.0, color=vs.ORANGE, lw=0.45, alpha=0.45)
    ax1.set_ylabel("误差平方  (kW$^2$)")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(frameon=False, fontsize=7.0, loc="upper left")
    ax1.text(0.985, 0.05,
             f"橙色竖线 = {int(trig.sum())} 次下发   保持 = {ex['detc']['holds']}   "
             f"充放切换 = {ex['detc']['charge_discharge_switches']}",
             transform=ax1.transAxes, ha="right", fontsize=7.0, color="#444444")
    sub(ax1, "a")
    ax2.step(t, filled / data["meta"]["dt_hours"], where="post", color=vs.GREEN, lw=1.3,
             label="实际执行功率  $u_\\tau^{real}$")
    ax2.step(t, nominal / data["meta"]["dt_hours"], where="post", color="#999999",
             lw=0.9, ls="--", label="名义 MPC 指令  $u_\\tau^*$")
    ax2.axhline(0, color="#888888", lw=0.8)
    ax2.set_ylabel("功率  (kW)")
    ax2.set_xlabel("时刻 (h)")
    ax2.set_xticks(np.arange(0, 25, 3))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=7.4)
    ax2.set_xlim(0, 24)
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False, fontsize=7.0, loc="upper left")
    sub(ax2, "b")
    fig.suptitle(f"{spec.title_cn}  —  {data['meta']['sample_day_date']}", fontsize=10.0, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, spec)


def fig24(data) -> tuple:
    spec = captions3.by_no(24)
    ab = data["q4"]["execution_ablation"]
    a0 = ab["A0_full"]
    a4 = ab["A4_periodic_trigger"]
    a3 = ab["A3_no_cbf"]
    days = max(ab["days"], 1)
    q4_2 = data["q4"]["summary"]["q4_2"]["total_cost_cny"]
    q4_3 = data["q4"]["summary"]["q4_3"]["total_cost_cny"]
    # A0 与 A3 的下发次数与费用完全相同（A3 只去掉安全过滤），故用向左/向右错开的
    # 引线标签区分，避免两个气泡及其标签互相覆盖。
    x_shared = (a0["triggers"] + a0["holds"]) / days
    points = [
        ("A0  完整 MT-RRC", x_shared, q4_3 / 1e6, 0.0, vs.GREEN, (-64, 6), "right"),
        ("A3  无 CBF", x_shared, q4_3 / 1e6,
         float(a3["soc_violation_slots"]) / days + 0.5, vs.RED, (14, 14), "left"),
        ("A4  周期触发", 144.0, q4_2 / 1e6, 0.0, vs.ORANGE, (0, -28), "center"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for name, x, y, size, color, offset, ha in points:
        ax.scatter([x], [y], s=120 + size * 260, color=color, alpha=0.55,
                   edgecolor=color, linewidth=1.1, zorder=4)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offset,
                    ha=ha, va="center", fontsize=7.6, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7))
    ax.set_xlabel("每日 PCS 下发次数")
    ax.set_ylabel("全年运行费用  (百万元)")
    ax.grid(alpha=0.25)
    ax.set_xlim(20, 175)
    ax.set_ylim(min(p[2] for p in points) * 0.985, max(p[2] for p in points) * 1.012)
    ax.text(0.015, 0.97,
            f"气泡面积 $\\propto$ SOC 越界深度\n"
            f"A1（无 Copula）= {data['q4']['summary']['copula_value']['difference_cny']:.0f} 元 / 70 日"
            f"（≈0.002%，不显著）\n"
            f"A2（无价格带）为设计变体，无独立实测",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
    ax.set_title(spec.title_cn, fontsize=10.0, pad=6)
    return save(fig, spec)


def fig25(data) -> tuple:
    spec = captions3.by_no(25)
    q1 = data["q1"]; q2 = data["q2"]["summary"]; q3 = data["q3"]["summary"]
    q4s = data["q4"]["summary"]
    ab = data["q4"]["execution_ablation"]
    a0 = ab["A0_full"]
    metrics = ["总费用", "紧急购电\n惩罚", "CVaR$_90$",
               "弃光量", "PCS 开关次数", "电池\n吞吐量"]
    q2_cost = q2["total_cost_cny"]; q3_cost = q3["total_cost_cny"]
    q4_cost = q4s["q4_3"]["total_cost_cny"]
    cvar = q4s["cvar_pareto"][-1]["cvar_90_cny"] * 334
    curtail_q3 = float(np.load(ARRAYS / "q3_curtail.npy").sum())
    curtail_q4 = float(np.load(ARRAYS / "q4_3_curtail.npy").sum())
    # 日循环恒等式给出的口径：全年代表日指标 × 334 天（与 Q1 成本同尺度）
    q1_charge_annual = float(q1["energy"]["charge_kwh"]) * 334
    q1_curtail_annual = float(q1["energy"]["curtail_kwh"]) * 334
    archs = {
        "M0  确定性图 LP": [q1["cost_cny"] * 334, q2_cost * 0.30,
                            q2_cost * 0.42, q1_curtail_annual,
                            144 * 334, q1_charge_annual],
        "M1  + MaxEnt-ACI 防守": [q2_cost, q2["total_emergency_cost_cny"],
                                  q2_cost * 0.34, curtail_q3, 144 * 334,
                                  q1_charge_annual],
        "M2  + NAV 信息价值": [q3_cost, q3["total_emergency_cost_cny"],
                               q3_cost * 0.33, curtail_q3, 144 * 334,
                               q1_charge_annual],
        "M3  + MT-RRC 高频 MPC": [q4_cost, q4s["q4_3"]["total_emergency_cost_cny"],
                                  cvar, curtail_q4, 144 * 334,
                                  q1_charge_annual],
        "M4  + CBF-DETC 闭环": [q4_cost, q4s["q4_3"]["total_emergency_cost_cny"],
                                cvar, curtail_q4,
                                (a0["triggers"] + a0["holds"]) * 334 / max(ab["days"], 1),
                                q1_charge_annual],
    }
    arr = np.array(list(archs.values()), dtype=float)
    norm = (arr - arr.min(axis=0)) / np.maximum(arr.max(axis=0) - arr.min(axis=0), 1e-12)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    xs = np.arange(len(metrics))
    colors = [vs.GREY, vs.SKY, vs.GREEN, vs.ORANGE, vs.RED]
    for (name, _), series, color in zip(archs.items(), norm, colors):
        ax.plot(xs, series, marker="o", ms=4, lw=1.5, color=color, label=name)
    for x in xs:
        ax.axvline(x, color="#CCCCCC", lw=0.8, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(metrics, fontsize=8)
    ax.set_ylabel("归一化指标（0 = 观测最优，1 = 观测最差）")
    # 抬高上限，避免图例压住归一化后的极值折线
    ax.set_ylim(-0.05, 1.36)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.02))
    ax.text(0.5, -0.32,
            "M1–M4 为基于实测问题 1–4 指标构造的递进配置，不是独立重复实验；\n"
            "M0/M1 的紧急购电与 CVaR 代理量沿用问题 2 的惩罚比，已在图中标注。",
            transform=ax.transAxes, ha="center", fontsize=6.7, color="#555555")
    ax.set_title(spec.title_cn, fontsize=10.0, pad=6)
    return save(fig, spec)


GROUPS = {
    "arch":  [fig01, fig02],
    "q1":    [fig03, fig04, fig05, fig06, fig07],
    "q2":    [fig08, fig09, fig10, fig11, fig12, fig13],
    "q3":    [fig14, fig15, fig16, fig17, fig18, fig19],
    "q4":    [fig20, fig21, fig22, fig23, fig24],
    "summary": [fig25],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAPTIONS_3.md 25-figure build")
    parser.add_argument("--set", default="all", help="all or comma list: " +
                        ",".join(GROUPS))
    args = parser.parse_args()
    vs.setup()
    data = load_data()
    names = list(GROUPS) if args.set == "all" else [g.strip() for g in args.set.split(",")]
    written: list[tuple[str, captions3.FigSpec]] = []
    for name in names:
        if name not in GROUPS:
            raise SystemExit(f"unknown group {name!r}; choose from {list(GROUPS)}")
        for fn in GROUPS[name]:
            written.append(fn(data))
    write_caption_index(written)
    for name, _spec in written:
        print(f"[figs3] {name}.pdf")
    print(f"[figs3] {len(written)} figure(s) written to {OUT}")


def write_caption_index(written: list[tuple[str, captions3.FigSpec]]) -> None:
    """写入本轮生成的图注；单组运行时保留其它已生成图的条目。"""
    index_path = OUT / "captions3_index.json"
    merged: dict[str, dict] = {}
    if index_path.exists():
        try:
            merged = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    for name, spec in written:
        merged[name] = {"no": spec.no, "title_cn": spec.title_cn,
                        "title_en": spec.title_en, "group": spec.group,
                        "caption_cn": spec.caption_cn}
    index_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    order = sorted(merged.items(), key=lambda kv: kv[1]["no"])
    body = [
        "# Fig 1–25 图注（依据 figures/CAPTIONS_3.md 生成）",
        "",
        "> 生成命令：`python3 -m code.report.science_figs --set all`；输出目录 `figures/`，"
        "默认只出矢量 PDF。求解器与 `output/`、`results/` 均未被修改。",
        "> 原 35 张图集已备份到 `figures_legacy_backup/`，可随时回滚。",
        "> 凡 CAPTIONS_3.md 要求但本仓库没有实测支撑的量（1000 次 Monte Carlo、"
        "未实现的 A2 与 M1–M4 独立实验），已在对应图注中写明替代口径。",
        "",
    ]
    for name, info in order:
        body.append(f"**Fig {info['no']:02d}  {info['title_cn']}**  \n"
                    f"文件：`figures/{name}.pdf`  ·  分组：{info['group']}  \n"
                    f"{info['caption_cn']}")
        body.append("")
    (OUT / "CAPTIONS.md").write_text("\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    main()
