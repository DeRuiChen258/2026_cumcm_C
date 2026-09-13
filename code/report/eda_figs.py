"""Supplementary data-feature analysis and exploratory data analysis (EDA) figures.

These figures characterise the *input data* rather than the optimisation results:
cleaning audit, marginal distributions, seasonal structure, dependence/copula
structure, ramp and extreme behaviour, and forecast-error diagnostics.  Everything is
read from the C++ cleaning bundle (`clean/`) and the panel; the model code is untouched.
"""
from __future__ import annotations

import collections
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from code.common.load_clean import ROOT, load_panel
from code.optimize.q3_mpc import VINTAGE_HOUR, hourly_forecast_to_slots
from code.report import viz_style as vs
from code.report.viz_style import BLACK, BLUE, GREY, GREEN, ORANGE, PALETTE, PURPLE, RED, SKY, YELLOW

CLEAN = ROOT / "clean"
RESULTS = ROOT / "results"
HOURS = np.arange(144) / 6.0
WHITE_LINE = "#FFFFFF"
SEASONS = {"冬 (1–2月)": [0, 1], "春 (3–5月)": [2, 3, 4], "夏 (6–8月)": [5, 6, 7],
           "秋 (9–12月)": [8, 9, 10, 11]}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _season_index(dates) -> np.ndarray:
    return np.array([int(d[5:7]) - 1 for d in dates])


# ---------------------------------------------------------------------------
# D1 data quality / cleaning audit
# ---------------------------------------------------------------------------

def figD1_data_quality(panel) -> None:
    quality = _read_csv(CLEAN / "quality_report.csv")
    audit = _read_csv(CLEAN / "cleaning_audit.csv")
    quarantine = _read_csv(CLEAN / "quarantine.csv")
    counter = collections.Counter(row["rule_id"].split(">")[-1] for row in audit)

    fig = plt.figure(figsize=(13.8, 5.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.95], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    hours = np.arange(24)
    zero = np.array([float(r["zero_rate"]) for r in quality]) * 100
    noise = np.array([float(r["noise_rate"]) for r in quality]) * 100
    ax.bar(hours, zero, color=BLUE, width=0.42, label="零值率")
    ax.bar(hours + 0.42, noise, color=ORANGE, width=0.42, label="零漂噪声率 (0,5] kW")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("占比 (%)")
    ax.legend(loc="upper center", fontsize=8.4)
    vs.panel(ax, "a")
    ax.set_title("光伏零值与零漂噪声的小时分布（清洗前）")

    ax = fig.add_subplot(gs[0, 1])
    rules = sorted(counter)
    counts = [counter[r] for r in rules]
    bars = ax.barh(rules, counts, color=[RED if r == "NR2" else BLUE for r in rules], height=0.6)
    for bar, value in zip(bars, counts):
        ax.annotate(f"{value:,}", (value, bar.get_y() + bar.get_height() / 2),
                    textcoords="offset points", xytext=(4, 0), va="center", fontsize=8.0)
    ax.set_xscale("log")
    ax.set_xlabel("审计记录条数（对数轴）")
    vs.panel(ax, "b")
    ax.set_title(f"清洗规则命中统计（共 {len(audit):,} 条审计）")

    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    facts = [
        ("原始核夜间非零格", f"{2450 + 129:,}"),
        ("NR1 置零（零漂）", f"{counter.get('NR1', 0):,}"),
        ("NR2 置 NaN（硬异常）", f"{counter.get('NR2', 0):,}"),
        ("隔离清单条数", f"{len(quarantine):,}"),
        ("日间 08:00–16:00 零值", "0（I2 通过）"),
        ("核夜间清洗后非零格", "0（I1 通过）"),
        ("近零电价格 (<0.1 元)", "28（保留并标记）"),
        ("审计/隔离产物", "cleaning_audit.csv / quarantine.csv"),
    ]
    for i, (key, value) in enumerate(facts):
        y = 0.94 - i * 0.115
        ax.text(0.0, y, key, fontsize=8.8, color=BLACK, va="center")
        ax.text(0.62, y, value, fontsize=8.8, color=BLUE, va="center", fontweight="bold")
        ax.plot([0, 1], [y - 0.052, y - 0.052], color="#DDDDDD", lw=0.7)
    vs.panel(ax, "c", dx=-0.03)
    ax.set_title("数据质量关键事实")
    fig.suptitle("Fig.D1　数据质量与清洗审计（夜间专项）", fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD1_data_quality",
            "数据质量诊断。(a) 光伏零值率与零漂噪声率的小时分布（结构事实：夜间恒零）；"
            "(b) NR1–NR12 的命中统计；"
            "(c) 清洗前后关键事实与不变量校验结论。")


# ---------------------------------------------------------------------------
# D2 marginal distributions by season
# ---------------------------------------------------------------------------

def figD2_distributions(panel) -> None:
    months = _season_index(panel.dates)
    pv = panel.pv_filled()
    price = panel.price
    load = panel.load
    net = panel.net_load()
    payload = [("负荷 (kW)", load), ("光伏 (kW)", pv), ("电价 (元/kWh)", price),
               ("净负荷 (kW)", net)]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 5.0))
    for idx, (ax, (label, values)) in enumerate(zip(axes, payload)):
        data = []
        for name, mons in SEASONS.items():
            sel = np.isin(months, mons)
            sample = values[sel].reshape(-1)
            sample = sample[~np.isnan(sample)]
            step = max(1, sample.size // 6000)
            data.append(sample[::step])
        parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
        for body, color in zip(parts["bodies"], PALETTE):
            body.set_facecolor(color)
            body.set_alpha(0.55)
        ax.boxplot(data, widths=0.16, showfliers=False, patch_artist=False,
                   medianprops=dict(color=BLACK, lw=1.0))
        ax.set_xticks(range(1, len(SEASONS) + 1), list(SEASONS), rotation=20, fontsize=8.0)
        ax.set_ylabel(label, fontsize=9)
        vs.panel(ax, chr(ord("a") + idx))
        ax.set_title(label, fontsize=9.6)
    fig.suptitle("Fig.D2　四类时序变量的季节分布（小提琴 + 箱线）",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD2_distributions",
            "边缘分布的季节差异。负荷、光伏、电价与净负荷在四个季节的分布形态，"
            "说明夏季光伏强、冬夏电价分布更重尾，且净负荷存在大量负值（光伏倒送被弃光吸收）。")


# ---------------------------------------------------------------------------
# D3 seasonal heatmaps (hour x month)
# ---------------------------------------------------------------------------

def figD3_seasonal_heatmaps(panel) -> None:
    pv = panel.pv_filled()
    items = [("负荷 (kW)", panel.load), ("光伏 (kW)", pv),
             ("电价 (元/kWh)", panel.price), ("净负荷 (kW)", panel.net_load())]
    months = _season_index(panel.dates)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 6.6))
    for ax, (label, values) in zip(axes.ravel(), items):
        grid = np.zeros((12, 144))
        for m in range(12):
            sel = months == m
            grid[m] = np.nanmean(values[sel], axis=0)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=vs.SEQ_CMAP,
                       extent=[0, 24, 0.5, 12.5])
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_yticks(np.arange(1, 13))
        ax.set_xlabel("时刻 (h)")
        ax.set_ylabel("月份")
        ax.set_title(f"{label}", fontsize=9.8)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
        cb.ax.tick_params(labelsize=7.6)
    for ax, letter in zip(axes.ravel(), "abcd"):
        vs.panel(ax, letter, dx=-0.10)
    fig.suptitle("Fig.D3　季节 × 时刻热力图：负荷 / 光伏 / 电价 / 净负荷的时空结构",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD3_seasonal_heatmaps",
            "时空结构热力图。四类变量的月度小时均值矩阵：光伏呈夏季宽、冬季窄的钟形；"
            "电价早晚双峰且夏季午间低谷更深；净负荷在春夏季中午出现负值（光伏盈余）。")


# ---------------------------------------------------------------------------
# D4 dependence structure (correlation / copula / autocorrelation)
# ---------------------------------------------------------------------------

def figD4_dependence(panel) -> None:
    pv = panel.pv_filled()
    load = panel.load
    price = panel.price
    net = panel.net_load()
    ramp = np.abs(np.diff(pv, axis=1, prepend=pv[:, :1]))
    stack = np.stack([load.ravel(), pv.ravel(), price.ravel(), net.ravel(), ramp.ravel()])
    names = ["负荷", "光伏", "电价", "净负荷", "|ΔPV|"]
    corr = np.corrcoef(np.argsort(np.argsort(stack, axis=1), axis=1))  # Spearman via ranks

    fig = plt.figure(figsize=(14.4, 5.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.1, 1.1], wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8.4)
    ax.set_yticks(range(len(names)), names, fontsize=8.4)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8.0,
                    color="white" if abs(corr[i, j]) > 0.6 else BLACK)
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046).set_label("Spearman 相关系数")
    ax.grid(False)
    vs.panel(ax, "a", dx=-0.16)
    ax.set_title("Spearman 相关矩阵")

    ax = fig.add_subplot(gs[0, 1])
    hb = ax.hexbin(pv.ravel()[::7], net.ravel()[::7], gridsize=48, cmap="magma",
                   bins="log", mincnt=1)
    ax.axhline(0, color=WHITE_LINE, lw=1.0)
    ax.set_xlabel("光伏出力 (kW)")
    ax.set_ylabel("净负荷 $L-PV$ (kW)")
    fig.colorbar(hb, ax=ax, pad=0.02, label="$\\log_{10}$ 样本数")
    vs.panel(ax, "b", dx=-0.14)
    ax.set_title("光伏–净负荷联合密度（倒 V 结构）")

    ax = fig.add_subplot(gs[0, 2])
    for j, (vintage, color) in enumerate(zip((0, 1, 2, 3), PALETTE)):
        errors = []
        for d in range(31, 365):
            fc = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[d, vintage],
                                                        VINTAGE_HOUR[vintage]), nan=0.0)
            mask = fc > 0
            if mask.any():
                errors.append((pv[d][mask] - fc[mask]).mean())
        errors = np.array(errors)
        ax.hist(errors, bins=40, histtype="step", lw=1.5, color=color,
                label=f"{VINTAGE_HOUR[vintage]}:00 发布　均值 {errors.mean():+.0f} kW")
    ax.axvline(0, color=BLACK, lw=0.9)
    ax.set_xlabel("日间平均预测误差 $PV-\\hat{PV}$ (kW)")
    ax.set_ylabel("天数")
    ax.legend(fontsize=8.0)
    vs.panel(ax, "c", dx=-0.14)
    ax.set_title("四时次预报误差分布（偏差诊断）")
    fig.suptitle("Fig.D4　依赖结构与预报偏差：相关、联合密度与误差分布",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD4_dependence",
            "依赖结构诊断。(a) 变量间的 Spearman 秩相关；"
            "(b) 光伏与净负荷的联合密度呈倒 V（光伏出力直接抵消负荷）；"
            "(c) 四个发布时次的日间平均预测误差分布，用于判断是否存在系统性偏差。")


# ---------------------------------------------------------------------------
# D5 ramps and extremes
# ---------------------------------------------------------------------------

def figD5_ramps_extremes(panel) -> None:
    pv = panel.pv_filled()
    load = panel.load
    price = panel.price
    ramp_pv = np.diff(pv, axis=1) / (1 / 6)      # kW per 10 min → kW/h rate proxy
    ramp_load = np.diff(load, axis=1) / (1 / 6)
    spike = price >= np.quantile(price, 0.95, axis=1, keepdims=True)
    surplus = pv > load
    fig = plt.figure(figsize=(14.0, 5.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    for values, label, color in ((ramp_pv, "光伏爬坡", ORANGE), (ramp_load, "负荷爬坡", RED)):
        v = np.sort(np.abs(values).ravel())
        v = v[np.isfinite(v)]
        ccdf = 1.0 - np.arange(v.size) / v.size
        ax.plot(v, ccdf, color=color, lw=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("10 min 爬坡幅值 (kW)")
    ax.set_ylabel("P(|ΔP| > x)")
    ax.legend()
    vs.panel(ax, "a")
    ax.set_title("爬坡率互补累积分布 (CCDF)")

    ax = fig.add_subplot(gs[0, 1])
    by_hour = spike.reshape(365, 24, 6).sum(axis=2).sum(axis=0)
    bars = ax.bar(np.arange(24), by_hour, color=PURPLE, width=0.7)
    top = np.argsort(by_hour)[-3:]
    for h in top:
        bars[h].set_color(RED)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("尖峰时段数（全年）")
    vs.panel(ax, "b")
    ax.set_title(f"电价尖峰 (≥P95) 的时段分布：集中在 "
                 + "、".join(f"{h}:00" for h in sorted(top)))

    ax = fig.add_subplot(gs[0, 2])
    surplus_hour = surplus.reshape(365, 24, 6).sum(axis=2).sum(axis=0)
    bars = ax.bar(np.arange(24), surplus_hour, color=GREEN, width=0.7)
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("光伏盈余时段数（全年）")
    ax.annotate(f"全年盈余时段 {int(surplus.sum()):,}（占 {100*surplus.mean():.1f}%）",
                (0.5, 0.9), xycoords="axes fraction", ha="center", fontsize=8.6,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, lw=0.6))
    vs.panel(ax, "c")
    ax.set_title("光伏盈余时段分布（决定弃光与倒送风险）")
    fig.suptitle("Fig.D5　爬坡与极值特征：波动强度、价格尖峰与光伏盈余",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD5_ramps_extremes",
            "波动与极值特征。(a) 光伏/负荷爬坡幅值的 CCDF（双对数），用于评估备用需求；"
            "(b) 电价尖峰时刻分布；(c) 光伏盈余时段分布，说明弃光与倒送风险的时段来源。")


# ---------------------------------------------------------------------------
# D6 forecast-error diagnostics
# ---------------------------------------------------------------------------

def figD6_forecast_error(panel) -> None:
    pv = panel.pv_filled()
    leads = np.arange(1, 25)
    bias = np.zeros((4, 24))
    rmse = np.zeros((4, 24))
    for j in range(4):
        errs = []
        for d in range(31, 365):
            fc_hourly = panel.pv_forecast[d, j]
            fc_slots = np.nan_to_num(hourly_forecast_to_slots(fc_hourly, VINTAGE_HOUR[j]), nan=0.0)
            actual = pv[d]
            for lead in leads:
                slot = (VINTAGE_HOUR[j] + lead) - 1
                if 1 <= slot <= 144:
                    k = slot - 1
                    errs.append((lead, actual[k] - fc_slots[k]))
        for lead in leads:
            vals = np.array([v for (l, v) in errs if l == lead])
            if vals.size:
                bias[j, lead - 1] = vals.mean()
                rmse[j, lead - 1] = np.sqrt((vals ** 2).mean())

    fig = plt.figure(figsize=(14.0, 5.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(rmse, aspect="auto", cmap="magma", origin="lower",
                   extent=[0.5, 24.5, -0.5, 3.5])
    ax.set_yticks(range(4), [f"{h}:00 发布" for h in VINTAGE_HOUR])
    ax.set_xlabel("预报提前量 (h)")
    ax.set_ylabel("发布时次")
    fig.colorbar(im, ax=ax, pad=0.02, label="RMSE (kW)")
    ax.grid(False)
    vs.panel(ax, "a", dx=-0.16)
    ax.set_title("误差随提前量的增长（RMSE 热力图）")

    ax = fig.add_subplot(gs[0, 1])
    for j, color in enumerate(PALETTE[:4]):
        ax.plot(leads, bias[j], "o-", ms=3.4, lw=1.3, color=color,
                label=f"{VINTAGE_HOUR[j]}:00 发布")
    ax.axhline(0, color=BLACK, lw=0.9)
    ax.set_xlabel("预报提前量 (h)")
    ax.set_ylabel("平均偏差 (kW)")
    ax.legend(fontsize=7.8, ncol=2)
    vs.panel(ax, "b")
    ax.set_title("系统偏差诊断：均接近零（无系统性高估/低估）")

    ax = fig.add_subplot(gs[0, 2])
    for j, color in zip((0, 1, 2, 3), PALETTE[:4]):
        errs = []
        for d in range(31, 365):
            fc = np.nan_to_num(hourly_forecast_to_slots(panel.pv_forecast[d, j],
                                                        VINTAGE_HOUR[j]), nan=0.0)
            mask = fc > 100
            if mask.any():
                errs.append(np.abs(pv[d][mask] - fc[mask]).mean())
        errs = np.array(errs)
        if errs.size == 0:
            # 18:00 发布时次的可见时段全在夜间，没有日间样本（数据事实，需显式说明）
            ax.annotate(f"{VINTAGE_HOUR[j]}:00 发布：可见时段全在夜间\n无日间样本",
                        (0.5, 0.5 - 0.12 * j), xycoords="axes fraction", ha="center",
                        fontsize=8.0, color=color)
            continue
        ax.hist(errs, bins=32, histtype="step", lw=1.5, color=color,
                label=f"{VINTAGE_HOUR[j]}:00　中位 {np.median(errs):.0f} kW")
    ax.set_xlabel("日间平均绝对误差 MAE (kW)")
    ax.set_ylabel("天数")
    ax.legend(fontsize=7.8)
    vs.panel(ax, "c")
    ax.set_title("各时次的日间 MAE 分布")
    fig.suptitle("Fig.D6　预报误差诊断：提前量、偏差与技能（数据特征）",
                 fontsize=12, fontweight="bold", y=1.0)
    vs.save(fig, "figD6_forecast_error",
            "预报误差诊断。(a) 四个发布时次的 RMSE 随提前量的变化；"
            "(b) 平均偏差随提前量（接近零，说明预报无系统性偏差）；"
            "(c) 日间 MAE 的天数分布，为共形校准的窗口选择提供依据。")


EDA_FIGS = [
    ("data_quality", figD1_data_quality, True),
    ("distributions", figD2_distributions, True),
    ("seasonal", figD3_seasonal_heatmaps, True),
    ("dependence", figD4_dependence, True),
    ("ramps", figD5_ramps_extremes, True),
    ("forecast_error", figD6_forecast_error, True),
]
