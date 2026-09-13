"""全项目共用的绘图风格。

中英文字体、Okabe-Ito 色盲友好配色、内向刻度、面板编号、轴标签带单位都在这设好，
默认只出矢量 PDF；每次出图的图注统一收进 figures/CAPTIONS.md，写论文直接抄。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"

# Okabe-Ito 色盲友好配色
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
GREY = "#7F7F7F"
BLACK = "#1A1A1A"
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY]
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"

CAPTIONS: dict[str, str] = {}
EXPORT_FORMATS: tuple[str, ...] = ("pdf",)   # overridden by the CLI `--formats`


def setup(font: str = "Noto Sans CJK SC") -> None:
    rcParams.update({
        "font.family": [font, "Noto Sans CJK JP", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "axes.titlepad": 7,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": BLACK,
        "axes.labelcolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.color": "#BFBFBF",
        "grid.alpha": 0.35,
        "grid.linewidth": 0.5,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


def panel(ax, letter: str, dx: float = -0.085, dy: float = 1.045) -> None:
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")


def save(fig, name: str, caption: str, formats=None) -> list[str]:
    """存图：默认写 figures/<名字>.pdf，临时要位图就把 formats 传成 ('pdf', 'png')。"""
    FIGDIR.mkdir(exist_ok=True)
    formats = EXPORT_FORMATS if formats is None else formats
    written = []
    for fmt in formats:
        path = FIGDIR / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        written.append(path.name)
    CAPTIONS[name] = caption
    plt.close(fig)
    return written


def flush_captions() -> None:
    """把这次的图注并进索引，不覆盖之前跑过的那些条目。"""
    index_path = FIGDIR / "figures_index.json"
    merged: dict[str, str] = {}
    if index_path.exists():
        try:
            merged = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:      # 索引坏了就用本次结果重建
            merged = {}
    merged.update(CAPTIONS)
    (FIGDIR / "CAPTIONS.md").write_text(
        "# 图目录与图注（SCI 投稿用）\n\n"
        "> 全部图由 `python3 -m code.report.visualize --set all` 生成，**统一为矢量 PDF**；"
        "如需位图可用 `--formats pdf,png` 追加导出。\n\n"
        + "\n\n".join(f"**{name}** — {text}" for name, text in sorted(merged.items()))
        + "\n", encoding="utf-8")
    index_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def annotate_bars(ax, bars, values, fmt="{:.0f}", dy=0.01, fontsize=8.2) -> None:
    span = max(values) - min(0.0, min(values))
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value + dy * span),
                    ha="center", va="bottom", fontsize=fontsize)


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20250911):
    """均值的百分位自助法置信区间，种子写死，重跑结果一样。"""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
