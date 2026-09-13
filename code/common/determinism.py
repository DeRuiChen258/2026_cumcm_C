"""确定性检查：把关键阶段再跑一遍，比文件哈希是不是一模一样。

整个流程没做随机采样，输入不变就应该跑出同样的结果。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from code.common.load_clean import ROOT

TARGETS = ["results/q1.json", "results/q2.json", "results/q3.json",
           "results/arrays/q2_q_plan.npy", "results/arrays/q3_q_adjust.npy",
           "output/result1.xlsx", "output/result2.xlsx", "output/result3.xlsx",
           # v2: the Q4 execution layer and the xlsx writer it feeds.
           "results/q4.json", "results/arrays/q4_2_charge.npy",
           "output/result4-2.xlsx", "output/result4-3.xlsx"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot() -> dict[str, str]:
    return {name: sha256(ROOT / name) for name in TARGETS}


def run_stages() -> None:
    import numpy as np

    from code.common.load_clean import load_panel
    from code.optimize.q2_stochastic import save as save_q2, solve_q2
    from code.optimize.q3_mpc import save as save_q3, solve_q3
    from code.optimize.q4_price import main as q4_main
    from code.optimize.q1_lp import main as q1_main
    from code.report.write_results import main as write_main

    q1_main()
    panel = load_panel()
    tariff = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    save_q2(solve_q2(panel, tariff, tag="q2"), "q2", list(panel.dates))
    save_q3(solve_q3(panel, tariff, tag="q3"), "q3")
    q4_main()
    write_main()


def main() -> None:
    before = snapshot()
    run_stages()
    after = snapshot()
    report = {
        "files": {name: {"before": before[name], "after": after[name],
                         "identical": before[name] == after[name]}
                  for name in TARGETS},
        "all_identical": all(before[name] == after[name] for name in TARGETS),
        "note": "second run of Q1/Q2/Q3 + workbook writer on identical inputs",
    }
    (ROOT / "results" / "determinism.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, info in report["files"].items():
        print(f"[determinism] {'OK ' if info['identical'] else 'DIFF'} {name}")
    print(f"[determinism] all_identical = {report['all_identical']}")
    sys.exit(0 if report["all_identical"] else 1)


if __name__ == "__main__":
    main()
