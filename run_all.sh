#!/usr/bin/env bash
# 全流程一键重跑（单机 CPU）。
#   bash run_all.sh          跑全套
#   bash run_all.sh --quick  跳过 Q4 风险扫描
# 顺序：C++ 清洗 → Q1 → 共形层 → Q2 → Q3 → Q4 → 出表 → 画图 → 论文数字 → 门禁自检
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3}
BUILD=build/cleaning
T0=$(date +%s)

log() { printf '\n=== %s ===\n' "$1"; }

log "0/10 configure + build cleaning_cpp (C++17, -Wall -Wextra)"
cmake -S code/cleaning_cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$BUILD" -j "$(nproc)" 2>&1 | grep -Ev '^\[|Building|Built target' || true

log "1/10 self-test: golden cases + 200 synthetic panels"
"$BUILD/cleaning_cpp" selftest

log "2/10 clean + export + verify (NR1-NR12, invariants I1-I8)"
"$BUILD/cleaning_cpp" clean --in Data --out clean --templates "templates/附件5"
"$BUILD/cleaning_cpp" verify --in Data --out clean | tail -12

log "3/10 Q1 LP + DP cross-check + result1 payloads"
"$PY" -m code.optimize.q1_lp

log "4/10 conformal layer (0.80 / 0.25 / 0.70 coverage + ACI)"
"$PY" -m code.forecast.calibrate

log "5/10 Q2 two-stage stochastic backtest"
"$PY" -m code.optimize.q2_stochastic

log "6/10 Q3 rolling MPC + subset experiments S0-S5"
"$PY" -m code.optimize.q3_mpc

if [[ "${1:-}" == "--quick" ]]; then
  log "7/10 Q4 risk sweep skipped (--quick)"
else
  log "7/10 Q4 fluctuating tariff + CVaR/Copula/soft-constraint evidence"
  "$PY" -m code.optimize.q4_price
fi

log "8/10 write the five result workbooks from the attachment-5 templates"
"$PY" -m code.report.write_results

log "9/10 figures + paper numbers"
"$PY" -m code.forecast.uncertainty      # U1-U5：PDF 未明确项 → 共形区间
"$PY" -m code.report.visualize --set all   # 出版级制图：主图 + 灵敏度 + 计算架构（含基准测量）
"$PY" -m code.report.paper_numbers

log "10/10 verification gates G0-G9 + sensitivity"
"$PY" -m code.common.checks | tail -25

log "extra: independent xlsx cross-check (openpyxl read-only channel, optional)"
if /usr/bin/python3 -c "import openpyxl" >/dev/null 2>&1; then
  /usr/bin/python3 tools/cross_verify.py | tail -6
  "$PY" -m code.common.determinism | tail -3
else
  echo "openpyxl channel not available on /usr/bin/python3 — skipped"
fi

T1=$(date +%s)
printf '\n=== pipeline finished in %s s ===\n' "$((T1 - T0))"
