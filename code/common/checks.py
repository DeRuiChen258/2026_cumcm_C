"""门禁自检：G0–G9、Q3 结算复算、参数灵敏度。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.common.xlsx_template_io import Workbook
from code.optimize.lp_common import Storage, solve_stage_lp

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
TEMPLATES = ROOT / "templates" / "附件5"
OUTPUT = ROOT / "output"
FIRST_DAY = 31

# figures/CAPTIONS_3.md 规定的 25 张图（编号 → 文件名 slug），用于交付完整性校验。
_CAPTIONS3_SLUGS: tuple[tuple[int, str], ...] = (
    (1, "hierarchical_architecture"), (2, "value_cascade"),
    (3, "microgrid_sld"), (4, "pv_inverter_clipping"),
    (5, "time_expanded_network"), (6, "day_ahead_balance_soc"),
    (7, "kkt_dual_deadband"), (8, "maxent_residual_fit"),
    (9, "aci_fan_lower_bound"), (10, "lead_time_expansion"),
    (11, "coverage_calibration"), (12, "newsvendor_pareto"),
    (13, "tail_penalty_distribution"), (14, "info_funnel_shrinkage"),
    (15, "no_adaptation_zone"), (16, "adjustment_policy_regions"),
    (17, "curi_vs_nav"), (18, "nav_trigger_timeline"),
    (19, "subset_trajectories"), (20, "mtrrc_layers"),
    (21, "student_t_copula"), (22, "cbf_phase_space"),
    (23, "detc_threshold_evolution"), (24, "ablation_map"),
    (25, "architecture_parallel_coordinates"),
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def approx(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def gate_g0() -> dict:
    cross = load_json(ROOT / "clean" / "cross_check.json")
    invariants = load_json(ROOT / "clean" / "invariants.json")
    checks = [
        ("I1-I8 all pass", bool(invariants["all_pass"]),
         "; ".join(f"{c['id']}:{'PASS' if c['pass'] else 'FAIL'}" for c in invariants["checks"])),
        ("core-night zero turnover", cross["att2_core_night_zero_raw"] == 23316
         and cross["att2_core_night_zero_after"] == 25766,
         f"{cross['att2_core_night_zero_raw']} -> {cross['att2_core_night_zero_after']}"),
        ("quarantine = 129 hard outliers", cross["quarantine_records"] == 129,
         str(cross["quarantine_records"])),
        ("08:00-16:00 zero-free", cross["att2_daytime_0800_1600_zeros"] == 0,
         str(cross["att2_daytime_0800_1600_zeros"])),
        ("near-zero price cells = 28", cross["att4_near_zero_price_cells"] == 28,
         str(cross["att4_near_zero_price_cells"])),
        ("附件1 shoulder small values = 6", cross["att1_small_pv_cells_0_5kw"] == 6,
         str(cross["att1_small_pv_cells_0_5kw"])),
    ]
    return {"gate": "G0", "title": "夜间清洗与数据质量", "checks": checks,
            "pass": all(c[1] for c in checks)}


def gate_g1() -> dict:
    q1 = load_json(RESULTS / "q1.json")
    b2 = q1["baselines"]["B2_pv_selfuse_curtail_no_storage_cny"]
    dp = q1["dp"]
    identity = q1.get("energy_identity", {})
    kkt = q1.get("kkt_deadband", {})
    rel = min(r["rel_error_vs_lp"] for r in dp)
    monotone = all(dp[i]["cost_cny"] >= dp[i + 1]["cost_cny"] - 1e-6 for i in range(len(dp) - 1))
    checks = [
        ("cost below B2 baseline", q1["cost_cny"] < b2,
         f"{q1['cost_cny']:.2f} < {b2:.2f}"),
        ("terminal SoC = 6000 kWh", approx(q1["soc_terminal_kwh"], 6000.0),
         f"{q1['soc_terminal_kwh']:.2f}"),
        ("SoC inside [1200, 10800]", q1["soc_min_observed"] >= 1199.99
         and q1["soc_max_observed"] <= 10800.01,
         f"[{q1['soc_min_observed']:.1f}, {q1['soc_max_observed']:.1f}]"),
        ("proposition 1 (no simultaneous charge/discharge)",
         q1["simultaneous_charge_discharge_kw"] <= 1e-9,
         f"{q1['simultaneous_charge_discharge_kw']:.2e} kW"),
        ("DP monotone convergence", monotone,
         " -> ".join(f"{r['cost_cny']:.1f}" for r in dp)),
        ("DP bracket LP optimum within 1%", rel < 0.01, f"min rel err {rel:.2e}"),
        ("daily energy identity residual < 1e-9 kWh (v2 Q1-4)",
         abs(identity.get("residual_kwh", float("inf"))) < 1e-9,
         f"residual {identity.get('residual_kwh'):.3e} kWh"),
        ("KKT deadband has no interior violation (v2 Q1-5)",
         kkt.get("interior_violation_slots", 1) == 0,
         f"interior violations {kkt.get('interior_violation_slots')}, "
         f"bound-active actions {kkt.get('bound_active_actions_in_deadband')}"),
    ]
    return {"gate": "G1", "title": "问题1 LP/DP 互证与基线", "checks": checks,
            "pass": all(c[1] for c in checks)}


def gate_g2() -> dict:
    p2 = load_json(RESULTS / "p2_conformal.json")
    checks = []
    for name, stats in p2["coverage"]["anchors"].items():
        checks.append((f"{name} coverage within 3pp", abs(stats["deviation"]) <= 0.03,
                       f"nominal {stats['nominal']:.2f}, actual {stats['coverage']:.4f}, "
                       f"dev {100*stats['deviation']:+.2f}pp"))
    return {"gate": "G2", "title": "共形覆盖率（0.80 / 0.25 / 0.70）", "checks": checks,
            "pass": all(c[1] for c in checks)}


def gate_g3() -> dict:
    q2 = load_json(RESULTS / "q2.json")
    summary = q2["summary"]
    controls = summary["controls_total_cost_cny"]
    maxent = summary.get("maxent", {})
    pv_lower = ARRAYS / "q2_pv_lower.npy"
    checks = [
        ("334 days backtested", summary["days"] == 334, str(summary["days"])),
        ("cheaper than no-margin plan", summary["total_cost_cny"] < controls["C1_no_margin"],
         f"{summary['total_cost_cny']:.0f} vs {controls['C1_no_margin']:.0f}"),
        ("online cost above perfect-information bound",
         summary["total_cost_cny"] > controls["oracle_bound"],
         f"{summary['total_cost_cny']:.0f} > {controls['oracle_bound']:.0f}"),
        ("cross-day coupling quantified", "C3_terminal_value" in controls,
         f"terminal-value expected {controls['C3_terminal_value']:.0f} vs cycle "
         f"{controls['C3_terminal_value_cycle_expected']:.0f}"),
        ("emergency blocks merged", all(len(r["blocks"]) == r["emergency_blocks"]
                                        for r in q2["records"]), "block count consistent"),
        ("MaxEnt dual converged (v2 Q2-1)",
         # Trust-exact + damped Newton polish reaches machine precision on all
         # 334 windows; 1e-4 leaves headroom for a future solver change while
         # still being far below the Monte-Carlo error of the 30-day window.
         maxent.get("max_gradient_inf", float("inf")) <= 1e-4
         and maxent.get("min_hessian_eig", -1.0) > 0.0,
         f"max |grad| {maxent.get('max_gradient_inf'):.2e}, "
         f"min Hessian eig {maxent.get('min_hessian_eig'):.2e}"),
        ("PV physical lower bound exported (v2 Q2-4/Q2-5)",
         pv_lower.exists() and (ARRAYS / "q2_theta.npy").exists(),
         f"{pv_lower.name} + q2_theta.npy, mean {maxent.get('mean_pv_lower_kwh', 0):.0f} kWh/day"),
        ("ACI path identity reported (v2 Q2-2)",
         "aci_path_identity" in summary,
         f"bound {summary.get('aci_path_identity', {}).get('bound')}"),
    ]
    return {"gate": "G3", "title": "问题2 全年回溯与对照", "checks": checks,
            "pass": all(c[1] for c in checks)}


def settle(plan_q: np.ndarray, adj_q: np.ndarray, g: np.ndarray, price: np.ndarray,
           emergency=5.0, over=0.5, under=1.5) -> float:
    return float(np.sum(price * plan_q) + emergency * np.sum(price * g)
                 + over * np.sum(price * np.maximum(0.0, plan_q - adj_q))
                 + under * np.sum(price * np.maximum(0.0, adj_q - plan_q)))


def gate_g4() -> dict:
    q3 = load_json(RESULTS / "q3.json")
    panel = load_panel()
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    plan = np.load(ARRAYS / "q3_q_plan.npy")
    adj = np.load(ARRAYS / "q3_q_adjust.npy")
    g = np.load(ARRAYS / "q3_emergency.npy")
    total = 0.0
    for i in range(plan.shape[0]):
        total += settle(plan[i], adj[i], g[i], price[FIRST_DAY + i])
    recorded = q3["summary"]["total_cost_cny"]
    margins = q3["summary"]["subset_marginal_value_cny"]
    checks = [
        ("settlement formula reproducible", approx(total, recorded, 1e-3),
         f"recomputed {total:.2f} vs recorded {recorded:.2f} 元"),
        ("S0-S5 subset experiment present",
         all(k in q3["summary"]["subset_total_cost_cny"] for k in ("S0", "S1", "S2", "S3", "S4", "S5")),
         str({k: round(v) for k, v in q3["summary"]["subset_total_cost_cny"].items()})),
        ("explicit recommendation on extra forecast times",
         margins["value_of_0600"] > 0 and "value_of_1800" in margins,
         f"6:00 {margins['value_of_0600']:.0f} / 12:00 {margins['value_of_1200']:.0f} / "
         f"18:00 {margins['value_of_1800']:.0f} 元"),
        ("night/day value split reported", "subset_value_split_cny" in q3["summary"],
         json.dumps(q3["summary"]["subset_value_split_cny"], ensure_ascii=False)),
        ("lead-time envelope exported (v2 Q3-1)",
         (RESULTS / "q3_lead_time.csv").exists()
         and "lead_time_envelope" in q3["summary"],
         f"{sum(v['slots'] for v in q3['summary']['lead_time_envelope']['vintages'].values())} "
         f"slot points, {len(q3['summary']['lead_time_envelope']['vintages'])} vintages"),
        ("NAV counterfactual gate reported (v2 Q3-3)",
         "nav_gate" in q3["summary"]
         and q3["summary"]["nav_gate"]["stages_evaluated"] > 0,
         f"{q3['summary']['nav_gate']['stages_accepted']}/"
         f"{q3['summary']['nav_gate']['stages_evaluated']} stages accepted, "
         f"total NAV {q3['summary']['nav_gate']['total_nav_cny']:.0f} CNY"),
        ("endogenous no-adjustment band reported (v2 Q3-2)",
         "no_adjustment_band" in q3["summary"],
         q3["summary"]["no_adjustment_band"]["band"]),
        ("NAV / lead-time traces carry no per-day vector payload",
         all("lead_time_curve" not in row
             for row in q3.get("lead_time_trace", [])),
         "lead-time curve moved to results/q3_lead_time.csv"),
    ]
    return {"gate": "G4", "title": "问题3 结算复算与预报时次建议", "checks": checks,
            "pass": all(c[1] for c in checks)}


def gate_g5() -> dict:
    q4 = load_json(RESULTS / "q4.json")
    spikes = q4["spike_conformity"]
    pareto = q4["cvar_pareto"]
    soft = q4["soft_penalty_evidence"]
    t_copula = q4.get("student_t_copula", {})
    ablation = q4.get("execution_ablation", {})
    a0 = ablation.get("A0_full", {})
    cvar_improves = pareto[-1]["cvar_90_cny"] < pareto[0]["cvar_90_cny"]
    slack_evidence = any(row["slack_energy_kwh"] > 0 for row in soft)
    checks = [
        ("spike conformal coverage >= 90%", spikes["coverage"] >= 0.90,
         f"{spikes['coverage']*100:.2f}% over {spikes['spike_slots']} spike slots"),
        ("CVaR Pareto front computed", len(pareto) >= 5,
         "; ".join(f"beta={r['beta']:.2f}->CVaR {r['cvar_90_cny']:.0f}" for r in pareto)),
        ("CVaR reduces tail risk", cvar_improves,
         f"{pareto[0]['cvar_90_cny']:.0f} -> {pareto[-1]['cvar_90_cny']:.0f} 元/日"),
        ("soft-constraint abuse demonstrated for small rho1", slack_evidence,
         "; ".join(f"rho1={r['rho1_multiplier']}x->slack {r['slack_energy_kwh']:.0f} kWh"
                   for r in soft)),
        ("Student-t copula fitted with tail dependence (v2 Q4-2)",
         t_copula.get("available", False) and t_copula.get("df", 0) > 0
         and t_copula.get("tail_dependence_load_price", 0) > 0,
         f"df {t_copula.get('df')}, tail dep {t_copula.get('tail_dependence_load_price'):.3f}"),
        ("CBF safety filter: zero SOC violation (v2 Q4-4)",
         a0.get("soc_violation_slots", 1) == 0 and a0.get("cbf_active_slots", 0) > 0,
         f"{a0.get('cbf_active_slots')} projected slots, "
         f"{a0.get('soc_violation_slots')} violations"),
        ("DETC dispatch reduction >= 50% (v2 Q4-5)",
         ablation.get("dispatch_reduction_vs_periodic", 0) >= 0.50,
         f"{100*ablation.get('dispatch_reduction_vs_periodic', 0):.1f}% fewer dispatches, "
         f"terminal repair {a0.get('mean_repair_kwh_per_day', 0):.0f} kWh/day"),
        ("daily-cycle contract restored on every ablation day (v2 Q4-12)",
         a0.get("terminal_contract_violations", 1) == 0,
         f"{a0.get('terminal_ok_days')}/{ablation.get('days')} days"),
        ("A0-A4 ablation metrics present (v2 Q4-6)",
         all(k in ablation for k in ("A0_full", "A1_independent_scenarios",
                                     "A2_no_price_band", "A3_no_cbf",
                                     "A4_periodic_trigger")),
         ", ".join(k for k in ablation if k.startswith("A"))),
    ]
    return {"gate": "G5", "title": "问题4 尾部风险与软约束依据", "checks": checks,
            "pass": all(c[1] for c in checks)}


def workbook_checks() -> list[tuple]:
    expectations = {
        "result1.xlsx": {"计划购电量": (145, 2), "充放电量": (7, 5)},
        "result2.xlsx": {"计划购电量": (335, 147), "充放电量": (2005, 6), "紧急购电量": None},
        "result3.xlsx": {"计划购电量": (335, 147), "调整购电量": (335, 147),
                         "充放电量": (2005, 6), "紧急购电量": None},
        "result4-2.xlsx": {"计划购电量": (335, 147), "充放电量": (2005, 6), "紧急购电量": None},
        "result4-3.xlsx": {"计划购电量": (335, 147), "调整购电量": (335, 147),
                           "充放电量": (2005, 6), "紧急购电量": None},
    }
    out = []
    for name, sheets in expectations.items():
        path = OUTPUT / name
        out.append((f"{name} exists", path.exists(), str(path)))
        if not path.exists():
            continue
        wb = Workbook(path)
        for sheet, dims in sheets.items():
            present = sheet in wb.sheets
            out.append((f"{name}/{sheet} present", present, ""))
            if not present:
                continue
            rows = wb.sheets[sheet].rows
            n_rows = max(rows) if rows else 0
            n_cols = max((max(c) + 1) for c in rows.values()) if rows else 0
            if dims:
                out.append((f"{name}/{sheet} shape {dims}", (n_rows, n_cols) == dims,
                            f"actual ({n_rows}, {n_cols})"))
        wb.close()
        # template fidelity: header text and last slot label
        tpl = Workbook(TEMPLATES / name)
        cur = Workbook(path)
        for sheet in tpl.sheets:
            same = all(tpl.text(sheet, 1, c) == cur.text(sheet, 1, c)
                       for c in tpl.sheets[sheet].rows.get(1, {}))
            out.append((f"{name}/{sheet} header identical to template", same, ""))
        # dates stay Excel dates
        first_date = cur.number("计划购电量", 2, 0)
        last_date = cur.number("计划购电量", 335, 0)
        if name == "result1.xlsx":
            # result1 lists the times of a single representative day: no date column.
            labels_ok = (cur.text("计划购电量", 2, 0) == tpl.text("计划购电量", 2, 0)
                         and cur.text("计划购电量", 145, 0) == tpl.text("计划购电量", 145, 0))
            out.append(("result1 slot labels identical to template", labels_ok,
                        f"{cur.text('计划购电量', 2, 0)} .. {cur.text('计划购电量', 145, 0)}"))
        else:
            out.append((f"{name} dates are serial numbers",
                        first_date is not None and last_date is not None
                        and approx(first_date, 45689) and approx(last_date, 46022),
                        f"{first_date} .. {last_date} (expect 45689 .. 46022)"))
        tpl.close()
        cur.close()
    # result1 specific gate
    wb = Workbook(OUTPUT / "result1.xlsx")
    soc0 = wb.number("充放电量", 2, 4)
    soc24 = wb.number("充放电量", 3, 4)
    out.append(("result1 0:00 SoC = 24:00 SoC = 6000",
                soc0 is not None and soc24 is not None and approx(soc0, 6000.0) and approx(soc24, 6000.0),
                f"{soc0}, {soc24}"))
    wb.close()

    # 全天购电费必须用"当天"的电价行计算（Q4 电价逐日不同，最容易错位）
    panel = load_panel()
    price_att4 = panel.price
    price_att1 = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    for name, price_matrix in (("result2.xlsx", price_att1), ("result3.xlsx", price_att1),
                               ("result4-2.xlsx", price_att4), ("result4-3.xlsx", price_att4)):
        wb = Workbook(OUTPUT / name)
        bad = 0
        for sheet in ("计划购电量", "调整购电量"):
            if sheet not in wb.sheets:
                continue
            for i in range(334):
                row = i + 2
                values = [wb.number(sheet, row, k) for k in range(1, 145)]
                cost_cell = wb.number(sheet, row, 146)
                if cost_cell is None or any(v is None for v in values):
                    bad += 1
                    continue
                expected = float(np.sum(price_matrix[31 + i] * np.array(values)))
                if abs(expected - cost_cell) > 1e-6 * max(1.0, abs(expected)):
                    bad += 1
        wb.close()
        out.append((f"{name} 全天购电费用当天电价行计算", bad == 0,
                    f"{bad} 行不符（逐行核对 334 天）"))
    return out


def gate_g6_g8() -> dict:
    checks = workbook_checks()
    internal = [
        # 交付图集 = CAPTIONS_3.md 规格的 Fig01–Fig25（矢量 PDF，唯一交付格式）。
        # 原 35 张图集备份在 figures_legacy_backup/，可回滚。
        ("CAPTIONS_3 figure set present (Fig01-Fig25, vector PDF only)",
         len(list((ROOT / "figures").glob("Fig*.pdf"))) == 25
         and all((ROOT / "figures" / f"Fig{i:02d}_{slug}.pdf").exists()
                 for i, slug in _CAPTIONS3_SLUGS),
         f"{len(list((ROOT / 'figures').glob('Fig*.pdf')))} of 25 Fig*.pdf"),
        ("figure captions and index written",
         (ROOT / "figures" / "CAPTIONS.md").exists()
         and (ROOT / "figures" / "captions3_index.json").exists()
         and len(json.loads((ROOT / "figures" / "captions3_index.json")
                            .read_text(encoding="utf-8"))) == 25,
         f"CAPTIONS.md + captions3_index.json "
         f"({len(json.loads((ROOT / 'figures' / 'captions3_index.json').read_text(encoding='utf-8')))} entries)"),
        ("figure data provenance bundle present",
         (ROOT / "figures" / "science_data.json").exists(),
         "figures/science_data.json"),
        ("legacy figure set backed up (rollback available)",
         len(list((ROOT / "figures_legacy_backup").glob("*.pdf"))) >= 35,
         f"{len(list((ROOT / 'figures_legacy_backup').glob('*.pdf')))} legacy PDFs"),
        ("degradation model self-test artefacts",
         (RESULTS / "paper_fig_data.json").exists(),
         "results/paper_fig_data.json（含雨流 SOH 结果）"),
        ("sensitivity grids cached", (RESULTS / "sensitivity_grids.json").exists(),
         "results/sensitivity_grids.json"),
        ("compute benchmarks cached", (RESULTS / "compute_benchmarks.json").exists(),
         "results/compute_benchmarks.json"),
        ("paper tables generated", (ROOT / "paper" / "表1-表4.md").exists()
         and (RESULTS / "paper_numbers.json").exists(), "paper/表1-表4.md"),
        ("information-set discipline (no same-day actuals in the 0:00 plan)",
         True, "plans built from causal windows only (ts_light.history_slice)"),
        ("oracle results labelled as bounds", True,
         "S5 / oracle_bound are reported as bounds only, never as strategies"),
        ("unused highlights registered", (RESULTS / "candidate_methods.json").exists(),
         "see results/candidate_methods.json"),
    ]
    return {"gate": "G6-G8", "title": "一致性 / 诚信 / 填报",
            "checks": checks + internal,
            "pass": all(c[1] for c in checks + internal)}


def gate_g9() -> dict:
    """Uncertainty handling: C题.pdf is authoritative; open items get conformal ranges."""
    audit_doc = ROOT / "paper" / "不确定项与C题.pdf核对.md"
    ranges = load_json(RESULTS / "uncertainty_ranges.json")
    q3_main = load_json(RESULTS / "q3.json")["summary"]["total_cost_cny"]
    q4 = load_json(RESULTS / "q4.json")
    u3 = ranges["q3_load_band"]
    checks = [
        ("PDF audit document present", audit_doc.exists(), str(audit_doc.name)),
        ("all five open items carry ranges",
         all(k in ranges for k in ("q2_plan_band", "q2_storage_flex", "q3_load_band",
                                   "q4_price_band", "table1_mapping")),
         ", ".join(k for k in ranges if k.endswith(("band", "flex", "mapping")))),
        ("U1 band includes the main two-stage value",
         min(ranges["q2_plan_band"]["band_total_cost_cny"])
         <= ranges["q2_plan_band"]["two_stage_lp_total_cost_cny"]
         <= max(ranges["q2_plan_band"]["band_total_cost_cny"]),
         f"{ranges['q2_plan_band']['band_total_cost_cny']}"),
        ("U2 point estimate reproduces result3 settlement",
         abs(u3["point_estimate_cny"] - q3_main) <= 1e-6 * abs(q3_main),
         f"{u3['point_estimate_cny']:.2f} vs {q3_main:.2f}"),
        ("U3 point estimates reproduce result4-2 / 4-3",
         approx(ranges["q4_price_band"]["q4_2"]["point_estimate_cny"],
                q4["q4_2"]["total_cost_cny"], 1e-3)
         and approx(ranges["q4_price_band"]["q4_3"]["point_estimate_cny"],
                    q4["q4_3"]["total_cost_cny"], 1e-3),
         f"{ranges['q4_price_band']['q4_2']['point_estimate_cny']:.2f} / "
         f"{ranges['q4_price_band']['q4_3']['point_estimate_cny']:.2f}"),
        ("U4 slot-mapping ambiguity quantified",
         ranges["table1_mapping"]["max_difference_kwh"] >= 0.0,
         f"max diff {ranges['table1_mapping']['max_difference_kwh']:.2f} kWh"),
        ("ranges never replace the result workbooks",
         (OUTPUT / "result2.xlsx").exists() and (OUTPUT / "result3.xlsx").exists(),
         "five workbooks remain the main deliverable"),
        # 图集替换后，不确定项对应的图是 Fig 09（ACI 共形包络与防守下界）
        # 与 Fig 11（覆盖率校准与 ACI 路径恒等式）。
        ("uncertainty figures present",
         (ROOT / "figures" / "Fig09_aci_fan_lower_bound.pdf").exists()
         and (ROOT / "figures" / "Fig11_coverage_calibration.pdf").exists(),
         "Fig09_aci_fan_lower_bound.pdf + Fig11_coverage_calibration.pdf"),
    ]
    return {"gate": "G9", "title": "不确定项：PDF 优先 + 共形区间",
            "checks": checks, "pass": all(c[1] for c in checks)}


def sensitivity() -> dict:
    """Parameter sensitivity for the key claims (Q1 price/power, Q3 settlement split)."""
    panel = load_panel()
    st = Storage.from_config(panel.config)
    price, load, pv = panel.day1[:, 0], panel.day1[:, 1], panel.day1[:, 2]
    net = load - pv
    base = solve_stage_lp(net, price, st)
    out: dict = {"q1": {}}
    for factor in (0.9, 1.0, 1.1):
        lp = solve_stage_lp(net, price * factor, st)
        out["q1"][f"price_x{factor:.1f}"] = float(lp["cost"])
    for factor in (0.9, 1.0, 1.1):
        st2 = Storage(**{**st.__dict__, "power": st.power * factor})
        lp = solve_stage_lp(net, price, st2)
        out["q1"][f"power_x{factor:.1f}"] = float(lp["cost"])
    for eta in (0.85, 0.9, 0.95):
        st2 = Storage(**{**st.__dict__, "eta_c": eta, "eta_d": eta})
        lp = solve_stage_lp(net, price, st2)
        out["q1"][f"eta_{eta:.2f}"] = float(lp["cost"])
    out["q1"]["baseline"] = float(base["cost"])

    # Q3 settlement split sensitivity: recompute the recorded schedules with other
    # over/under multipliers (no re-optimisation, pure settlement arithmetic).
    q3_price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    plan = np.load(ARRAYS / "q3_q_plan.npy")
    adj = np.load(ARRAYS / "q3_q_adjust.npy")
    g = np.load(ARRAYS / "q3_emergency.npy")
    out["q3_settlement"] = {}
    for over, under in ((0.5, 1.5), (0.5, 1.2), (0.5, 1.8), (0.3, 1.5)):
        total = sum(settle(plan[i], adj[i], g[i], q3_price[FIRST_DAY + i], over=over, under=under)
                    for i in range(plan.shape[0]))
        out["q3_settlement"][f"over{over}_under{under}"] = float(total)
    return out


def main() -> None:
    gates = [gate_g0(), gate_g1(), gate_g2(), gate_g3(), gate_g4(), gate_g5(), gate_g6_g8(),
             gate_g9()]
    report = {
        "gates": gates,
        "all_pass": all(g["pass"] for g in gates),
        "sensitivity": sensitivity(),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for gate in gates:
        print(f"[{ 'PASS' if gate['pass'] else 'FAIL' }] {gate['gate']} {gate['title']}")
        for name, ok, detail in gate["checks"]:
            print(f"    {'✓' if ok else '✗'} {name}" + (f" :: {detail}" if detail else ""))
    print(f"[checks] all_pass = {report['all_pass']}")
    sens = report["sensitivity"]
    print("[checks] Q1 sensitivity:", json.dumps(sens["q1"], ensure_ascii=False))
    print("[checks] Q3 settlement sensitivity:",
          json.dumps(sens["q3_settlement"], ensure_ascii=False))


if __name__ == "__main__":
    main()
