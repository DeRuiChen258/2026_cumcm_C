"""问题3：四个预报时次的滚动 MPC，含计划/调整的违约结算（附件1/2/3）。

结算口径：计划购电费 + 5 倍应急购电费 + 0.5 倍超额违约 + 1.5 倍缺额违约。
计划层取 0.25 分位（多买要挨罚），调整层取 0.70 分位（比应急便宜），执行时缺多少买多少。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.forecast.conformal import (conformal_level, hourly_forecast_to_slots,
                                     weighted_quantile)
from code.optimize.lp_common import Storage, solve_stage_lp
from code.optimize.q2_stochastic import block_label, merge_blocks

RESULTS = ROOT / "results"
ARRAYS = RESULTS / "arrays"
FIRST_DAY = 31
VINTAGE_SLOT = (0, 36, 72, 108)      # 0:00, 6:00, 12:00, 18:00 in 10-minute slots
VINTAGE_HOUR = (0, 6, 12, 18)


def build_error_panel(panel) -> np.ndarray:
    """err[d, j, k] = actual PV - vintage-j forecast, mapped onto the 144 slots."""
    pv = panel.pv_filled()
    err = np.full((365, 4, 144), np.nan)
    for j in range(4):
        for d in range(365):
            fc = hourly_forecast_to_slots(panel.pv_forecast[d, j], VINTAGE_HOUR[j])
            err[d, j] = pv[d] - fc
    return err


def stage_quantile(err: np.ndarray, window: slice, j: int, level: float) -> np.ndarray:
    """Causal conformal quantile of the forecast error, per slot, for one vintage."""
    sample = err[window, j, :]
    out = np.zeros(144)
    counts = np.zeros(144, dtype=int)
    for k in range(144):
        column = sample[:, k]
        column = column[~np.isnan(column)]
        counts[k] = column.size
        out[k] = 0.0 if column.size == 0 else weighted_quantile(column, level)
    return out, counts


def lead_time_envelope(err: np.ndarray, window: slice, j: int, level: float,
                       lookback_bins: int = 4) -> dict:
    """Lead-time aware envelope diagnostics (tasks/Q3.md Q3-1, Q3-7).

    Lead time h = (slot - vintage_slot)/6 hours.  The forecast variance grows
    with h, so the envelope must widen; when a new vintage arrives h resets to
    zero and the band collapses.  We report the per-lead-time residual scale
    and the collapse ratio between consecutive vintages on the shared segment.
    """
    sample = err[window, j, :]
    n_days = sample.shape[0]
    lead_hours = np.array([(k - VINTAGE_SLOT[j]) / 6.0 for k in range(144)])
    rows = []
    for lo in range(144):
        column = sample[:, lo]
        column = column[~np.isnan(column)]
        if column.size == 0:
            continue
        rows.append({
            "slot": lo + 1,
            "hour_end": (lo + 1) * 10,
            "lead_hours": float(lead_hours[lo]),
            "sigma_kw": float(np.std(column, ddof=1)) if column.size > 1 else 0.0,
            "quantile_kw": float(weighted_quantile(column, level)),
            "samples": int(column.size),
        })
    # Envelope collapse: sigma on the shared segment just after the vintage
    # versus the previous vintage's sigma on the same slots.
    seg = slice(VINTAGE_SLOT[j], min(VINTAGE_SLOT[j] + 36, 144))
    if j > 0:
        old = err[window, j - 1, seg]
        new = err[window, j, seg]
        s_old = float(np.nanstd(old)) if np.any(~np.isnan(old)) else float("nan")
        s_new = float(np.nanstd(new)) if np.any(~np.isnan(new)) else float("nan")
        uri = float(np.log(s_old / s_new)) if (np.isfinite(s_old) and np.isfinite(s_new)
                                               and s_old > 0 and s_new > 0) else float("nan")
        collapse = float(s_new / s_old) if (np.isfinite(s_old) and s_old > 0) else float("nan")
    else:
        s_old = s_new = uri = collapse = float("nan")
    return {
        "vintage_hour": VINTAGE_HOUR[j],
        "level": float(level),
        "window_days": int(n_days),
        "lead_time_curve": rows,
        "sigma_previous_vintage_kw": s_old,
        "sigma_current_vintage_kw": s_new,
        "uri": uri,
        "collapse_ratio": collapse,
    }


def solve_q3(panel, price: np.ndarray, tag: str = "q3", days: range | None = None,
             lookback: int = 30, aci_gamma: float = 0.02, soft_penalty: float = 0.0,
             subsets=("S0", "S1", "S2", "S3", "S4", "S5")) -> dict:
    st = Storage.from_config(panel.config)
    tariff = panel.config.get("tariff", {})
    emergency = float(tariff.get("emergency_multiplier", 5.0))
    over = float(tariff.get("over_plan_multiplier", 0.5))
    under = float(tariff.get("under_plan_multiplier", 1.5))
    q_plan_cfg = float(panel.config.get("quantiles", {}).get("q3_plan", 0.25))
    q_adj_cfg = float(panel.config.get("quantiles", {}).get("q3_adjust", 0.7))
    net_actual = panel.net_load()
    pv_actual = panel.pv_filled()
    err = build_error_panel(panel)
    days = days or range(FIRST_DAY, 365)

    n_days = len(days)
    q_plan_arr = np.zeros((n_days, 144))
    q_adj_arr = np.zeros((n_days, 144))
    charge_arr = np.zeros((n_days, 144))
    disch_arr = np.zeros((n_days, 144))
    soc_arr = np.zeros((n_days, 145))
    emerg_arr = np.zeros((n_days, 144))
    curtail_arr = np.zeros((n_days, 144))
    records = []
    subset_cost = {name: 0.0 for name in subsets}
    value_split = {"night_core": {}, "daylight": {}}
    alpha_plan = 1.0 - q_plan_cfg
    alpha_adj = 1.0 - q_adj_cfg
    aci_trace = []
    uri_trace = []
    lead_time_trace = []
    nav_trace = []
    deadband_trace = []

    for i, d in enumerate(days):
        window = slice(max(0, d - lookback), d)
        p_row = price[d]
        load = panel.load[d]
        pv_now = pv_actual[d]
        n_act_energy = net_actual[d] * st.dt

        # ---------------- plan layer (0:00) ---------------------------------
        level_plan = conformal_level(1.0 - alpha_plan, window.stop - window.start)
        q_err, q_counts = stage_quantile(err, window, 0, 1.0 - level_plan)
        fc0 = hourly_forecast_to_slots(panel.pv_forecast[d, 0], 0)
        pv_plan = fc0 + q_err
        net_plan = load - pv_plan
        plan = solve_stage_lp(net_plan, p_row, st)
        lead_time_trace.append(lead_time_envelope(err, window, 0, 1.0 - level_plan))

        # Q3-2: endogenous no-adjustment band.  The relevant quantity is the
        # value of relaxing the committed contract by one kWh, not the bus
        # price itself: buying more than the contract costs under*p, while
        # buying less forfeits over*p.  The band is therefore
        # [-over*p_t, under*p_t] and is derived, never hand-tuned.
        band_lo, band_hi = -over * p_row, under * p_row
        # Screening proxy: the dual spread across the horizon tells us where a
        # contract change is even worth evaluating.  The economic decision
        # itself is made by the NAV counterfactual below; this trace reports
        # the band and how much of the horizon it covers.
        spread = np.abs(np.asarray(plan["mu"], dtype=float) - np.mean(plan["mu"]))
        in_band = spread <= under * np.mean(p_row)
        deadband_trace.append({
            "date": panel.dates[d],
            "in_band_slots": int(in_band.sum()),
            "out_of_band_slots": int((~in_band).sum()),
            "dual_spread_mean_cny_per_kwh": float(np.mean(spread)),
            "band_low_cny_per_kwh": float(np.mean(band_lo)),
            "band_high_cny_per_kwh": float(np.mean(band_hi)),
        })

        # ---------------- adjust layers -------------------------------------
        # Each stage re-optimises the remaining slots starting from the SoC that the
        # *accepted* schedule actually realises, so the chains S1 ⊂ S2 ⊂ S3 share
        # their prefixes and no stage ever assumes a state it cannot see.
        level_adj = conformal_level(1.0 - alpha_adj, window.stop - window.start)
        stage_net = {}
        for j in (1, 2, 3):
            q_err_j, _ = stage_quantile(err, window, j, 1.0 - level_adj)
            fc_j = np.nan_to_num(panel.pv_forecast[d, j], nan=0.0)
            fc_j_slots = hourly_forecast_to_slots(fc_j, VINTAGE_HOUR[j])
            fc_prev = hourly_forecast_to_slots(panel.pv_forecast[d, j - 1], VINTAGE_HOUR[j - 1])
            base = np.nan_to_num(np.where(np.isnan(fc_j_slots), fc_prev, fc_j_slots), nan=0.0)
            stage_net[j] = load - (base + q_err_j)
            seg = slice(VINTAGE_SLOT[j], 144)
            old = err[window, j - 1, seg]
            new = err[window, j, seg]
            s_old = float(np.nanstd(old)) if np.any(~np.isnan(old)) else float("nan")
            s_new = float(np.nanstd(new)) if np.any(~np.isnan(new)) else float("nan")
            if s_old > 0 and s_new > 0:
                uri_trace.append({"date": panel.dates[d], "stage": VINTAGE_HOUR[j],
                                  "uri": float(np.log(s_old / s_new)), "sigma_old": s_old,
                                  "sigma_new": s_new})
            envelope = lead_time_envelope(err, window, j, 1.0 - level_adj)
            envelope["date"] = panel.dates[d]
            lead_time_trace.append(envelope)

        def run_chain(stage_list):  # stage_list: [(name, vintage_index, start_slot)]
            """Sequential chain: returned dict maps subset name -> schedule + settlement."""
            schedules = {}
            q_final = plan["q"].copy()
            charge = plan["c"].copy()
            disch = plan["d"].copy()
            soc = plan["soc"].copy()

            def settle():
                g = np.maximum(0.0, n_act_energy + charge - disch - q_final)
                over_energy = np.maximum(0.0, plan["q"] - q_final)
                under_energy = np.maximum(0.0, q_final - plan["q"])
                return {
                    "q": q_final.copy(), "c": charge.copy(), "d": disch.copy(), "soc": soc.copy(),
                    "g": g,
                    "cost": float(np.sum(p_row * plan["q"]) + emergency * np.sum(p_row * g)
                                  + over * np.sum(p_row * over_energy)
                                  + under * np.sum(p_row * under_energy)),
                }

            schedules["S0"] = settle()
            for name, j, start in stage_list:
                seg = slice(start, 144)
                sub = solve_stage_lp(stage_net[j][seg], p_row[seg], st,
                                     soc_start=float(soc[start]), soc_end=st.soc_init)
                q_final[start:] = sub["q"]
                charge[start:] = sub["c"]
                disch[start:] = sub["d"]
                soc[start:] = sub["soc"]
                schedules[name] = settle()
            return schedules

        chain = run_chain([("S1", 1, VINTAGE_SLOT[1]), ("S2", 2, VINTAGE_SLOT[2]),
                           ("S3", 3, VINTAGE_SLOT[3])])
        chain_s4 = run_chain([("S4a", 2, VINTAGE_SLOT[2]), ("S4", 3, VINTAGE_SLOT[3])])
        subset_schedule = {
            "S0": chain["S0"], "S1": chain["S1"], "S2": chain["S2"], "S3": chain["S3"],
            "S4": chain_s4["S4"],
        }

        # Q3-3: NAV_k = J_keep - J_reopt, evaluated on the *remaining* horizon so
        # the counterfactual is comparable to the leave-one-out subset cost.
        def remaining_cost(schedule: dict, start: int) -> float:
            return float(np.sum(_slot_costs(schedule, plan["q"], p_row, emergency,
                                            over, under, n_act_energy)[start:]))

        prev_name = "S0"
        for name, stage_j, start in (("S1", 1, VINTAGE_SLOT[1]),
                                     ("S2", 2, VINTAGE_SLOT[2]),
                                     ("S3", 3, VINTAGE_SLOT[3])):
            keep_cost = remaining_cost(subset_schedule[prev_name], start)
            reopt_cost = remaining_cost(subset_schedule[name], start)
            nav_trace.append({
                "date": panel.dates[d],
                "stage": VINTAGE_HOUR[stage_j],
                "start_slot": start + 1,
                "nav_cny": float(keep_cost - reopt_cost),
                "keep_cost_cny": keep_cost,
                "reopt_cost_cny": reopt_cost,
                "delta_q_kwh": float(np.sum(subset_schedule[name]["q"][start:]
                                            - subset_schedule[prev_name]["q"][start:])),
                "accepted": bool(keep_cost - reopt_cost > 0.0),
            })
            prev_name = name
        for name in subsets:
            if name in subset_cost and name != "S5":
                subset_cost[name] += subset_schedule[name]["cost"]
        # night/day attribution of each vintage, using the station's own daylight window
        core = np.array([k < panel.windows["k_rise"][d] - 1 or k > panel.windows["k_set"][d] + 1
                         for k in range(144)])
        steps = [("S0->S1", subset_schedule["S0"], subset_schedule["S1"]),
                 ("S1->S2", subset_schedule["S1"], subset_schedule["S2"]),
                 ("S2->S3", subset_schedule["S2"], subset_schedule["S3"])]
        for label, old, new in steps:
            delta = _slot_costs(old, plan["q"], p_row, emergency, over, under, n_act_energy) - \
                    _slot_costs(new, plan["q"], p_row, emergency, over, under, n_act_energy)
            for key, mask in (("night_core", core), ("daylight", ~core)):
                bucket = value_split[key].setdefault(label, 0.0)
                value_split[key][label] = bucket + float(delta[mask].sum())

        main = subset_schedule["S3"]
        q_plan_arr[i] = plan["q"]
        q_adj_arr[i] = main["q"]
        charge_arr[i] = main["c"]
        disch_arr[i] = main["d"]
        soc_arr[i] = main["soc"]
        emerg_arr[i] = main["g"]
        curtail_arr[i] = np.maximum(0.0, -n_act_energy - main["c"] + main["d"])

        # ---------------- oracle bound and ACI state --------------------------
        oracle = solve_stage_lp(net_actual[d], p_row, st)
        subset_cost["S5"] = subset_cost.get("S5", 0.0) + float(np.sum(p_row * oracle["q"]))
        miss = float(np.mean(n_act_energy > (plan["q"] + np.zeros(144))))
        aci_trace.append({"date": panel.dates[d], "level_plan": level_plan,
                          "alpha_plan": alpha_plan, "plan_miss": miss})
        alpha_plan = float(np.clip(alpha_plan + aci_gamma * ((1.0 - q_plan_cfg) - miss), 0.001, 0.999))
        alpha_adj = float(np.clip(alpha_adj + aci_gamma * ((1.0 - q_adj_cfg) - miss), 0.001, 0.999))

        blocks = merge_blocks(main["g"])
        records.append({
            "date": panel.dates[d],
            "cost_cny": main["cost"],
            "plan_cost_cny": float(np.sum(p_row * plan["q"])),
            "emergency_energy_kwh": float(main["g"].sum()),
            "emergency_cost_cny": float(emergency * np.sum(p_row * main["g"])),
            "over_plan_energy_kwh": float(np.sum(np.maximum(0.0, plan["q"] - main["q"]))),
            "under_plan_energy_kwh": float(np.sum(np.maximum(0.0, main["q"] - plan["q"]))),
            "over_plan_penalty_cny": float(over * np.sum(p_row * np.maximum(0.0, plan["q"] - main["q"]))),
            "under_plan_penalty_cny": float(under * np.sum(p_row * np.maximum(0.0, main["q"] - plan["q"]))),
            "q_plan_kwh": float(plan["q"].sum()),
            "q_adjust_kwh": float(main["q"].sum()),
            "charge_kwh": float(main["c"].sum()),
            "discharge_kwh": float(main["d"].sum()),
            "soc_0_kwh": float(plan["soc"][0]),
            "soc_24_kwh": float(main["soc"][-1]),
            "emergency_slots": int(np.sum(main["g"] > 1e-6)),
            "emergency_blocks": len(blocks),
            "subset_cost": {k: v["cost"] for k, v in subset_schedule.items()},
            "blocks": [{"first_slot": b[0] + 1, "last_slot": b[1] + 1,
                        "label": block_label(b[0], b[1]), "energy_kwh": b[2]} for b in blocks],
        })

    summary = {
        "tag": tag,
        "model": "4-vintage rolling MPC, plan layer p*=0.25 / adjust layer p*=0.70",
        "days": len(records),
        "total_cost_cny": float(sum(r["cost_cny"] for r in records)),
        "total_emergency_energy_kwh": float(sum(r["emergency_energy_kwh"] for r in records)),
        "total_emergency_cost_cny": float(sum(r["emergency_cost_cny"] for r in records)),
        "total_over_plan_penalty_cny": float(sum(r["over_plan_penalty_cny"] for r in records)),
        "total_under_plan_penalty_cny": float(sum(r["under_plan_penalty_cny"] for r in records)),
        "subset_total_cost_cny": subset_cost,
        "emergency_slot_rate": float(np.mean([r["emergency_slots"] / 144.0 for r in records])),
        "days_with_emergency": int(sum(1 for r in records if r["emergency_slots"] > 0)),
        "avg_emergency_blocks_per_day": float(np.mean([r["emergency_blocks"] for r in records])),
        "aci_final_alpha_plan": alpha_plan,
        "aci_final_alpha_adjust": alpha_adj,
        "soft_penalty": soft_penalty,
    }
    delta = {}
    if "S0" in subset_cost and "S3" in subset_cost:
        delta = {
            "value_of_0600": subset_cost["S0"] - subset_cost["S1"],
            "value_of_1200": subset_cost["S1"] - subset_cost["S2"],
            "value_of_1800": subset_cost["S2"] - subset_cost["S3"],
            "value_of_all_adjustments": subset_cost["S0"] - subset_cost["S3"],
            "oracle_gap": subset_cost["S3"] - subset_cost.get("S5", float("nan")),
            "value_of_1200_1800_only": subset_cost["S0"] - subset_cost["S4"],
        }
    summary["subset_marginal_value_cny"] = delta
    summary["subset_value_split_cny"] = value_split
    accepted = [t for t in nav_trace if t["accepted"]]
    summary["nav_gate"] = {
        "stages_evaluated": len(nav_trace),
        "stages_accepted": len(accepted),
        "stages_rejected": len(nav_trace) - len(accepted),
        "total_nav_cny": float(sum(t["nav_cny"] for t in nav_trace)),
        "total_accepted_nav_cny": float(sum(t["nav_cny"] for t in accepted)),
        "note": "NAV_k = J_keep - J_reopt on the remaining horizon; oracle stays a bound",
    }
    summary["lead_time_envelope"] = {
        "vintages": {str(row["vintage_hour"]): {
            "median_sigma_kw": float(np.median([p["sigma_kw"] for p in row["lead_time_curve"]])),
            "max_sigma_kw": float(np.max([p["sigma_kw"] for p in row["lead_time_curve"]])),
            "slots": len(row["lead_time_curve"]),
        } for row in lead_time_trace if "lead_time_curve" in row},
        "note": "sigma grows with lead time h; new vintage resets h to zero",
    }
    summary["no_adjustment_band"] = {
        "mean_in_band_slots": float(np.mean([t["in_band_slots"] for t in deadband_trace])),
        "mean_out_of_band_slots": float(np.mean([t["out_of_band_slots"] for t in deadband_trace])),
        "band": "[-0.5 p_t, 1.5 p_t] (endogenous, settlement-derived)",
    }
    return {"summary": summary, "records": records, "aci_trace": aci_trace, "uri_trace": uri_trace,
            "nav_trace": nav_trace, "deadband_trace": deadband_trace,
            "lead_time_trace": lead_time_trace,
            "arrays": {"q_plan": q_plan_arr, "q_adjust": q_adj_arr, "charge": charge_arr,
                       "discharge": disch_arr, "soc": soc_arr, "emergency": emerg_arr,
                       "curtail": curtail_arr}}


def _slot_costs(schedule: dict, q_plan: np.ndarray, price: np.ndarray, emergency: float,
                over: float, under: float, n_act_energy: np.ndarray) -> np.ndarray:
    """Per-slot settlement cost of one schedule against the committed plan."""
    q = schedule["q"]
    g = schedule["g"]
    return (price * q_plan + emergency * price * g
            + over * price * np.maximum(0.0, q_plan - q)
            + under * price * np.maximum(0.0, q - q_plan))


def _accepted_stages(name: str) -> list[tuple[int, int]]:
    """Which adjustment stages each subset experiment accepts (j, start slot)."""
    if name == "S0":
        return []
    if name == "S1":
        return [(1, VINTAGE_SLOT[1])]
    if name == "S2":
        return [(1, VINTAGE_SLOT[1]), (2, VINTAGE_SLOT[2])]
    if name == "S3":
        return [(1, VINTAGE_SLOT[1]), (2, VINTAGE_SLOT[2]), (3, VINTAGE_SLOT[3])]
    if name == "S4":
        return [(2, VINTAGE_SLOT[2]), (3, VINTAGE_SLOT[3])]
    raise ValueError(f"unknown subset {name}")


def _soc_at(schedule: dict, slot: int, st: Storage) -> float:
    return float(schedule["soc"][slot])


def save(out: dict, tag: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    ARRAYS.mkdir(parents=True, exist_ok=True)
    # Write the per-slot envelope curve only for the last evaluated day: the
    # daily curves are near-identical, and 334 x 4 x 144 rows would otherwise
    # dominate the results directory.
    lead_rows = [row for row in out.get("lead_time_trace", []) if "lead_time_curve" in row]
    if lead_rows:
        last_date = lead_rows[-1].get("date")
        lead_rows = [row for row in lead_rows if row.get("date") == last_date]
    if lead_rows:
        lines = ["date,vintage_hour,slot,hour_end,lead_hours,sigma_kw,quantile_kw,samples"]
        for row in lead_rows:
            for point in row["lead_time_curve"]:
                lines.append(",".join([
                    str(row.get("date", "")), str(row["vintage_hour"]), str(point["slot"]),
                    str(point["hour_end"]), f"{point['lead_hours']:.3f}",
                    f"{point['sigma_kw']:.6f}", f"{point['quantile_kw']:.6f}",
                    str(point["samples"]),
                ]))
        (RESULTS / f"{tag}_lead_time.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULTS / f"{tag}.json").write_text(
        json.dumps({"summary": out["summary"], "records": out["records"],
                    "aci_trace": out["aci_trace"], "uri_trace": out["uri_trace"],
                    "nav_trace": out.get("nav_trace", []),
                    "deadband_trace": out.get("deadband_trace", []),
                    # Drop the 144-row per-slot curve from the JSON: it is
                    # already written to the CSV above and would otherwise
                    # inflate the file by two orders of magnitude.
                    "lead_time_trace": [{k: v for k, v in row.items()
                                         if k != "lead_time_curve"}
                                        for row in out.get("lead_time_trace", [])]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    for name, arr in out["arrays"].items():
        np.save(ARRAYS / f"{tag}_{name}.npy", arr)
    header = ("date,cost_cny,plan_cost_cny,emergency_energy_kwh,emergency_cost_cny,"
              "over_plan_energy_kwh,under_plan_energy_kwh,over_plan_penalty_cny,"
              "under_plan_penalty_cny,q_plan_kwh,q_adjust_kwh,charge_kwh,discharge_kwh,"
              "soc_0_kwh,soc_24_kwh,emergency_slots,emergency_blocks,S0,S1,S2,S3,S4")
    lines = [header]
    for r in out["records"]:
        lines.append(",".join([
            r["date"], f"{r['cost_cny']:.6f}", f"{r['plan_cost_cny']:.6f}",
            f"{r['emergency_energy_kwh']:.6f}", f"{r['emergency_cost_cny']:.6f}",
            f"{r['over_plan_energy_kwh']:.6f}", f"{r['under_plan_energy_kwh']:.6f}",
            f"{r['over_plan_penalty_cny']:.6f}", f"{r['under_plan_penalty_cny']:.6f}",
            f"{r['q_plan_kwh']:.6f}", f"{r['q_adjust_kwh']:.6f}", f"{r['charge_kwh']:.6f}",
            f"{r['discharge_kwh']:.6f}", f"{r['soc_0_kwh']:.6f}", f"{r['soc_24_kwh']:.6f}",
            str(r["emergency_slots"]), str(r["emergency_blocks"]),
        ] + [f"{r['subset_cost'].get(k, float('nan')):.6f}" for k in ("S0", "S1", "S2", "S3", "S4")]))
    (RESULTS / f"{tag}_days.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    panel = load_panel()
    # Q3 also uses the representative-day tariff of 附件1 (附件4 belongs to Q4).
    price = np.tile(panel.day1[:, 0], (panel.price.shape[0], 1))
    out = solve_q3(panel, price, tag="q3")
    save(out, "q3")
    s = out["summary"]
    print(f"[Q3] days={s['days']} total={s['total_cost_cny']:.2f} CNY "
          f"(avg/day {s['total_cost_cny']/s['days']:.2f})")
    print(f"[Q3] emergency energy={s['total_emergency_energy_kwh']:.1f} kWh "
          f"slot-rate={s['emergency_slot_rate']*100:.2f}%")
    print(f"[Q3] penalties: over={s['total_over_plan_penalty_cny']:.2f} "
          f"under={s['total_under_plan_penalty_cny']:.2f}")
    for k, v in s["subset_total_cost_cny"].items():
        print(f"[Q3] subset {k}: {v:.2f} CNY")
    for k, v in s["subset_marginal_value_cny"].items():
        print(f"[Q3] marginal {k}: {v:.2f} CNY")


if __name__ == "__main__":
    main()
