"""问题1：用附件1 的代表性日算最优计划购电策略（附件1 + 附录1）。

本身是个线性规划：命题1 说明最优解不会同时充放电，所以不需要 0-1 变量；
再用离散 DP 独立解一遍做互证。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from code.common.load_clean import ROOT, load_panel
from code.optimize.dp_ops import solve_soc_dp
from code.optimize.lp_common import Storage, solve_stage_lp

RESULTS = ROOT / "results"


def baselines(price: np.ndarray, load: np.ndarray, pv: np.ndarray, dt: float) -> dict:
    """B0 all-purchase, B1 PV netting (no storage), B2 PV self-use + curtailment."""
    b0 = float(np.sum(price * load) * dt)
    # B1 = PV netting with a credit for surplus (no curtailment, no storage)
    b1 = float(np.sum(price * (load - pv)) * dt)
    # B2 = PV self-consumption + curtailment, still without storage
    surplus = np.maximum(0.0, pv - load)
    b2 = float(np.sum(price * np.maximum(0.0, load - pv)) * dt)
    return {
        "B0_all_purchase_cny": b0,
        "B1_pv_netting_cny": b1,
        "B2_pv_selfuse_curtail_no_storage_cny": b2,
        "pv_energy_kwh": float(np.sum(pv) * dt),
        "load_energy_kwh": float(np.sum(load) * dt),
        "surplus_pv_energy_kwh": float(np.sum(surplus) * dt),
        "pv_value_at_slot_price_cny": float(np.sum(price * pv) * dt),
        "b0_minus_pv_value_cny": b0 - float(np.sum(price * pv) * dt),
    }


def solve_q1(panel, dp_steps=(120.0, 60.0, 30.0, 15.0)) -> dict:
    st = Storage.from_config(panel.config)
    price = panel.day1[:, 0]
    load = panel.day1[:, 1]
    pv = panel.day1[:, 2]
    net = load - pv

    lp = solve_stage_lp(net, price, st)

    # Proposition 1 check: no simultaneous charge/discharge at the optimum.
    simultaneous = float(np.minimum(lp["c"], lp["d"]).max())

    dp_runs = []
    for step in dp_steps:
        dp = solve_soc_dp(net, price, st, step)
        rel = abs(dp.cost - lp["cost"]) / lp["cost"]
        dp_runs.append({"step_kwh": step, "states": dp.states, "cost_cny": dp.cost,
                        "rel_error_vs_lp": float(rel),
                        "grid_energy_kwh": float(dp.grid_energy.sum()),
                        "charge_kwh": float(dp.charge_energy.sum()),
                        "discharge_kwh": float(dp.discharge_energy.sum())})

    # Daily conservation identity from tasks/Q1.md.  It is an audit equation,
    # not a second optimization constraint.
    lhs_purchase = float(lp["q"].sum())
    rhs_purchase = float(np.sum(net) * st.dt + lp["curtail"].sum()
                         + (1.0 - st.eta_c * st.eta_d) * lp["c"].sum())
    identity_residual = lhs_purchase - rhs_purchase

    # KKT deadband diagnostics.  Bound-active actions are allowed at a
    # threshold; the audit therefore reports, rather than rejects, them.
    # Unit discipline: the LP duals are CNY per kWh of slot energy, while the
    # price is CNY per kWh of power * time.  Multiplying by dt puts both sides
    # on the same footing as the dual of the slot-balance constraint.
    lambda_value = np.asarray(lp["lambda"], dtype=float)
    charge_floor = st.dt * st.eta_c * lambda_value
    discharge_floor = st.dt * lambda_value / st.eta_d
    deadband = (price >= charge_floor - 1e-7) & (price <= discharge_floor + 1e-7)
    interior_deadband = deadband & (lp["c"] <= 1e-7) & (lp["d"] <= 1e-7)
    action_in_deadband = deadband & ((lp["c"] > 1e-7) | (lp["d"] > 1e-7))
    # A KKT violation would be a strictly interior action inside the bands.
    # Actions riding a power bound are legitimate, so separate the two cases.
    at_charge_bound = np.isclose(lp["c"], st.max_charge_energy, atol=1e-6)
    at_discharge_bound = np.isclose(lp["d"], st.max_discharge_energy, atol=1e-6)
    interior_violation = action_in_deadband & ~at_charge_bound & ~at_discharge_bound

    energy = {
        # LP variables are already slot energies (kWh) — never multiply by dt again.
        "purchase_kwh": float(lp["q"].sum()),
        "charge_kwh": float(lp["c"].sum()),
        "discharge_kwh": float(lp["d"].sum()),
        "curtail_kwh": float(lp["curtail"].sum()),
    }
    energy["cycling_kwh"] = energy["charge_kwh"]

    night = panel.windows["k_rise"][0]  # 附件1 is a standalone day: use its own window
    return {
        "model": "single-stage LP (time-expanded network form) + proposition-1 relaxation",
        "cost_cny": float(lp["cost"]),
        "lp": {k: lp[k].tolist() for k in ("q", "c", "d", "soc", "mu", "lambda", "curtail")},
        "energy": energy,
        "simultaneous_charge_discharge_kw": simultaneous,
        "storage_lambda": lambda_value.tolist(),
        "energy_identity": {
            "purchase_lhs_kwh": lhs_purchase,
            "purchase_rhs_kwh": rhs_purchase,
            "residual_kwh": identity_residual,
            "tolerance_kwh": 1e-9,
        },
        "kkt_deadband": {
            "charge_threshold": charge_floor.tolist(),
            "discharge_threshold": discharge_floor.tolist(),
            "deadband_slots": int(deadband.sum()),
            "interior_deadband_no_action_slots": int(interior_deadband.sum()),
            "action_in_deadband_slots": int(action_in_deadband.sum()),
            "action_in_deadband_rate": float(action_in_deadband.sum() / max(deadband.sum(), 1)),
            "bound_active_actions_in_deadband": int((action_in_deadband & ~interior_violation).sum()),
            "interior_violation_slots": int(interior_violation.sum()),
        },
        "soc_min_observed": float(lp["soc"].min()),
        "soc_max_observed": float(lp["soc"].max()),
        "soc_terminal_kwh": float(lp["soc"][-1]),
        "dp": dp_runs,
        "baselines": baselines(price, load, pv, st.dt),
        "inputs": {"price": price.tolist(), "load": load.tolist(), "pv": pv.tolist()},
        "day1_window": {"k_rise": float(panel.windows["k_rise"][0]),
                        "k_set": float(panel.windows["k_set"][0])},
    }


def block_sum(values: np.ndarray, hours: int = 4) -> np.ndarray:
    """Sum the 10-minute values into 6 blocks of `hours` hours (kWh when values are kWh)."""
    per_block = hours * 6
    return values.reshape(-1, per_block).sum(axis=1)


def main() -> None:
    panel = load_panel()
    out = solve_q1(panel)
    st = Storage.from_config(panel.config)

    RESULTS.mkdir(exist_ok=True)
    q = np.array(out["lp"]["q"])          # kWh per slot
    c = np.array(out["lp"]["c"])          # kWh per slot
    d = np.array(out["lp"]["d"])          # kWh per slot
    soc = np.array(out["lp"]["soc"])
    mu = np.array(out["lp"]["mu"])
    price, load, pv = panel.day1[:, 0], panel.day1[:, 1], panel.day1[:, 2]

    charge_energy = c
    discharge_energy = d
    blocks_charge = block_sum(charge_energy)
    blocks_discharge = block_sum(discharge_energy)
    soc_blocks = np.array([soc[0], soc[24], soc[48], soc[72], soc[96], soc[120], soc[144]])

    table1_slots = {"10:00-10:10": 60, "12:00-12:10": 72, "14:00-14:10": 84,
                    "16:00-16:10": 96, "18:00-18:10": 108, "20:00-20:10": 120}
    table1 = {"periods": {k: float(q[v - 1]) for k, v in table1_slots.items()},
              "day_purchase_kwh": float(q.sum()),
              "day_cost_cny": float(np.sum(price * q))}
    out["table1"] = table1
    out["table2"] = {
        "blocks": [f"{4*i}:00-{4*i+4}:00" for i in range(6)],
        "charge_kwh": blocks_charge.tolist(),
        "discharge_kwh": blocks_discharge.tolist(),
        "soc_0_00_kwh": float(soc[0]),
        "soc_24_00_kwh": float(soc[-1]),
        "soc_checkpoints": soc_blocks.tolist(),
    }
    out["night_statistics"] = night_statistics(panel, q, c, d, st)

    (RESULTS / "q1.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    header = ("slot_index,slot_label,price_cny_per_kwh,load_kw,pv_kw,purchase_kwh,charge_kwh,"
              "discharge_kwh,soc_end_kwh,marginal_mu_cny_per_kwh,curtail_kwh")
    lines = [header]
    for k in range(144):
        lines.append(",".join([
            str(k + 1), panel.slots[k], f"{price[k]:.6f}", f"{load[k]:.6f}", f"{pv[k]:.6f}",
            f"{q[k]:.6f}", f"{charge_energy[k]:.6f}", f"{discharge_energy[k]:.6f}",
            f"{soc[k+1]:.6f}", f"{mu[k]:.6f}",
            f"{np.maximum(0.0, d[k] - c[k] - (load[k] - pv[k]) * st.dt):.6f}",
        ]))
    (RESULTS / "q1_slots.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[Q1] LP cost = {out['cost_cny']:.2f} CNY  (B2 baseline "
          f"{out['baselines']['B2_pv_selfuse_curtail_no_storage_cny']:.2f})")
    print(f"[Q1] purchase={out['energy']['purchase_kwh']:.1f} kWh "
          f"charge={out['energy']['charge_kwh']:.1f} kWh discharge={out['energy']['discharge_kwh']:.1f}")
    print(f"[Q1] SOC range [{out['soc_min_observed']:.1f}, {out['soc_max_observed']:.1f}] "
          f"terminal={out['soc_terminal_kwh']:.1f} simultaneous={out['simultaneous_charge_discharge_kw']:.2e}")
    for run in out["dp"]:
        print(f"[Q1] DP de={run['step_kwh']:.0f} kWh states={run['states']:4d} "
              f"cost={run['cost_cny']:.2f} rel_err={run['rel_error_vs_lp']:.2e}")


def night_statistics(panel, q, c, d, st) -> dict:
    """Share of charging/discharging that happens in the core-night window."""
    k_rise = int(panel.windows["k_rise"][0])
    k_set = int(panel.windows["k_set"][0])
    core = np.array([k < k_rise - 1 or k > k_set + 1 for k in range(144)])
    shoulder = np.array([k in (k_rise - 1, k_rise, k_set, k_set + 1) for k in range(144)])
    charge_energy = c
    discharge_energy = d
    total_c = charge_energy.sum()
    total_d = discharge_energy.sum()
    return {
        "core_night_slots": int(core.sum()),
        "charge_share_core_night": float(charge_energy[core].sum() / total_c) if total_c else 0.0,
        "discharge_share_core_night": float(discharge_energy[core].sum() / total_d) if total_d else 0.0,
        "charge_share_shoulder": float(charge_energy[shoulder].sum() / total_c) if total_c else 0.0,
        "discharge_share_shoulder": float(discharge_energy[shoulder].sum() / total_d) if total_d else 0.0,
    }


if __name__ == "__main__":
    main()
