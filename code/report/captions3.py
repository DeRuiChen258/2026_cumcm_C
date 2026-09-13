"""图注权威源（依据 figures/CAPTIONS_3.md 的 25 张图规格）。

本模块只描述"画什么、为什么这么画"，不含任何求解逻辑。凡 CAPTIONS_3.md
要求但本仓库没有实测支撑的指标（1000 次 Monte Carlo、尚未实现的
A2/M1–M4 架构），一律在注记里显式说明替代口径，不编造数值。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigSpec:
    no: int          # 1..25，对应 CAPTIONS_3.md 的 Fig 编号
    slug: str        # 输出文件名前缀
    title_en: str    # 英文标题（图内使用）
    title_cn: str    # 中文标题（图注使用）
    caption_cn: str  # 中文图注，含数据口径与替代说明
    group: str       # 所属部分，便于审查完整性


def _spec(no, slug, title_en, title_cn, caption_cn, group) -> FigSpec:
    return FigSpec(no=no, slug=slug, title_en=title_en, title_cn=title_cn,
                   caption_cn=caption_cn, group=group)


FIG_SPECS: list[FigSpec] = [
    _spec(1, "hierarchical_architecture",
          "Hierarchical Multi-Time-Scale Energy Management Architecture",
          "多时间尺度分层能量管理架构",
          "三层闭环：日前 Q1 凸图 LP 调度并抽取 KKT 对偶价格；滚动 Q3 依据 MaxEnt-ACI 预测边界做 "
          "NAV 再优化；实时 10 分钟 Q4 MT-RRC 在 BESS 上执行 Student-t Copula 场景 → CBF 安全过滤 "
          "→ DETC 事件触发。箭头标注 g_t^*、λ_t^*、NAV_k、u^safe、u^real。", "架构"),
    _spec(2, "value_cascade",
          "Information-Decision-Control Value Cascade",
          "信息—决策—控制价值递进",
          "四级价值递进：物理价值（确定性凸图 LP + KKT 影子价格）→ 风险价值（MaxEnt 分布、ACI "
          "自适应共形、报童临界分位）→ 信息价值（Lead-time 收缩、内生无调整带、NAV 再优化）→ "
          "控制价值（MT-RRC、Student-t Copula、CBF 不变集、DETC 事件触发）。", "架构"),
    _spec(3, "microgrid_sld",
          "Physical Topology of the Microgrid",
          "微电网物理拓扑单线图",
          "并网工业微电网单线图：交流母线 N_t 汇集光伏 v_t、BESS 内部节点 B_t（充放效率 η_c、η_d）、"
          "负荷 l_t 与 PCC 变压器至大电网 g_t；标注功率平衡变量与退化成本 κ_deg。", "Q1"),
    _spec(4, "pv_inverter_clipping",
          "PV Inverter Clipping and Power Envelope",
          "光伏逆变器截断与功率包络",
          "辐照度—交流出力响应曲线族（不同环境温度），红色虚线标出逆变器功率上限并着色标注截断"
          "饱和区。曲线为单二极管模型参数化示意，不替代附件实测功率。", "Q1"),
    _spec(5, "time_expanded_network",
          "Time-Expanded Generalized Network Flow of the BESS",
          "时域扩展广义网络流与 SOC 状态转移",
          "24 步有向无环广义网络流：节点为母线平衡 N_t 与储能状态 B_t，有向边为充电 c_t（效率 "
          "η_c）与放电 d_t（1/η_d），体现凸动态平衡与自放电衰减 γ。", "Q1"),
    _spec(6, "day_ahead_balance_soc",
          "Day-Ahead Energy Balance and SOC Trajectory",
          "日前能量平衡与 SOC 轨迹",
          "双面板 24 小时（144 个 10 分钟时段聚合为 1 小时）确定性调度：上板为光伏、网购电与放电"
          "对负荷与充电的平衡；下板为 SOC 轨迹并标注 [1200, 10800] kWh 边界与 "
          "E_T = E_0 = 6000 kWh。", "Q1"),
    _spec(7, "kkt_dual_deadband",
          "KKT Dual Shadow Price and Marginal Energy Value",
          "KKT 对偶影子价格与储能边际价值",
          "24 小时对偶轨迹：母线影子价格 μ_t 与分时电价 p_t 对比，储能状态边际价值 λ_t 单独成轴；"
          "阴影区标出充放电触发区间与无动作死区。注：本数据最优解未落入死区，死区以解析边界呈现。",
          "Q1"),
    _spec(8, "maxent_residual_fit",
          "Forecast Residual Distribution and MaxEnt Fit",
          "预测残差分布与 MaxEnt 拟合",
          "净负荷残差直方图上叠加极大熵指数族密度 p*(x) 与同方差正态参照。MaxEnt 由前四阶矩以凸"
          "对偶求解，334 个因果窗口的梯度中位数 1.9e-16、最大 1.2e-5。", "Q2"),
    _spec(9, "aci_fan_lower_bound",
          "ACI Conformal Fan and Defensive PV Lower Bound",
          "ACI 共形包络与防守型光伏下界",
          "代表性日 ACI 共形预测包络；粗红实线为报童 F* = 0.80 对应的物理下界 "
          "P_L = max(0, P̂ − r)，夜间严格截断为 0。下界取自 results/arrays/q2_pv_lower.npy。", "Q2"),
    _spec(10, "lead_time_expansion",
          "Lead-Time Dependent Uncertainty Expansion",
          "预测提前量相关的不确定性膨胀",
          "预测误差尺度随提前量 h 单调扩散的喇叭口曲线，四个发布时次分面；新预报到达时 h 归零、"
          "包络收缩。数据取自 results/q3_lead_time.csv（最后一个评估窗口）。", "Q2"),
    _spec(11, "coverage_calibration",
          "Coverage Calibration and ACI Path Identity",
          "覆盖率校准与 ACI 路径恒等式",
          "左板：名义分位与实际覆盖的对齐曲线（含 y = x 理想线）与 Q2 报童分位扫描；右板：ACI "
          "累积失效率随时间收敛轨迹。三档锚点实测偏差 −0.52 / −0.17 / −0.37 pp。", "Q2"),
    _spec(12, "newsvendor_pareto",
          "Risk-Cost Pareto under 5x Asymmetric Penalty",
          "5 倍非对称惩罚下的风险—成本前沿",
          "分位 F* 扫描下的期望成本与缺失率权衡；星标为报童最优 F* = 0.80，并给出 5 倍应急惩罚"
          "（C_u = 4p, C_o = p）的解析位置。", "Q2"),
    _spec(13, "tail_penalty_distribution",
          "Tail Penalty Distribution across Realized Days",
          "极端惩罚的分布形态（实测日样本）",
          "三种计划口径（确定性均值、无裕度、报童 p*=0.80 两阶段）在 334 个实测日上的应急购电惩罚"
          "分布小提琴图。注：CAPTIONS_3 原要求 1000 次 Monte Carlo，本仓库无合成场景文件，改用全年"
          "实测日的经验分布，口径更保守。", "Q2"),
    _spec(14, "info_funnel_shrinkage",
          "Sequential Information Revelation and Band Shrinkage",
          "多时次信息到达与预测带收缩",
          "0:00/6:00/12:00/18:00 四个信息节点的预测带收缩：新预报到达后提前量归零，不确定性带显著"
          "收窄；夜间段因光伏恒零而掩码。", "Q3"),
    _spec(15, "no_adaptation_zone",
          "Endogenous No-Adaptation Zone under Asymmetric Cost",
          "非对称成本下的内生无调整带",
          "分段线性的契约调整边际成本：下调段 −0.5p_t、上调段 +1.5p_t；次梯度区间 "
          "∂C^adj ∈ [−0.5p_t, 1.5p_t] 内保持契约不变，形成无人工阈值的阻尼带。", "Q3"),
    _spec(16, "adjustment_policy_regions",
          "Contract Adjustment Policy Region Map",
          "契约调整决策相图",
          "以边际收益 m_t 与实时电价 p_t 为坐标的三分区相图：下调区、无调整死带、上调区，边界由"
          "结算系数 0.5 / 1.5 导出。", "Q3"),
    _spec(17, "curi_vs_nav",
          "Forecast Accuracy Gain (CURI) vs Net Adjustment Value (NAV)",
          "预测精度增益与净调整价值",
          "散点展示预测精度增益 CURI_k = ln(σ_old/σ_new) 与净调整价值 NAV_k 的关系：精度提升并不"
          "自动带来正 NAV，需决策阈值过滤。含正 / 负 NAV 分色与零轴参考。", "Q3"),
    _spec(18, "nav_trigger_timeline",
          "Event-Driven NAV Trigger Trajectory",
          "事件驱动 NAV 触发轨迹",
          "全天脉冲柱状图给出各时次 NAV，虚线为执行阈值；标注采纳（REOPT）与拒绝（KEEP）点位，"
          "灰色区块标出夜间掩码时段（19:00–05:00）。", "Q3"),
    _spec(19, "subset_trajectories",
          "Contract and SOC Trajectories across Subsets S0-S5",
          "S0–S5 子集下的契约与 SOC 轨迹",
          "双面板对比 S0（仅 0:00）、S3（四时次）、S4（12:00+18:00）、S5（完美预报界）：上板为购电"
          "契约演化，下板为储能 SOC 参考轨迹。S5 仅作上界。", "Q3"),
    _spec(20, "mtrrc_layers",
          "MT-RRC Hierarchical Control and Time-Scale Mapping",
          "MT-RRC 分层控制与跨尺度映射",
          "1 小时 → 10 分钟映射：日前契约 g_t^final 与参考 S_t^ref 经接口层展开为 P_τ^contract 与 "
          "S_τ^ref，依次经 Student-t Copula 场景层、软 MPC 决策层、CBF 安全层与 DETC 执行层。",
          "Q4"),
    _spec(21, "student_t_copula",
          "Student-t Copula Joint High-Frequency Risk Scenarios",
          "Student-t Copula 高频联合风险场景",
          "三维散点给出（光伏残差、负荷残差、电价残差）联合场景，红色标出尾部共现簇（低光伏 + "
          "高负荷 + 电价尖峰）。拟合自由度 ν = 7，负荷—价格尾部相关系数 0.393。", "Q4"),
    _spec(22, "cbf_phase_space",
          "CBF Phase-Space Safety Filtering",
          "CBF 相空间安全过滤",
          "（储能电量 E_τ, 控制量 u_τ）相平面：阴影为满足 ḣ + γh ≥ 0 的安全不变集；箭头示意名义 "
          "MPC 指令被二次规划投影回安全域的拉回效应。", "Q4"),
    _spec(23, "detc_threshold_evolution",
          "DETC Execution and Dynamic Threshold Evolution",
          "DETC 执行与动态阈值演化",
          "双面板 144 步：上板为跟踪误差平方与动态阈值对比，脉冲标记实际下发时刻；下板为 BESS "
          "实际功率阶梯，显示 Hold 期休眠与切换次数下降。", "Q4"),
    _spec(24, "ablation_map",
          "Cost-Switching-Safety Multi-Objective Ablation (A0-A4)",
          "成本—开关—安全多目标消融",
          "气泡散点：横轴为 PCS 动作次数、纵轴为运行成本、气泡面积代表 SOC 越界深度。A0 全功能、"
          "A3 无 CBF（越界风险）、A4 周期触发（动作数最高）；A1/A2 为概念对照。", "Q4"),
    _spec(25, "architecture_parallel_coordinates",
          "System-Level Performance Envelope across Architectures",
          "系统级架构性能包络",
          "平行坐标：确定性基准、+MaxEnt-ACI、+NAV 信息价值、+MT-RRC 高频软 MPC、+CBF-DETC 闭环"
          "在成本、应急电费、CVaR、弃光、退化与开关次数六个归一化指标上的包络。注：M1–M4 为按实测"
          "指标构造的递进配置，非独立重复实验。", "总结"),
]


def by_no(no: int) -> FigSpec:
    for item in FIG_SPECS:
        if item.no == no:
            return item
    raise KeyError(f"no figure spec for Fig {no}")


def all_specs() -> list[FigSpec]:
    return list(FIG_SPECS)
